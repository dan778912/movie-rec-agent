"""Generate P2 agent training samples for the six action types.

Implements the action-supervision plan in the midterm report (section 3.4)
and the roadmap's P2 example construction. Each sample is one agent decision:
a rendered prompt (``src/profiles/agent_text``) plus the JSON action the model
should emit. Samples come from simulated episodes. An episode fixes a user, an
anchor time ``t`` inside the training period, and a visible profile built only
from ratings before ``t`` -- either the canonical profile or a masked one that
shows only the user's last few ratings. The user's training ratings at or after
``t`` are the held-out interactions that supervise each decision:

* ``assess_sufficiency`` / ``ask_clarification`` / ``evaluate_candidates`` /
  ``retrieve_candidates``: every action sequence within the budget (at most two
  template questions answered by the simulator from hidden pre-``t`` history,
  then at most one wider retrieval from 50 to 100 candidates) is scored by
  NDCG@10 against the held-out positives; the episode follows the sequence with
  the highest NDCG@10, preferring fewer actions on ties.
* ``rank_candidates``: held-out positives first, unlabeled movies in pool
  order, held-out explicit dislikes last.
* ``explain_recommendations``: template justifications citing only titles,
  genres, tags and answers shown in the prompt.

Splits ``train`` and ``dev`` (user-disjoint, for checkpoint selection) both use
training ratings only: validation and test targets never supply answers or
labels. No validation or test rating file is opened. Users in the configured
evaluation lists are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from src.data import clarification_sim as sim
from src.data.build import _genome_tags, _movies, _user_tags, csv_rows, sha256, write_jsonl
from src.data.p2_actions import validate_action, validate_episode
from src.profiles.agent_text import PROMPT_VERSION, display_tags, render_prompt, serialize_action
from src.profiles.text import movie_document, user_profile

SCHEMA_VERSION = "p2-sample-v1"
EPISODE_SCHEMA_VERSION = "p2-train-episode-v1"
INDEX_VERSION = "p2-train-feature-index-v1"
DATA = Path("data/processed/ml25m-global-v1")
RAW = Path("data/raw/ml-25m")
CONFIG = Path("configs/p2_samples_v1.json")
OUTPUT = Path("data/processed/p2-v1")
CONSUMED = ("catalog.csv", "movie_documents.jsonl", "statistics.json", "train_ratings.csv")
SPLITS = ("train", "dev")
YEAR = re.compile(r"\((\d{4})(?:[-–]\d{0,4})?\)\s*$")


# --------------------------------------------------------------------------- metadata

def year_of(title: str) -> int | None:
    match = YEAR.search(title)
    return int(match.group(1)) if match else None


def load_metadata(raw: Path, data: Path, data_config: dict, validation_cutoff: int):
    """Movie metadata identical to the frozen build (titles, genres, tag slices)."""
    movies = _movies(raw / "movies.csv")
    catalog_rows = list(csv_rows(data / "catalog.csv"))
    catalog = {int(row["movie_id"]) for row in catalog_rows}
    popularity = {int(row["movie_id"]): int(row["train_positives"]) for row in catalog_rows}
    genome = _genome_tags(raw, data_config["genome_tags_per_movie"])
    user_tags = _user_tags(raw, validation_cutoff, catalog, data_config["user_tags_per_movie"])
    meta = {}
    for movie_id, (title, genres) in movies.items():
        year = year_of(title)
        meta[movie_id] = {
            "movie_id": movie_id, "title": title, "raw_genres": genres,
            "genres": [] if genres == "(no genres listed)" else genres.split("|"),
            "genome_tags": genome.get(movie_id, []), "user_tags": user_tags.get(movie_id, []),
            "tags": display_tags(genome.get(movie_id, []), user_tags.get(movie_id, [])),
            "year": year, "decade": f"{year // 10 * 10}s" if year else None,
        }
    return meta, catalog, popularity


def check_documents(data: Path, meta: dict, catalog: set[int]) -> int:
    """Recomputed metadata must reproduce the frozen movie documents exactly."""
    count = 0
    with (data / "movie_documents.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            m = meta[record["movie_id"]]
            expected = movie_document(m["title"], m["raw_genres"], m["genome_tags"], m["user_tags"])
            if record["movie_id"] not in catalog or record["text"] != expected:
                raise ValueError(f"Movie document mismatch for {record['movie_id']}")
            count += 1
    if count != len(catalog):
        raise ValueError("Movie document count differs from catalog")
    return count


def display(meta: dict, movie_id: int) -> dict:
    m = meta[movie_id]
    return {"movie_id": movie_id, "title": m["title"], "genres": m["genres"], "tags": m["tags"]}


# --------------------------------------------------------------------------- vectors

def _add(acc: dict, vector: dict, weight: float) -> None:
    for key, value in vector.items():
        acc[key] = acc.get(key, 0.0) + weight * value


def _normalize(vector: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vector.values()))
    return {k: v / norm for k, v in vector.items() if v} if norm else {}


def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class FeatureSpace:
    """IDF-weighted genre/tag/decade vectors and a small inverted-index retriever.

    This is a deterministic, dependency-free stand-in for the online vector index,
    used only to construct training pools and labels.
    """

    def __init__(self, meta: dict, catalog: set[int], popularity: dict, config: dict):
        self.meta = meta
        self.weights = config["feature_weights"]
        raw = {mid: self._raw(mid) for mid in sorted(catalog)}
        df = Counter(f for feats in raw.values() for f in feats)
        n = len(catalog)
        self.idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
        self.default_idf = math.log(n + 1) + 1.0
        self.cache: dict[int, dict] = {}
        # Retrieval postings use pivoted length normalization (norm ** alpha, as in
        # BM25-style scoring) so metadata-poor movies do not win on genre overlap
        # alone; label similarities keep exact unit-vector cosines.
        alpha = config["retrieval_length_norm"]
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for mid in sorted(catalog):
            raw_weights = {f: v * self.idf[f] for f, v in raw[mid].items()}
            norm = math.sqrt(sum(v * v for v in raw_weights.values())) ** alpha or 1.0
            for feature, weight in raw_weights.items():
                self.postings[feature].append((mid, weight / norm))
        top = math.log1p(max(popularity.values()))
        self.prior = {mid: config["popularity_prior"] * math.log1p(popularity[mid]) / top for mid in catalog}
        self.by_popularity = sorted(catalog, key=lambda mid: (-popularity[mid], mid))

    def popular_in_genres(self, genres: list[str], k: int, exclude: set[int]) -> list[int]:
        """Most popular unwatched movies in the given genres (a realistic off-profile retrieval)."""
        wanted = set(genres)
        return [mid for mid in self.by_popularity
                if mid not in exclude and wanted & set(self.meta[mid]["genres"])][:k]

    def _raw(self, movie_id: int) -> dict:
        m, w = self.meta[movie_id], self.weights
        feats = {"g:" + g: w["genre"] for g in m["genres"]}
        for tag in m["genome_tags"]:
            feats["t:" + tag.lower()] = max(feats.get("t:" + tag.lower(), 0.0), w["genome_tag"])
        for tag in m["user_tags"]:
            feats["t:" + tag.lower()] = max(feats.get("t:" + tag.lower(), 0.0), w["user_tag"])
        if m["decade"]:
            feats["d:" + m["decade"]] = w["decade"]
        return feats

    def vector(self, movie_id: int) -> dict:
        if movie_id not in self.cache:
            raw = self._raw(movie_id)
            self.cache[movie_id] = _normalize({f: v * self.idf.get(f, self.default_idf) for f, v in raw.items()})
        return self.cache[movie_id]

    def concept(self, prefix: str, values: list[str]) -> dict:
        return _normalize({prefix + v: self.idf.get(prefix + v, self.default_idf) for v in values})

    def retrieve(self, query: dict, k: int, exclude: set[int]) -> list[int]:
        terms = sorted(query.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:40]
        scores: dict[int, float] = {}
        for feature, weight in terms:
            for mid, value in self.postings.get(feature, ()):
                scores[mid] = scores.get(mid, 0.0) + weight * value
        ranked = heapq.nlargest(k, ((s + self.prior[mid], -mid) for mid, s in scores.items()
                                    if s > 0 and mid not in exclude))
        result = [-neg for _, neg in ranked]
        if len(result) < k:
            chosen = set(result)
            result += [mid for mid in self.by_popularity if mid not in exclude and mid not in chosen][:k - len(result)]
        return result


def fact_contribution(space: FeatureSpace, fact: dict) -> tuple[dict, float]:
    kind = fact["kind"]
    if kind == "liked_movie":
        return space.vector(fact["movie_id"]), 1.0
    if kind == "disliked_movie":
        return space.vector(fact["movie_id"]), -0.5
    if kind == "favorite_genres":
        return space.concept("g:", fact["genres"]), 1.0
    if kind == "disliked_genre":
        return space.concept("g:", [fact["genre"]]), -0.5
    if kind == "themes":
        return space.concept("t:", fact["tags"]), 1.0
    return space.concept("d:", [fact["decade"]]), 0.5


def state_vector(space: FeatureSpace, state: dict) -> dict:
    acc: dict[str, float] = {}
    for item in state["liked"]:
        _add(acc, space.vector(item["movie_id"]), 1.0)
    for item in state["disliked"]:
        _add(acc, space.vector(item["movie_id"]), -0.5)
    for turn in state["turns"]:
        if turn["informative"]:
            vector, weight = fact_contribution(space, turn["fact"])
            _add(acc, vector, weight)
    return _normalize(acc)


# --------------------------------------------------------------------------- profiles

def profile_item(meta: dict, row: tuple) -> dict:
    movie_id, rating, ts = row
    m = meta[movie_id]
    return {"movie_id": movie_id, "title": m["title"], "rating": rating, "timestamp": ts,
            "genres": m["genres"], "tags": m["tags"]}


def select_profile(meta: dict, rows: list[tuple], likes: int, dislikes: int) -> tuple[list[dict], list[dict]]:
    """Exactly the canonical build rule (src/data/build.py::_export_profiles)."""
    liked = sorted((r for r in rows if r[1] >= 4), key=lambda r: (-r[1], -r[2], r[0]))[:likes]
    disliked = sorted((r for r in rows if r[1] <= 2.5), key=lambda r: (r[1], -r[2], r[0]))[:dislikes]
    return [profile_item(meta, r) for r in liked], [profile_item(meta, r) for r in disliked]


def visible_rows(history: list[tuple], catalog: set[int], level: str, sparse_k: int | None) -> list[tuple]:
    """Canonical level shows the full history; sparse level the last k catalog ratings (a live user)."""
    if level == "canonical":
        return history
    return [row for row in history if row[0] in catalog][-sparse_k:]


def clarifications(state: dict) -> list[dict]:
    return [{"question": t["question"], "answer": t["answer"]} for t in state["turns"]]


def prompt_text(state: dict) -> str:
    return state["profile_text"] + " " + " ".join(t["answer"] for t in state["turns"])


# --------------------------------------------------------------------------- query and explanations

def retrieval_query(space: FeatureSpace, state: dict, vector: dict) -> tuple[str, dict, dict]:
    """Query text built only from preferences displayed in the prompt, plus its execution vector."""
    turns = [t for t in state["turns"] if t["informative"]]
    shown_genres = {g for item in state["liked"] for g in item["genres"]}
    shown_tags = {t.lower() for item in state["liked"] for t in item["tags"]}
    titles, decade, avoid = [], None, []
    for turn in reversed(turns):
        fact = turn["fact"]
        if fact["kind"] == "liked_movie":
            titles.append((fact["title"], fact["movie_id"]))
        elif fact["kind"] == "favorite_genres":
            shown_genres.update(fact["genres"])
        elif fact["kind"] == "themes":
            shown_tags.update(fact["tags"])
        elif fact["kind"] == "favorite_decade":
            decade = fact["decade"]
        elif fact["kind"] == "disliked_genre":
            avoid.append(fact["genre"])
    titles += [(item["title"], item["movie_id"]) for item in state["liked"]]
    titles = list(dict.fromkeys(titles))[:4]
    ranked = sorted(((v, k) for k, v in vector.items() if v > 0), key=lambda p: (-p[0], p[1]))
    genres = [k[2:] for _, k in ranked if k.startswith("g:") and k[2:] in shown_genres and k[2:] not in avoid][:3]
    themes = [k[2:] for _, k in ranked if k.startswith("t:") and k[2:] in shown_tags][:5]
    parts = []
    if titles:
        parts.append("Movies similar to " + ", ".join(t for t, _ in titles) + ".")
    if genres:
        parts.append("Genres: " + ", ".join(genres) + ".")
    if themes:
        parts.append("Themes: " + ", ".join(themes) + ".")
    if decade:
        parts.append(f"Preferably from the {decade}.")
    if avoid:
        parts.append("Avoid: " + ", ".join(avoid) + ".")
    text = " ".join(parts) or "Highly rated popular movies."
    acc: dict[str, float] = {}
    for _, movie_id in titles:
        _add(acc, space.vector(movie_id), 1.0)
    _add(acc, space.concept("g:", genres) if genres else {}, 1.0)
    _add(acc, space.concept("t:", themes) if themes else {}, 1.0)
    _add(acc, space.concept("d:", [decade]) if decade else {}, 0.5)
    _add(acc, space.concept("g:", avoid) if avoid else {}, -0.5)
    terms = {"titles": [t for t, _ in titles], "genres": genres, "themes": themes, "decade": decade, "avoid": avoid}
    return text, _normalize(acc), terms


def _quote_join(values: list[str]) -> str:
    return sim._join([f'"{v}"' for v in values])


def explanation_evidence(candidate: dict, decade: str | None, state: dict) -> dict:
    """Overlap between a displayed candidate and displayed preferences (the only allowed evidence)."""
    # Tags containing double quotes cannot be cited unambiguously inside quoted text.
    genres, tags = candidate["genres"], [t.lower() for t in candidate["tags"] if '"' not in t]
    best, best_key = None, None
    for index, item in enumerate(state["liked"]):
        shared_genres = [g for g in genres if g in item["genres"]]
        item_tags = {t.lower() for t in item["tags"]}
        shared_tags = list(dict.fromkeys(t for t in tags if t in item_tags))
        key = (2 * len(shared_tags) + len(shared_genres), item["rating"], -index)
        if (shared_genres or shared_tags) and (best_key is None or key > best_key):
            best, best_key = (item, shared_genres, shared_tags), key
    answer_genres, answer_tags, answer_decade = [], [], None
    for turn in state["turns"]:
        fact = turn["fact"] if turn["informative"] else None
        if not fact:
            continue
        if fact["kind"] == "favorite_genres":
            answer_genres += [g for g in fact["genres"] if g in genres and g not in answer_genres]
        elif fact["kind"] == "themes":
            answer_tags += [t for t in fact["tags"] if t in tags and t not in answer_tags]
        elif fact["kind"] == "favorite_decade" and fact["decade"] == decade:
            answer_decade = decade
    evidence = {"movie_id": candidate["movie_id"], "candidate_genres": genres,
                "answer_genres": answer_genres, "answer_tags": answer_tags, "answer_decade": answer_decade}
    if best:
        item, shared_genres, shared_tags = best
        evidence.update({"liked_movie_id": item["movie_id"], "liked_title": item["title"],
                         "liked_rating": item["rating"], "shared_genres": shared_genres, "shared_tags": shared_tags})
    else:
        evidence.update({"liked_movie_id": None, "liked_title": None, "liked_rating": None,
                         "shared_genres": [], "shared_tags": []})
    return evidence


def explanation_text(evidence: dict) -> str:
    parts = []
    if evidence["liked_movie_id"] is not None:
        traits = []
        if evidence["shared_genres"]:
            traits.append("is listed under " + sim._join(evidence["shared_genres"]))
        if evidence["shared_tags"]:
            traits.append("is tagged " + _quote_join(evidence["shared_tags"]))
        parts.append(f"Like {evidence['liked_title']}, which you rated {evidence['liked_rating']:g}/5, "
                     f"it {' and '.join(traits)}.")
    if evidence["answer_genres"]:
        parts.append(f"It fits your stated preference for {sim._join(evidence['answer_genres'])} movies.")
    if evidence["answer_tags"]:
        parts.append(f"It is tagged {_quote_join(evidence['answer_tags'])}, which you said you are drawn to.")
    if evidence["answer_decade"]:
        parts.append(f"It is from the {evidence['answer_decade']}, the decade you said you enjoy most.")
    if not parts:
        listed = sim._join(evidence["candidate_genres"]) if evidence["candidate_genres"] else "no genre"
        parts.append(f"It is listed under {listed}, but it shares no listed genre or tag "
                     "with the movies you rated highly.")
    return " ".join(parts)


# --------------------------------------------------------------------------- episodes

def hash_key(seed, split: str, user_id: int) -> int:
    return int(hashlib.sha256(f"{seed}:{split}:{user_id}".encode()).hexdigest()[:16], 16)


def feasible_anchors(history: list[tuple], catalog: set[int], config: dict) -> list[int]:
    """Indices i where ratings before history[i].timestamp form a usable history."""
    n = len(history)
    catalog_prefix = [0]
    positive_prefix = [0]
    for mid, rating, _ in history:
        catalog_prefix.append(catalog_prefix[-1] + (mid in catalog))
        positive_prefix.append(positive_prefix[-1] + (mid in catalog and rating >= 4))
    horizon = config["future_horizon_ratings"]
    need_catalog = max(config["train"]["sparse_rating_counts"])
    anchors = []
    for i in range(config["min_history_before_anchor"], n):
        if history[i][2] == history[i - 1][2]:
            continue  # anchors fall on timestamp boundaries; ties stay in the future side
        if catalog_prefix[i] < need_catalog:
            continue
        if positive_prefix[min(n, i + horizon)] - positive_prefix[i] >= config["min_future_positives"]:
            anchors.append(i)
    return anchors


def ndcg_at_10(ranked: list[int], relevant: set[int]) -> float:
    """Binary NDCG@10, the project's primary metric."""
    dcg = sum(1.0 / math.log2(i + 2) for i, mid in enumerate(ranked[:10]) if mid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return dcg / ideal if ideal else 0.0


def rank_pool(space: FeatureSpace, vector: dict, pool: list[int]) -> list[int]:
    """Order a pool by the preference vector (ties keep pool order)."""
    order = sorted(range(len(pool)), key=lambda i: (-_dot(vector, space.vector(pool[i])), i))
    return [pool[i] for i in order]


class EpisodeBuilder:
    def __init__(self, meta, catalog, space: FeatureSpace, config: dict, data_config: dict):
        self.meta, self.catalog, self.space = meta, catalog, space
        self.config, self.data_config = config, data_config
        genre_counts = Counter(g for mid in catalog for g in meta[mid]["genres"])
        self.common_genres = [g for g, _ in sorted(genre_counts.items(), key=lambda p: (-p[1], p[0]))[:12]]

    def plan(self, state: dict, facts: dict, pool: list[int], watched: set[int], relevant: set[int]):
        """Midterm section 3.4 labelling rule.

        Enumerate every action sequence within the budget (0..max_questions
        clarifying questions, each answered by the simulator from pre-anchor
        history, then optionally one wider retrieval), rank the resulting pool
        with the preference vector, score NDCG@10 against held-out training-period
        positives, and choose the highest NDCG@10, preferring fewer actions on ties.
        """
        cfg, space = self.config, self.space
        sequences = [((), state)]
        frontier = sequences
        for _ in range(cfg["max_questions"]):
            grown = []
            for intents, current in frontier:
                asked = [sim.question_for(i, 0) for i in intents]
                for intent in sim.INTENTS:
                    fact = sim.next_fact(intent, facts, asked)
                    if fact is not None:
                        child = {**current, "turns": current["turns"] + [{"informative": True, "fact": fact}]}
                        grown.append((intents + (intent,), child))
            sequences = sequences + grown
            frontier = grown
        results = []
        for intents, current in sequences:
            vector = state_vector(space, current)
            results.append({"questions": list(intents), "retrieve": False,
                            "ndcg": ndcg_at_10(rank_pool(space, vector, pool), relevant)})
            if cfg["max_retrievals"]:
                _, query_vector, _ = retrieval_query(space, current, vector)
                added = space.retrieve(query_vector, cfg["retrieval_k"], watched | set(pool))
                results.append({"questions": list(intents), "retrieve": True,
                                "ndcg": ndcg_at_10(rank_pool(space, vector, pool + added), relevant)})
        for result in results:
            result["actions"] = len(result["questions"]) + result["retrieve"]
        best = min(results, key=lambda r: (-r["ndcg"], r["actions"],
                                           [sim.INTENTS.index(i) for i in r["questions"]], r["retrieve"]))
        return best, results

    def build(self, split: str, user_id: int, episode_index: int, history: list[tuple], future: list[tuple],
              anchor: int, boundary: int, level: str, sparse_k: int | None) -> tuple[list[dict], dict]:
        cfg, space, meta = self.config, self.space, self.meta
        rng = random.Random(f"{cfg['seed']}:{split}:{user_id}:{episode_index}")
        episode_id = f"{split}:{user_id}:{episode_index}"
        if any(ts >= anchor for _, _, ts in history) or any(not anchor <= ts < boundary for _, _, ts in future):
            raise AssertionError(f"Temporal boundary violated in {episode_id}")
        watched = {row[0] for row in history}
        shown_rows = visible_rows(history, self.catalog, level, sparse_k)
        liked, disliked = select_profile(meta, shown_rows, self.data_config["liked_movies_per_profile"],
                                         self.data_config["disliked_movies_per_profile"])
        state = {"liked": liked, "disliked": disliked, "turns": [], "profile_text": user_profile(liked, disliked)}
        visible_ids = {item["movie_id"] for item in liked + disliked}
        facts = sim.build_fact_pool(history, meta, visible_ids, state["profile_text"], cutoff=anchor)
        future_ratings = {mid: (rating, ts) for mid, rating, ts in future}
        relevant = {r[0] for r in future if r[0] in self.catalog and r[1] >= 4}
        positives = [r for r in future if r[0] in self.catalog and r[1] >= 4][:cfg["future_positives"]]
        negatives = [r for r in future if r[0] in self.catalog and r[1] <= 2.5][:cfg["future_negatives"]]
        samples, events = [], []
        step = 0

        def emit(task, action, candidates=None, basis=None, explain_ids=None):
            nonlocal step
            validate_action(action, candidate_ids=None if task != "rank_candidates" else candidates,
                            explain_ids=explain_ids)
            shown = [display(meta, mid) for mid in candidates] if candidates is not None else None
            samples.append({
                "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
                "sample_id": f"{episode_id}:{step:02d}:{task}", "episode_id": episode_id, "step": step,
                "task": task, "split": split, "user_id": user_id,
                "state": {"profile_text": state["profile_text"], "clarifications": clarifications(state),
                          "candidate_movie_ids": list(candidates) if candidates is not None else None},
                "prompt": render_prompt(task, state["profile_text"], clarifications(state), shown),
                "target": action, "completion": serialize_action(action),
                "provenance": {"anchor_timestamp": anchor, "boundary_timestamp": boundary, "level": level,
                               "label_basis": basis or {}},
            })
            step += 1

        # Initial pool: retrieval from the starting profile (the B3 analogue), or a simulated miss.
        start_vector = state_vector(space, state)
        regime = "offtarget" if rng.random() < cfg["offtarget_pool_fraction"] else "profile"
        if regime == "profile":
            pool = space.retrieve(start_vector, cfg["pool_size"], watched)
        else:
            weakest = sorted(self.common_genres, key=lambda g: (start_vector.get("g:" + g, 0.0), rng.random()))[:2]
            pool = space.popular_in_genres(weakest, cfg["pool_size"], watched | {r[0] for r in positives})
        initial_pool = list(pool)

        best, results = self.plan(state, facts, initial_pool, watched, relevant)

        def best_where(condition):
            values = [r["ndcg"] for r in results if condition(r)]
            return round(max(values), 6) if values else None

        # Clarification loop: follow the chosen question sequence.
        for k in range(len(best["questions"]) + 1):
            prefix = best["questions"][:k]
            ask = k < len(best["questions"])
            basis = {"ndcg_if_stop": best_where(lambda r: r["questions"] == prefix),
                     "ndcg_if_continue": best_where(lambda r: r["questions"][:k] == prefix and len(r["questions"]) > k),
                     "chosen_questions": best["questions"], "chosen_ndcg": round(best["ndcg"], 6)}
            emit("assess_sufficiency", {"action": "assess_sufficiency", "sufficient": not ask}, basis=basis)
            events.append({"action": "assess_sufficiency", "sufficient": not ask})
            if not ask:
                break
            intent = best["questions"][k]
            asked = [t["question"] for t in state["turns"]]
            question = sim.question_for(intent, rng.randrange(3))
            emit("ask_clarification", {"action": "ask_clarification", "question": question},
                 basis={"intent": intent, "chosen_questions": best["questions"]})
            reply = sim.answer_question(question, facts, asked, prompt_text(state))
            if reply["intent"] != intent or not reply["informative"]:
                raise AssertionError(f"Simulator did not answer intended question in {episode_id}")
            if reply["fact"]["source_max_timestamp"] >= anchor:
                raise AssertionError(f"Clarification answer used a future rating in {episode_id}")
            state["turns"].append({"question": question, **reply})
            events.append({"action": "ask_clarification", "question": question, "answer": reply["answer"],
                           "answer_source": sim.SIMULATOR_VERSION, "intent": intent,
                           "informative": True, "new_fact": reply["new_fact"],
                           "source_movie_ids": reply["source_movie_ids"]})

        # Pool decision: follow the chosen retrieval choice; the retrieval budget then ends the loop.
        vector = state_vector(space, state)
        same = lambda r: r["questions"] == best["questions"]
        pool_basis = {"ndcg_keep_pool": best_where(lambda r: same(r) and not r["retrieve"]),
                      "ndcg_after_retrieval": best_where(lambda r: same(r) and r["retrieve"]),
                      "future_positive_hits": sum(1 for mid in pool if mid in relevant)}
        emit("evaluate_candidates", {"action": "evaluate_candidates", "pool_sufficient": not best["retrieve"]},
             candidates=pool, basis=pool_basis)
        events.append({"action": "evaluate_candidates", "candidate_movie_ids": list(pool),
                       "pool_sufficient": not best["retrieve"]})
        if best["retrieve"]:
            query, query_vector, terms = retrieval_query(space, state, vector)
            added = space.retrieve(query_vector, cfg["retrieval_k"], watched | set(pool))
            emit("retrieve_candidates", {"action": "retrieve_candidates", "query": query}, candidates=pool,
                 basis={"query_terms": terms, "added_movie_ids": added,
                        "added_future_positive_hits": sum(1 for mid in added if mid in relevant)})
            events.append({"action": "retrieve_candidates", "added_movie_ids": added, "index_version": INDEX_VERSION})
            pool = pool + added
            emit("evaluate_candidates", {"action": "evaluate_candidates", "pool_sufficient": True},
                 candidates=pool, basis={"retrieval_budget_used": True, "chosen_ndcg": round(best["ndcg"], 6)})
            events.append({"action": "evaluate_candidates", "candidate_movie_ids": list(pool), "pool_sufficient": True})

        # Ranking over the final pool with held-out positives/dislikes injected for supervision.
        ranked_pool = list(pool)
        absent_pos = [r[0] for r in positives if r[0] not in pool]
        absent_neg = [r[0] for r in negatives if r[0] not in pool]
        injected = (rng.sample(absent_pos, min(cfg["inject_positives"], len(absent_pos)))
                    + absent_neg[:cfg["inject_negatives"]])
        for mid in injected:
            ranked_pool.insert(rng.randrange(len(ranked_pool) + 1), mid)
        rating_of = {mid: future_ratings[mid] for mid in ranked_pool if mid in future_ratings}
        top = sorted((m for m in ranked_pool if rating_of.get(m, (0,))[0] >= 4),
                     key=lambda m: (-rating_of[m][0], rating_of[m][1], m))
        bottom = [m for m in ranked_pool if m in rating_of and rating_of[m][0] <= 2.5]
        middle = [m for m in ranked_pool if m not in set(top) | set(bottom)]
        order = top + middle + bottom
        emit("rank_candidates", {"action": "rank_candidates", "movie_ids": order}, candidates=ranked_pool,
             basis={"injected_movie_ids": injected, "positive_movie_ids": top, "negative_movie_ids": bottom,
                    "future_ratings": {str(m): rating_of[m][0] for m in sorted(rating_of)}})
        events.append({"action": "rank_candidates", "movie_ids": order, "injected_movie_ids": injected})

        shown_top = order[:cfg["explain_k"]]
        evidence = [explanation_evidence(display(meta, m), meta[m]["decade"], state) for m in shown_top]
        explanations = [{"movie_id": e["movie_id"], "text": explanation_text(e)} for e in evidence]
        emit("explain_recommendations", {"action": "explain_recommendations", "explanations": explanations},
             candidates=shown_top, explain_ids=shown_top, basis={"evidence": evidence})
        events.append({"action": "explain_recommendations", "explanations": explanations})
        summary = validate_episode(events, initial_pool)
        episode = {
            "schema_version": EPISODE_SCHEMA_VERSION, "episode_id": episode_id, "split": split, "user_id": user_id,
            "anchor_timestamp": anchor, "boundary_timestamp": boundary, "level": level, "sparse_k": sparse_k,
            "pool_regime": regime, "history_count": len(history),
            "history_latest_timestamp": max(ts for _, _, ts in history),
            "visible_movie_ids": [item["movie_id"] for item in liked + disliked],
            "profile_text": state["profile_text"],
            "relevant_count": len(relevant),
            "future_positive_movie_ids": [r[0] for r in positives],
            "future_positive_timestamps": [r[2] for r in positives],
            "future_negative_movie_ids": [r[0] for r in negatives],
            "initial_candidate_movie_ids": initial_pool, "events": events,
            "final_candidate_movie_ids": pool, "injected_movie_ids": injected,
            "termination_reason": "recommend",
            "plan": {"rule": "max NDCG@10, fewest actions on ties (midterm section 3.4)",
                     "chosen_questions": best["questions"], "chosen_retrieve": best["retrieve"],
                     "chosen_ndcg": round(best["ndcg"], 6), "sequences_evaluated": len(results),
                     "no_action_ndcg": round(results[0]["ndcg"], 6),
                     "decided_by_tie_break": any(r is not best and r["ndcg"] == best["ndcg"] for r in results)},
            "summary": summary,
            "sample_ids": [s["sample_id"] for s in samples],
        }
        return samples, episode


# --------------------------------------------------------------------------- selection and driver

def load_user_list(path: Path) -> set[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value["user_ids"] if isinstance(value, dict) else value
    if not all(type(i) is int for i in ids):
        raise ValueError(f"{path} must contain integer user IDs")
    return set(ids)


def stream_histories(path: Path, cutoff: int):
    previous = -1
    for user_id, rows in groupby(csv_rows(path), key=lambda r: int(r["user_id"])):
        if user_id <= previous:
            raise ValueError("train_ratings.csv must be sorted by user_id")
        previous = user_id
        history = [(int(r["movie_id"]), float(r["rating"]), int(r["timestamp"])) for r in rows]
        if any(ts >= cutoff for _, _, ts in history):
            raise AssertionError(f"Training rating at or after validation cutoff for user {user_id}")
        yield user_id, history


def choose_level(rng: random.Random, split_config: dict) -> tuple[str, int | None]:
    if rng.random() < split_config["canonical_fraction"]:
        return "canonical", None
    return "sparse", rng.choice(split_config["sparse_rating_counts"])


def provenance(paths: list[Path]) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], capture_output=True, text=True)
    return {"git_commit": revision.stdout.strip() or "unknown", "worktree_dirty": bool(status.stdout.strip()),
            "source_sha256": {str(p): sha256(p) for p in paths}}


def generate(data: Path, raw: Path, output: Path, config: dict) -> dict:
    started = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != config["data_version"]:
        raise ValueError("Processed data version differs from config")
    for name in CONSUMED:
        if sha256(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    stats = json.loads((data / "statistics.json").read_text(encoding="utf-8"))
    val_cutoff = stats["validation_cutoff_unix"]
    data_config = manifest["config"]
    meta, catalog, popularity = load_metadata(raw, data, data_config, val_cutoff)
    documents = check_documents(data, meta, catalog)
    space = FeatureSpace(meta, catalog, popularity, config)
    builder = EpisodeBuilder(meta, catalog, space, config, data_config)
    excluded = set()
    for path in config["exclude_user_lists"]:
        excluded |= load_user_list(Path(path))
    seed = config["seed"]

    # One user-level draw from training-period data; dev users are a disjoint subset of it.
    heap: list[tuple] = []  # max-heap on hash key via negation
    limit = config["train"]["users"] + config["dev"]["users"]
    population = Counter()
    for user_id, history in stream_histories(data / "train_ratings.csv", val_cutoff):
        population["training_users"] += 1
        if user_id in excluded:
            population["excluded_evaluation_users"] += 1
            continue
        key = hash_key(seed, "users", user_id)
        if len(heap) == limit and -heap[0][0] <= key:
            continue
        anchors = feasible_anchors(history, catalog, config)
        if not anchors:
            population["examined_users_without_feasible_anchor"] += 1
            continue
        entry = (-key, user_id, history, anchors)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)
    chosen = sorted(heap, key=lambda entry: hash_key(seed, "dev", entry[1]))
    assignment = {"dev": chosen[:config["dev"]["users"]], "train": chosen[config["dev"]["users"]:]}

    output.mkdir(parents=True, exist_ok=True)
    report_splits = {}
    for split in SPLITS:
        split_config = config[split]
        samples_path = output / f"p2_samples_{split}.jsonl"
        episodes_path = output / f"p2_episodes_{split}.jsonl"
        tasks, labels, levels, regimes, intents = Counter(), Counter(), Counter(), Counter(), Counter()
        answers, plans = Counter(), Counter()
        prompt_chars, completion_chars = defaultdict(list), defaultdict(list)
        episodes = []
        users = sorted((uid, history, anchors) for _, uid, history, anchors in assignment[split])

        def records():
            for user_id, history, anchors in users:
                user_rng = random.Random(f"{seed}:{split}:{user_id}:anchors")
                picks = sorted(user_rng.sample(anchors, min(split_config["episodes_per_user"], len(anchors))))
                for index, i in enumerate(picks):
                    level, sparse_k = choose_level(random.Random(f"{seed}:{split}:{user_id}:{index}:level"),
                                                   split_config)
                    samples, episode = builder.build(split, user_id, index, history[:i], history[i:],
                                                     history[i][2], val_cutoff, level, sparse_k)
                    episodes.append(episode)
                    levels[level if level == "canonical" else f"sparse_{sparse_k}"] += 1
                    regimes[episode["pool_regime"]] += 1
                    plan = episode["plan"]
                    plans[f"questions_{len(plan['chosen_questions'])}"] += 1
                    plans["retrieve" if plan["chosen_retrieve"] else "no_retrieve"] += 1
                    plans["no_action_ndcg_zero"] += plan["no_action_ndcg"] == 0
                    plans["chosen_ndcg_zero"] += plan["chosen_ndcg"] == 0
                    plans["decided_by_tie_break"] += plan["decided_by_tie_break"]
                    for event in episode["events"]:
                        if event["action"] == "ask_clarification":
                            intents[event["intent"]] += 1
                            answers["new_fact" if event["new_fact"] else "restated_fact"] += 1
                    for sample in samples:
                        task = sample["task"]
                        tasks[task] += 1
                        target = sample["target"]
                        if task == "assess_sufficiency":
                            labels[f"{task}:{'sufficient' if target['sufficient'] else 'insufficient'}"] += 1
                        elif task == "evaluate_candidates":
                            labels[f"{task}:{'sufficient' if target['pool_sufficient'] else 'insufficient'}"] += 1
                        prompt_chars[task].append(len(sample["prompt"]))
                        completion_chars[task].append(len(sample["completion"]))
                        yield sample

        write_jsonl(samples_path, records())
        write_jsonl(episodes_path, episodes)

        def lengths(values):
            values = sorted(values)
            return {"p50": values[len(values) // 2], "p95": values[int(len(values) * 0.95)], "max": values[-1]}

        report_splits[split] = {
            "users": len({e["user_id"] for e in episodes}), "episodes": len(episodes),
            "samples": sum(tasks.values()), "tasks": dict(sorted(tasks.items())),
            "decision_labels": dict(sorted(labels.items())), "levels": dict(sorted(levels.items())),
            "pool_regimes": dict(sorted(regimes.items())), "plans": dict(sorted(plans.items())),
            "clarification_intents": dict(sorted(intents.items())), "clarification_answers": dict(answers),
            "control_events": dict(sorted(Counter(e["summary"]["control_events"] for e in episodes).items())),
            "prompt_chars": {t: lengths(v) for t, v in sorted(prompt_chars.items())},
            "completion_chars": {t: lengths(v) for t, v in sorted(completion_chars.items())},
            "samples_sha256": sha256(samples_path), "episodes_sha256": sha256(episodes_path),
        }

    sources = sorted(Path("src/data").glob("*.py")) + [Path("src/profiles/text.py"), Path("src/profiles/agent_text.py")]
    report = {
        "schema_version": SCHEMA_VERSION, "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION, "simulator_version": sim.SIMULATOR_VERSION, "index_version": INDEX_VERSION,
        "config": config, "data_version": manifest["version"],
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "consumed_sha256": {name: manifest["file_sha256"][name] for name in CONSUMED},
        "test_files_read": False, "documents_verified": documents,
        "validation_cutoff_unix": val_cutoff, "labels_from": "training-period ratings only",
        "excluded_users": len(excluded), "population": dict(population), "splits": report_splits,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        **provenance(sources),
        "limitations": [
            "Clarification answers come from a deterministic simulator over pre-anchor ratings, not real users.",
            "Action labels compare NDCG@10 of a genre/tag/decade preference-vector ranker, not the P2 model itself.",
            "Training pools come from a dependency-free feature index, not Member 3's vector index or B3 pools.",
            "Off-target pools simulate retrieval misses; ranking pools inject held-out positives and dislikes.",
            "Explanations are templates citing displayed overlap; they are grounded but not human-written.",
            "Genome tags remain static transductive metadata, as in the Phase 1 data contract.",
        ],
    }
    (output / "p2_samples.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-users", type=int, help="Override training user count for a smoke run")
    parser.add_argument("--dev-users", type=int, help="Override dev user count for a smoke run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name, value in (("train", args.train_users), ("dev", args.dev_users)):
        if value is not None:
            if value < 1:
                parser.error("user overrides must be positive")
            config[name]["users"] = value
    report = generate(args.data, args.raw, args.output, config)
    print(json.dumps({k: report[k] for k in ("splits", "runtime_seconds", "population")}, indent=2))


if __name__ == "__main__":
    main()
