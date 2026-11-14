"""Data-quality and temporal-boundary checks for Member 1's Phase 2 handoff.

Checks, independently of the generator's own assertions:

* base artifacts: frozen checksums, profile cutoffs, split time ranges, catalog
  metadata gaps, profile observations relevant to training/evaluation parity;
* P2 samples: every episode is replayed from source ratings truncated at the
  episode's anchor (history) and split boundary (labels) and must reproduce the
  stored samples byte for byte; histories, simulated answers, candidates, labels,
  action grammar, episode order, and explanation grounding are also asserted
  directly;
* catalog seed: coverage, canonical text, and training-only aggregates.

Writes a JSON report and exits nonzero when any check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.data import clarification_sim as sim
from src.data.build import csv_rows, sha256
from src.data.p2_actions import ActionError, parse_action, validate_action, validate_episode
from src.data.p2_samples import (CONFIG, DATA, OUTPUT, RAW, EpisodeBuilder, FeatureSpace, display,
                                 load_metadata, load_user_list, select_profile, stream_histories,
                                 visible_rows)
from src.data.verify import verify
from src.profiles.agent_text import ACTIONS, render_prompt, serialize_action
from src.profiles.text import user_profile

MAX_ERRORS = 50


class Errors:
    def __init__(self):
        self.count = Counter()
        self.examples: list[str] = []

    def add(self, kind: str, message: str) -> None:
        self.count[kind] += 1
        if len(self.examples) < MAX_ERRORS:
            self.examples.append(f"{kind}: {message}")

    def require(self, condition: bool, kind: str, message: str) -> None:
        if not condition:
            self.add(kind, message)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


# --------------------------------------------------------------------------- base data

def time_range(path: Path) -> dict:
    low = high = None
    rows = 0
    for row in csv_rows(path):
        ts = int(row["timestamp"])
        low = ts if low is None or ts < low else low
        high = ts if high is None or ts > high else high
        rows += 1
    return {"rows": rows, "min_timestamp": low, "max_timestamp": high}


def base_checks(data: Path, meta: dict, catalog: set[int], errors: Errors) -> dict:
    stats = json.loads((data / "statistics.json").read_text(encoding="utf-8"))
    val_cutoff, test_cutoff = stats["validation_cutoff_unix"], stats["test_cutoff_unix"]
    result = {"verify": verify(data), "cutoffs": {"validation": val_cutoff, "test": test_cutoff,
                                                  "validation_utc": stats["validation_cutoff_utc"],
                                                  "test_utc": stats["test_cutoff_utc"]}}
    ranges = {split: time_range(data / f"{split}_ratings.csv") for split in ("train", "validation", "test")}
    errors.require(ranges["train"]["max_timestamp"] < val_cutoff, "split_range", "training rating after cutoff")
    errors.require(val_cutoff <= ranges["validation"]["min_timestamp"] and
                   ranges["validation"]["max_timestamp"] < test_cutoff, "split_range", "validation range")
    errors.require(ranges["test"]["min_timestamp"] >= test_cutoff, "split_range", "test rating before cutoff")
    result["split_time_ranges"] = ranges
    titles = Counter(meta[m]["title"] for m in catalog)
    result["catalog_quality"] = {
        "movies": len(catalog),
        "without_genres": sum(not meta[m]["genres"] for m in catalog),
        "without_genome_tags": sum(not meta[m]["genome_tags"] for m in catalog),
        "without_user_tags": sum(not meta[m]["user_tags"] for m in catalog),
        "without_any_tag": sum(not meta[m]["tags"] for m in catalog),
        "without_year": sum(meta[m]["year"] is None for m in catalog),
        "duplicate_title_strings": sum(c for c in titles.values() if c > 1),
    }
    observations = {}
    for split in ("validation", "test"):
        total = out_of_catalog = short_likes = no_likes = 0
        for record in jsonl(data / f"{split}_profiles.jsonl"):
            total += 1
            liked = record["liked_movies"]
            out_of_catalog += any(item["movie_id"] not in catalog for item in liked)
            short_likes += len(liked) < 10
            no_likes += not liked
        observations[split] = {"profiles": total, "with_out_of_catalog_liked_movie": out_of_catalog,
                               "with_fewer_than_10_liked": short_likes, "with_no_liked": no_likes}
    result["profile_observations"] = observations
    return result


# --------------------------------------------------------------------------- explanation grounding

GENRES = ("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western")
_LIKE = re.compile(r"^Like (?P<title>.+?), which you rated (?P<rating>\d+(?:\.\d+)?)/5, it ")
_DECADE = re.compile(r"from the (\d{4}s)")


def grounding_problems(text: str, candidate: dict, decade: str | None, state: dict) -> list[str]:
    """Independent check that every cited title, rating, genre, tag, and decade is displayed."""
    problems = []
    liked = {item["title"]: item for item in state["liked"]}
    facts = [t["fact"] for t in state["turns"] if t["informative"]]
    answer_genres = {g for f in facts if f["kind"] == "favorite_genres" for g in f["genres"]}
    answer_tags = {t for f in facts if f["kind"] == "themes" for t in f["tags"]}
    answer_decades = {f["decade"] for f in facts if f["kind"] == "favorite_decade"}
    cand_tags = {t.lower() for t in candidate["tags"]}
    rest = text
    source_genres, source_tags = set(answer_genres), set(answer_tags)
    match = _LIKE.match(text)
    if match:
        item = liked.get(match["title"])
        if item is None:
            problems.append(f"cited title not in visible liked movies: {match['title']}")
        else:
            if float(match["rating"]) != item["rating"]:
                problems.append("cited rating differs from profile")
            source_genres |= set(item["genres"])
            source_tags |= {t.lower() for t in item["tags"]}
        rest = text[match.end():]
    elif text.startswith("Like "):
        problems.append("unparseable liked-movie citation")
    for tag in re.findall(r'"([^"]+)"', rest):
        if tag not in cand_tags or tag not in source_tags:
            problems.append(f"tag not displayed for both candidate and source: {tag}")
    no_quotes = re.sub(r'"[^"]*"', "", rest)
    fallback = "shares no listed genre or tag" in no_quotes
    for genre in GENRES:
        if re.search(rf"(?<![\w-]){re.escape(genre)}(?![\w-])", no_quotes):
            if genre not in candidate["genres"] or (not fallback and genre not in source_genres):
                problems.append(f"genre not displayed for both candidate and source: {genre}")
    for cited in _DECADE.findall(rest):
        if cited != decade or cited not in answer_decades:
            problems.append(f"decade not supported: {cited}")
    return problems


# --------------------------------------------------------------------------- sample checks

def replay_state(episode: dict, history: list[tuple], meta: dict, catalog: set[int], data_config: dict):
    rows = visible_rows(history, catalog, episode["level"], episode["sparse_k"])
    liked, disliked = select_profile(meta, rows, data_config["liked_movies_per_profile"],
                                     data_config["disliked_movies_per_profile"])
    state = {"liked": liked, "disliked": disliked, "turns": [], "profile_text": user_profile(liked, disliked)}
    visible_ids = {item["movie_id"] for item in liked + disliked}
    facts = sim.build_fact_pool(history, meta, visible_ids, state["profile_text"], cutoff=episode["anchor_timestamp"])
    return state, facts


def sample_checks(data: Path, raw_output: Path, meta: dict, catalog: set[int], config: dict,
                  data_config: dict, errors: Errors) -> dict:
    stats = json.loads((data / "statistics.json").read_text(encoding="utf-8"))
    val_cutoff, test_cutoff = stats["validation_cutoff_unix"], stats["test_cutoff_unix"]
    popularity = {int(r["movie_id"]): int(r["train_positives"]) for r in csv_rows(data / "catalog.csv")}
    space = FeatureSpace(meta, catalog, popularity, config)
    builder = EpisodeBuilder(meta, catalog, space, config, data_config)
    excluded = set()
    for path in config["exclude_user_lists"]:
        excluded |= load_user_list(Path(path))
    official = {r["user_id"]: r["text"] for r in jsonl(data / "validation_profiles.jsonl")}
    episodes = {split: list(jsonl(raw_output / f"p2_episodes_{split}.jsonl")) for split in ("train", "validation")}
    users = {split: {e["user_id"] for e in eps} for split, eps in episodes.items()}
    errors.require(not users["train"] & users["validation"], "user_overlap", "train and validation users overlap")
    errors.require(not (users["train"] | users["validation"]) & excluded, "excluded_user", "evaluation user used")
    wanted = users["train"] | users["validation"]
    train_hist = {uid: h for uid, h in stream_histories(data / "train_ratings.csv", val_cutoff) if uid in wanted}
    val_rows = defaultdict(list)
    for row in csv_rows(data / "validation_ratings.csv"):
        uid = int(row["user_id"])
        if uid in users["validation"]:
            val_rows[uid].append((int(row["movie_id"]), float(row["rating"]), int(row["timestamp"])))

    result = {}
    for split in ("train", "validation"):
        samples = defaultdict(list)
        seen_ids = set()
        tasks = Counter()
        for sample in jsonl(raw_output / f"p2_samples_{split}.jsonl"):
            errors.require(sample["sample_id"] not in seen_ids, "duplicate_sample", sample["sample_id"])
            seen_ids.add(sample["sample_id"])
            errors.require(sample["split"] == split, "sample_split", sample["sample_id"])
            tasks[sample["task"]] += 1
            samples[sample["episode_id"]].append(sample)
        errors.require(set(tasks) == set(ACTIONS), "task_coverage", f"{split} tasks: {sorted(tasks)}")
        replayed = grounded = explanations = answers_checked = leak_checked = 0
        for episode in episodes[split]:
            eid, uid, anchor = episode["episode_id"], episode["user_id"], episode["anchor_timestamp"]
            source = train_hist.get(uid, [])
            past = [r for r in source if r[2] < anchor]
            if split == "train":
                errors.require(anchor < val_cutoff and episode["boundary_timestamp"] == val_cutoff,
                               "boundary", f"{eid} train anchor/boundary")
                future = [r for r in source if r[2] >= anchor]
            else:
                errors.require(anchor == val_cutoff and episode["boundary_timestamp"] == test_cutoff,
                               "boundary", f"{eid} validation anchor/boundary")
                errors.require(len(past) == len(source), "boundary", f"{eid} history must be all training ratings")
                future = val_rows[uid]
            errors.require(all(ts < anchor for _, _, ts in past) and len(past) == episode["history_count"]
                           and max(ts for _, _, ts in past) == episode["history_latest_timestamp"],
                           "history", f"{eid} history count/latest timestamp")
            errors.require(all(anchor <= ts < episode["boundary_timestamp"] for _, _, ts in future),
                           "future", f"{eid} future outside [anchor, boundary)")
            watched = {r[0] for r in past}
            future_by_id = {r[0]: r for r in future}
            expected_pos = [r[0] for r in future if r[0] in catalog and r[1] >= 4][:config["future_positives"]]
            errors.require(episode["future_positive_movie_ids"] == expected_pos, "future_labels", eid)
            errors.require(set(episode["visible_movie_ids"]) <= watched, "visible", f"{eid} visible movie not in history")
            pools = [episode["initial_candidate_movie_ids"], episode["final_candidate_movie_ids"]]
            pools += [ev["added_movie_ids"] for ev in episode["events"] if ev["action"] == "retrieve_candidates"]
            for pool in pools:
                errors.require(all(m in catalog and m not in watched for m in pool), "candidate",
                               f"{eid} candidate watched or outside catalog")
            errors.require(set(episode["injected_movie_ids"]) <= set(future_by_id), "injection",
                           f"{eid} injected movie is not a held-out rating")
            if split == "validation" and episode["level"] == "canonical":
                errors.require(episode["profile_text"] == official[uid], "canonical_profile", eid)
            try:
                validate_episode(episode["events"], episode["initial_candidate_movie_ids"])
            except ActionError as error:
                errors.add("episode_order", f"{eid}: {error}")

            # Simulated answers replayed from pre-anchor history only.
            state, facts = replay_state(episode, past, meta, catalog, data_config)
            errors.require(state["profile_text"] == episode["profile_text"], "profile_text", eid)
            future_titles = [meta[m]["title"] for m in episode["future_positive_movie_ids"]]
            asked = []
            for event in episode["events"]:
                if event["action"] != "ask_clarification":
                    continue
                reply = sim.answer_question(event["question"], facts, asked,
                                            state["profile_text"] + " " + " ".join(t["answer"] for t in state["turns"]))
                errors.require(reply["answer"] == event["answer"] and event["answer_source"] == sim.SIMULATOR_VERSION,
                               "answer_replay", eid)
                errors.require(set(reply["source_movie_ids"]) <= watched, "answer_source", f"{eid} answer outside history")
                asked.append(event["question"])
                state["turns"].append({"question": event["question"], **reply})
                answers_checked += 1
            visible_text = state["profile_text"] + " " + " ".join(t["answer"] for t in state["turns"])
            leak = [t for t in future_titles if t in visible_text and t not in {i["title"] for i in state["liked"]}]
            errors.require(not leak, "future_title_leak", f"{eid}: {leak[:2]}")
            leak_checked += 1

            # Explanation grounding, checked independently of the generator's evidence.
            explain = [ev for ev in episode["events"] if ev["action"] == "explain_recommendations"][0]
            for item in explain["explanations"]:
                movie_id = item["movie_id"]
                problems = grounding_problems(item["text"], display(meta, movie_id), meta[movie_id]["decade"], state)
                explanations += 1
                grounded += not problems
                for problem in problems:
                    errors.add("grounding", f"{eid}/{movie_id}: {problem}")

            # Samples: grammar, completion, prompt rendering.
            for sample in samples.get(eid, []):
                target = sample["target"]
                candidates = sample["state"]["candidate_movie_ids"]
                try:
                    validate_action(target, candidate_ids=candidates if sample["task"] == "rank_candidates" else None,
                                    explain_ids=candidates if sample["task"] == "explain_recommendations" else None)
                    errors.require(parse_action(sample["completion"]) == target and
                                   serialize_action(target) == sample["completion"], "completion", sample["sample_id"])
                except ActionError as error:
                    errors.add("action", f"{sample['sample_id']}: {error}")
                shown = [display(meta, m) for m in candidates] if candidates is not None else None
                expected_prompt = render_prompt(sample["task"], sample["state"]["profile_text"],
                                                sample["state"]["clarifications"], shown)
                errors.require(sample["prompt"] == expected_prompt, "prompt_render", sample["sample_id"])
                errors.require(sample["task"] == target["action"], "task_mismatch", sample["sample_id"])

            # Full replay from boundary-truncated inputs must reproduce the stored output.
            index = int(eid.rsplit(":", 1)[1])
            replay_samples, replay_episode = builder.build(split, uid, index, past, future, anchor,
                                                           episode["boundary_timestamp"], episode["level"],
                                                           episode["sparse_k"])
            errors.require(replay_episode == episode and replay_samples == samples.get(eid), "replay", eid)
            replayed += 1
        errors.require(set(samples) == {e["episode_id"] for e in episodes[split]}, "orphan_samples", split)
        result[split] = {"episodes": len(episodes[split]), "samples": sum(tasks.values()), "tasks": dict(sorted(tasks.items())),
                         "users": len(users[split]), "episodes_replayed": replayed,
                         "answers_replayed": answers_checked, "episodes_leak_checked": leak_checked,
                         "explanations": explanations, "explanations_grounded": grounded}
    return result


def seed_checks(data: Path, output: Path, catalog: set[int], errors: Errors) -> dict:
    path = output / "catalog_seed.jsonl"
    if not path.exists():
        errors.add("seed_missing", str(path))
        return {}
    rows = {int(r["movie_id"]): r for r in csv_rows(data / "catalog.csv")}
    docs = {r["movie_id"]: r["text"] for r in jsonl(data / "movie_documents.jsonl")}
    ids = []
    for record in jsonl(path):
        mid = record["movie_id"]
        ids.append(mid)
        row = rows.get(mid)
        errors.require(row is not None and record["text"] == docs.get(mid), "seed_text", str(mid))
        errors.require(row is not None and record["rating_count"] == int(row["train_ratings"])
                       and record["train_positives"] == int(row["train_positives"]), "seed_counts", str(mid))
        errors.require(0.5 <= record["rating_avg"] <= 5.0, "seed_average", str(mid))
        errors.require(bool(record["display_title"].strip()), "seed_title", str(mid))
    errors.require(sorted(ids) == sorted(catalog) and len(ids) == len(set(ids)), "seed_coverage", "seed IDs differ")
    return {"movies": len(ids), "sha256": sha256(path)}


def run(data: Path, raw: Path, output: Path, config: dict) -> dict:
    started = time.perf_counter()
    errors = Errors()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((data / "statistics.json").read_text(encoding="utf-8"))
    meta, catalog, _ = load_metadata(raw, data, manifest["config"], stats["validation_cutoff_unix"])
    report = {"data_version": manifest["version"], "data_manifest_sha256": sha256(data / "manifest.json")}
    report["base"] = base_checks(data, meta, catalog, errors)
    report["samples"] = sample_checks(data, output, meta, catalog, config, manifest["config"], errors)
    report["catalog_seed"] = seed_checks(data, output, catalog, errors)
    report["artifact_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.jsonl"))}
    report["error_count"] = sum(errors.count.values())
    report["errors_by_kind"] = dict(sorted(errors.count.items()))
    report["error_examples"] = errors.examples
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    (output / "p2_quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_report = json.loads((args.output / "p2_samples.report.json").read_text(encoding="utf-8"))
    def without_user_counts(value):
        return {**value, **{split: {**value[split], "users": None} for split in ("train", "validation")}}

    if without_user_counts(sample_report["config"]) != without_user_counts(config):
        parser.error("Sample report config differs from --config (only user-count overrides are allowed)")
    config = sample_report["config"]
    report = run(args.data, args.raw, args.output, config)
    print(json.dumps({k: report[k] for k in ("samples", "catalog_seed", "error_count", "errors_by_kind",
                                             "error_examples", "runtime_seconds")}, indent=2))
    sys.exit(1 if report["error_count"] else 0)


if __name__ == "__main__":
    main()
