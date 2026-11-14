import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from src.data import clarification_sim as sim
from src.data.build import build, sha256
from src.data.catalog_seed import build_seed, display_title
from src.data.p2_actions import ActionError, parse_action, validate_action, validate_episode
from src.data.p2_checks import grounding_problems, run as run_checks
from src.data.p2_samples import generate
from src.profiles.agent_text import ACTIONS, render_prompt, serialize_action

GENRES = ["Drama", "Comedy", "Sci-Fi", "Horror", "Romance", "Thriller"]
TAGS = ["dark", "funny", "space", "romantic", "heist", "family", "twist ending", "atmospheric"]
DATA_CONFIG = {
    "version": "fixture-v1", "minimum_training_ratings_per_user": 20,
    "minimum_training_ratings_per_catalog_movie": 2, "positive_rating_at_least": 4.0,
    "test_quantile": 0.9, "validation_quantile_within_pretest": 0.9,
    "genome_tags_per_movie": 4, "user_tags_per_movie": 2,
    "liked_movies_per_profile": 4, "disliked_movies_per_profile": 2,
}


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def make_raw(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir()
    rng = random.Random(7)
    movies = []
    for i in range(1, 61):
        genres = "|".join(sorted(rng.sample(GENRES, 2)))
        title = f"Movie {i}, The ({1960 + i % 50})" if i % 7 == 0 else f"Movie {i} ({1960 + i % 50})"
        movies.append((i, title, genres))
    write_csv(raw / "movies.csv", ("movieId", "title", "genres"), movies)
    write_csv(raw / "links.csv", ("movieId", "imdbId", "tmdbId"), [(i, f"{i:07d}", i + 100) for i in range(1, 61)])
    ratings = []
    for user in range(1, 13):
        favorite = GENRES[user % len(GENRES)]
        order = rng.sample(range(1, 61), 50)
        for step, movie in enumerate(order):
            liked = favorite in movies[movie - 1][2]
            rating = rng.choice([4.0, 4.5, 5.0]) if liked or rng.random() < 0.45 else rng.choice([1.0, 2.0, 3.0])
            ratings.append((user, movie, rating, 1000 + step * 20 + user))
    write_csv(raw / "ratings.csv", ("userId", "movieId", "rating", "timestamp"), ratings)
    write_csv(raw / "tags.csv", ("userId", "movieId", "tag", "timestamp"),
              [(1, i, rng.choice(TAGS), 1001) for i in range(1, 61, 2)])
    write_csv(raw / "genome-tags.csv", ("tagId", "tag"), [(i + 1, t) for i, t in enumerate(TAGS)])
    write_csv(raw / "genome-scores.csv", ("movieId", "tagId", "relevance"),
              [(i, j + 1, round(rng.random(), 3)) for i in range(1, 61) for j in range(len(TAGS))])
    return raw


def p2_config(exclude_path: Path) -> dict:
    return {
        "version": "p2-fixture", "data_version": "fixture-v1", "seed": 11,
        "exclude_user_lists": [str(exclude_path)],
        "train": {"users": 5, "episodes_per_user": 2, "canonical_fraction": 0.5, "sparse_rating_counts": [1, 2, 3]},
        "validation": {"users": 3, "episodes_per_user": 1, "canonical_fraction": 0.5, "sparse_rating_counts": [1, 2, 3]},
        "min_history_before_anchor": 8, "future_horizon_ratings": 20, "min_future_positives": 2,
        "future_positives": 5, "future_negatives": 1, "max_questions": 2, "max_retrievals": 1,
        "sufficiency_min_gain": 0.02, "pool_min_gap": 0.02, "offtarget_pool_fraction": 0.5,
        "pool_size": 5, "retrieval_k": 5, "quality_top_m": 3, "inject_positives": 2, "inject_negatives": 1,
        "explain_k": 10, "popularity_prior": 0.2, "retrieval_length_norm": 0.5,
        "feature_weights": {"genre": 1.0, "genome_tag": 1.0, "user_tag": 0.5, "decade": 0.5},
    }


class FixturePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.raw = make_raw(root)
        cls.data = root / "processed"
        build(cls.raw, cls.data, DATA_CONFIG)
        cls.exclude = root / "exclude.json"
        cls.exclude.write_text(json.dumps({"user_ids": [1]}))
        cls.config = p2_config(cls.exclude)
        cls.output = root / "p2"
        cls.report = generate(cls.data, cls.raw, cls.output, cls.config)
        cls.seed_report = build_seed(cls.data, cls.raw, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def samples(self, split="train"):
        with (self.output / f"p2_samples_{split}.jsonl").open(encoding="utf-8") as source:
            return [json.loads(line) for line in source]

    def episodes(self, split="train"):
        with (self.output / f"p2_episodes_{split}.jsonl").open(encoding="utf-8") as source:
            return [json.loads(line) for line in source]

    def test_all_six_actions_and_valid_targets(self):
        samples = self.samples() + self.samples("validation")
        self.assertEqual({s["task"] for s in samples}, set(ACTIONS))
        for sample in samples:
            self.assertEqual(parse_action(sample["completion"]), sample["target"])
            self.assertNotIn("anchor_timestamp", sample["prompt"])
        users = {s["user_id"] for s in samples}
        self.assertNotIn(1, users)
        train_users = {s["user_id"] for s in self.samples()}
        self.assertFalse(train_users & {s["user_id"] for s in self.samples("validation")})

    def test_temporal_boundaries_in_episodes(self):
        for split in ("train", "validation"):
            for episode in self.episodes(split):
                self.assertLess(episode["history_latest_timestamp"], episode["anchor_timestamp"])
                self.assertTrue(all(episode["anchor_timestamp"] <= ts < episode["boundary_timestamp"]
                                    for ts in episode["future_positive_timestamps"]))
                self.assertLessEqual(episode["summary"]["control_events"], 8)

    def test_quality_checks_pass_and_replay(self):
        report = run_checks(self.data, self.raw, self.output, self.config)
        self.assertEqual(report["error_count"], 0, report["error_examples"])
        for split in ("train", "validation"):
            stats = report["samples"][split]
            self.assertEqual(stats["episodes_replayed"], stats["episodes"])
            self.assertEqual(stats["explanations_grounded"], stats["explanations"])
        self.assertEqual(report["catalog_seed"]["movies"], self.seed_report["movies"])

    def test_generation_is_deterministic(self):
        second = Path(self.temp.name) / "p2-second"
        report = generate(self.data, self.raw, second, self.config)
        for split in ("train", "validation"):
            self.assertEqual(report["splits"][split]["samples_sha256"], self.report["splits"][split]["samples_sha256"])
            self.assertEqual(sha256(second / f"p2_episodes_{split}.jsonl"),
                             sha256(self.output / f"p2_episodes_{split}.jsonl"))

    def test_checks_detect_tampering(self):
        tampered = Path(self.temp.name) / "p2-tampered"
        tampered.mkdir()
        for path in self.output.iterdir():
            (tampered / path.name).write_bytes(path.read_bytes())
        episodes = self.episodes()
        target = next(e for e in episodes if e["future_positive_movie_ids"])
        explain = target["events"][-1]["explanations"][0]
        explain["text"] = explain["text"] + ' It is tagged "invented tag".'
        target["initial_candidate_movie_ids"][0] = target["future_positive_movie_ids"][0]
        with (tampered / "p2_episodes_train.jsonl").open("w", encoding="utf-8") as out:
            for episode in episodes:
                out.write(json.dumps(episode, sort_keys=True) + "\n")
        report = run_checks(self.data, self.raw, tampered, self.config)
        self.assertIn("grounding", report["errors_by_kind"])
        self.assertIn("replay", report["errors_by_kind"])

    def test_seed_matches_catalog(self):
        with (self.output / "catalog_seed.jsonl").open(encoding="utf-8") as source:
            seed = [json.loads(line) for line in source]
        with (self.data / "catalog.csv").open(encoding="utf-8") as source:
            catalog = {int(r["movie_id"]): r for r in csv.DictReader(source)}
        self.assertEqual(sorted(r["movie_id"] for r in seed), sorted(catalog))
        for record in seed:
            self.assertEqual(record["rating_count"], int(catalog[record["movie_id"]]["train_ratings"]))
            self.assertTrue(record["imdb_id"].startswith("tt"))

    def test_generator_never_opens_test_files(self):
        for name in ("p2_samples.py", "clarification_sim.py", "catalog_seed.py"):
            source = (Path("src/data") / name).read_text(encoding="utf-8")
            for forbidden in ("test_ratings", "test_targets", "test_profiles"):
                self.assertNotIn(forbidden, source, f"{name} references {forbidden}")


class ActionGrammarTest(unittest.TestCase):
    def test_parse_and_reject(self):
        self.assertEqual(parse_action('{"action": "assess_sufficiency", "sufficient": false}')["sufficient"], False)
        bad = [
            'not json',
            '{"action": "assess_sufficiency", "sufficient": "no"}',
            '{"action": "assess_sufficiency", "sufficient": true, "sufficient": false}',
            '{"action": "ask_clarification", "question": "  "}',
            '{"action": "rank_candidates", "movie_ids": [1, 1]}',
            '{"action": "rank_candidates", "movie_ids": [1, NaN]}',
            '{"action": "retrieve", "query": "x"}',
            '{"action": "evaluate_candidates", "pool_sufficient": true, "extra": 1}',
        ]
        for text in bad:
            with self.assertRaises(ActionError, msg=text):
                parse_action(text)
        with self.assertRaises(ActionError):
            validate_action({"action": "rank_candidates", "movie_ids": [1, 2]}, candidate_ids=[1, 3])
        with self.assertRaises(ActionError):
            validate_action({"action": "explain_recommendations", "explanations": [{"movie_id": 2, "text": "x"}]},
                            explain_ids=[1])

    def test_serialization_order(self):
        text = serialize_action({"sufficient": True, "action": "assess_sufficiency"})
        self.assertEqual(text, '{"action": "assess_sufficiency", "sufficient": true}')

    def test_episode_rules(self):
        ok = [
            {"action": "assess_sufficiency", "sufficient": False},
            {"action": "ask_clarification", "question": "q", "answer": "a", "answer_source": "s"},
            {"action": "assess_sufficiency", "sufficient": True},
            {"action": "evaluate_candidates", "candidate_movie_ids": [1, 2], "pool_sufficient": False},
            {"action": "retrieve_candidates", "added_movie_ids": [3], "index_version": "v"},
            {"action": "evaluate_candidates", "candidate_movie_ids": [1, 2, 3], "pool_sufficient": True},
            {"action": "rank_candidates", "movie_ids": [3, 1, 2]},
            {"action": "explain_recommendations", "explanations": [
                {"movie_id": 3, "text": "a"}, {"movie_id": 1, "text": "b"}, {"movie_id": 2, "text": "c"}]},
        ]
        self.assertEqual(validate_episode(ok, [1, 2])["control_events"], 6)
        ask_first = [ok[1]] + ok[2:]
        stale_pool = ok[:3] + [dict(ok[3], candidate_movie_ids=[2, 1])] + ok[4:]
        rank_early = ok[:3] + ok[6:]
        too_many = ok[:2] + [ok[0], ok[1]] * 3 + ok[2:]
        for events in (ask_first, stale_pool, rank_early, too_many):
            with self.assertRaises(ActionError):
                validate_episode(events, [1, 2])


class SimulatorTest(unittest.TestCase):
    def test_templates_round_trip(self):
        for intent, templates in sim.QUESTION_TEMPLATES.items():
            for template in templates:
                self.assertEqual(sim.classify_question(template), intent, template)
        self.assertIsNone(sim.classify_question("How are you today?"))

    def test_answers_use_only_available_history(self):
        meta = {
            1: {"title": "A (1990)", "genres": ["Drama"], "tags": ["dark"], "decade": "1990s"},
            2: {"title": "B (1991)", "genres": ["Drama"], "tags": ["dark"], "decade": "1990s"},
            3: {"title": "C (1992)", "genres": ["Comedy"], "tags": ["funny"], "decade": "1990s"},
        }
        history = [(1, 5.0, 10), (2, 4.0, 11), (3, 1.0, 12)]
        with self.assertRaises(ValueError):
            sim.build_fact_pool(history, meta, set(), "", cutoff=12)
        pool = sim.build_fact_pool(history, meta, {1}, "Highly rated movies: A (1990)", cutoff=13)
        first = sim.answer_question("Can you name a movie you really enjoyed?", pool, [], "A (1990)")
        self.assertEqual(first["answer"], "I really enjoyed B (1991) (4/5).")
        self.assertTrue(first["new_fact"])
        again = sim.answer_question("What is one of your favorite movies?", pool,
                                    ["Can you name a movie you really enjoyed?"], "")
        self.assertFalse(again["informative"])
        self.assertEqual(sim.answer_question("Hello?", pool, [], "")["answer"], sim.UNRECOGNIZED_ANSWER)

    def test_quoted_tags_are_never_revealed(self):
        meta = {i: {"title": f"M{i} (1990)", "genres": ["Drama"], "tags": ['"quoted tag"', "plain"],
                    "decade": "1990s"} for i in (1, 2)}
        pool = sim.build_fact_pool([(1, 5.0, 1), (2, 5.0, 2)], meta, set(), "", cutoff=3)
        self.assertEqual([f["tags"] for f in pool["themes"]], [["plain"]])


class GroundingTest(unittest.TestCase):
    def test_flags_unsupported_claims(self):
        state = {"liked": [{"movie_id": 1, "title": "A (1990)", "rating": 5.0, "genres": ["Drama"],
                            "tags": ["dark"]}], "turns": []}
        candidate = {"movie_id": 9, "title": "Z (1995)", "genres": ["Drama", "Horror"], "tags": ["dark", "gore"]}
        good = 'Like A (1990), which you rated 5/5, it is listed under Drama and is tagged "dark".'
        self.assertEqual(grounding_problems(good, candidate, "1990s", state), [])
        for bad in ('Like A (1990), which you rated 4/5, it is listed under Drama.',
                    'Like B (1990), which you rated 5/5, it is listed under Drama.',
                    'Like A (1990), which you rated 5/5, it is listed under Horror.',
                    'Like A (1990), which you rated 5/5, it is tagged "gore".',
                    'It is from the 1990s, the decade you said you enjoy most.'):
            self.assertTrue(grounding_problems(bad, candidate, "1990s", state), bad)

    def test_prompt_requires_candidates(self):
        with self.assertRaises(ValueError):
            render_prompt("rank_candidates", "profile", [])
        with self.assertRaises(ValueError):
            render_prompt("assess_sufficiency", "profile", [], [{"movie_id": 1}])

    def test_display_title(self):
        self.assertEqual(display_title("Matrix, The (1999)"), "The Matrix")
        self.assertEqual(display_title("Amélie (Fabuleux destin d'Amélie Poulain, Le) (2001)"),
                         "Amélie (Fabuleux destin d'Amélie Poulain, Le)")
        self.assertEqual(display_title("Lord of the Rings, The (1978)"), "The Lord of the Rings")
        self.assertEqual(display_title("Untitled"), "Untitled")


if __name__ == "__main__":
    unittest.main()
