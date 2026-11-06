"""Synthetic integration checks for the predeclared P1/P2 failure breakdown.

The two movie users below are invented fixture users. Their results are never
evidence about the real P2 system.
"""

import contextlib
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate import evaluate, write_results
from src.evaluation.p2_diagnostics import analyze_p2, write_diagnostics
from src.evaluation.phase2_failure_analysis import (
    _spec, analyze_failure_modes, main, write_failure_analysis,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(user: int, ranking: list[int]) -> dict:
    return {"user_id": user, "split": "validation", "method": "p2",
            "recommendations": [{"movie_id": movie, "rank": rank, "score": 4 - rank}
                                for rank, movie in enumerate(ranking, start=1)]}


def _trace(user: int, initial: list[int], final_pool: list[int], ranking: list[int],
           events: list[dict]) -> dict:
    return {
        "schema_version": "p2-trace-v1", "user_id": user, "split": "validation", "method": "p2",
        "initial_candidate_movie_ids": initial,
        "events": [*events, {"action": "rank_candidates", "movie_ids": ranking},
                   {"action": "explain_recommendations", "explanations": [
                       {"movie_id": movie, "text": f"Catalog movie {movie}."} for movie in ranking]}],
        "final_candidate_movie_ids": final_pool, "termination_reason": "recommend",
    }


class Phase2FailureAnalysisTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)

        # Bind a deliberately short and long canonical text to the frozen
        # synthetic source chain, then regenerate its manifest/user hashes.
        profiles = self.data / "validation_profiles.jsonl"
        rows = [json.loads(line) for line in profiles.read_text(encoding="utf-8").splitlines()]
        rows[0]["text"] = "short"
        rows[1]["text"] = "a considerably longer starting profile"
        _write_jsonl(profiles, rows)
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_sha256"]["validation_profiles.jsonl"] = _digest(profiles)
        _write_json(manifest_path, manifest)
        users_path = self.data / "users.json"
        users = json.loads(users_path.read_text(encoding="utf-8"))
        users["source_sha256"].update(manifest=_digest(manifest_path), profiles=_digest(profiles))
        _write_json(users_path, users)
        self.config_path = self.work / "config.json"
        config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(self.data), users_file=str(users_path),
                      manifest_sha256=_digest(manifest_path), users_sha256=_digest(users_path))
        _write_json(self.config_path, config)

        self.p2_predictions = self.work / "p2.jsonl"
        self.trace_path = self.work / "p2_trace.jsonl"
        self.pool = self.data / "embedding.jsonl"
        _write_jsonl(self.p2_predictions, [
            _prediction(1, [30, 10, 20]), _prediction(2, [40, 50, 60]),
        ])
        _write_jsonl(self.trace_path, [
            _trace(1, [10, 30, 20], [10, 30, 20], [30, 10, 20], [
                {"action": "assess_sufficiency", "sufficient": False},
                {"action": "ask_clarification", "question": "A genre?", "answer": "Comedy",
                 "answer_source": "synthetic_pre_cutoff_history"},
                {"action": "assess_sufficiency", "sufficient": True},
                {"action": "evaluate_candidates", "candidate_movie_ids": [10, 30, 20],
                 "pool_sufficient": True},
            ]),
            _trace(2, [60, 40, 70], [60, 40, 70, 50], [40, 50, 60], [
                {"action": "assess_sufficiency", "sufficient": True},
                {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70],
                 "pool_sufficient": False},
                {"action": "retrieve_candidates", "added_movie_ids": [50],
                 "index_version": "synthetic-index-v1"},
                {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70, 50],
                 "pool_sufficient": True},
            ]),
        ])
        self.p1_dir, self.p2_dir = self.work / "p1", self.work / "p2"
        p1 = evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                      candidate_pool_path=self.pool, repository_root=self.work)
        p2 = evaluate(self.config_path, self.p2_predictions, method="p2",
                      candidate_pool_path=self.pool, trace_path=self.trace_path,
                      repository_root=self.work)
        write_results(self.p1_dir, p1)
        write_results(self.p2_dir, p2)
        self.diagnostics_dir = self.work / "diagnostics"
        diagnostics = analyze_p2(self.config_path, self.pool, self.p2_predictions,
                                 self.trace_path, repository_root=self.work)
        write_diagnostics(self.diagnostics_dir, diagnostics)
        self.spec_path = self.work / "analysis_spec.json"
        self.spec = {
            "schema_version": "phase2-failure-analysis-spec-v1", "status": "draft_for_validation",
            "selection_basis": "validation_only", "rank_cutoff": 2,
            "profile_length_threshold_chars": 10,
        }
        _write_json(self.spec_path, self.spec)

    def analyze(self):
        return analyze_failure_modes(self.p1_dir, self.p2_dir, self.diagnostics_dir,
                                     self.spec_path)

    def test_predeclared_partitions_pair_every_user_and_keep_retrieval_outcome_distinct(self):
        result = self.analyze()
        summary = result.summary
        self.assertEqual(summary["status"], "analyzed")
        self.assertEqual(summary["user_count"], 2)
        self.assertEqual(summary["difference_direction"], "p2_minus_p1")
        self.assertEqual([row["user_id"] for row in result.per_user], [1, 2])
        self.assertEqual([row["profile_length_group"] for row in result.per_user],
                         ["short_profile", "long_profile"])
        self.assertEqual([row["ndcg_change_group"] for row in result.per_user],
                         ["p2_worse", "p2_better"])
        self.assertEqual(summary["groups"]["p2_better"]["user_count"], 1)
        self.assertEqual(summary["groups"]["p2_worse"]["user_count"], 1)
        self.assertEqual(summary["groups"]["p2_tied"]["user_count"], 0)
        self.assertIsNone(summary["groups"]["p2_tied"]["difference_ndcg_mean"])
        self.assertEqual(summary["groups"]["retrieved"]["user_count"], 1)
        self.assertEqual(summary["groups"]["expansion_added_positive"]["user_count"], 1)
        self.assertEqual(summary["groups"]["asked"]["user_count"], 1)
        self.assertEqual(result.per_user[1]["expanded_pool_recall"], 1.0)
        self.assertEqual(result.per_user[1]["final_ranking_recall"], 1.0)
        self.assertEqual(summary["paired"]["ndcg"]["better_count"], 1)
        self.assertEqual(summary["paired"]["ndcg"]["worse_count"], 1)
        self.assertEqual(summary["paired"]["ndcg"]["user_count"], 2)
        self.assertEqual(summary["input_sha256"]["diagnostics_per_user.csv"],
                         _digest(self.diagnostics_dir / "per_user.csv"))

    def test_cli_creates_immutable_results_and_rejects_changed_directory(self):
        output = self.work / "analysis"
        args = ["--p1-dir", str(self.p1_dir), "--p2-dir", str(self.p2_dir),
                "--diagnostics-dir", str(self.diagnostics_dir),
                "--analysis-spec", str(self.spec_path), "--output-dir", str(output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(args), 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual({path.name for path in output.iterdir()}, {"summary.json", "per_user.csv"})
        self.assertEqual(json.loads(stdout.getvalue()), json.loads((output / "summary.json").read_text()))
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        (output / "per_user.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing failure analysis differs"):
            write_failure_analysis(output, self.analyze())

    def test_tampered_diagnostics_and_unmatched_scorer_are_rejected(self):
        path = self.diagnostics_dir / "per_user.csv"
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            columns, rows = reader.fieldnames, list(reader)
        rows[1]["expanded_pool_hits"] = "1"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "final hit count|pool hit counts|added relevant count"):
            self.analyze()
        self.assertFalse((self.work / "analysis").exists())

        # Restore diagnostics and break a shared scored-run source hash.
        diagnostics = analyze_p2(self.config_path, self.pool, self.p2_predictions,
                                 self.trace_path, repository_root=self.work)
        shutil.rmtree(self.diagnostics_dir)
        write_diagnostics(self.diagnostics_dir, diagnostics)
        summary_path = self.p1_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source_sha256"]["targets"] = "0" * 64
        _write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "different targets source hashes"):
            self.analyze()

    def test_test_split_requires_pretest_frozen_analysis_spec(self):
        with self.assertRaisesRegex(ValueError, "frozen before test"):
            _spec(self.spec_path, "test", [2])
        self.spec["status"] = "frozen_before_test"
        _write_json(self.spec_path, self.spec)
        self.assertEqual(_spec(self.spec_path, "test", [2])["profile_length_threshold_chars"], 10)

    def test_missing_diagnostic_user_and_changed_profile_source_are_rejected(self):
        path = self.diagnostics_dir / "per_user.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cover every scored user"):
            self.analyze()
        diagnostics = analyze_p2(self.config_path, self.pool, self.p2_predictions,
                                 self.trace_path, repository_root=self.work)
        shutil.rmtree(self.diagnostics_dir)
        write_diagnostics(self.diagnostics_dir, diagnostics)
        profiles = self.data / "validation_profiles.jsonl"
        profiles.write_bytes(profiles.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "profile file hash"):
            self.analyze()

    def test_rejects_output_in_input_bundle_or_data_directory(self):
        result = self.analyze()
        for output in (self.p1_dir / "nested", self.p2_dir / "nested",
                       self.diagnostics_dir / "nested", self.data / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                write_failure_analysis(output, result)


if __name__ == "__main__":
    unittest.main()
