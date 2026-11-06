"""Predeclared paired P1/P2 failure breakdown over authoritative saved reports.

Run from the repository root::

    python -m src.evaluation.phase2_failure_analysis \
      --p1-dir outputs/reports/phase2_validation/p1 \
      --p2-dir outputs/reports/phase2_validation/p2 \
      --diagnostics-dir outputs/reports/phase2_validation/p2_diagnostics \
      --analysis-spec configs/phase2_failure_analysis_spec.json \
      --output-dir outputs/reports/phase2_validation/failure_analysis

The analysis spec is a JSON object with ``schema_version`` equal to
``phase2-failure-analysis-spec-v1``, ``status`` equal to
``draft_for_validation`` or ``frozen_before_test``, ``selection_basis`` equal
to ``validation_only``, a positive integer ``rank_cutoff`` present in the
scoring configuration, and a nonnegative integer
``profile_length_threshold_chars``. Test runs require ``frozen_before_test``.
The threshold is inclusive for the short-profile group and counts Unicode
code points in the canonical profile ``text`` field. Review and freeze the
spec before test inference; a self-declared status alone cannot prove when
that happened. This command does not choose thresholds or inspect examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .bootstrap import _load_run, compare_runs
from .evaluate import _canonical_bytes, _json_object, _sha256
from .p2_diagnostics import CSV_COLUMNS, DIAGNOSTICS_VERSION


ANALYSIS_VERSION = "phase2-failure-analysis-v1"
SPEC_VERSION = "phase2-failure-analysis-spec-v1"
OUTPUT_FILES = ("summary.json", "per_user.csv")
GROUPS = (
    "initial_positive_present", "initial_positive_absent",
    "retrieved", "not_retrieved", "expansion_added_positive",
    "expansion_added_no_positive", "asked", "not_asked",
    "short_profile", "long_profile", "p2_better", "p2_tied", "p2_worse",
)
INTEGER_DIAGNOSTIC_COLUMNS = (
    "user_id", "relevant_targets", "initial_pool_size", "final_pool_size",
    "final_ranking_size", "initial_pool_hits", "expanded_pool_hits",
    "final_ranking_hits", "added_candidate_count", "added_relevant_count",
    "clarification_count", "retrieval_count", "fallback_count",
)
RECALL_DIAGNOSTIC_COLUMNS = (
    ("initial_pool_recall", "initial_pool_hits"),
    ("expanded_pool_recall", "expanded_pool_hits"),
    ("final_ranking_recall", "final_ranking_hits"),
)
PER_USER_COLUMNS = (
    "user_id", "split", "profile_length_chars", "profile_length_group",
    "initial_positive_group", "retrieval_group", "expansion_group",
    "clarification_group", "ndcg_change_group", "initial_pool_hits",
    "expanded_pool_hits", "final_ranking_hits", "initial_pool_recall",
    "expanded_pool_recall", "final_ranking_recall", "added_candidate_count",
    "added_relevant_count", "final_pool_size", "clarification_count",
    "retrieval_count", "fallback_count", "p1_ndcg", "p2_ndcg",
    "difference_ndcg", "p1_recall", "p2_recall", "difference_recall",
)


@dataclass(frozen=True)
class FailureAnalysisResult:
    summary: dict
    per_user: list[dict]


def _nonnegative_integer(raw: str, label: str) -> int:
    if not isinstance(raw, str) or re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(raw)


def _fraction(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number in [0, 1]") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a number in [0, 1]")
    return value


def _same(actual: float, expected: float, label: str) -> None:
    if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{label} disagrees with its authoritative source")


def _spec(path: Path, split: str, ranking_cutoffs: list[int]) -> dict:
    spec = _json_object(path)
    required = {"schema_version", "status", "selection_basis", "rank_cutoff",
                "profile_length_threshold_chars"}
    if set(spec) != required or spec["schema_version"] != SPEC_VERSION:
        raise ValueError("analysis spec has an unsupported schema or fields")
    if spec["status"] not in ("draft_for_validation", "frozen_before_test"):
        raise ValueError("analysis spec status is invalid")
    if split == "test" and spec["status"] != "frozen_before_test":
        raise ValueError("test analysis requires a spec frozen before test inference")
    if spec["selection_basis"] != "validation_only":
        raise ValueError("analysis spec selection_basis must be validation_only")
    if type(spec["rank_cutoff"]) is not int or spec["rank_cutoff"] not in ranking_cutoffs:
        raise ValueError("analysis spec rank_cutoff must be a configured ranking cutoff")
    threshold = spec["profile_length_threshold_chars"]
    if type(threshold) is not int or threshold < 0:
        raise ValueError("profile_length_threshold_chars must be a nonnegative integer")
    return spec


def _diagnostics(directory: Path, p2_summary: dict, p2_rows: dict[int, dict]) -> tuple[dict, dict[int, dict], dict[str, Path]]:
    paths = {name: Path(directory).resolve() / name for name in OUTPUT_FILES}
    summary = _json_object(paths["summary.json"])
    for key, expected in (
        ("status", "diagnosed"), ("diagnostics_version", DIAGNOSTICS_VERSION),
        ("method", "p2"), ("split", p2_summary["split"]),
        ("user_count", len(p2_rows)), ("dataset_version", p2_summary["dataset_version"]),
        ("evaluation_version", p2_summary["evaluation_version"]),
        ("config_status", p2_summary["config_status"]),
        ("source_sha256", p2_summary["source_sha256"]),
        ("source_paths", p2_summary["source_paths"]),
        ("score_summary_sha256", hashlib.sha256(_canonical_bytes(p2_summary)).hexdigest()),
    ):
        if summary.get(key) != expected:
            raise ValueError(f"P2 diagnostics {key} disagrees with the scored P2 bundle")
    if not isinstance(summary.get("aggregate"), dict):
        raise ValueError("P2 diagnostics aggregate is missing")
    rows: dict[int, dict] = {}
    with paths["per_user.csv"].open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(CSV_COLUMNS):
            raise ValueError("P2 diagnostics per_user.csv has invalid columns")
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("P2 diagnostics per_user.csv has a malformed row")
            row = dict(raw)
            for key in INTEGER_DIAGNOSTIC_COLUMNS:
                row[key] = _nonnegative_integer(row[key], f"P2 diagnostics {key}")
            user = row["user_id"]
            if user in rows or user not in p2_rows:
                raise ValueError("P2 diagnostics has a duplicate or unexpected user")
            if row["split"] != p2_summary["split"] or row["relevant_targets"] != p2_rows[user]["relevant_targets"]:
                raise ValueError("P2 diagnostics split or truth count disagrees with scorer")
            required_count = p2_summary["settings"]["required_predictions_per_user"]
            if row["initial_pool_size"] != required_count or row["final_ranking_size"] != required_count:
                raise ValueError("P2 diagnostics initial pool or final ranking size is invalid")
            if row["final_pool_size"] != required_count + row["added_candidate_count"]:
                raise ValueError("P2 diagnostics final pool size disagrees with added candidates")
            truth = row["relevant_targets"]
            if truth < 1 or not 0 <= row["initial_pool_hits"] <= row["expanded_pool_hits"] <= truth:
                raise ValueError("P2 diagnostics pool hit counts are invalid")
            if not 0 <= row["final_ranking_hits"] <= row["expanded_pool_hits"]:
                raise ValueError("P2 diagnostics final hit count is invalid")
            if row["added_relevant_count"] != row["expanded_pool_hits"] - row["initial_pool_hits"]:
                raise ValueError("P2 diagnostics added relevant count is invalid")
            for key, hit_key in RECALL_DIAGNOSTIC_COLUMNS:
                row[key] = _fraction(row[key], f"P2 diagnostics {key}")
                _same(row[key], row[hit_key] / truth, f"P2 diagnostics {key}")
            full_recall = f"recall_at_{required_count}"
            _same(row["final_ranking_recall"], p2_rows[user][full_recall], "P2 final-ranking recall")
            if row["added_candidate_count"]:
                row["added_candidate_yield"] = _fraction(row["added_candidate_yield"], "added_candidate_yield")
                _same(row["added_candidate_yield"], row["added_relevant_count"] / row["added_candidate_count"],
                      "P2 added-candidate yield")
            elif row["added_candidate_yield"] != "":
                raise ValueError("P2 added-candidate yield must be null when no candidates were added")
            else:
                row["added_candidate_yield"] = None
            if row["fallback_count"] not in (0, 1) or row["termination_reason"] not in ("recommend", "fallback"):
                raise ValueError("P2 diagnostics fallback or termination reason is invalid")
            if row["fallback_count"] != int(row["termination_reason"] == "fallback"):
                raise ValueError("P2 diagnostics fallback flag disagrees with termination reason")
            if row["retrieval_count"] == 0 and row["added_candidate_count"] != 0:
                raise ValueError("P2 diagnostics added candidates without retrieval")
            rows[user] = row
    if rows.keys() != p2_rows.keys():
        raise ValueError("P2 diagnostics must cover every scored user")
    aggregate = summary["aggregate"]
    count = len(rows)
    for key, row_key in (
        ("initial_pool_recall_macro", "initial_pool_recall"),
        ("expanded_pool_recall_macro", "expanded_pool_recall"),
        ("final_ranking_recall_macro", "final_ranking_recall"),
    ):
        _same(aggregate.get(key), math.fsum(row[row_key] for row in rows.values()) / count,
              f"P2 aggregate {key}")
    for key, row_key in (
        ("added_candidate_count", "added_candidate_count"),
        ("added_relevant_count", "added_relevant_count"),
        ("clarification_event_count", "clarification_count"),
        ("retrieval_event_count", "retrieval_count"),
        ("fallback_user_count", "fallback_count"),
    ):
        if aggregate.get(key) != sum(row[row_key] for row in rows.values()):
            raise ValueError(f"P2 aggregate {key} disagrees with per-user diagnostics")
    return summary, rows, paths


def _profile_lengths(path: Path, expected_hash: str, users: set[int]) -> dict[int, int]:
    if _sha256(path) != expected_hash:
        raise ValueError("canonical profile file hash disagrees with scored bundles")
    lengths: dict[int, int] = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                record = json.loads(line, object_pairs_hook=_unique_pairs,
                                    parse_constant=lambda value: _reject_constant(value))
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"profile line {number} is invalid JSON: {error}") from error
            if not isinstance(record, dict) or type(record.get("user_id")) is not int:
                raise ValueError(f"profile line {number} has invalid user_id")
            user = record["user_id"]
            if user not in users:
                continue
            if user in lengths or not isinstance(record.get("text"), str):
                raise ValueError(f"profile user {user} is duplicated or lacks canonical text")
            lengths[user] = len(record["text"])
    if lengths.keys() != users:
        raise ValueError("canonical profiles do not cover every scored user")
    return lengths


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _group_summary(rows: list[dict], selected: list[dict]) -> dict:
    count = len(selected)
    if not count:
        return {"user_count": 0, "fraction_of_all_users": 0.0,
                "p1_ndcg_mean": None, "p2_ndcg_mean": None, "difference_ndcg_mean": None,
                "p1_recall_mean": None, "p2_recall_mean": None, "difference_recall_mean": None,
                "initial_pool_recall_mean": None, "expanded_pool_recall_mean": None,
                "final_ranking_recall_mean": None, "retrieval_user_rate": None,
                "clarification_user_rate": None, "fallback_user_rate": None}
    def mean(key: str) -> float:
        return math.fsum(row[key] for row in selected) / count
    return {
        "user_count": count, "fraction_of_all_users": count / len(rows),
        **{f"{key}_mean": mean(key) for key in (
            "p1_ndcg", "p2_ndcg", "difference_ndcg", "p1_recall", "p2_recall",
            "difference_recall", "initial_pool_recall", "expanded_pool_recall",
            "final_ranking_recall")},
        "retrieval_user_rate": sum(row["retrieval_count"] > 0 for row in selected) / count,
        "clarification_user_rate": sum(row["clarification_count"] > 0 for row in selected) / count,
        "fallback_user_rate": sum(row["fallback_count"] > 0 for row in selected) / count,
    }


def analyze_failure_modes(p1_dir: Path, p2_dir: Path, diagnostics_dir: Path,
                          analysis_spec_path: Path) -> FailureAnalysisResult:
    """Validate compatible saved outputs and summarize every predeclared group."""
    p1_dir, p2_dir, diagnostics_dir, analysis_spec_path = map(
        lambda path: Path(path).resolve(), (p1_dir, p2_dir, diagnostics_dir, analysis_spec_path))
    diagnostic_paths = {name: diagnostics_dir / name for name in OUTPUT_FILES}
    diagnostic_hashes_before = {name: _sha256(path) for name, path in diagnostic_paths.items()}
    comparison = compare_runs(p1_dir, p2_dir)
    if comparison.summary["baseline_method"] != "p1" or comparison.summary["challenger_method"] != "p2":
        raise ValueError("failure analysis requires P1 baseline and P2 challenger")
    p1, p1_rows, p1_paths, p1_hashes = _load_run(p1_dir)
    p2, p2_rows, p2_paths, p2_hashes = _load_run(p2_dir)
    for prefix, hashes in (("baseline", p1_hashes), ("challenger", p2_hashes)):
        for name, digest in hashes.items():
            if comparison.summary["source_sha256"].get(f"{prefix}_{name.split('.')[0]}") != digest:
                raise ValueError("scored bundle changed during paired analysis")
    spec = _spec(analysis_spec_path, p2["split"], p2["settings"]["ranking_k"])
    diagnostic_summary, diagnostics, diagnostic_paths = _diagnostics(diagnostics_dir, p2, p2_rows)
    profiles_path = Path(p2["source_paths"]["profiles"])
    profile_lengths = _profile_lengths(profiles_path, p2["source_sha256"]["profiles"], set(p2_rows))
    k = spec["rank_cutoff"]
    ndcg, recall = f"ndcg_at_{k}", f"recall_at_{k}"
    rows = []
    for user in sorted(p2_rows):
        d = diagnostics[user]
        left, right = p1_rows[user], p2_rows[user]
        delta_ndcg = right[ndcg] - left[ndcg]
        delta_recall = right[recall] - left[recall]
        profile_group = "short_profile" if profile_lengths[user] <= spec["profile_length_threshold_chars"] else "long_profile"
        change = "p2_better" if delta_ndcg > 1e-12 else "p2_worse" if delta_ndcg < -1e-12 else "p2_tied"
        rows.append({
            "user_id": user, "split": p2["split"], "profile_length_chars": profile_lengths[user],
            "profile_length_group": profile_group,
            "initial_positive_group": "initial_positive_present" if d["initial_pool_hits"] else "initial_positive_absent",
            "retrieval_group": "retrieved" if d["retrieval_count"] else "not_retrieved",
            "expansion_group": "expansion_added_positive" if d["added_relevant_count"] else "expansion_added_no_positive",
            "clarification_group": "asked" if d["clarification_count"] else "not_asked",
            "ndcg_change_group": change,
            **{key: d[key] for key in (
                "initial_pool_hits", "expanded_pool_hits", "final_ranking_hits",
                "initial_pool_recall", "expanded_pool_recall", "final_ranking_recall",
                "added_candidate_count", "added_relevant_count", "final_pool_size",
                "clarification_count", "retrieval_count", "fallback_count")},
            "p1_ndcg": left[ndcg], "p2_ndcg": right[ndcg], "difference_ndcg": delta_ndcg,
            "p1_recall": left[recall], "p2_recall": right[recall], "difference_recall": delta_recall,
        })
    groups = {group: _group_summary(rows, [row for row in rows if group in (
        row["profile_length_group"], row["initial_positive_group"], row["retrieval_group"],
        row["expansion_group"], row["clarification_group"], row["ndcg_change_group"])])
        for group in GROUPS}
    for partition in (("initial_positive_present", "initial_positive_absent"),
                      ("retrieved", "not_retrieved"),
                      ("expansion_added_positive", "expansion_added_no_positive"),
                      ("asked", "not_asked"), ("short_profile", "long_profile"),
                      ("p2_better", "p2_tied", "p2_worse")):
        if sum(groups[group]["user_count"] for group in partition) != len(rows):
            raise ValueError("predeclared group partition failed to cover all users")
    deltas_ndcg = [row["difference_ndcg"] for row in rows]
    deltas_recall = [row["difference_recall"] for row in rows]
    bootstrap = comparison.summary["metrics"]
    _same(bootstrap[ndcg]["mean_difference"], math.fsum(deltas_ndcg) / len(rows), "paired NDCG difference")
    _same(bootstrap[recall]["mean_difference"], math.fsum(deltas_recall) / len(rows), "paired Recall difference")
    input_paths = {
        **{f"p1_{name}": str(path) for name, path in p1_paths.items()},
        **{f"p2_{name}": str(path) for name, path in p2_paths.items()},
        **{f"diagnostics_{name}": str(path) for name, path in diagnostic_paths.items()},
        "analysis_spec": str(analysis_spec_path), "profiles": str(profiles_path),
        "data_dir": p2["source_paths"]["data_dir"],
    }
    input_hashes = {
        **{f"p1_{name}": digest for name, digest in p1_hashes.items()},
        **{f"p2_{name}": digest for name, digest in p2_hashes.items()},
        **{f"diagnostics_{name}": _sha256(path) for name, path in diagnostic_paths.items()},
        "analysis_spec": _sha256(analysis_spec_path), "profiles": _sha256(profiles_path),
    }
    if any(input_hashes[f"diagnostics_{name}"] != digest for name, digest in diagnostic_hashes_before.items()):
        raise ValueError("P2 diagnostics changed during failure analysis")
    if input_hashes["profiles"] != p2["source_sha256"]["profiles"]:
        raise ValueError("canonical profiles changed during failure analysis")
    for prefix, paths, hashes in (("p1", p1_paths, p1_hashes), ("p2", p2_paths, p2_hashes)):
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise ValueError(f"{prefix.upper()} scored bundle changed during failure analysis")
    summary = {
        "status": "analyzed", "analysis_version": ANALYSIS_VERSION,
        "analysis_spec": spec, "split": p2["split"], "dataset_version": p2["dataset_version"],
        "evaluation_version": p2["evaluation_version"], "config_status": p2["config_status"],
        "user_count": len(rows), "difference_direction": "p2_minus_p1",
        "rank_cutoff": k, "group_policy": "predeclared overlapping groups; empty groups have null means and rates",
        "tie_policy": "absolute NDCG difference at most 1e-12",
        "profile_length_policy": "Unicode code points in canonical profile text; short is <= threshold",
        "paired": {
            "ndcg": {**bootstrap[ndcg], "median_difference": statistics.median(deltas_ndcg),
                     "better_count": groups["p2_better"]["user_count"],
                     "tie_count": groups["p2_tied"]["user_count"],
                     "worse_count": groups["p2_worse"]["user_count"]},
            "recall": {**bootstrap[recall], "median_difference": statistics.median(deltas_recall),
                       "better_count": sum(value > 1e-12 for value in deltas_recall),
                       "tie_count": sum(abs(value) <= 1e-12 for value in deltas_recall),
                       "worse_count": sum(value < -1e-12 for value in deltas_recall)},
        },
        "groups": groups,
        "p2_aggregate_diagnostics": diagnostic_summary["aggregate"],
        "not_assessed_here": ["qualitative case labels", "explanation grounding", "service latency", "causal attribution"],
        "input_paths": input_paths, "input_sha256": input_hashes,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
    }
    return FailureAnalysisResult(summary, rows)


def _serialized(result: FailureAnalysisResult) -> dict[str, bytes]:
    if not result.per_user or result.summary.get("user_count") != len(result.per_user):
        raise ValueError("failure analysis requires all scored users")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PER_USER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(result.per_user)
    return {"summary.json": _canonical_bytes(result.summary),
            "per_user.csv": output.getvalue().encode("utf-8")}


def write_failure_analysis(output_dir: Path, result: FailureAnalysisResult) -> None:
    """Atomically save immutable output, never inside or over an input bundle."""
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    paths = result.summary.get("input_paths")
    if not isinstance(paths, dict) or "data_dir" not in paths:
        raise ValueError("failure analysis requires input paths")
    forbidden_dirs = [Path(paths["data_dir"]).resolve()]
    forbidden_dirs += [Path(paths[f"{prefix}_summary.json"]).resolve().parent
                       for prefix in ("p1", "p2", "diagnostics")]
    if any(resolved == directory or directory in resolved.parents for directory in forbidden_dirs):
        raise ValueError("output directory must be outside processed data and input report directories")
    payloads = _serialized(result)
    for key, raw_path in paths.items():
        if key == "data_dir":
            continue
        source = Path(raw_path).resolve()
        if resolved == source:
            raise ValueError("output directory must not overwrite an input")
        for name in OUTPUT_FILES:
            destination = (output_dir / name).resolve()
            if destination == source or (destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError("output file must not overwrite or alias an input")
    if output_dir.exists():
        if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != set(payloads) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw
            for name, raw in payloads.items()
        ):
            raise ValueError("existing failure analysis differs; choose a new output directory")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, raw in payloads.items():
            (staged / name).write_bytes(raw)
        if output_dir.exists():
            raise ValueError("output directory appeared during staging; choose a new output directory")
        os.rename(staged, output_dir)
    finally:
        if staged.exists():
            if staged.resolve().parent != output_dir.parent.resolve():
                raise ValueError("temporary output escaped its parent")
            shutil.rmtree(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_failure_modes(args.p1_dir, args.p2_dir, args.diagnostics_dir, args.analysis_spec)
        write_failure_analysis(args.output_dir, result)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error, KeyError) as error:
        print(f"Phase 2 failure analysis failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result.summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
