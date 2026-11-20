"""Score P2 candidate-pool opportunity and agent decisions on frozen users.

This is a label-aware evaluation command. It reads held-out targets only after
the shared scorer has checked the data manifest, frozen users, predictions, and
complete p2-trace-v1 file. Do not use its output to construct clarification
answers, select a checkpoint, or tune retrieval on the final test split.

Run from the repository root::

    python -m src.evaluation.p2_diagnostics --config CONFIG \
        --candidate-pool FROZEN_B3.jsonl --predictions P2.jsonl \
        --trace P2_TRACE.jsonl --output-dir OUTPUT_DIR

The command writes immutable ``summary.json`` and ``per_user.csv``. A second
run accepts an identical output directory and refuses to replace changed data.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .evaluate import _canonical_bytes, _sha256, _targets, evaluate
from .validate import PredictionValidationError, load_context, validate_predictions


DIAGNOSTICS_VERSION = "p2-diagnostics-v1"
OUTPUT_FILES = ("summary.json", "per_user.csv")
CSV_COLUMNS = (
    "user_id", "split", "relevant_targets", "initial_pool_size", "final_pool_size",
    "final_ranking_size", "initial_pool_hits", "expanded_pool_hits",
    "final_ranking_hits", "initial_pool_recall", "expanded_pool_recall",
    "final_ranking_recall", "added_candidate_count", "added_relevant_count",
    "added_candidate_yield", "clarification_count", "retrieval_count",
    "fallback_count", "termination_reason",
)


@dataclass(frozen=True)
class P2DiagnosticsResult:
    summary: dict
    per_user: list[dict]


def _recall(hits: int, target_count: int) -> float:
    if target_count <= 0:
        raise ValueError("every selected user must have an in-catalog positive target")
    return hits / target_count


def _percentile(ordered: list[int], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _pool_sizes(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "min": ordered[0], "mean": math.fsum(values) / len(values),
        "p50": _percentile(ordered, 0.5), "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "percentile_convention": "linear_interpolation_at_(n-1)*p",
    }


def analyze_p2(
    config_path: Path, candidate_pool_path: Path, predictions_path: Path,
    trace_path: Path, *, repository_root: Path | None = None,
) -> P2DiagnosticsResult:
    """Validate all frozen inputs, then diagnose every selected P2 episode.

    Pool recall is macro-averaged over the complete selected population using
    all distinct in-catalog positives as each user's denominator. Added-item
    yield pools added-movie hits over added-movie count; when nothing is added,
    it is null rather than zero. Fallback users remain in every rate and mean.
    """
    config_path = Path(config_path).resolve()
    candidate_pool_path = Path(candidate_pool_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    trace_path = Path(trace_path).resolve()
    scored = evaluate(
        config_path, predictions_path, method="p2",
        candidate_pool_path=candidate_pool_path, trace_path=trace_path,
        repository_root=repository_root,
    )
    source_paths = scored.summary["source_paths"]
    context = load_context(
        Path(source_paths["catalog"]), Path(source_paths["profiles"]),
        Path(source_paths["users"]), scored.summary["split"],
    )
    relevant, _ = _targets(Path(source_paths["targets"]), context)
    validated = validate_predictions(
        predictions_path, context, method="p2",
        required_count=scored.summary["settings"]["required_predictions_per_user"],
        candidate_pool_path=candidate_pool_path, trace_path=trace_path,
    )
    for key, expected in scored.summary["source_sha256"].items():
        if _sha256(Path(source_paths[key])) != expected:
            raise ValueError(f"{key} changed during diagnostics; restart with immutable inputs")
    if validated.report["source_sha256"] != {
        key: scored.summary["source_sha256"][key]
        for key in validated.report["source_sha256"]
    }:
        raise ValueError("validated P2 sources changed during diagnostics")

    selected = sorted(context.expected_user_ids)
    if set(validated.traces) != set(selected) or set(validated.records) != set(selected):
        raise ValueError("validated P2 exports must cover every frozen user")
    rows: list[dict] = []
    actions: Counter[str] = Counter()
    for user_id in selected:
        trace = validated.traces[user_id]
        events = trace["events"]
        actions.update(event["action"] for event in events)
        initial = trace["initial_candidate_movie_ids"]
        expanded = trace["final_candidate_movie_ids"]
        ranking = [item["movie_id"] for item in validated.records[user_id]["recommendations"]]
        added = [movie_id for event in events if event["action"] == "retrieve_candidates"
                 for movie_id in event["added_movie_ids"]]
        truth = relevant[user_id]
        initial_hits = len(set(initial) & truth)
        expanded_hits = len(set(expanded) & truth)
        ranking_hits = len(set(ranking) & truth)
        added_hits = len(set(added) & truth)
        clarification_count = sum(event["action"] == "ask_clarification" for event in events)
        retrieval_count = sum(event["action"] == "retrieve_candidates" for event in events)
        rows.append({
            "user_id": user_id, "split": scored.summary["split"],
            "relevant_targets": len(truth), "initial_pool_size": len(initial),
            "final_pool_size": len(expanded), "final_ranking_size": len(ranking),
            "initial_pool_hits": initial_hits, "expanded_pool_hits": expanded_hits,
            "final_ranking_hits": ranking_hits,
            "initial_pool_recall": _recall(initial_hits, len(truth)),
            "expanded_pool_recall": _recall(expanded_hits, len(truth)),
            "final_ranking_recall": _recall(ranking_hits, len(truth)),
            "added_candidate_count": len(added), "added_relevant_count": added_hits,
            "added_candidate_yield": added_hits / len(added) if added else None,
            "clarification_count": clarification_count, "retrieval_count": retrieval_count,
            "fallback_count": int(trace["termination_reason"] == "fallback"),
            "termination_reason": trace["termination_reason"],
        })

    count = len(rows)
    full_recall_key = f"recall_at_{scored.summary['settings']['required_predictions_per_user']}"
    if full_recall_key not in scored.summary["metrics"]:
        raise ValueError(f"P2 diagnostics require scorer metric {full_recall_key}")
    scored_by_user = {row["user_id"]: row for row in scored.per_user}
    for row in rows:
        if not math.isclose(row["final_ranking_recall"], scored_by_user[row["user_id"]][full_recall_key],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("final-ranking recall disagrees with authoritative scorer")
    added_count = sum(row["added_candidate_count"] for row in rows)
    added_hits = sum(row["added_relevant_count"] for row in rows)
    user_counts = {
        "clarification": sum(row["clarification_count"] > 0 for row in rows),
        "retrieval": sum(row["retrieval_count"] > 0 for row in rows),
        "fallback": sum(row["fallback_count"] > 0 for row in rows),
    }
    aggregate = {
        "initial_pool_recall_macro": math.fsum(row["initial_pool_recall"] for row in rows) / count,
        "expanded_pool_recall_macro": math.fsum(row["expanded_pool_recall"] for row in rows) / count,
        "final_ranking_recall_macro": math.fsum(row["final_ranking_recall"] for row in rows) / count,
        "expanded_minus_initial_pool_recall_macro": math.fsum(
            row["expanded_pool_recall"] - row["initial_pool_recall"] for row in rows
        ) / count,
        "added_candidate_count": added_count, "added_relevant_count": added_hits,
        "added_candidate_yield": added_hits / added_count if added_count else None,
        "clarification_event_count": sum(row["clarification_count"] for row in rows),
        "retrieval_event_count": sum(row["retrieval_count"] for row in rows),
        "fallback_user_count": sum(row["fallback_count"] for row in rows),
        "users_with_clarification": user_counts["clarification"],
        "users_with_retrieval": user_counts["retrieval"],
        "users_with_fallback": user_counts["fallback"],
        "clarification_user_rate": user_counts["clarification"] / count,
        "retrieval_user_rate": user_counts["retrieval"] / count,
        "fallback_user_rate": user_counts["fallback"] / count,
        "action_event_counts": dict(sorted(actions.items())),
        "final_pool_size": _pool_sizes([row["final_pool_size"] for row in rows]),
    }
    if not math.isclose(aggregate["final_ranking_recall_macro"],
                        scored.summary["metrics"][full_recall_key], rel_tol=0, abs_tol=1e-12):
        raise ValueError("macro final-ranking recall disagrees with authoritative scorer")
    summary = {
        "status": "diagnosed", "diagnostics_version": DIAGNOSTICS_VERSION,
        "dataset_version": scored.summary["dataset_version"],
        "evaluation_version": scored.summary["evaluation_version"],
        "config_status": scored.summary["config_status"],
        "method": "p2", "split": scored.summary["split"], "user_count": count,
        "denominator_policy": "all frozen users, including fallbacks; all distinct in-catalog positive targets for each user",
        "yield_policy": "added relevant movies / all added movies; null if no movies were added",
        "aggregate": aggregate,
        "source_sha256": dict(sorted(scored.summary["source_sha256"].items())),
        "source_paths": source_paths,
        "score_summary_sha256": _sha256_bytes(_canonical_bytes(scored.summary)),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
    }
    return P2DiagnosticsResult(summary=summary, per_user=rows)


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _serialized(result: P2DiagnosticsResult) -> dict[str, bytes]:
    if not result.per_user or result.summary.get("user_count") != len(result.per_user):
        raise ValueError("diagnostics require every frozen user")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(result.per_user)
    return {
        "summary.json": _canonical_bytes(result.summary),
        "per_user.csv": output.getvalue().encode("utf-8"),
    }


def write_diagnostics(output_dir: Path, result: P2DiagnosticsResult) -> None:
    """Atomically create two immutable files outside the inputs and data dir."""
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    sources = result.summary.get("source_paths")
    if not isinstance(sources, dict) or "data_dir" not in sources:
        raise ValueError("diagnostics require source_paths and data_dir")
    data_dir = Path(sources["data_dir"]).resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise ValueError("output directory must be outside the processed data directory")
    payloads = _serialized(result)
    for key, source_name in sources.items():
        if key == "data_dir":
            continue
        source = Path(source_name).resolve()
        if resolved == source:
            raise ValueError("output directory must not overwrite an input")
        for name in OUTPUT_FILES:
            destination = (output_dir / name).resolve()
            if destination == source or (destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError("output must not overwrite or alias an input")
    if output_dir.exists():
        if not output_dir.is_dir() or {item.name for item in output_dir.iterdir()} != set(payloads) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw
            for name, raw in payloads.items()
        ):
            raise ValueError("existing diagnostics differ; preserve them and choose a new output directory")
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
                raise ValueError("temporary diagnostics path escaped output parent")
            shutil.rmtree(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_p2(args.config, args.candidate_pool, args.predictions, args.trace)
        write_diagnostics(args.output_dir, result)
    except PredictionValidationError as error:
        print(str(error), file=sys.stderr)
        print(_canonical_bytes(error.report).decode("utf-8"), end="")
        return 1
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"P2 diagnostics failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result.summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
