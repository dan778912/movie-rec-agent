"""Score validated Phase 1 rankings against a frozen, complete held-out truth.

Configuration paths resolve from the repository root; CLI paths resolve from the
working directory. This command performs no training or bootstrap comparison.
Invalid inputs never produce partial scores or an output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .metrics import hit_rate_at_k, mean_metrics, ndcg_at_k, recall_at_k
from .validate import METHODS, SPLITS, PredictionValidationError, load_context, validate_predictions


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict
    per_user: list[dict]
    validation_report: dict


def _canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite constant {value!r} is not valid JSON")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite JSON number {value!r}")
    return result


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_json_float)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _check_hash(actual: str, expected, description: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{description}: requires a lowercase SHA-256 hash")
    if actual != expected:
        raise ValueError(f"{description}: SHA-256 mismatch; use matching frozen artifacts")


def _configuration(path: Path) -> dict:
    config = _json_object(path)
    fixed = {
        "path_resolution": "repository_root", "relevance": "binary", "averaging": "macro",
        "ground_truth": "all_distinct_in_catalog_positive_targets",
        "failure_policy": "reject_invalid_or_missing_prediction_records",
        "p1_candidate_policy": "exact_permutation_of_frozen_embedding_pool",
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"config.{key} must be {expected!r}")
    threshold = config.get("positive_rating_at_least")
    if type(threshold) not in (int, float) or threshold != 4.0:
        raise ValueError("config.positive_rating_at_least must be 4.0")
    for key in ("evaluation_version", "dataset_version", "status", "data_dir", "users_file"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"config.{key} must be a nonempty string")
    if config.get("split") not in SPLITS:
        raise ValueError(f"config.split must be one of {SPLITS}")
    for key in ("sample_size", "required_predictions_per_user", "candidate_pool_size"):
        if not _positive_int(config.get(key)):
            raise ValueError(f"config.{key} must be a positive integer, excluding booleans")
    count = config["required_predictions_per_user"]
    if config["candidate_pool_size"] != count:
        raise ValueError("config.candidate_pool_size must equal required_predictions_per_user")
    for key in ("ranking_k", "retrieval_k"):
        values = config.get(key)
        if not isinstance(values, list) or not values or any(not _positive_int(k) or k > count for k in values):
            raise ValueError(f"config.{key} must contain positive integer cutoffs no larger than {count}")
        if len(values) != len(set(values)):
            raise ValueError(f"config.{key} contains duplicate cutoffs")
    if type(config.get("include_hit_rate")) is not bool:
        raise ValueError("config.include_hit_rate must be a Boolean")
    if "selection_seed" in config and type(config["selection_seed"]) is not int:
        raise ValueError("config.selection_seed must be an integer, excluding booleans")
    return config


def _csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"user_id", "movie_id", "rating", "in_catalog"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: target CSV requires columns {', '.join(sorted(required))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate target CSV column names")
        for row in reader:
            location = f"{path}, line {reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{location}: malformed target CSV row")
            yield row, location


def _targets(path: Path, context) -> tuple[dict[int, set[int]], Counter]:
    relevant, positives, seen = defaultdict(set), Counter(), set()
    for row, location in _csv_rows(path):
        try:
            user_id, movie_id = int(row["user_id"]), int(row["movie_id"])
            rating = float(row["rating"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location}: invalid target IDs or rating") from error
        if user_id <= 0 or movie_id <= 0:
            raise ValueError(f"{location}: target IDs must be positive integers")
        if not math.isfinite(rating) or not 4 <= rating <= 5:
            raise ValueError(f"{location}: target rating must be finite and between 4 and 5")
        if row["in_catalog"] not in ("0", "1"):
            raise ValueError(f"{location}: in_catalog must be 0 or 1")
        in_catalog = movie_id in context.catalog_ids
        if (row["in_catalog"] == "1") != in_catalog:
            raise ValueError(f"{location}: in_catalog disagrees with catalog membership")
        if (user_id, movie_id) in seen:
            raise ValueError(f"{location}: duplicate target for user {user_id}, movie {movie_id}")
        seen.add((user_id, movie_id))
        if movie_id in context.histories.get(user_id, set()):
            raise ValueError(f"{location}: positive target overlaps the user's watched history")
        positives[user_id] += 1
        if in_catalog:
            relevant[user_id].add(movie_id)
    if set(relevant) != set(context.histories):
        raise ValueError("profile users must equal users with an in-catalog positive target")
    return dict(relevant), positives


def _coverage(users, relevant: dict[int, set[int]], positives: Counter) -> dict:
    all_positive = sum(positives[user] for user in users)
    in_catalog = sum(len(relevant.get(user, set())) for user in users)
    return {
        "positive_targets": all_positive, "in_catalog_positive_targets": in_catalog,
        "out_of_catalog_positive_targets": all_positive - in_catalog,
        "catalog_positive_coverage": in_catalog / all_positive if all_positive else 0.0,
    }


def _implementation(root: Path) -> dict:
    """Record source identity even when the working tree is uncommitted."""
    directory = Path(__file__).resolve().parent
    provenance = {"source_sha256": {
        name: _sha256(directory / f"{name}.py") for name in ("evaluate", "metrics", "validate")
    }, "git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True, timeout=3)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                cwd=root, check=True, capture_output=True, text=True, timeout=3)
        provenance.update(git_commit=commit.stdout.strip(), git_dirty=bool(status.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return provenance


def evaluate(
    config_path: Path, predictions_path: Path, *, method: str,
    candidate_pool_path: Path | None = None, repository_root: Path | None = None,
) -> EvaluationResult:
    """Validate provenance and predictions, then macro-average the frozen users.

    Every selected user's truth includes all their in-catalog held-out positives,
    including those absent from retrieval. No users are silently discarded.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    config_path, predictions_path = Path(config_path).resolve(), Path(predictions_path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    config = _configuration(config_path)
    split = config["split"]
    data_dir = (root / config["data_dir"]).resolve()
    paths = {
        "config": config_path, "manifest": data_dir / "manifest.json",
        "catalog": data_dir / "catalog.csv", "profiles": data_dir / f"{split}_profiles.jsonl",
        "targets": data_dir / f"{split}_targets.csv", "users": (root / config["users_file"]).resolve(),
        "predictions": predictions_path,
    }
    if candidate_pool_path is not None:
        paths["candidate_pool"] = Path(candidate_pool_path).resolve()
    hashes = {key: _sha256(paths[key]) for key in ("config", "manifest", "catalog", "profiles", "targets", "users")}
    _check_hash(hashes["manifest"], config.get("manifest_sha256"), "config.manifest_sha256")
    _check_hash(hashes["users"], config.get("users_sha256"), "config.users_sha256")
    manifest = _json_object(paths["manifest"])
    if manifest.get("version") != config["dataset_version"]:
        raise ValueError("manifest version disagrees with config.dataset_version")
    recorded = manifest.get("file_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("manifest.file_sha256 must be an object")
    for key in ("catalog", "profiles", "targets"):
        _check_hash(hashes[key], recorded.get(paths[key].name), f"manifest {paths[key].name}")
    manifest_config = manifest.get("config", {})
    if not isinstance(manifest_config, dict):
        raise ValueError("manifest.config must be an object when present")
    if "positive_rating_at_least" in manifest_config and (
        type(manifest_config["positive_rating_at_least"]) not in (int, float)
        or manifest_config["positive_rating_at_least"] != 4.0
    ):
        raise ValueError("manifest positive threshold disagrees with the evaluation threshold")
    users = _json_object(paths["users"])
    if users.get("dataset_version") != config["dataset_version"] or users.get("split") != split:
        raise ValueError("frozen user list dataset_version/split disagrees with configuration")
    frozen = users.get("source_sha256")
    if not isinstance(frozen, dict):
        raise ValueError("frozen user list requires source_sha256 provenance")
    for key in ("manifest", "catalog", "profiles", "targets"):
        _check_hash(hashes[key], frozen.get(key), f"frozen user list {key}")
    context = load_context(paths["catalog"], paths["profiles"], paths["users"], split)
    if len(context.expected_user_ids) != config["sample_size"]:
        raise ValueError("config.sample_size disagrees with the frozen user count")
    if "user_ids_sha256" in users:
        _check_hash(hashlib.sha256(_canonical_bytes(users["user_ids"])).hexdigest(),
                    users["user_ids_sha256"], "frozen user_ids_sha256")
    selection = users.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("frozen user list selection must be an object when present")
    for name, expected in (("size", config["sample_size"]), ("eligible_users", len(context.histories))):
        if name in selection and (type(selection[name]) is not int or selection[name] != expected):
            raise ValueError(f"frozen user list selection.{name} disagrees with the input population")
    if "seed" in selection and type(selection["seed"]) is not int:
        raise ValueError("frozen user list selection.seed must be an integer")
    if "selection_seed" in config and selection.get("seed") != config["selection_seed"]:
        raise ValueError("config.selection_seed disagrees with the frozen user list")
    relevant, positives = _targets(paths["targets"], context)
    validated = validate_predictions(predictions_path, context, method=method,
                                     required_count=config["required_predictions_per_user"],
                                     candidate_pool_path=paths.get("candidate_pool"))
    hashes.update(validated.report["source_sha256"])
    rows, score_rows = [], []
    selected = sorted(context.expected_user_ids)
    for user_id in selected:
        ranking = [item["movie_id"] for item in validated.records[user_id]["recommendations"]]
        scores = {}
        for k in sorted(set(config["ranking_k"]) | set(config["retrieval_k"])):
            scores[f"recall_at_{k}"] = recall_at_k(ranking, relevant[user_id], k)
        for k in sorted(config["ranking_k"]):
            scores[f"ndcg_at_{k}"] = ndcg_at_k(ranking, relevant[user_id], k)
            if config["include_hit_rate"]:
                scores[f"hit_rate_at_{k}"] = hit_rate_at_k(ranking, relevant[user_id], k)
        score_rows.append(scores)
        rows.append({
            "user_id": user_id, "method": method, "split": split,
            "relevant_targets": len(relevant[user_id]), "positive_targets": positives[user_id],
            "out_of_catalog_positive_targets": positives[user_id] - len(relevant[user_id]),
            "prediction_count": len(ranking), **dict(sorted(scores.items())),
        })
    summary = {
        "status": "evaluated", "evaluation_version": config["evaluation_version"],
        "dataset_version": config["dataset_version"], "config_status": config["status"],
        "method": method, "split": split, "user_count": len(selected), "metrics": mean_metrics(score_rows),
        "metric_description": "Binary relevance; macro mean over every frozen user; all in-catalog positive targets form truth, including retrieval misses. Each recall_at_K scores the submitted ranking order. For P1, recall_at_20 describes reranked positions, not original B3 retrieval; recall at the full pool size is candidate coverage and cannot improve through reordering.",
        "coverage": {"full_split": _coverage(positives, relevant, positives),
                     "selected_users": _coverage(selected, relevant, positives)},
        "coverage_description": "Catalog positive coverage is in-catalog positive targets / all positive targets, counted per user/movie pair; full_split includes users with only out-of-catalog targets.",
        "prediction_user_coverage": validated.report["prediction_user_coverage"],
        "source_sha256": hashes, "source_paths": {**{key: str(path) for key, path in paths.items()}, "data_dir": str(data_dir)},
        "settings": config, "bootstrap_applied": False,
        "runtime": {"python_version": platform.python_version(), "implementation": platform.python_implementation()},
        "implementation": _implementation(root),
    }
    return EvaluationResult(summary=summary, per_user=rows, validation_report=validated.report)


def _serialized_results(result: EvaluationResult) -> dict[str, bytes]:
    if not result.per_user:
        raise ValueError("cannot write evaluation results without per-user scores")
    columns = ["user_id", "method", "split", "relevant_targets", "positive_targets",
               "out_of_catalog_positive_targets", "prediction_count"]
    columns += sorted(set(result.per_user[0]) - set(columns))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(result.per_user)
    return {"summary.json": _canonical_bytes(result.summary),
            "per_user.csv": buffer.getvalue().encode("utf-8"),
            "validation.json": _canonical_bytes(result.validation_report)}


def write_results(output_dir: Path, result: EvaluationResult) -> None:
    """Create an immutable result directory, or accept identical existing files.

    A fresh directory is fully staged before rename. Different existing results
    require a new output directory; input aliases and processed-data paths fail.
    """
    output_dir = Path(output_dir)
    sources = result.summary.get("source_paths")
    if not isinstance(sources, dict) or "data_dir" not in sources:
        raise ValueError("result requires source_paths for safe output handling")
    data_dir = Path(sources["data_dir"]).resolve()
    resolved = output_dir.resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise ValueError("output directory must be outside data_dir to protect input artifacts")
    inputs = [Path(path) for key, path in sources.items() if key != "data_dir"]
    payloads = _serialized_results(result)
    for source in inputs:
        if resolved == source.resolve():
            raise ValueError(f"output directory must not overwrite an input: {source}")
        for name in payloads:
            destination = output_dir / name
            if destination.resolve() == source.resolve() or (
                destination.exists() and source.exists() and destination.samefile(source)
            ):
                raise ValueError(f"output must not overwrite or alias an input: {source}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("output directory is an existing file")
        if {item.name for item in output_dir.iterdir()} != set(payloads) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw
            for name, raw in payloads.items()
        ):
            raise ValueError("existing results differ; preserve them and choose a new output directory")
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
            shutil.rmtree(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.config, args.predictions, method=args.method,
                          candidate_pool_path=args.candidate_pool)
        write_results(args.output_dir, result)
    except PredictionValidationError as error:
        print(str(error), file=sys.stderr)
        print(_canonical_bytes(error.report).decode("utf-8"), end="")
        return 1
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result.summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
