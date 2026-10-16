"""Compare compatible evaluation reports with a paired-user percentile bootstrap.

Differences always mean challenger minus baseline. The saved reports are the
inputs: the original dataset and predictions do not need to be available.
Confidence intervals describe sampling uncertainty, not a significance test.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .evaluate import _canonical_bytes, _json_object, _sha256
from .validate import METHODS, SPLITS

RNG = "python.random.Random"
PERCENTILE = "linear_interpolation_at_(B-1)*p"
SHARED_SOURCES = ("config", "manifest", "catalog", "profiles", "targets", "users")
REPORT_FILES = ("summary.json", "per_user.csv", "validation.json")
COUNTS = ("relevant_targets", "positive_targets", "out_of_catalog_positive_targets", "prediction_count")


@dataclass(frozen=True)
class ComparisonResult:
    summary: dict
    per_user: list[dict]


def _integer(value, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _number(value, label: str, lower=0.0, upper=1.0) -> float:
    # Compare bounds before converting arbitrary-size Python integers to floats.
    if type(value) not in (int, float) or not lower <= value <= upper or not math.isfinite(value):
        raise ValueError(f"{label} must be finite and between {lower} and {upper}")
    return float(value)


def _bootstrap_settings(resamples, confidence, seed) -> dict:
    _integer(resamples, "bootstrap.resamples", 1)
    _integer(seed, "bootstrap.seed")
    _number(confidence, "bootstrap.confidence")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap.confidence must be strictly between 0 and 1")
    return {"resamples": resamples, "confidence": confidence, "seed": seed,
            "method": "paired_user_percentile"}


def _scores(values, label: str) -> dict[int, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a nonempty mapping from user IDs to scores")
    result = {}
    for user, score in values.items():
        _integer(user, f"{label} user ID", 1)
        if user in result:
            raise ValueError(f"{label} contains duplicate user {user}")
        result[user] = _number(score, f"{label} user {user} score")
    return result


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(baseline: Mapping[int, float], challenger: Mapping[int, float], *,
                     resamples=10000, confidence=0.95, seed=20261002) -> dict:
    """Resample paired users and return a two-sided percentile interval.

    Each call starts a dedicated seeded RNG, giving every metric the same
    sequence of sampled user indices. Input ordering does not affect results.
    """
    settings = _bootstrap_settings(resamples, confidence, seed)
    baseline, challenger = _scores(baseline, "baseline"), _scores(challenger, "challenger")
    if baseline.keys() != challenger.keys():
        raise ValueError("baseline and challenger must contain exactly the same user IDs")
    users = sorted(baseline)
    count = len(users)
    differences = [challenger[user] - baseline[user] for user in users]
    rng = random.Random(seed)
    draws = sorted(math.fsum(differences[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(resamples))
    tail = (1 - confidence) / 2
    return {"baseline_mean": math.fsum(baseline.values()) / count,
            "challenger_mean": math.fsum(challenger.values()) / count,
            "mean_difference": math.fsum(differences) / count,
            "ci_lower": _percentile(draws, tail), "ci_upper": _percentile(draws, 1 - tail),
            "user_count": count, **{key: settings[key] for key in ("resamples", "confidence", "seed")},
            "percentile_convention": PERCENTILE, "rng": RNG}


def _hashes(value, required, label: str) -> dict:
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError(f"{label} requires hashes for {', '.join(required)}")
    for key, digest in value.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{label}.{key} requires a lowercase SHA-256 hash")
    return value


def _metric_names(settings: dict) -> set[str]:
    names = set()
    limit = _integer(settings.get("required_predictions_per_user"), "required_predictions_per_user", 1)
    if _integer(settings.get("candidate_pool_size"), "candidate_pool_size", 1) != limit:
        raise ValueError("candidate_pool_size must equal required_predictions_per_user")
    for key in ("ranking_k", "retrieval_k"):
        cutoffs = settings.get(key)
        if not isinstance(cutoffs, list) or not cutoffs:
            raise ValueError(f"settings.{key} must be a nonempty cutoff list")
        for cutoff in cutoffs:
            if _integer(cutoff, f"settings.{key} cutoff", 1) > limit:
                raise ValueError(f"settings.{key} cutoff exceeds prediction count")
            names.add(f"recall_at_{cutoff}")
        if len(set(cutoffs)) != len(cutoffs):
            raise ValueError(f"settings.{key} contains duplicate cutoffs")
    if type(settings.get("include_hit_rate")) is not bool:
        raise ValueError("settings.include_hit_rate must be Boolean")
    for cutoff in settings["ranking_k"]:
        names.add(f"ndcg_at_{cutoff}")
        if settings["include_hit_rate"]:
            names.add(f"hit_rate_at_{cutoff}")
    return names


def _coverage(value, label: str) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a coverage object")
    positive = _integer(value.get("positive_targets"), f"{label}.positive_targets", 1)
    inside = _integer(value.get("in_catalog_positive_targets"), f"{label}.in_catalog_positive_targets", 1)
    outside = _integer(value.get("out_of_catalog_positive_targets"), f"{label}.out_of_catalog_positive_targets", 0)
    ratio = _number(value.get("catalog_positive_coverage"), f"{label}.catalog_positive_coverage")
    if positive != inside + outside or not math.isclose(ratio, inside / positive, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label} has inconsistent target counts or coverage")
    return positive, inside, outside


def _load_run(directory: Path) -> tuple[dict, dict[int, dict], dict, dict]:
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in REPORT_FILES}
    summary, validation = _json_object(paths["summary.json"]), _json_object(paths["validation.json"])
    if summary.get("status") != "evaluated":
        raise ValueError("summary status must be evaluated")
    method, split = summary.get("method"), summary.get("split")
    if method not in METHODS or split not in SPLITS:
        raise ValueError("summary method or split is invalid")
    count = _integer(summary.get("user_count"), "summary.user_count", 1)
    settings = summary.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("summary.settings must be an object")
    for key in ("evaluation_version", "dataset_version", "split"):
        if not isinstance(summary.get(key), str) or not summary[key] or summary[key] != settings.get(key):
            raise ValueError(f"summary.{key} disagrees with settings")
    if summary.get("config_status") != settings.get("status") or not isinstance(summary.get("config_status"), str):
        raise ValueError("summary.config_status disagrees with settings")
    if _integer(settings.get("sample_size"), "settings.sample_size", 1) != count:
        raise ValueError("summary.user_count disagrees with settings.sample_size")
    fixed = {"averaging": "macro", "relevance": "binary", "path_resolution": "repository_root",
             "ground_truth": "all_distinct_in_catalog_positive_targets",
             "failure_policy": "reject_invalid_or_missing_prediction_records",
             "p1_candidate_policy": "exact_permutation_of_frozen_embedding_pool"}
    if any(settings.get(key) != value for key, value in fixed.items()):
        raise ValueError("summary.settings contains unsupported evaluation rules")
    threshold = settings.get("positive_rating_at_least")
    if type(threshold) not in (int, float) or threshold != 4.0:
        raise ValueError("settings.positive_rating_at_least must be 4.0, excluding booleans")
    bootstrap = settings.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"resamples", "confidence", "seed", "method"}:
        raise ValueError("settings.bootstrap requires method, resamples, confidence, and seed")
    if bootstrap.get("method") != "paired_user_percentile":
        raise ValueError("settings.bootstrap.method must be paired_user_percentile")
    _bootstrap_settings(bootstrap["resamples"], bootstrap["confidence"], bootstrap["seed"])
    metrics = _metric_names(settings)
    if not isinstance(summary.get("metrics"), dict) or set(summary["metrics"]) != metrics:
        raise ValueError("summary metric columns disagree with settings")
    for name in metrics:
        _number(summary["metrics"][name], f"summary metric {name}")
    hashes = _hashes(summary.get("source_sha256"), (*SHARED_SOURCES, "predictions"), "summary.source_sha256")
    for key in ("manifest", "users"):
        if settings.get(f"{key}_sha256") != hashes[key]:
            raise ValueError(f"settings.{key}_sha256 disagrees with summary source hashes")
    implementation = summary.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("summary.implementation must be an object")
    _hashes(implementation.get("source_sha256"), ("evaluate", "metrics", "validate"), "scoring implementation")
    sources = summary.get("source_paths")
    if not isinstance(sources, dict) or any(key not in sources for key in (*SHARED_SOURCES, "predictions", "data_dir")):
        raise ValueError("summary.source_paths is incomplete")
    if any(not isinstance(path, str) or not path or not Path(path).is_absolute() for path in sources.values()):
        raise ValueError("summary.source_paths must contain nonempty absolute paths")
    if method == "p1":
        if "candidate_pool" not in sources:
            raise ValueError("P1 source_paths requires candidate_pool")
        _hashes(hashes, ("candidate_pool",), "P1 source_sha256")
    elif "candidate_pool" in hashes:
        raise ValueError("only P1 reports may contain a candidate_pool hash")
    if validation.get("status") != "valid" or validation.get("method") != method or validation.get("split") != split:
        raise ValueError("validation report status/method/split disagrees with summary")
    for key in ("expected_users", "users_with_records", "valid_users"):
        if _integer(validation.get(key), f"validation.{key}", 1) != count:
            raise ValueError(f"validation.{key} disagrees with summary.user_count")
    if _integer(validation.get("lines_read"), "validation.lines_read", 1) < count:
        raise ValueError("validation.lines_read is smaller than the user count")
    if _integer(validation.get("required_count"), "validation.required_count", 1) != settings["required_predictions_per_user"]:
        raise ValueError("validation.required_count disagrees with settings")
    if _integer(validation.get("error_count"), "validation.error_count", 0) != 0:
        raise ValueError("validation report contains errors")
    for key, expected in (("errors", []), ("errors_by_code", {}), ("missing_users", []), ("unexpected_users", [])):
        if validation.get(key) != expected:
            raise ValueError(f"validation.{key} must be empty")
    for value in (summary.get("prediction_user_coverage"), validation.get("prediction_user_coverage")):
        if _number(value, "prediction_user_coverage") != 1:
            raise ValueError("comparison requires complete prediction user coverage")
    validation_hashes = _hashes(validation.get("source_sha256"), ("catalog", "profiles", "users", "predictions"), "validation.source_sha256")
    if set(validation_hashes) != ({"catalog", "profiles", "users", "predictions", "candidate_pool"} if method == "p1" else {"catalog", "profiles", "users", "predictions"}):
        raise ValueError("validation source hashes contain unexpected or missing inputs")
    if any(hashes.get(key) != digest for key, digest in validation_hashes.items()):
        raise ValueError("validation source hashes disagree with summary")
    rows = {}
    columns = {"user_id", "method", "split", *COUNTS, *metrics}
    with paths["per_user.csv"].open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != columns:
            raise ValueError("per_user.csv has missing, extra, or duplicate columns")
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("per_user.csv has a malformed row")
            row = dict(record)
            for key in ("user_id", *COUNTS):
                if re.fullmatch(r"0|[1-9][0-9]*", row[key]) is None:
                    raise ValueError(f"per_user.csv {key} must be a nonnegative integer")
                row[key] = int(row[key])
                _integer(row[key], f"per_user.csv {key}", 0 if key == "out_of_catalog_positive_targets" else 1)
            user = row["user_id"]
            if user in rows:
                raise ValueError(f"per_user.csv contains duplicate user {user}")
            if row["method"] != method or row["split"] != split:
                raise ValueError("per_user.csv method/split disagrees with summary")
            if row["positive_targets"] != row["relevant_targets"] + row["out_of_catalog_positive_targets"]:
                raise ValueError("per_user.csv target counts disagree")
            if row["prediction_count"] != settings["required_predictions_per_user"]:
                raise ValueError("per_user.csv prediction_count disagrees with settings")
            for metric in metrics:
                row[metric] = _number(float(row[metric]), f"per_user.csv {metric}")
                if metric.startswith("hit_rate_at_") and row[metric] not in (0, 1):
                    raise ValueError("per-user hit rate must be zero or one")
            rows[user] = row
    if len(rows) != count:
        raise ValueError("per_user.csv user count disagrees with summary")
    for metric in metrics:
        actual = math.fsum(row[metric] for row in rows.values()) / count
        if not math.isclose(actual, summary["metrics"][metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"per_user.csv mean for {metric} disagrees with summary")
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("summary.coverage must be an object")
    selected = _coverage(coverage.get("selected_users"), "selected_users coverage")
    full = _coverage(coverage.get("full_split"), "full_split coverage")
    actual = tuple(sum(row[key] for row in rows.values()) for key in ("positive_targets", "relevant_targets", "out_of_catalog_positive_targets"))
    if selected != actual or any(a > b for a, b in zip(selected, full)):
        raise ValueError("summary coverage disagrees with per-user target counts")
    return summary, rows, paths, {name: _sha256(path) for name, path in paths.items()}


def compare_runs(baseline_dir: Path, challenger_dir: Path) -> ComparisonResult:
    """Load complete scorer reports, verify pairing, and compare every metric."""
    baseline, left, left_paths, left_hashes = _load_run(baseline_dir)
    challenger, right, right_paths, right_hashes = _load_run(challenger_dir)
    for key in ("dataset_version", "evaluation_version", "split", "config_status", "user_count", "coverage"):
        if _canonical_bytes(baseline[key]) != _canonical_bytes(challenger[key]):
            raise ValueError(f"runs have different {key}")
    if _canonical_bytes(baseline["settings"]) != _canonical_bytes(challenger["settings"]):
        raise ValueError("runs have different settings")
    if baseline["implementation"]["source_sha256"] != challenger["implementation"]["source_sha256"]:
        raise ValueError("runs use different scoring implementation source hashes")
    for key in SHARED_SOURCES:
        if baseline["source_sha256"][key] != challenger["source_sha256"][key]:
            raise ValueError(f"runs have different {key} source hashes")
    if left.keys() != right.keys():
        raise ValueError("runs must contain exactly the same user IDs")
    for user in left:
        if any(left[user][key] != right[user][key] for key in COUNTS):
            raise ValueError(f"runs have different target or prediction counts for user {user}")
    if "p1" in (baseline["method"], challenger["method"]):
        if {baseline["method"], challenger["method"]} != {"p1", "embedding"}:
            raise ValueError("P1 comparison requires its embedding counterpart")
        p1, embedding = (baseline, challenger) if baseline["method"] == "p1" else (challenger, baseline)
        if p1["source_sha256"]["candidate_pool"] != embedding["source_sha256"]["predictions"]:
            raise ValueError("P1 candidate pool hash does not match embedding predictions")
    settings = baseline["settings"]["bootstrap"]
    metrics = {}
    for metric in sorted(baseline["metrics"]):
        metrics[metric] = paired_bootstrap({user: row[metric] for user, row in left.items()},
                                         {user: row[metric] for user, row in right.items()},
                                         **{key: settings[key] for key in ("resamples", "confidence", "seed")})
    rows = []
    for user in sorted(left):
        row = {"user_id": user, "baseline_method": baseline["method"],
               "challenger_method": challenger["method"], "split": baseline["split"]}
        for metric in metrics:
            row.update({f"baseline_{metric}": left[user][metric],
                        f"challenger_{metric}": right[user][metric],
                        f"difference_{metric}": right[user][metric] - left[user][metric]})
        rows.append(row)
    summary = {
        "status": "compared", "difference_direction": "challenger_minus_baseline",
        "baseline_method": baseline["method"], "challenger_method": challenger["method"],
        "user_count": len(rows), "metrics": metrics, "config_status": baseline["config_status"],
        "bootstrap": {**settings, "rng": RNG, "percentile_convention": PERCENTILE},
        "shared_experiment": {key: baseline[key] for key in ("dataset_version", "evaluation_version", "split", "settings")},
        "source_sha256": {**{f"baseline_{name.split('.')[0]}": value for name, value in left_hashes.items()},
                          **{f"challenger_{name.split('.')[0]}": value for name, value in right_hashes.items()}},
        "source_paths": {**{f"baseline_{name.split('.')[0]}": str(path) for name, path in left_paths.items()},
                         **{f"challenger_{name.split('.')[0]}": str(path) for name, path in right_paths.items()},
                         "baseline_data_dir": baseline["source_paths"]["data_dir"],
                         "challenger_data_dir": challenger["source_paths"]["data_dir"],
                         **{f"baseline_original_{key}": value for key, value in baseline["source_paths"].items() if key != "data_dir"},
                         **{f"challenger_original_{key}": value for key, value in challenger["source_paths"].items() if key != "data_dir"}},
        "shared_source_sha256": {key: baseline["source_sha256"][key] for key in SHARED_SOURCES},
        "implementation": {"source_sha256": {"bootstrap": _sha256(Path(__file__).resolve()),
                                                **baseline["implementation"]["source_sha256"]}},
        "interval_description": "Paired-user two-sided percentile interval for challenger minus baseline. Users are sampled with replacement; identical sampled indices are used across metrics. No p-value or significance decision is produced.",
    }
    return ComparisonResult(summary, rows)


def write_comparison(output_dir: Path, result: ComparisonResult) -> None:
    """Atomically save an immutable comparison; identical existing files pass."""
    if not result.per_user:
        raise ValueError("cannot write a comparison without users")
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    sources = result.summary.get("source_paths")
    if not isinstance(sources, dict) or any(key not in sources for key in ("baseline_data_dir", "challenger_data_dir", "baseline_summary", "challenger_summary")):
        raise ValueError("comparison requires source_paths for safe output handling")
    for key in ("baseline_data_dir", "challenger_data_dir"):
        data_dir = Path(sources[key]).resolve()
        if resolved == data_dir or data_dir in resolved.parents:
            raise ValueError("output directory must be outside the original data_dir")
    for key in ("baseline_summary", "challenger_summary"):
        report_dir = Path(sources[key]).resolve().parent
        if resolved == report_dir or report_dir in resolved.parents:
            raise ValueError("output directory must be outside input report directories")
    buffer = io.StringIO(newline="")
    columns = ["user_id", "baseline_method", "challenger_method", "split"]
    columns += [f"{prefix}_{metric}" for metric in sorted(result.summary["metrics"])
                for prefix in ("baseline", "challenger", "difference")]
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(result.per_user)
    payloads = {"comparison.json": _canonical_bytes(result.summary),
                "per_user_differences.csv": buffer.getvalue().encode("utf-8")}
    for key, value in sources.items():
        if key.endswith("_data_dir"):
            continue
        source = Path(value)
        if resolved == source.resolve():
            raise ValueError("output directory must not overwrite an input")
        for name in payloads:
            destination = output_dir / name
            if destination.resolve() == source.resolve() or (destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError("output must not overwrite or alias an input")
    if output_dir.exists():
        if not output_dir.is_dir() or {item.name for item in output_dir.iterdir()} != set(payloads) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw for name, raw in payloads.items()
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
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--challenger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare_runs(args.baseline_dir, args.challenger_dir)
        write_comparison(args.output_dir, result)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result.summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
