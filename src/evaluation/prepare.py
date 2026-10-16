"""Prepare teammates' complete prediction exports for the frozen evaluation.

Every source row and recommendation is validated before selecting users or
keeping ranking prefixes. Original files are preserved, and provenance records
the explicit source population, artifact label, and deterministic transformation.
No recommendation models are run by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .evaluate import (
    _canonical_bytes, _check_hash, _configuration, _json_object, _positive_int,
    _sha256, _targets,
)
from .validate import (
    METHODS, PredictionValidationError, ValidationContext, load_context,
    validate_predictions,
)


@dataclass(frozen=True)
class PreparedPredictions:
    summary: dict
    records: list[dict]
    candidate_records: list[dict] | None
    source_validation: dict
    derived_validation: dict


def _inputs(config_path: Path, repository_root: Path | None):
    """Bind configuration, saved population, and all relevant data artifacts."""
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    config = _configuration(config_path)
    split = config["split"]
    data_dir = (root / config["data_dir"]).resolve()
    paths = {
        "config": config_path, "manifest": data_dir / "manifest.json",
        "catalog": data_dir / "catalog.csv", "profiles": data_dir / f"{split}_profiles.jsonl",
        "targets": data_dir / f"{split}_targets.csv", "users": (root / config["users_file"]).resolve(),
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
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
    if "subset_version" in users and users["subset_version"] != "evaluation-users-v1":
        raise ValueError("unsupported frozen user list subset_version")
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
    for key, expected in (("size", config["sample_size"]), ("eligible_users", len(context.histories))):
        if key in selection and (type(selection[key]) is not int or selection[key] != expected):
            raise ValueError(f"frozen user list selection.{key} disagrees with the input population")
    if "seed" in selection and type(selection["seed"]) is not int:
        raise ValueError("frozen user list selection.seed must be an integer")
    if "selection_seed" in config and selection.get("seed") != config["selection_seed"]:
        raise ValueError("config.selection_seed disagrees with the frozen user list")
    _targets(paths["targets"], context)
    return config, context, {**paths, "data_dir": data_dir}, hashes


def _jsonl(records: list[dict]) -> bytes:
    return b"".join(_canonical_bytes(record) for record in records)


def _identity(config: dict, paths: dict, hashes: dict) -> dict:
    return {
        "evaluation_version": config["evaluation_version"], "dataset_version": config["dataset_version"],
        "config_status": config["status"], "split": config["split"],
        "source_sha256": hashes, "source_paths": {key: str(path) for key, path in paths.items()},
        "serialization": "UTF-8 JSON with sorted object keys, compact separators, finite numbers, and one newline per record",
        "implementation": {"source_sha256": {"prepare": _sha256(Path(__file__).resolve())}},
    }


def prepare_predictions(
    config_path: Path, source_path: Path, *, method: str, source_count: int,
    source_users: str, source_label: str, candidate_pool_path: Path | None = None,
    expected_source_sha256: str | None = None, source_metadata_path: Path | None = None,
    repository_root: Path | None = None,
) -> PreparedPredictions:
    """Validate the entire declared source, then select frozen users and prefixes.

    ``source_label`` and optional metadata record owner-supplied artifact identity;
    they do not establish which model actually produced the recommendations.
    P1 must already rank exactly the configured candidate count and use the
    matching source-population embedding pool. P1 is never truncated.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if source_users not in ("all", "selected"):
        raise ValueError("source_users must be 'all' or 'selected'")
    if not _positive_int(source_count):
        raise ValueError("source_count must be a positive integer, excluding booleans")
    if not isinstance(source_label, str) or not source_label.strip():
        raise ValueError("source_label must be a nonempty artifact/model label")
    if method == "p1" and candidate_pool_path is None:
        raise ValueError("p1 requires --candidate-pool with the matching frozen embedding JSONL")
    if method != "p1" and candidate_pool_path is not None:
        raise ValueError("--candidate-pool is supported only for p1")
    config, selected_context, paths, hashes = _inputs(config_path, repository_root)
    count = config["required_predictions_per_user"]
    if source_count < count:
        raise ValueError("source_count must be at least required_predictions_per_user")
    if method == "p1" and source_count != count:
        raise ValueError("P1 source_count must equal the configured candidate count; P1 cannot be truncated")
    paths["predictions"] = Path(source_path).resolve()
    hashes["predictions"] = _sha256(paths["predictions"])
    if expected_source_sha256 is not None:
        _check_hash(hashes["predictions"], expected_source_sha256, "expected source predictions")
    metadata = None
    if source_metadata_path is not None:
        paths["source_metadata"] = Path(source_metadata_path).resolve()
        hashes["source_metadata"] = _sha256(paths["source_metadata"])
        metadata = _json_object(paths["source_metadata"])
    if candidate_pool_path is not None:
        paths["candidate_pool"] = Path(candidate_pool_path).resolve()
        hashes["candidate_pool"] = _sha256(paths["candidate_pool"])
    source_context = ValidationContext(
        catalog_ids=selected_context.catalog_ids, histories=selected_context.histories,
        expected_user_ids=(set(selected_context.histories) if source_users == "all"
                           else set(selected_context.expected_user_ids)),
        split=selected_context.split, source_sha256=dict(selected_context.source_sha256),
    )
    # This validation includes users not selected and positions beyond the prefix.
    source = validate_predictions(paths["predictions"], source_context, method=method,
                                  required_count=source_count, candidate_pool_path=paths.get("candidate_pool"))
    for key in ("predictions", "candidate_pool"):
        if key in hashes:
            _check_hash(source.report["source_sha256"][key], hashes[key], f"source {key} changed during validation")
    selected = sorted(selected_context.expected_user_ids)
    records = [
        {**source.records[user], "recommendations": source.records[user]["recommendations"][:count]}
        for user in selected
    ]
    candidates = None
    if method == "p1":
        pool = validate_predictions(paths["candidate_pool"], source_context,
                                    method="embedding", required_count=count)
        candidates = [{**pool.records[user], "recommendations": pool.records[user]["recommendations"][:count]}
                      for user in selected]
    prediction_bytes = _jsonl(records)
    pool_bytes = _jsonl(candidates) if candidates is not None else None
    # Temporary validation writes never escape into a saved report or output.
    with tempfile.TemporaryDirectory(prefix="evaluation-preparation-") as scratch:
        scratch_path = Path(scratch)
        derived_path = scratch_path / "predictions.jsonl"
        derived_path.write_bytes(prediction_bytes)
        derived_pool = None
        if pool_bytes is not None:
            derived_pool = scratch_path / "candidate_pool.jsonl"
            derived_pool.write_bytes(pool_bytes)
        try:
            derived = validate_predictions(derived_path, selected_context, method=method,
                                           required_count=count, candidate_pool_path=derived_pool)
        except PredictionValidationError as error:
            report = {**error.report, "predictions": "derived predictions.jsonl"}
            raise PredictionValidationError(report) from error
    source_report = {**source.report, "declared_population": source_users,
                     "expected_user_ids_sha256": hashlib.sha256(_canonical_bytes(sorted(source_context.expected_user_ids))).hexdigest()}
    derived_report = {**derived.report, "predictions": "derived predictions.jsonl"}
    output_hashes = {"predictions": hashlib.sha256(prediction_bytes).hexdigest()}
    if pool_bytes is not None:
        output_hashes["candidate_pool"] = hashlib.sha256(pool_bytes).hexdigest()
    summary = {
        **_identity(config, paths, hashes), "status": "prepared", "method": method,
        "source_label": source_label, "source_label_verification": "owner_supplied_unverified",
        "source_users": source_users, "source_count": source_count,
        "source_user_count": len(source.records), "selected_users": len(selected), "output_count": count,
        "output_sha256": output_hashes, "expected_source_sha256_verified": expected_source_sha256 is not None,
        "transformation": {
            "user_selection": "exact frozen user IDs, output records sorted by user_id",
            "recommendations": "retain the first configured count in original array order; preserve ranks, scores, and extra fields",
            "source_preserved": True, "scores_resorted": False,
            "whole_source_validated_before_selection": True,
        },
    }
    if metadata is not None:
        summary.update(source_metadata=metadata, source_metadata_verification="owner_supplied_unverified")
    return PreparedPredictions(summary, records, candidates, source_report, derived_report)


def _write_directory(output_dir: Path, payloads: dict[str, bytes], summary: dict) -> None:
    """Publish a complete immutable directory while protecting every input."""
    output_dir = Path(output_dir)
    sources = summary.get("source_paths")
    if not isinstance(sources, dict) or "data_dir" not in sources:
        raise ValueError("preparation provenance requires source_paths for safe output handling")
    resolved, data_dir = output_dir.resolve(), Path(sources["data_dir"]).resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise ValueError("output directory must be outside data_dir to protect input artifacts")
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    for key, raw_path in sources.items():
        if key == "data_dir":
            continue
        source = Path(raw_path)
        if resolved == source.resolve():
            raise ValueError(f"output directory must not overwrite an input: {source}")
        for name in payloads:
            destination = output_dir / name
            if destination.is_symlink() or destination.resolve() == source.resolve() or (
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
            raise ValueError("existing preparation differs; preserve it and choose a new output directory")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, raw in payloads.items():
            (staged / name).write_bytes(raw)
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("output directory appeared during staging; choose a new output directory")
        os.rename(staged, output_dir)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def write_prepared(output_dir: Path, result: PreparedPredictions) -> dict:
    """Write validated predictions and audit reports, or accept identical files."""
    predictions = _jsonl(result.records)
    if hashlib.sha256(predictions).hexdigest() != result.summary.get("output_sha256", {}).get("predictions"):
        raise ValueError("prepared prediction records disagree with their recorded hash")
    payloads = {"predictions.jsonl": predictions, "preparation.json": _canonical_bytes(result.summary),
                "source_validation.json": _canonical_bytes(result.source_validation),
                "derived_validation.json": _canonical_bytes(result.derived_validation)}
    if result.candidate_records is not None:
        candidates = _jsonl(result.candidate_records)
        if hashlib.sha256(candidates).hexdigest() != result.summary.get("output_sha256", {}).get("candidate_pool"):
            raise ValueError("prepared candidate records disagree with their recorded hash")
        payloads["candidate_pool.jsonl"] = candidates
    _write_directory(output_dir, payloads, result.summary)
    return result.summary


def export_users(config_path: Path, output_dir: Path, *, repository_root: Path | None = None) -> dict:
    """Export the frozen IDs as a plain array for teammates' existing loaders."""
    config, context, paths, hashes = _inputs(config_path, repository_root)
    user_bytes = _canonical_bytes(sorted(context.expected_user_ids))
    summary = {
        **_identity(config, paths, hashes), "status": "users_exported",
        "selected_users": len(context.expected_user_ids),
        "output_sha256": {"user_ids": hashlib.sha256(user_bytes).hexdigest()},
        "transformation": "export sorted frozen user IDs as a plain JSON array; no resampling",
    }
    _write_directory(Path(output_dir), {"user_ids.json": user_bytes,
                     "user_ids.provenance.json": _canonical_bytes(summary)}, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predictions = subparsers.add_parser("predictions", help="validate an entire export, then select frozen users and prefixes")
    predictions.add_argument("--config", required=True, type=Path)
    predictions.add_argument("--source", required=True, type=Path)
    predictions.add_argument("--method", required=True, choices=METHODS)
    predictions.add_argument("--source-count", required=True, type=int)
    predictions.add_argument("--source-users", required=True, choices=("all", "selected"))
    predictions.add_argument("--source-label", required=True)
    predictions.add_argument("--output-dir", required=True, type=Path)
    predictions.add_argument("--candidate-pool", type=Path)
    predictions.add_argument("--expected-source-sha256")
    predictions.add_argument("--source-metadata", type=Path)
    users = subparsers.add_parser("users", help="export the frozen IDs as a plain JSON array")
    users.add_argument("--config", required=True, type=Path)
    users.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "users":
            summary = export_users(args.config, args.output_dir)
        else:
            result = prepare_predictions(
                args.config, args.source, method=args.method, source_count=args.source_count,
                source_users=args.source_users, source_label=args.source_label,
                candidate_pool_path=args.candidate_pool, expected_source_sha256=args.expected_source_sha256,
                source_metadata_path=args.source_metadata,
            )
            summary = write_prepared(args.output_dir, result)
    except PredictionValidationError as error:
        print(str(error), file=sys.stderr)
        print(_canonical_bytes(error.report).decode("utf-8"), end="")
        return 1
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
