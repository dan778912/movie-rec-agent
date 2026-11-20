"""Freeze a seeded evaluation-user subset bound to verified data artifacts.

Only the catalog, manifest, the requested split's profiles and positive targets,
and matching statistics (if recorded in the manifest) are read. Selection never
uses model predictions, profile text quality, activity thresholds, or the other
split's targets.

Canonical JSON bytes use UTF-8, sorted object keys, compact separators, no NaN,
and one trailing newline. ``user_ids_sha256`` hashes the same serializer applied
to the sorted ID list alone. The complete output-file hash is reported outside
the file, avoiding a self-referential hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path


DEFAULT_DATA_DIR = Path("data/processed/ml25m-global-v1")
SPLITS = ("validation", "test")


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


def _parse_json(raw: bytes, location: str):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{location}: invalid JSON: {error}") from error


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _csv_id(value, location: str, name: str) -> int:
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{location}: {name} must be a positive integer") from error
    if not _positive_int(result):
        raise ValueError(f"{location}: {name} must be a positive integer")
    return result


def _csv_rows(path: Path, required: set[str]):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: CSV requires columns {', '.join(sorted(required))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate CSV column names")
        for row in reader:
            location = f"{path}, line {reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{location}: malformed CSV row")
            yield row, location


def _load_catalog(path: Path) -> set[int]:
    catalog = set()
    for row, location in _csv_rows(path, {"movie_id"}):
        movie_id = _csv_id(row["movie_id"], location, "movie_id")
        if movie_id in catalog:
            raise ValueError(f"{location}: duplicate catalog movie_id {movie_id}")
        catalog.add(movie_id)
    if not catalog:
        raise ValueError(f"{path}: catalog must not be empty")
    return catalog


def _load_profiles(path: Path, split: str, cutoff: int | None = None) -> dict[int, set[int]]:
    histories = {}
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            location = f"{path}, line {line_number}"
            profile = _parse_json(raw, location)
            if not isinstance(profile, dict) or not _positive_int(profile.get("user_id")):
                raise ValueError(f"{location}: profile requires a positive integer user_id, excluding booleans")
            user_id = profile["user_id"]
            if user_id in histories:
                raise ValueError(f"{location}: duplicate profile for user {user_id}")
            if profile.get("split") != split:
                raise ValueError(f"{location}: profile split must be {split!r}")
            if cutoff is not None:
                latest = profile.get("history_latest_timestamp")
                if type(latest) is not int or not 0 <= latest < cutoff:
                    raise ValueError(f"{location}: history_latest_timestamp must precede the {split} cutoff")
            history = profile.get("history_movie_ids")
            if not isinstance(history, list) or any(not _positive_int(item) for item in history):
                raise ValueError(f"{location}: history_movie_ids must be a list of positive integer IDs, excluding booleans")
            if len(history) != len(set(history)):
                raise ValueError(f"{location}: history_movie_ids contains duplicate IDs")
            histories[user_id] = set(history)
    return histories


def _check_target_population(path: Path, catalog: set[int], histories: dict[int, set[int]]) -> None:
    eligible_users = set()
    seen = set()
    required = {"user_id", "movie_id", "rating", "in_catalog"}
    for row, location in _csv_rows(path, required):
        user_id = _csv_id(row["user_id"], location, "user_id")
        movie_id = _csv_id(row["movie_id"], location, "movie_id")
        try:
            rating = float(row["rating"])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{location}: rating must be a finite positive-target rating between 4 and 5") from error
        if not math.isfinite(rating) or not 4 <= rating <= 5:
            raise ValueError(f"{location}: rating must be a finite positive-target rating between 4 and 5")
        if row["in_catalog"] not in ("0", "1"):
            raise ValueError(f"{location}: in_catalog must be 0 or 1")
        in_catalog = movie_id in catalog
        if (row["in_catalog"] == "1") != in_catalog:
            raise ValueError(f"{location}: in_catalog disagrees with catalog membership for movie_id {movie_id}")
        pair = (user_id, movie_id)
        if pair in seen:
            raise ValueError(f"{location}: duplicate positive target for user {user_id}, movie {movie_id}")
        seen.add(pair)
        if user_id in histories and movie_id in histories[user_id]:
            raise ValueError(f"{location}: positive target movie_id {movie_id} for user {user_id} is already watched")
        if in_catalog:
            eligible_users.add(user_id)
    profile_users = set(histories)
    if profile_users != eligible_users:
        missing = sorted(eligible_users - profile_users)
        extra = sorted(profile_users - eligible_users)
        raise ValueError(
            f"{path}: profile users must equal users with an in-catalog positive target; "
            f"missing profiles={missing[:10]}, profiles without eligible targets={extra[:10]}"
        )


def build_subset(
    data_dir: Path, *, split: str = "validation", size: int = 300, seed: int = 20261001
) -> dict:
    """Return a deterministic user-list payload after checking source integrity.

    The stored ID list is authoritative. The Python version is recorded because
    byte-identical regeneration across different Python versions is not promised.
    Users having only out-of-catalog targets may legitimately lack a profile.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer, excluding booleans")
    if type(seed) is not int:
        raise ValueError("seed must be an integer, excluding booleans")
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = _parse_json(manifest_path.read_bytes(), str(manifest_path))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("version"), str) or not manifest["version"].strip():
        raise ValueError(f"{manifest_path}: manifest requires a nonempty string version")
    recorded = manifest.get("file_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(f"{manifest_path}: manifest requires a file_sha256 object")
    paths = {
        "catalog": data_dir / "catalog.csv",
        "profiles": data_dir / f"{split}_profiles.jsonl",
        "targets": data_dir / f"{split}_targets.csv",
    }
    if "statistics.json" in recorded:
        paths["statistics"] = data_dir / "statistics.json"
    source_hashes = {"manifest": _sha256(manifest_path)}
    for name, path in paths.items():
        expected = recorded.get(path.name)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"{manifest_path}: missing or invalid SHA-256 for {path.name}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 mismatch with manifest; rebuild or obtain matching artifacts")
        source_hashes[name] = actual
    stats = None
    cutoff = None
    if "statistics" in paths:
        stats = _parse_json(paths["statistics"].read_bytes(), str(paths["statistics"]))
        if not isinstance(stats, dict):
            raise ValueError(f"{paths['statistics']}: statistics must be a JSON object")
        cutoff = stats.get(f"{split}_cutoff_unix")
        if not _positive_int(cutoff):
            raise ValueError(f"{paths['statistics']}: requires a positive integer {split}_cutoff_unix")
    catalog = _load_catalog(paths["catalog"])
    histories = _load_profiles(paths["profiles"], split, cutoff)
    _check_target_population(paths["targets"], catalog, histories)
    if stats is not None:
        if type(stats.get("catalog_movies")) is not int or stats["catalog_movies"] != len(catalog):
            raise ValueError(f"{paths['statistics']}: catalog_movies disagrees with the catalog")
        evaluation = stats.get("evaluation")
        split_stats = evaluation.get(split) if isinstance(evaluation, dict) else None
        count = split_stats.get("users_with_in_catalog_positive") if isinstance(split_stats, dict) else None
        if type(count) is not int or count != len(histories):
            raise ValueError(f"{paths['statistics']}: {split} evaluable-user count disagrees with profiles")
    population = sorted(histories)
    if size > len(population):
        raise ValueError(f"size {size} exceeds the eligible {split} population of {len(population)} users")
    selected = sorted(random.Random(seed).sample(population, size))
    return {
        "subset_version": "evaluation-users-v1",
        "dataset_version": manifest["version"],
        "split": split,
        "selection": {
            "algorithm": "python.random.Random.sample_from_sorted_user_ids_v1",
            "seed": seed,
            "size": size,
            "eligible_users": len(population),
            "python_version": platform.python_version(),
        },
        "source_sha256": source_hashes,
        "user_ids": selected,
        "user_ids_sha256": hashlib.sha256(_canonical_bytes(selected)).hexdigest(),
    }


def write_subset(output: Path, payload: dict) -> str:
    """Create an immutable subset, or accept exactly identical existing bytes.

    The return value is the SHA-256 of the complete canonical file. A different
    population, seed, version, or artifact provenance requires a new output name.
    """
    output = Path(output)
    raw = _canonical_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as target:
            target.write(raw)
    except FileExistsError:
        if output.read_bytes() != raw:
            raise ValueError(f"{output}: existing subset differs; preserve it and choose a new versioned output filename")
    return hashlib.sha256(raw).hexdigest()


def _check_output_path(output: Path, data_dir: Path) -> None:
    resolved = output.resolve()
    root = data_dir.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("output must be outside data-dir to protect processed data inputs")
    if output.exists():
        for source in data_dir.iterdir():
            if source.is_file() and output.samefile(source):
                raise ValueError(f"output must not overwrite or alias a data input: {source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _check_output_path(args.output, args.data_dir)
        payload = build_subset(args.data_dir, split=args.split, size=args.size, seed=args.seed)
        file_hash = write_subset(args.output, payload)
    except (ValueError, OSError, csv.Error) as error:
        print(f"Subset selection failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "created_or_verified",
        "output": str(args.output),
        "split": payload["split"],
        "selected_users": len(payload["user_ids"]),
        "eligible_users": payload["selection"]["eligible_users"],
        "user_ids_sha256": payload["user_ids_sha256"],
        "file_sha256": file_hash,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
