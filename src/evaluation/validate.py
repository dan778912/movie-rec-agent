"""Validate Phase 1 recommendation JSONL before shared evaluation.

Validation never repairs predictions or silently drops users. A result containing
records is returned only when the entire file passes. This module uses only the
standard library and does not download data or run recommendation models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


METHODS = ("popularity", "bm25", "embedding", "p1")
SPLITS = ("validation", "test")


@dataclass(frozen=True)
class ValidationContext:
    catalog_ids: set[int]
    histories: dict[int, set[int]]
    expected_user_ids: set[int]
    split: str
    source_sha256: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidatedPredictions:
    records: dict[int, dict]
    report: dict


class PredictionValidationError(ValueError):
    """An invalid prediction file, with a complete JSON-compatible issue report."""

    def __init__(self, report: dict):
        self.report = report
        lines = [f"Prediction validation failed: {report['error_count']} issue(s) in {report['predictions']}"]
        for issue in report["errors"][:20]:
            location = []
            if issue["line"] is not None:
                location.append(f"line {issue['line']}")
            if issue["user_id"] is not None:
                location.append(f"user {issue['user_id']}")
            prefix = ", ".join(location) or "file"
            lines.append(f"[{issue['code']}] {prefix}: {issue['message']}")
        if report["error_count"] > 20:
            lines.append("Additional issues are available in the JSON report.")
        super().__init__("\n".join(lines))


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


def _parse_json(raw: bytes):
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _id_list(value, description: str, *, nonempty: bool = False) -> set[int]:
    if not isinstance(value, list) or any(not _positive_int(item) for item in value):
        raise ValueError(f"{description} must be a list of positive integer IDs, excluding booleans")
    if len(set(value)) != len(value):
        raise ValueError(f"{description} contains duplicate IDs")
    if nonempty and not value:
        raise ValueError(f"{description} must not be empty")
    return set(value)


def _check_context(context: ValidationContext) -> None:
    if context.split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not context.catalog_ids or any(not _positive_int(item) for item in context.catalog_ids):
        raise ValueError("catalog_ids must contain positive integer movie IDs")
    if not context.expected_user_ids or any(not _positive_int(user) for user in context.expected_user_ids):
        raise ValueError("expected_user_ids must contain positive integer user IDs")
    for user_id in context.expected_user_ids:
        if user_id not in context.histories:
            raise ValueError(f"expected user {user_id} is missing from the profile file")
        if any(not _positive_int(item) for item in context.histories[user_id]):
            raise ValueError(f"history for user {user_id} contains invalid movie IDs")


def load_context(
    catalog_path: Path, profiles_path: Path, users_path: Path, split: str
) -> ValidationContext:
    """Load a catalog, complete profile histories, and an explicit expected-user list.

    The users JSON object requires ``split`` and a nonempty, unique ``user_ids``
    list. Selection metadata may be included. Profiles can contain additional
    eligible users; prediction files must match the chosen subset exactly.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    catalog_path, profiles_path, users_path = map(Path, (catalog_path, profiles_path, users_path))
    catalog_ids = set()
    with catalog_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "movie_id" not in reader.fieldnames:
            raise ValueError(f"{catalog_path}: catalog CSV requires a movie_id column")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{catalog_path}: duplicate catalog CSV column names")
        for row in reader:
            try:
                movie_id = int(row["movie_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{catalog_path}, line {reader.line_num}: invalid movie_id") from error
            if not _positive_int(movie_id) or movie_id in catalog_ids:
                raise ValueError(f"{catalog_path}, line {reader.line_num}: nonpositive or duplicate movie_id {movie_id}")
            catalog_ids.add(movie_id)

    histories = {}
    with profiles_path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            location = f"{profiles_path}, line {line_number}"
            try:
                record = _parse_json(raw)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"{location}: invalid profile JSON: {error}") from error
            if not isinstance(record, dict) or not _positive_int(record.get("user_id")):
                raise ValueError(f"{location}: profile requires a positive integer user_id")
            user_id = record["user_id"]
            if user_id in histories:
                raise ValueError(f"{location}: duplicate profile for user {user_id}")
            if record.get("split") != split:
                raise ValueError(f"{location}: profile split must be {split!r}")
            histories[user_id] = _id_list(record.get("history_movie_ids"), f"{location}, history_movie_ids")

    try:
        users = _parse_json(users_path.read_bytes())
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{users_path}: invalid user-list JSON: {error}") from error
    if not isinstance(users, dict) or users.get("split") != split:
        raise ValueError(f"{users_path}: user list must be an object with split {split!r}")
    expected_user_ids = _id_list(users.get("user_ids"), f"{users_path}, user_ids", nonempty=True)
    source_sha256 = {
        "catalog": _sha256(catalog_path),
        "profiles": _sha256(profiles_path),
        "users": _sha256(users_path),
    }
    if "source_sha256" in users:
        frozen_sources = users["source_sha256"]
        if not isinstance(frozen_sources, dict):
            raise ValueError(f"{users_path}: source_sha256 must be an object")
        for name in ("catalog", "profiles"):
            expected = frozen_sources.get(name)
            if not isinstance(expected, str) or len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ValueError(f"{users_path}: source_sha256.{name} requires a lowercase SHA-256 hash")
            if source_sha256[name] != expected:
                raise ValueError(f"{name} hash does not match the frozen user list in {users_path}")
    context = ValidationContext(
        catalog_ids=catalog_ids,
        histories=histories,
        expected_user_ids=expected_user_ids,
        split=split,
        source_sha256=source_sha256,
    )
    _check_context(context)
    return context


def _issue(errors: list[dict], code: str, message: str, line=None, user_id=None) -> None:
    errors.append({"code": code, "message": message, "line": line, "user_id": user_id})


def _finite_score(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_predictions(
    path: Path,
    context: ValidationContext,
    *,
    method: str,
    required_count: int = 50,
    candidate_pool_path: Path | None = None,
) -> ValidatedPredictions:
    """Return records only if every row and the entire expected population pass.

    P1 requires an independently validated embedding candidate file and must
    return precisely the same candidate IDs for every user. Prediction errors
    raise PredictionValidationError with all collected issues. Bad reference
    inputs or options raise ValueError/OSError instead.
    """
    _check_context(context)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if type(required_count) is not int or required_count <= 0:
        raise ValueError("required_count must be a positive integer, excluding booleans")
    pools = None
    if method == "p1":
        if candidate_pool_path is None:
            raise ValueError("p1 requires --candidate-pool with the frozen embedding JSONL")
        try:
            pool_result = validate_predictions(
                Path(candidate_pool_path), context, method="embedding", required_count=required_count
            )
        except PredictionValidationError as error:
            raise ValueError(f"Frozen embedding candidate pool is invalid:\n{error}") from error
        pools = {
            user_id: {item["movie_id"] for item in record["recommendations"]}
            for user_id, record in pool_result.records.items()
        }
    elif candidate_pool_path is not None:
        raise ValueError("--candidate-pool is supported only for p1 in Phase 1")

    path = Path(path)
    errors = []
    records = {}
    seen_users = set()
    invalid_users = set()
    lines_read = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            lines_read = line_number
            try:
                record = _parse_json(raw)
            except (ValueError, UnicodeError) as error:
                _issue(errors, "invalid_json", str(error), line_number)
                continue
            if not isinstance(record, dict):
                _issue(errors, "invalid_record", "each line must be a JSON object", line_number)
                continue
            before = len(errors)
            user_id = record.get("user_id")
            location_user = user_id if _positive_int(user_id) else None
            for key in ("user_id", "split", "method", "recommendations"):
                if key not in record:
                    _issue(errors, "missing_field", f"missing required field {key!r}", line_number, location_user)
            if not _positive_int(user_id):
                _issue(errors, "invalid_user_id", "user_id must be a positive integer, excluding booleans", line_number)
                continue
            if user_id in seen_users:
                _issue(errors, "duplicate_user", "user has more than one prediction record", line_number, user_id)
            seen_users.add(user_id)
            if user_id not in context.expected_user_ids:
                _issue(errors, "unexpected_user", "user is outside the declared evaluation subset", line_number, user_id)
            if "split" in record and record["split"] != context.split:
                _issue(errors, "wrong_split", f"expected split {context.split!r}", line_number, user_id)
            if "method" in record and record["method"] != method:
                _issue(errors, "wrong_method", f"expected method {method!r}", line_number, user_id)

            recommendations = record.get("recommendations")
            movie_ids = []
            if "recommendations" in record and not isinstance(recommendations, list):
                _issue(errors, "invalid_recommendations", "recommendations must be an array", line_number, user_id)
            elif isinstance(recommendations, list):
                if len(recommendations) != required_count:
                    _issue(errors, "wrong_length", f"expected exactly {required_count} recommendations; got {len(recommendations)}", line_number, user_id)
                seen_movies = set()
                watched = context.histories.get(user_id, set())
                for position, item in enumerate(recommendations, start=1):
                    label = f"recommendation at position {position}"
                    if not isinstance(item, dict):
                        _issue(errors, "invalid_item", f"{label} must be an object", line_number, user_id)
                        continue
                    for key in ("movie_id", "score", "rank"):
                        if key not in item:
                            _issue(errors, "missing_field", f"{label} is missing {key!r}", line_number, user_id)
                    movie_id = item.get("movie_id")
                    if "movie_id" in item:
                        if not _positive_int(movie_id):
                            _issue(errors, "invalid_movie_id", f"{label}: movie_id must be a positive integer, excluding booleans", line_number, user_id)
                        else:
                            movie_ids.append(movie_id)
                            if movie_id not in context.catalog_ids:
                                _issue(errors, "unknown_movie", f"{label}: movie_id {movie_id} is outside the catalog", line_number, user_id)
                            if movie_id in watched:
                                _issue(errors, "watched_movie", f"{label}: movie_id {movie_id} is already in history_movie_ids", line_number, user_id)
                            if movie_id in seen_movies:
                                _issue(errors, "duplicate_movie", f"{label}: duplicate movie_id {movie_id}", line_number, user_id)
                            seen_movies.add(movie_id)
                    if "rank" in item and (not _positive_int(item["rank"]) or item["rank"] != position):
                        _issue(errors, "invalid_rank", f"{label}: rank must be integer {position} in array order", line_number, user_id)
                    if "score" in item and not _finite_score(item["score"]):
                        _issue(errors, "invalid_score", f"{label}: score must be a finite number, excluding booleans", line_number, user_id)
                if pools is not None and user_id in pools and set(movie_ids) != pools[user_id]:
                    missing = sorted(pools[user_id] - set(movie_ids))
                    extra = sorted(set(movie_ids) - pools[user_id])
                    _issue(errors, "candidate_pool_mismatch", f"P1 must permute the frozen B3 pool; missing IDs {missing}; extra IDs {extra}", line_number, user_id)

            if len(errors) == before:
                records[user_id] = record
            else:
                invalid_users.add(user_id)

    for user_id in invalid_users:
        records.pop(user_id, None)
    missing_users = sorted(context.expected_user_ids - seen_users)
    if missing_users:
        shown = missing_users[:20]
        suffix = " (first 20 shown)" if len(missing_users) > 20 else ""
        _issue(errors, "missing_users", f"no prediction record for expected user IDs {shown}{suffix}")
    source_sha256 = {**context.source_sha256, "predictions": _sha256(path)}
    if candidate_pool_path is not None:
        source_sha256["candidate_pool"] = _sha256(Path(candidate_pool_path))
    report = {
        "status": "invalid" if errors else "valid",
        "predictions": str(path.resolve()),
        "split": context.split,
        "method": method,
        "required_count": required_count,
        "lines_read": lines_read,
        "expected_users": len(context.expected_user_ids),
        "users_with_records": len(seen_users & context.expected_user_ids),
        "valid_users": len(records),
        "prediction_user_coverage": len(records) / len(context.expected_user_ids),
        "missing_users": missing_users,
        "unexpected_users": sorted(seen_users - context.expected_user_ids),
        "error_count": len(errors),
        "errors_by_code": dict(sorted(Counter(issue["code"] for issue in errors).items())),
        "errors": errors,
        "source_sha256": source_sha256,
    }
    if errors:
        raise PredictionValidationError(report)
    return ValidatedPredictions(records=records, report=report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="Member 1 catalog CSV")
    parser.add_argument("--profiles", required=True, type=Path, help="matching split profile JSONL")
    parser.add_argument("--users", required=True, type=Path, help="JSON object with split and user_ids")
    parser.add_argument("--predictions", required=True, type=Path, help="recommendation JSONL to validate")
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--required-count", type=int, default=50)
    parser.add_argument("--candidate-pool", type=Path, help="frozen embedding JSONL; required for p1")
    parser.add_argument("--report", type=Path, help="optionally save the JSON validation report")
    args = parser.parse_args()
    try:
        inputs = [args.catalog, args.profiles, args.users, args.predictions]
        if args.candidate_pool is not None:
            inputs.append(args.candidate_pool)
        if args.report is not None and args.report.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("--report must not overwrite an input file")
        context = load_context(args.catalog, args.profiles, args.users, args.split)
        result = validate_predictions(
            args.predictions, context, method=args.method,
            required_count=args.required_count, candidate_pool_path=args.candidate_pool,
        )
        report, exit_code = result.report, 0
    except PredictionValidationError as error:
        report, exit_code = error.report, 1
        print(str(error), file=sys.stderr)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    try:
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
    except OSError as error:
        print(f"Report output error: {error}", file=sys.stderr)
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
