"""Validate recommendation JSONL and Phase 2 execution traces before scoring.

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


METHODS = ("popularity", "bm25", "embedding", "p1", "p2")
SPLITS = ("validation", "test")
P2_TRACE_SCHEMA = "p2-trace-v1"
P2_MAX_CONTROL_EVENTS = 8
P2_MAX_EVENTS = P2_MAX_CONTROL_EVENTS + 2
P2_EXPLANATION_COUNT = 10
P2_ACTIONS = (
    "assess_sufficiency", "ask_clarification", "evaluate_candidates",
    "retrieve_candidates", "rank_candidates", "explain_recommendations",
)


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
    traces: dict[int, dict] = field(default_factory=dict)


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


def _trace_ids(value, label: str, errors: list[dict], line: int, user_id: int,
               context: ValidationContext, *, nonempty: bool = False) -> list[int] | None:
    if not isinstance(value, list) or any(not _positive_int(item) for item in value):
        _issue(errors, "trace_invalid_ids", f"{label} must be a list of positive integer IDs", line, user_id)
        return None
    if nonempty and not value:
        _issue(errors, "trace_invalid_ids", f"{label} must not be empty", line, user_id)
    if len(set(value)) != len(value):
        _issue(errors, "trace_duplicate_movie", f"{label} contains duplicate IDs", line, user_id)
    watched = context.histories.get(user_id, set())
    for movie_id in value:
        if movie_id not in context.catalog_ids:
            _issue(errors, "trace_unknown_movie", f"{label} contains catalog-unknown movie {movie_id}", line, user_id)
        if movie_id in watched:
            _issue(errors, "trace_watched_movie", f"{label} contains watched movie {movie_id}", line, user_id)
    return value


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_p2_traces(path: Path, context: ValidationContext, pools: dict[int, list[int]],
                        predictions: dict[int, dict], required_count: int,
                        errors: list[dict]) -> tuple[dict, dict[int, dict]]:
    """Check a complete one-record-per-user trace against frozen B3 and final rankings."""
    seen_users = set()
    trace_records = {}
    lines_read = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            lines_read = line_number
            try:
                trace = _parse_json(raw)
            except (ValueError, UnicodeError) as error:
                _issue(errors, "trace_invalid_json", f"trace JSON: {error}", line_number)
                continue
            if not isinstance(trace, dict):
                _issue(errors, "trace_invalid_record", "trace line must be an object", line_number)
                continue
            user_id = trace.get("user_id")
            if not _positive_int(user_id):
                _issue(errors, "trace_invalid_user_id", "trace user_id must be a positive integer", line_number)
                continue
            if user_id in seen_users:
                _issue(errors, "trace_duplicate_user", "user has more than one trace", line_number, user_id)
            seen_users.add(user_id)
            trace_records[user_id] = trace
            if user_id not in context.expected_user_ids:
                _issue(errors, "trace_unexpected_user", "trace user is outside the evaluation subset", line_number, user_id)
            for key in ("schema_version", "user_id", "split", "method", "initial_candidate_movie_ids",
                        "events", "final_candidate_movie_ids", "termination_reason"):
                if key not in trace:
                    _issue(errors, "trace_missing_field", f"trace is missing {key!r}", line_number, user_id)
            for key, expected in (("schema_version", P2_TRACE_SCHEMA), ("split", context.split),
                                  ("method", "p2")):
                if key in trace and trace[key] != expected:
                    _issue(errors, "trace_wrong_value", f"trace {key} must be {expected!r}", line_number, user_id)
            termination = trace.get("termination_reason")
            if termination not in ("recommend", "fallback"):
                _issue(errors, "trace_wrong_value", "termination_reason must be 'recommend' or 'fallback'",
                       line_number, user_id)
            if termination == "fallback" and not _text(trace.get("fallback_reason")):
                _issue(errors, "trace_missing_field", "fallback requires nonempty fallback_reason",
                       line_number, user_id)
            if termination == "fallback" and trace.get("fallback_source") != "frozen_b3_order":
                _issue(errors, "trace_wrong_value", "fallback_source must be 'frozen_b3_order'",
                       line_number, user_id)
            initial = _trace_ids(trace.get("initial_candidate_movie_ids"), "initial_candidate_movie_ids",
                                 errors, line_number, user_id, context)
            expected_initial = pools.get(user_id)
            if initial is not None and expected_initial is not None and initial != expected_initial:
                _issue(errors, "trace_initial_pool_mismatch", "initial pool must match frozen B3 IDs in rank order",
                       line_number, user_id)
            # Provenance is always reconstructed from the validated B3 input, not a
            # possibly corrupted initial pool declared by the trace.
            current_pool = list(expected_initial or [])
            events = trace.get("events")
            if not isinstance(events, list):
                _issue(errors, "trace_invalid_events", "events must be an array", line_number, user_id)
                events = []
            control_count = sum(isinstance(event, dict) and event.get("action") not in
                                ("rank_candidates", "explain_recommendations") for event in events)
            if control_count > P2_MAX_CONTROL_EVENTS or len(events) > P2_MAX_EVENTS:
                _issue(errors, "trace_loop_limit", f"at most {P2_MAX_CONTROL_EVENTS} control and {P2_MAX_EVENTS} total events are allowed", line_number, user_id)
            assessed = evaluated = ranked = explained = False
            latest_assessment = latest_evaluation = None
            ranking = None
            if user_id in predictions:
                ranking = [item["movie_id"] for item in predictions[user_id]["recommendations"]]
            for position, event in enumerate(events, start=1):
                label = f"event {position}"
                if not isinstance(event, dict):
                    _issue(errors, "trace_invalid_event", f"{label} must be an object", line_number, user_id)
                    continue
                action = event.get("action")
                if action not in P2_ACTIONS:
                    _issue(errors, "trace_invalid_action", f"{label} has unsupported action {action!r}",
                           line_number, user_id)
                    continue
                if explained or (ranked and action != "explain_recommendations"):
                    _issue(errors, "trace_action_order", f"{label} appears after final ranking/explanation",
                           line_number, user_id)
                if action == "assess_sufficiency":
                    assessed = True
                    latest_evaluation = None
                    if type(event.get("sufficient")) is not bool:
                        _issue(errors, "trace_invalid_decision", f"{label} requires Boolean sufficient",
                               line_number, user_id)
                    else:
                        latest_assessment = event["sufficient"]
                elif action == "ask_clarification":
                    if latest_assessment is not False:
                        _issue(errors, "trace_action_order", f"{label} requires a preceding insufficient assessment",
                               line_number, user_id)
                    for key in ("question", "answer", "answer_source"):
                        if not _text(event.get(key)):
                            _issue(errors, "trace_invalid_clarification", f"{label} requires nonempty {key}",
                                   line_number, user_id)
                    latest_assessment = None
                    latest_evaluation = None
                elif action == "evaluate_candidates":
                    evaluated = True
                    if latest_assessment is not True:
                        _issue(errors, "trace_action_order", f"{label} requires a sufficient information assessment",
                               line_number, user_id)
                    ids = _trace_ids(event.get("candidate_movie_ids"), f"{label}.candidate_movie_ids",
                                     errors, line_number, user_id, context, nonempty=True)
                    if ids is not None and ids != current_pool:
                        _issue(errors, "trace_pool_mismatch", f"{label} must evaluate the current pool in order",
                               line_number, user_id)
                    if type(event.get("pool_sufficient")) is not bool:
                        _issue(errors, "trace_invalid_decision", f"{label} requires Boolean pool_sufficient",
                               line_number, user_id)
                    else:
                        latest_evaluation = event["pool_sufficient"]
                elif action == "retrieve_candidates":
                    if latest_evaluation is not False:
                        _issue(errors, "trace_action_order", f"{label} requires a preceding insufficient pool evaluation",
                               line_number, user_id)
                    if not _text(event.get("index_version")):
                        _issue(errors, "trace_invalid_retrieval", f"{label} requires nonempty index_version",
                               line_number, user_id)
                    added = _trace_ids(event.get("added_movie_ids"), f"{label}.added_movie_ids",
                                       errors, line_number, user_id, context, nonempty=True)
                    if added is not None:
                        overlap = sorted(set(added) & set(current_pool))
                        if overlap:
                            _issue(errors, "trace_pool_mismatch", f"{label} retrieved IDs already in pool: {overlap}",
                                   line_number, user_id)
                        else:
                            current_pool.extend(added)
                    latest_evaluation = None
                elif action == "rank_candidates":
                    if ranked or (termination == "recommend" and
                                  (not assessed or not evaluated or latest_evaluation is None)):
                        _issue(errors, "trace_action_order", f"{label} requires one ranking and, for recommend, current assessments",
                               line_number, user_id)
                    ranked = True
                    ids = _trace_ids(event.get("movie_ids"), f"{label}.movie_ids",
                                     errors, line_number, user_id, context)
                    if ids is not None:
                        if len(ids) != required_count or not set(ids).issubset(current_pool):
                            _issue(errors, "trace_rank_pool_mismatch", f"{label} must rank {required_count} IDs from the final pool",
                                   line_number, user_id)
                        if ranking is not None and ids != ranking:
                            _issue(errors, "trace_rank_mismatch", f"{label} must match final prediction order",
                                   line_number, user_id)
                elif action == "explain_recommendations":
                    if not ranked or explained:
                        _issue(errors, "trace_action_order", f"{label} must occur once after ranking",
                               line_number, user_id)
                    explained = True
                    explanations = event.get("explanations")
                    count = min(P2_EXPLANATION_COUNT, required_count)
                    if not isinstance(explanations, list) or len(explanations) != count:
                        _issue(errors, "trace_invalid_explanations", f"{label} requires {count} explanations",
                               line_number, user_id)
                    else:
                        ids = []
                        for index, explanation in enumerate(explanations):
                            if not isinstance(explanation, dict) or not _positive_int(explanation.get("movie_id")) or not _text(explanation.get("text")):
                                _issue(errors, "trace_invalid_explanations", f"{label} requires movie_id and nonempty text",
                                       line_number, user_id)
                            else:
                                ids.append(explanation["movie_id"])
                                if ranking is not None:
                                    prediction_item = predictions[user_id]["recommendations"][index]
                                    if "explanation" in prediction_item and prediction_item["explanation"] != explanation["text"]:
                                        _issue(errors, "trace_explanation_mismatch", f"{label} text differs from prediction explanation at rank {index + 1}",
                                               line_number, user_id)
                        if ranking is not None and ids != ranking[:count]:
                            _issue(errors, "trace_explanation_mismatch", f"{label} must explain ranked top {count} in order",
                                   line_number, user_id)
            required_actions = ((ranked, "rank_candidates"), (explained, "explain_recommendations"))
            if termination == "recommend":
                required_actions += ((assessed, "assess_sufficiency"),
                                     (evaluated, "evaluate_candidates"))
            for present, action in required_actions:
                if not present:
                    _issue(errors, "trace_missing_action", f"trace requires {action}", line_number, user_id)
            if termination == "recommend" and (latest_assessment is not True or latest_evaluation is not True):
                _issue(errors, "trace_unresolved_decision", "recommend requires sufficient information and candidate pool",
                       line_number, user_id)
            if termination == "fallback" and trace.get("fallback_source") == "frozen_b3_order" and (
                ranking is not None and ranking != expected_initial
            ):
                _issue(errors, "trace_fallback_mismatch", "frozen_b3_order fallback must preserve B3 rank order",
                       line_number, user_id)
            final_pool = _trace_ids(trace.get("final_candidate_movie_ids"), "final_candidate_movie_ids",
                                    errors, line_number, user_id, context)
            if final_pool is not None and final_pool != current_pool:
                _issue(errors, "trace_final_pool_mismatch", "final pool must equal initial B3 plus retrieved IDs in event order",
                       line_number, user_id)
            if ranking is not None and not set(ranking).issubset(current_pool):
                _issue(errors, "trace_rank_pool_mismatch", "final predictions contain IDs outside the final pool",
                       line_number, user_id)
    missing_users = sorted(context.expected_user_ids - seen_users)
    if missing_users:
        _issue(errors, "trace_missing_users", f"no trace for expected user IDs {missing_users[:20]}")
    metadata = {"trace": str(path.resolve()), "trace_lines_read": lines_read,
            "trace_users_with_records": len(seen_users & context.expected_user_ids),
            "trace_missing_users": missing_users,
            "trace_unexpected_users": sorted(seen_users - context.expected_user_ids),
            "trace_schema_version": P2_TRACE_SCHEMA,
            "trace_max_control_events": P2_MAX_CONTROL_EVENTS,
            "trace_max_events": P2_MAX_EVENTS}
    return metadata, trace_records


def validate_predictions(
    path: Path,
    context: ValidationContext,
    *,
    method: str,
    required_count: int = 50,
    candidate_pool_path: Path | None = None,
    trace_path: Path | None = None,
) -> ValidatedPredictions:
    """Return records only if every row and the entire expected population pass.

    P1 exactly permutes an independently validated B3 pool. P2 starts with
    that same pool and requires an independently validated execution trace.
    Prediction/trace errors raise PredictionValidationError with all issues.
    Bad reference inputs or options raise ValueError/OSError instead.
    """
    _check_context(context)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if type(required_count) is not int or required_count <= 0:
        raise ValueError("required_count must be a positive integer, excluding booleans")
    pools = None
    if method in ("p1", "p2"):
        if candidate_pool_path is None:
            raise ValueError(f"{method} requires --candidate-pool with the frozen embedding JSONL")
        try:
            pool_result = validate_predictions(
                Path(candidate_pool_path), context, method="embedding", required_count=required_count
            )
        except PredictionValidationError as error:
            raise ValueError(f"Frozen embedding candidate pool is invalid:\n{error}") from error
        pools = {
            user_id: [item["movie_id"] for item in record["recommendations"]]
            for user_id, record in pool_result.records.items()
        }
    elif candidate_pool_path is not None:
        raise ValueError("--candidate-pool is supported only for p1 and p2")
    if method == "p2" and trace_path is None:
        raise ValueError("p2 requires --trace with one execution trace per user")
    if method != "p2" and trace_path is not None:
        raise ValueError("--trace is supported only for p2")

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
                    if method == "p2" and "explanation" in item and not _text(item["explanation"]):
                        _issue(errors, "invalid_explanation", f"{label}: explanation must be nonempty text",
                               line_number, user_id)
                if method == "p1" and pools is not None and user_id in pools and set(movie_ids) != set(pools[user_id]):
                    missing = sorted(set(pools[user_id]) - set(movie_ids))
                    extra = sorted(set(movie_ids) - set(pools[user_id]))
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
    trace_metadata, traces = {}, {}
    if method == "p2":
        trace_metadata, traces = _validate_p2_traces(Path(trace_path), context, pools, records,
                                                     required_count, errors)
    source_sha256 = {**context.source_sha256, "predictions": _sha256(path)}
    if candidate_pool_path is not None:
        source_sha256["candidate_pool"] = _sha256(Path(candidate_pool_path))
    if trace_path is not None:
        source_sha256["trace"] = _sha256(Path(trace_path))
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
        **trace_metadata,
    }
    if errors:
        raise PredictionValidationError(report)
    return ValidatedPredictions(records=records, report=report, traces=traces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="Member 1 catalog CSV")
    parser.add_argument("--profiles", required=True, type=Path, help="matching split profile JSONL")
    parser.add_argument("--users", required=True, type=Path, help="JSON object with split and user_ids")
    parser.add_argument("--predictions", required=True, type=Path, help="recommendation JSONL to validate")
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--required-count", type=int, default=50)
    parser.add_argument("--candidate-pool", type=Path, help="frozen B3 embedding JSONL; required for p1/p2")
    parser.add_argument("--trace", type=Path, help="P2 execution trace JSONL; required for p2")
    parser.add_argument("--report", type=Path, help="optionally save the JSON validation report")
    args = parser.parse_args()
    try:
        inputs = [args.catalog, args.profiles, args.users, args.predictions]
        if args.candidate_pool is not None:
            inputs.append(args.candidate_pool)
        if args.trace is not None:
            inputs.append(args.trace)
        if args.report is not None and args.report.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("--report must not overwrite an input file")
        context = load_context(args.catalog, args.profiles, args.users, args.split)
        result = validate_predictions(
            args.predictions, context, method=args.method,
            required_count=args.required_count, candidate_pool_path=args.candidate_pool,
            trace_path=args.trace,
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
