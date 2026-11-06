"""Validate a Phase 2 producer receipt without parsing or scoring held-out labels.

The p2-run-manifest-v1 policy carries the settings to freeze before inference;
the receipt adds output hashes and per-user measurements after inference. The
policy digest is only a consistency check. A dated, external pre-run record is
still needed to establish that the policy was actually frozen in advance.

Run from the repository root::

    python -m src.evaluation.p2_run_manifest --manifest RUN.json \
        --config CONFIG.json --candidate-pool B3.jsonl \
        --predictions P2.jsonl --trace TRACE.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

from .evaluate import _canonical_bytes, _check_hash, _configuration, _json_object, _sha256
from .validate import PredictionValidationError, load_context, validate_predictions


SCHEMA_VERSION = "p2-run-manifest-v1"
SOURCE_KEYS = (
    "config", "dataset_manifest", "catalog", "profiles", "targets", "users",
    "candidate_pool",
)
OUTPUT_KEYS = ("predictions", "trace")
LATENCY_KEYS = ("model", "retrieval", "end_to_end")
EPISODE_COUNTER_KEYS = (
    "model_call_count", "retrieval_call_count", "informative_answer_count",
    "noninformative_answer_count", "invalid_action_count",
)
AGGREGATE_COUNT_KEYS = (
    "user_count", "fallback_user_count", "invalid_action_user_count",
    "timeout_user_count", "loop_cap_user_count", *EPISODE_COUNTER_KEYS,
)


def _fields(value, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = required - value.keys()
    extra = value.keys() - required
    if missing or extra:
        raise ValueError(f"{label} has missing fields {sorted(missing)} or extra fields {sorted(extra)}")


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _count(value, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer, excluding booleans")
    return value


def _number(value, label: str, *, positive: bool = False) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a finite {qualifier} number, excluding booleans")
    return float(value)


def _sha(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} requires a lowercase SHA-256 digest")
    return value


def _validate_policy(policy: dict, config: dict, actual_sources: dict[str, str]) -> None:
    _fields(policy, {
        "dataset_version", "split", "user_count", "code_commit", "code_dirty",
        "source_sha256", "model", "retrieval", "simulator", "agent", "runtime",
    }, "policy")
    if policy["dataset_version"] != config["dataset_version"] or policy["split"] != config["split"]:
        raise ValueError("policy dataset_version/split disagrees with evaluation config")
    if _count(policy["user_count"], "policy.user_count", positive=True) != config["sample_size"]:
        raise ValueError("policy.user_count disagrees with frozen evaluation population")
    if not isinstance(policy["code_commit"], str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", policy["code_commit"]) is None:
        raise ValueError("policy.code_commit must be a full lowercase Git commit ID")
    if type(policy["code_dirty"]) is not bool:
        raise ValueError("policy.code_dirty must be Boolean")
    _fields(policy["source_sha256"], set(SOURCE_KEYS), "policy.source_sha256")
    for key, actual in actual_sources.items():
        _check_hash(actual, policy["source_sha256"][key], f"policy.source_sha256.{key}")

    model = policy["model"]
    _fields(model, {
        "p1_checkpoint_id", "p1_checkpoint_revision", "p2_checkpoint_id",
        "p2_checkpoint_revision", "tokenizer_id", "training_sample_version",
        "prompt_version", "decoding",
    }, "policy.model")
    for key in set(model) - {"decoding"}:
        _text(model[key], f"policy.model.{key}")
    decoding = model["decoding"]
    _fields(decoding, {"temperature", "top_p", "max_new_tokens", "seed", "score_kind"}, "policy.model.decoding")
    if _number(decoding["temperature"], "policy.model.decoding.temperature") > 2:
        raise ValueError("policy.model.decoding.temperature must be at most 2")
    if _number(decoding["top_p"], "policy.model.decoding.top_p", positive=True) > 1:
        raise ValueError("policy.model.decoding.top_p must be at most 1")
    _count(decoding["max_new_tokens"], "policy.model.decoding.max_new_tokens", positive=True)
    _count(decoding["seed"], "policy.model.decoding.seed")
    _text(decoding["score_kind"], "policy.model.decoding.score_kind")

    retrieval = policy["retrieval"]
    _fields(retrieval, {
        "index_version", "index_manifest_sha256", "ingestion_catalog_sha256",
        "embedding_model_id", "max_candidate_pool_size", "watched_exclusion",
    }, "policy.retrieval")
    for key in ("index_version", "embedding_model_id"):
        _text(retrieval[key], f"policy.retrieval.{key}")
    _sha(retrieval["index_manifest_sha256"], "policy.retrieval.index_manifest_sha256")
    if _sha(retrieval["ingestion_catalog_sha256"], "policy.retrieval.ingestion_catalog_sha256") != actual_sources["catalog"]:
        raise ValueError("retrieval ingestion catalog does not match the frozen evaluation catalog")
    if _count(retrieval["max_candidate_pool_size"], "policy.retrieval.max_candidate_pool_size", positive=True) < config["candidate_pool_size"]:
        raise ValueError("retrieval max_candidate_pool_size is smaller than the frozen B3 pool")
    if retrieval["watched_exclusion"] is not True:
        raise ValueError("policy.retrieval.watched_exclusion must be true")

    simulator = policy["simulator"]
    _fields(simulator, {"version", "code_sha256", "answer_source", "pre_cutoff_source"}, "policy.simulator")
    for key in ("version", "answer_source", "pre_cutoff_source"):
        _text(simulator[key], f"policy.simulator.{key}")
    _sha(simulator["code_sha256"], "policy.simulator.code_sha256")

    agent = policy["agent"]
    _fields(agent, {
        "parser_version", "max_control_events", "max_serialized_events", "timeout_ms",
        "fallback_source", "fallback_triggers", "explanation_policy_version",
    }, "policy.agent")
    for key in ("parser_version", "explanation_policy_version"):
        _text(agent[key], f"policy.agent.{key}")
    if agent["max_control_events"] != 8 or type(agent["max_control_events"]) is not int:
        raise ValueError("policy.agent.max_control_events must be 8")
    if agent["max_serialized_events"] != 10 or type(agent["max_serialized_events"]) is not int:
        raise ValueError("policy.agent.max_serialized_events must be 10")
    _count(agent["timeout_ms"], "policy.agent.timeout_ms", positive=True)
    if agent["fallback_source"] != "frozen_b3_order":
        raise ValueError("policy.agent.fallback_source must be frozen_b3_order")
    triggers = agent["fallback_triggers"]
    if not isinstance(triggers, list) or not triggers or any(not isinstance(item, str) or not item.strip() for item in triggers):
        raise ValueError("policy.agent.fallback_triggers must contain nonempty reason codes")
    if len(set(triggers)) != len(triggers):
        raise ValueError("policy.agent.fallback_triggers contains duplicates")

    runtime = policy["runtime"]
    _fields(runtime, {"hardware", "concurrency", "measurement_version", "latency_scope"}, "policy.runtime")
    _text(runtime["hardware"], "policy.runtime.hardware")
    _count(runtime["concurrency"], "policy.runtime.concurrency", positive=True)
    _text(runtime["measurement_version"], "policy.runtime.measurement_version")
    if runtime["latency_scope"] != "offline_episode_excludes_simulator_wait":
        raise ValueError("policy.runtime.latency_scope must exclude simulator waiting time")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validate_episodes(episodes: list, traces: dict[int, dict], policy: dict) -> dict:
    if not isinstance(episodes, list):
        raise ValueError("episodes must be an array")
    by_user = {}
    for episode in episodes:
        _fields(episode, {
            "user_id", "termination_reason", "fallback_reason", "fallback_trigger", "model_call_count",
            "retrieval_call_count", "informative_answer_count", "noninformative_answer_count",
            "invalid_action_count", "timed_out", "loop_cap_hit", "latency_ms", "cost_usd",
        }, "episode")
        user_id = _count(episode["user_id"], "episode.user_id", positive=True)
        if user_id in by_user:
            raise ValueError(f"duplicate episode for user {user_id}")
        by_user[user_id] = episode
    if set(by_user) != set(traces):
        raise ValueError(f"episode user coverage mismatch; missing {sorted(set(traces) - set(by_user))[:20]}, extra {sorted(set(by_user) - set(traces))[:20]}")

    aggregate = {key: 0 for key in AGGREGATE_COUNT_KEYS}
    aggregate["user_count"] = len(by_user)
    aggregate["fallback_by_trigger"] = {}
    latencies = {key: [] for key in LATENCY_KEYS}
    costs = []
    for user_id, episode in sorted(by_user.items()):
        prefix = f"episode user {user_id}"
        trace = traces[user_id]
        if episode["termination_reason"] != trace["termination_reason"]:
            raise ValueError(f"{prefix}: termination_reason disagrees with trace")
        fallback = episode["termination_reason"] == "fallback"
        if not fallback and episode["termination_reason"] != "recommend":
            raise ValueError(f"{prefix}: invalid termination_reason")
        if fallback:
            if _text(episode["fallback_reason"], f"{prefix}.fallback_reason") != trace["fallback_reason"]:
                raise ValueError(f"{prefix}: fallback_reason disagrees with trace")
            trigger = _text(episode["fallback_trigger"], f"{prefix}.fallback_trigger")
            if trigger not in policy["agent"]["fallback_triggers"]:
                raise ValueError(f"{prefix}: fallback_trigger is absent from frozen policy")
            aggregate["fallback_by_trigger"][trigger] = aggregate["fallback_by_trigger"].get(trigger, 0) + 1
        elif episode["fallback_reason"] is not None:
            raise ValueError(f"{prefix}: a normal recommendation cannot have a fallback_reason")
        elif episode["fallback_trigger"] is not None:
            raise ValueError(f"{prefix}: a normal recommendation cannot have a fallback_trigger")
        aggregate["fallback_user_count"] += int(fallback)
        for key in EPISODE_COUNTER_KEYS:
            aggregate[key] += _count(episode[key], f"{prefix}.{key}")
        if not fallback and episode["model_call_count"] == 0:
            raise ValueError(f"{prefix}: a normal P2 recommendation requires a model call")
        retrieval_events = [event for event in trace["events"] if event["action"] == "retrieve_candidates"]
        if episode["retrieval_call_count"] < len(retrieval_events):
            raise ValueError(f"{prefix}: retrieval_call_count is smaller than successful retrieval events")
        if any(event["index_version"] != policy["retrieval"]["index_version"] for event in retrieval_events):
            raise ValueError(f"{prefix}: trace index_version disagrees with policy")
        clarification_events = [event for event in trace["events"] if event["action"] == "ask_clarification"]
        if episode["informative_answer_count"] + episode["noninformative_answer_count"] != len(clarification_events):
            raise ValueError(f"{prefix}: answerability counts disagree with clarification events")
        if any(event["answer_source"] != policy["simulator"]["answer_source"] for event in clarification_events):
            raise ValueError(f"{prefix}: trace answer_source disagrees with simulator policy")
        for key in ("timed_out", "loop_cap_hit"):
            if type(episode[key]) is not bool:
                raise ValueError(f"{prefix}.{key} must be Boolean")
        aggregate["timeout_user_count"] += int(episode["timed_out"])
        aggregate["loop_cap_user_count"] += int(episode["loop_cap_hit"])
        aggregate["invalid_action_user_count"] += int(episode["invalid_action_count"] > 0)
        _fields(episode["latency_ms"], set(LATENCY_KEYS), f"{prefix}.latency_ms")
        for key in LATENCY_KEYS:
            latencies[key].append(_number(episode["latency_ms"][key], f"{prefix}.latency_ms.{key}"))
        if episode["latency_ms"]["model"] + episode["latency_ms"]["retrieval"] > episode["latency_ms"]["end_to_end"] + 1e-6:
            raise ValueError(f"{prefix}: stage latency exceeds end_to_end latency")
        costs.append(_number(episode["cost_usd"], f"{prefix}.cost_usd"))
    aggregate["latency_ms"] = {
        key: {"p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95)}
        for key, values in latencies.items()
    }
    aggregate["cost_usd"] = math.fsum(costs)
    return aggregate


def _validate_aggregate(declared: dict, actual: dict) -> None:
    _fields(declared, set(AGGREGATE_COUNT_KEYS) | {"fallback_by_trigger", "latency_ms", "cost_usd"}, "aggregate")
    for key in AGGREGATE_COUNT_KEYS:
        if _count(declared[key], f"aggregate.{key}") != actual[key]:
            raise ValueError(f"aggregate.{key} disagrees with per-user episodes")
    _fields(declared["fallback_by_trigger"], set(actual["fallback_by_trigger"]), "aggregate.fallback_by_trigger")
    for trigger, count in actual["fallback_by_trigger"].items():
        if _count(declared["fallback_by_trigger"][trigger], f"aggregate.fallback_by_trigger.{trigger}") != count:
            raise ValueError(f"aggregate.fallback_by_trigger.{trigger} disagrees with per-user episodes")
    _fields(declared["latency_ms"], set(LATENCY_KEYS), "aggregate.latency_ms")
    for stage in LATENCY_KEYS:
        _fields(declared["latency_ms"][stage], {"p50", "p95"}, f"aggregate.latency_ms.{stage}")
        for percentile in ("p50", "p95"):
            value = _number(declared["latency_ms"][stage][percentile], f"aggregate.latency_ms.{stage}.{percentile}")
            if not math.isclose(value, actual["latency_ms"][stage][percentile], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"aggregate.latency_ms.{stage}.{percentile} disagrees with per-user episodes")
    if not math.isclose(_number(declared["cost_usd"], "aggregate.cost_usd"), actual["cost_usd"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("aggregate.cost_usd disagrees with per-user episodes")


def validate_run_manifest(
    manifest_path: Path, config_path: Path, candidate_pool_path: Path,
    predictions_path: Path, trace_path: Path, *, repository_root: Path | None = None,
) -> dict:
    """Validate source identity, trace coverage, policy, and producer telemetry.

    This hashes the target file bytes after the freeze. It does not parse or score
    held-out labels or claim to verify checkpoint/index bytes or telemetry truth.
    """
    manifest_path, config_path, candidate_pool_path, predictions_path, trace_path = (
        Path(path).resolve() for path in (
            manifest_path, config_path, candidate_pool_path, predictions_path, trace_path,
        )
    )
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    config = _configuration(config_path)
    data_dir = (root / config["data_dir"]).resolve()
    paths = {
        "config": config_path, "dataset_manifest": data_dir / "manifest.json",
        "catalog": data_dir / "catalog.csv",
        "profiles": data_dir / f"{config['split']}_profiles.jsonl",
        "targets": data_dir / f"{config['split']}_targets.csv",
        "users": (root / config["users_file"]).resolve(),
        "candidate_pool": candidate_pool_path,
        "predictions": predictions_path, "trace": trace_path,
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    _check_hash(hashes["dataset_manifest"], config.get("manifest_sha256"), "config.manifest_sha256")
    _check_hash(hashes["users"], config.get("users_sha256"), "config.users_sha256")
    dataset_manifest = _json_object(paths["dataset_manifest"])
    if dataset_manifest.get("version") != config["dataset_version"]:
        raise ValueError("dataset manifest version disagrees with config")
    file_hashes = dataset_manifest.get("file_sha256")
    if not isinstance(file_hashes, dict):
        raise ValueError("dataset manifest.file_sha256 must be an object")
    for key in ("catalog", "profiles", "targets"):
        _check_hash(hashes[key], file_hashes.get(paths[key].name), f"dataset manifest {paths[key].name}")
    users = _json_object(paths["users"])
    if users.get("dataset_version") != config["dataset_version"] or users.get("split") != config["split"]:
        raise ValueError("frozen user list dataset_version/split disagrees with config")
    frozen_sources = users.get("source_sha256")
    if not isinstance(frozen_sources, dict):
        raise ValueError("frozen user list requires source_sha256")
    for key in ("catalog", "profiles", "targets"):
        _check_hash(hashes[key], frozen_sources.get(key), f"frozen user list {key}")
    _check_hash(hashes["dataset_manifest"], frozen_sources.get("manifest"), "frozen user list manifest")
    context = load_context(paths["catalog"], paths["profiles"], paths["users"], config["split"])
    if len(context.expected_user_ids) != config["sample_size"]:
        raise ValueError("config.sample_size disagrees with frozen user list")

    manifest_hash = _sha256(manifest_path)
    receipt = _json_object(manifest_path)
    if _sha256(manifest_path) != manifest_hash:
        raise ValueError("run manifest changed during validation; restart with immutable inputs")
    _fields(receipt, {"schema_version", "run_id", "policy", "policy_sha256", "output_sha256", "episodes", "aggregate"} | ({"owner_notes"} if "owner_notes" in receipt else set()), "run manifest")
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"run manifest schema_version must be {SCHEMA_VERSION}")
    if not isinstance(receipt["run_id"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", receipt["run_id"]) is None:
        raise ValueError("run_id must be 3–128 safe ID characters")
    if "owner_notes" in receipt:
        _text(receipt["owner_notes"], "owner_notes")
    policy = receipt["policy"]
    _validate_policy(policy, config, {key: hashes[key] for key in SOURCE_KEYS})
    _check_hash(hashlib.sha256(_canonical_bytes(policy)).hexdigest(), receipt["policy_sha256"], "policy_sha256")
    _fields(receipt["output_sha256"], set(OUTPUT_KEYS), "output_sha256")
    for key in OUTPUT_KEYS:
        _check_hash(hashes[key], receipt["output_sha256"][key], f"output_sha256.{key}")
    validated = validate_predictions(
        predictions_path, context, method="p2",
        required_count=config["required_predictions_per_user"],
        candidate_pool_path=candidate_pool_path, trace_path=trace_path,
    )
    actual_aggregate = _validate_episodes(receipt["episodes"], validated.traces, policy)
    _validate_aggregate(receipt["aggregate"], actual_aggregate)
    for key, path in paths.items():
        if _sha256(path) != hashes[key]:
            raise ValueError(f"{key} changed during run-manifest validation; restart with immutable inputs")
    if _sha256(manifest_path) != manifest_hash:
        raise ValueError("run manifest changed during validation; restart with immutable inputs")
    return {
        "status": "valid", "schema_version": SCHEMA_VERSION,
        "run_id": receipt["run_id"], "split": config["split"],
        "config_status": config["status"],
        "user_count": len(context.expected_user_ids),
        "policy_sha256": receipt["policy_sha256"],
        "run_manifest_sha256": manifest_hash,
        "source_sha256": dict(sorted(hashes.items())),
        "aggregate": actual_aggregate,
        "verified_scope": "file identity, declared schema, full trace/prediction/episode coverage, internal counts and latency arithmetic",
        "owner_assertions_unverified": [
            "policy existed before inference", "checkpoint and index contents",
            "simulator temporal safety", "model action provenance", "measurement accuracy",
            "explanation grounding",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_run_manifest(
            args.manifest, args.config, args.candidate_pool, args.predictions, args.trace,
        )
    except PredictionValidationError as error:
        print(str(error), file=sys.stderr)
        print(_canonical_bytes(error.report).decode("utf-8"), end="")
        return 1
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        print(f"P2 run manifest validation failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
