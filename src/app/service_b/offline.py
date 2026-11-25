"""Offline P2 producer. Official runs require an explicit, hash-bound team approval receipt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

from src.agent.decide import RetrievalResult, decide, resume
from src.agent.manifest import agent_policy, manifest_episode
from src.agent.trace import telemetry_record, to_trace_record
from src.data import clarification_sim as sim
from src.evaluation.evaluate import _configuration
from src.evaluation.p2_run_manifest import (_validate_episodes, _validate_policy, validate_run_manifest)
from src.evaluation.validate import load_context, validate_predictions
from src.profiles.agent_text import PROMPT_VERSION
from src.retrieval.online import canonical, file_hash
from .model import checkpoint_identity
from .runtime import Timed, load_resources
from .simulator import load_simulator_metadata
from .explanations import summarize_explanations

ROOT = Path(__file__).resolve().parents[3]


def source_paths(config_path, candidate_pool, root=ROOT):
    config = _configuration(Path(config_path))
    data = Path(root) / config["data_dir"]
    return config, {
        "config": Path(config_path), "dataset_manifest": data / "manifest.json",
        "catalog": data / "catalog.csv", "profiles": data / f"{config['split']}_profiles.jsonl",
        "targets": data / f"{config['split']}_targets.csv",
        "users": Path(root) / config["users_file"], "candidate_pool": Path(candidate_pool),
    }


def verify_sources(config, paths):
    """Check provenance before any model calls. Targets are hashed, never parsed here."""
    hashes = {k: file_hash(p) for k, p in paths.items()}
    manifest = json.loads(paths["dataset_manifest"].read_text())
    users = json.loads(paths["users"].read_text())
    if (manifest.get("version") != config["dataset_version"]
            or hashes["dataset_manifest"] != config["manifest_sha256"]
            or hashes["users"] != config["users_sha256"]
            or users.get("dataset_version") != config["dataset_version"]
            or users.get("split") != config["split"]
            or len(users.get("user_ids", [])) != config["sample_size"]):
        raise ValueError("Dataset or frozen-population provenance mismatch")
    for key in ("catalog", "profiles", "targets"):
        if (hashes[key] != manifest["file_sha256"].get(paths[key].name)
                or hashes[key] != users["source_sha256"].get(key)):
            raise ValueError(f"Source provenance mismatch: {key}")
    if hashes["dataset_manifest"] != users["source_sha256"].get("manifest"):
        raise ValueError("Frozen users/dataset manifest mismatch")
    return hashes


def source_snapshot():
    paths = list((ROOT / "src").rglob("*.py")) + list(ROOT.glob("requirements-service-b*.txt"))
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(args):
    config, paths = source_paths(args.config, args.candidate_pool)
    hashes = verify_sources(config, paths)
    service = json.loads(Path(args.service_config).read_text())
    if service["mode"] != "real":
        raise ValueError("Freeze requires a real service configuration")
    index_path = Path(service["index"]["manifest"])
    index = json.loads(index_path.read_text())
    from src.agent.config import AgentConfig
    agent = AgentConfig(**service.get("agent", {}))
    if agent.exhaustion_policy != "fallback_b3_order" or agent.ranking_size != config["required_predictions_per_user"]:
        raise ValueError("Agent policy is incompatible with Member 2's export contract")
    if agent.explanation_failure_policy != "template" or agent.grounding_policy != "flag":
        raise ValueError("First validation requires explanation_failure_policy='template' and grounding_policy='flag'")
    checkpoint = checkpoint_identity(service["model"]["adapter_path"])
    policy = {
        "dataset_version": config["dataset_version"], "split": config["split"], "user_count": config["sample_size"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": hashes,
        "model": {
            "p1_checkpoint_id": args.p1_checkpoint_id, "p1_checkpoint_revision": args.p1_checkpoint_revision,
            "p2_checkpoint_id": service["model"]["adapter_path"], "p2_checkpoint_revision": checkpoint["sha256"],
            "tokenizer_id": service["model"]["adapter_path"], "training_sample_version": args.training_sample_version,
            "prompt_version": PROMPT_VERSION,
            "decoding": {"temperature": 0, "top_p": 1, "max_new_tokens": service["model"]["max_new_tokens"],
                         "seed": service["model"]["seed"], "score_kind": "rank_surrogate"},
        },
        "retrieval": {"index_version": index["index_version"], "index_manifest_sha256": file_hash(index_path),
            "ingestion_catalog_sha256": index["catalog_sha256"], "embedding_model_id": index["embedding"]["model_id"],
            "max_candidate_pool_size": agent.max_pool_size, "watched_exclusion": True},
        "simulator": {"version": sim.SIMULATOR_VERSION, "answer_source": sim.SIMULATOR_VERSION,
            "pre_cutoff_source": "train_ratings.csv" + (" + validation_ratings.csv" if config["split"] == "test" else ""),
            "code_sha256": file_hash(Path(sim.__file__))},
        "agent": agent_policy(agent, timeout_ms=int(service["request_timeout_seconds"] * 1000)),
        "runtime": {"hardware": args.hardware, "concurrency": 1, "measurement_version": "service-b-perf-counter-v1",
                    "latency_scope": "offline_episode_excludes_simulator_wait"},
    }
    _validate_policy(policy, config, policy["source_sha256"])
    bundle = {"policy": policy, "policy_sha256": hashlib.sha256(canonical(policy)).hexdigest(),
              "runtime_config_sha256": file_hash(args.service_config), "source_snapshot": source_snapshot(),
              "agent_config": agent.to_dict(), "checkpoint_files": checkpoint["files"]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as out:
        out.write(canonical(bundle))
    return {"freeze_path": str(output), "freeze_sha256": file_hash(output),
            "policy_sha256": bundle["policy_sha256"], "status": "awaiting_team_approval"}


class FrozenRetriever:
    def __init__(self, ids, delegate):
        self.ids, self.delegate = list(ids), delegate
        self.index_version = delegate.index_version

    def initial_pool(self, **kwargs):
        return RetrievalResult(list(self.ids), self.index_version)

    def search(self, **kwargs):
        return self.delegate.search(**kwargs)


def run_episode(resources, profile, pool, facts, split, *, hourly_rate=0):
    """One complete episode; only controller work contributes to end-to-end latency."""
    model = Timed(resources.model)
    retrieval = Timed(FrozenRetriever(pool, resources.retriever))
    elapsed, informative, noninformative = 0.0, 0, 0
    answers, questions, previous_state = [], [], None
    limit = resources.config["request_timeout_seconds"]
    try:
        while True:
            if hasattr(resources.model, "deadline"):
                resources.model.deadline = time.monotonic() + max(0, limit - elapsed)
            start = time.perf_counter()
            try:
                if previous_state is None:
                    decision = decide({"profile_text": profile["text"], "history_movie_ids": profile["history_movie_ids"],
                        "clarifications": [], "user_id": profile["user_id"], "split": split},
                        model=model, retriever=retrieval, catalog=resources.catalog, config=resources.agent_config)
                else:
                    decision = resume(previous_state, answers[-1]["answer"], sim.SIMULATOR_VERSION,
                                      model=model, retriever=retrieval, catalog=resources.catalog)
            finally:
                elapsed += time.perf_counter() - start
            if decision.response["action"] != "ask":
                break
            # Simulator sees pre-cutoff history facts, never held-out targets.
            question = decision.response["question"]
            visible = profile["text"] + "\n" + "\n".join(x["answer"] for x in answers)
            answer = sim.answer_question(question, facts, questions, visible)
            informative += int(answer["informative"])
            noninformative += int(not answer["informative"])
            answers.append(answer)
            questions.append(question)
            previous_state = decision.state
        state = decision.state
        evidence = {"telemetry": telemetry_record(state), "answers": answers,
                    "model_calls": model.calls, "search_calls": max(0, retrieval.calls - 1),
                    "model_call_audit": model.model_call_audit}
        if decision.response["action"] == "error":
            raise EpisodeFailed(state.reason, evidence)
        trace = to_trace_record(state)
        prediction = {"user_id": state.user_id, "split": split, "method": "p2", "recommendations": [
            {"movie_id": mid, "score": round(1 - i / len(state.ranked), 6), "rank": i + 1}
            for i, mid in enumerate(state.ranked)]}
        measurements = manifest_episode(state, informative_answer_count=informative,
            noninformative_answer_count=noninformative, timed_out=model.timed_out or retrieval.timed_out,
            latency_ms={"model": model.seconds * 1000, "retrieval": retrieval.seconds * 1000,
                        "end_to_end": elapsed * 1000}, cost_usd=elapsed / 3600 * hourly_rate)
        return prediction, trace, measurements, evidence
    finally:
        if hasattr(resources.model, "deadline"):
            resources.model.deadline = None


class EpisodeFailed(RuntimeError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


def histories_for_simulator(paths, selected, split, profiles, catalog, raw=None):
    data = paths["profiles"].parent
    manifest = json.loads(paths["dataset_manifest"].read_text())
    stats = json.loads((data / "statistics.json").read_text())
    if file_hash(data / "statistics.json") != manifest["file_sha256"]["statistics.json"]:
        raise ValueError("Statistics checksum mismatch")
    cutoff = stats[f"{split}_cutoff_unix"]
    metadata = load_simulator_metadata(data, Path(raw) if raw else ROOT / "data/raw/ml-25m")
    if not set(catalog).issubset(metadata):
        raise ValueError("Simulator metadata does not cover the retrieval catalog")
    history = {u: [] for u in selected}
    files = ["train_ratings.csv"] + (["validation_ratings.csv"] if split == "test" else [])
    for name in files:
        if file_hash(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"History checksum mismatch: {name}")
        with (data / name).open() as source:
            for row in csv.DictReader(source):
                uid = int(row["user_id"])
                if uid in history:
                    history[uid].append((int(row["movie_id"]), float(row["rating"]), int(row["timestamp"])))
    facts = {}
    for uid in selected:
        p = profiles[uid]
        if {r[0] for r in history[uid]} != set(p["history_movie_ids"]):
            raise ValueError(f"Simulator/profile history mismatch for {uid}")
        visible = {m["movie_id"] for key in ("liked_movies", "disliked_movies") for m in p.get(key, [])}
        facts[uid] = sim.build_fact_pool(history[uid], metadata, visible, p["text"], cutoff=cutoff)
    return facts


def run(args):
    config, paths = source_paths(args.config, args.candidate_pool)
    bundle = json.loads(Path(args.freeze).read_text())
    policy = bundle["policy"]
    if hashlib.sha256(canonical(policy)).hexdigest() != bundle["policy_sha256"]:
        raise ValueError("Freeze policy digest mismatch")
    if bundle["runtime_config_sha256"] != file_hash(args.service_config) or bundle["source_snapshot"] != source_snapshot():
        raise ValueError("Runtime configuration or source code changed since freeze")
    hashes = verify_sources(config, paths)
    _validate_policy(policy, config, hashes)
    if not args.smoke:
        if not args.approval_record:
            raise ValueError("Official inference requires --approval-record from the team")
        approval = json.loads(Path(args.approval_record).read_text())
        if (approval.get("approved") is not True or approval.get("freeze_sha256") != file_hash(args.freeze)
                or approval.get("split") != config["split"] or not approval.get("approved_by")
                or not approval.get("approved_at")):
            raise ValueError("Missing or mismatched dated team approval record")
    elif config["split"] == "test":
        raise ValueError("Test inference requires explicit approval, including smoke runs")
    if not math.isfinite(args.hourly_rate) or args.hourly_rate < 0:
        raise ValueError("hourly_rate must be finite and nonnegative")
    context = load_context(paths["catalog"], paths["profiles"], paths["users"], config["split"])
    pools = validate_predictions(paths["candidate_pool"], context, method="embedding",
        required_count=config["candidate_pool_size"]).records
    profiles = {}
    with paths["profiles"].open() as source:
        for line in source:
            p = json.loads(line)
            if p["user_id"] in context.expected_user_ids:
                profiles[p["user_id"]] = p
    selected = sorted(context.expected_user_ids)
    if args.smoke:
        selected = selected[:args.smoke]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    resources = load_resources(args.service_config)
    predictions, traces, episodes, failures = [], [], [], []
    completed_evidence = {}
    try:
        if resources.mode != "real":
            raise ValueError("Offline production requires real model and retrieval")
        if (resources.agent_config.to_dict() != bundle["agent_config"]
                or resources.model.identity["files"] != bundle["checkpoint_files"]
                or resources.retriever.index_version != policy["retrieval"]["index_version"]
                or file_hash(resources.config["index"]["manifest"]) != policy["retrieval"]["index_manifest_sha256"]):
            raise ValueError("Loaded resources disagree with freeze")
        facts = histories_for_simulator(paths, selected, config["split"], profiles, resources.catalog,
                                       resources.config.get("simulator_raw_dir"))
        with (output / "telemetry.jsonl").open("wb") as audit:
            for uid in selected:
                try:
                    pred, trace, episode, evidence = run_episode(resources, profiles[uid],
                        [i["movie_id"] for i in pools[uid]["recommendations"]], facts[uid], config["split"],
                        hourly_rate=args.hourly_rate)
                    predictions.append(pred); traces.append(trace); episodes.append(episode)
                    completed_evidence[uid] = evidence
                    audit.write(canonical({"user_id": uid, **evidence}))
                except Exception as error:
                    failure = {"user_id": uid, "error": f"{type(error).__name__}: {error}",
                               "evidence": getattr(error, "evidence", None)}
                    failures.append(failure)
                    audit.write(canonical(failure))
                audit.flush()
    finally:
        resources.close()
    for name, rows in (("p2.jsonl", predictions), ("p2_trace.jsonl", traces), ("failures.jsonl", failures)):
        (output / name).write_bytes(b"".join(canonical(r) for r in rows))
    explanation_summary = summarize_explanations(traces, completed_evidence)
    explanation_path = output / "explanation_fallback_summary.json"
    explanation_path.write_bytes(canonical(explanation_summary))
    result = {"status": "incomplete" if failures else ("smoke_only" if args.smoke else "awaiting_validation"),
              "attempted_users": len(selected), "failed_users": len(failures),
              "explanation_fallback_users": explanation_summary["catalog_metadata_fallback_user_count"],
              "cost_basis": {"hourly_rate_usd": args.hourly_rate, "scope": "measured episode work only; excludes loading and idle time"}}
    (output / "status.json").write_bytes(canonical(result))
    if failures:
        raise RuntimeError("Incomplete run; failures retained. No valid official manifest was published.")
    if args.smoke:
        return result
    trace_by_user = {t["user_id"]: t for t in traces}
    receipt = {"schema_version": "p2-run-manifest-v1", "run_id": args.run_id,
        "policy": policy, "policy_sha256": bundle["policy_sha256"],
        "output_sha256": {"predictions": file_hash(output / "p2.jsonl"), "trace": file_hash(output / "p2_trace.jsonl")},
        "episodes": episodes, "aggregate": _validate_episodes(episodes, trace_by_user, policy),
        "owner_notes": "Estimated marginal episode compute cost; see status.json. "
            "Full source snapshot is bound by the approval receipt. "
            f"Explanation provenance/counts: explanation_fallback_summary.json SHA-256 {file_hash(explanation_path)}; "
            f"catalog_metadata_fallback users: {explanation_summary['catalog_metadata_fallback_user_count']}. "
            f"Per-call errors/digests: telemetry.jsonl SHA-256 {file_hash(output / 'telemetry.jsonl')}."}
    candidate = output / "run_manifest.pending.json"
    candidate.write_bytes(canonical(receipt))
    report = validate_run_manifest(candidate, paths["config"], paths["candidate_pool"],
                                   output / "p2.jsonl", output / "p2_trace.jsonl")
    for key, path in paths.items():
        if file_hash(path) != hashes[key]:
            raise ValueError(f"Input changed during inference: {key}")
    if source_snapshot() != bundle["source_snapshot"]:
        raise ValueError("Source code changed during inference")
    candidate.rename(output / "run_manifest.json")
    (output / "validation.json").write_bytes(canonical(report))
    result["status"] = "validated"
    (output / "status.json").write_bytes(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = subs.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--candidate-pool", required=True)
        p.add_argument("--service-config", required=True)
        if name == "prepare":
            p.add_argument("--output", required=True)
            p.add_argument("--p1-checkpoint-id", required=True)
            p.add_argument("--p1-checkpoint-revision", required=True)
            p.add_argument("--training-sample-version", required=True)
            p.add_argument("--hardware", default=platform.platform())
        else:
            p.add_argument("--freeze", required=True)
            p.add_argument("--approval-record")
            p.add_argument("--output-dir", required=True)
            p.add_argument("--run-id", required=True)
            p.add_argument("--hourly-rate", type=float, required=True)
            p.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    if getattr(args, "smoke", 0) < 0:
        parser.error("--smoke must be nonnegative")
    print(json.dumps(prepare(args) if args.command == "prepare" else run(args), indent=2))


if __name__ == "__main__":
    main()
