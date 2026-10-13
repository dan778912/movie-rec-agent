"""Index, tune on validation only, and export standard B2 predictions.

Run ``python -m src.retrieval.run_bm25 --help`` for commands. Diagnostic
metrics here support Member 4's tuning/report; Member 2's evaluator remains
authoritative for the team's comparison table.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from src.data.build import sha256, write_jsonl
from src.retrieval.bm25 import (
    BM25Index, TOKENIZER_VERSION, build_index, document_text, read_jsonl, tokenize,
)

DEFAULT_DATA = Path("data/processed/ml25m-global-v1")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def checked_manifest(data_dir: Path, filenames: list[str]) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename in filenames:
        if sha256(data_dir / filename) != manifest["file_sha256"][filename]:
            raise ValueError(f"Dataset checksum mismatch: {filename}")
    return manifest


def catalog_ids(data_dir: Path) -> set[int]:
    with (data_dir / "catalog.csv").open(encoding="utf-8", newline="") as source:
        ids = [int(row["movie_id"]) for row in csv.DictReader(source)]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate movie IDs in catalog")
    return set(ids)


def load_profiles(data_dir: Path, split: str, user_ids: Path | None = None) -> list[dict]:
    profiles = read_jsonl(data_dir / f"{split}_profiles.jsonl")
    ids = [row["user_id"] for row in profiles]
    if len(ids) != len(set(ids)) or any(row["split"] != split for row in profiles):
        raise ValueError("Duplicate users or incorrect split in profiles")
    if user_ids is not None:
        selected = json.loads(user_ids.read_text(encoding="utf-8"))
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or any(type(user_id) is not int for user_id in selected)):
            raise ValueError("User subset must be a nonempty JSON array of unique integer IDs")
        missing = set(selected) - set(ids)
        if missing:
            raise ValueError(f"Subset contains missing profile users: {sorted(missing)}")
        selected_set = set(selected)
        profiles = [row for row in profiles if row["user_id"] in selected_set]
    if not profiles:
        raise ValueError(f"No evaluation profiles for {split}")
    return sorted(profiles, key=lambda row: row["user_id"])


def load_validation_targets(data_dir: Path, profiles: list[dict], catalog: set[int]):
    # Deliberately no split argument: tuning and local diagnostics cannot read test targets.
    relevant = {row["user_id"]: set() for row in profiles}
    watched = {row["user_id"]: set(row["history_movie_ids"]) for row in profiles}
    total = included = split_total = split_included = 0
    with (data_dir / "validation_targets.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            user_id, movie_id = int(row["user_id"]), int(row["movie_id"])
            in_catalog = int(row["in_catalog"])
            if in_catalog not in (0, 1) or bool(in_catalog) != (movie_id in catalog):
                raise ValueError("Target catalog flag is inconsistent with the frozen catalog")
            split_total += 1
            split_included += in_catalog
            if user_id not in relevant:
                continue
            if float(row["rating"]) < 4.0 or movie_id in watched[user_id]:
                raise ValueError("Validation target is nonpositive or already watched")
            total += 1
            included += in_catalog
            if in_catalog:
                relevant[user_id].add(movie_id)
    if any(not items for items in relevant.values()):
        raise ValueError("Every evaluation profile must have an in-catalog positive target")
    return relevant, {"scope": "selected profile users", "positive_target_rows": total,
                      "in_catalog_target_rows": included,
                      "excluded_target_rows": total - included,
                      "positive_target_catalog_coverage": included / total,
                      "split_positive_target_rows": split_total,
                      "split_in_catalog_target_rows": split_included,
                      "split_excluded_target_rows": split_total - split_included,
                      "split_positive_target_catalog_coverage": split_included / split_total}


def diagnostic_metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    if not relevant:
        raise ValueError("Recall and NDCG require at least one relevant target")
    result = {f"recall@{k}": len(set(ranked[:k]) & relevant) / len(relevant)
              for k in (10, 20, 50, 100)}
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(ranked[:10], 1)
              if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    result["ndcg@10"] = dcg / ideal
    return result


def validate_record(record: dict, profile: dict, catalog: set[int], top_k: int) -> None:
    if (record["user_id"] != profile["user_id"] or record["split"] != profile["split"]
            or record["method"] != "bm25"):
        raise ValueError("Incorrect prediction user, split, or method")
    watched = set(profile["history_movie_ids"])
    seen, previous = set(), math.inf
    if len(record["recommendations"]) != min(top_k, len(catalog - watched)):
        raise ValueError("Prediction has an incorrect recommendation count")
    for rank, item in enumerate(record["recommendations"], 1):
        movie_id, score = item["movie_id"], item["score"]
        if movie_id not in catalog or movie_id in watched or movie_id in seen:
            raise ValueError("Invalid, duplicate, or watched recommendation")
        if item["rank"] != rank or not math.isfinite(score) or score > previous:
            raise ValueError("Invalid rank or nonfinite/non-descending score")
        seen.add(movie_id)
        previous = score


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_predictions(index: BM25Index, profiles: list[dict], config: dict,
                    catalog: set[int], relevant: dict | None = None):
    if set(index.movie_ids) != catalog:
        raise ValueError("Index movie IDs do not match the frozen catalog")
    index.prepare(config["k1"], config["b"])
    # Warm page/code paths; these queries are excluded from the reported distribution.
    for _ in range(3):
        index.search(profiles[0]["text"], set(profiles[0]["history_movie_ids"]),
                     config["top_k"], config["k1"], config["b"])
    records, timings, per_user = [], [], []
    recommended = set()
    zero_queries = shortfalls = 0
    for profile in profiles:
        start = time.perf_counter()
        ranked = index.search(profile["text"], set(profile["history_movie_ids"]),
                              config["top_k"], config["k1"], config["b"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        record = {"user_id": profile["user_id"], "split": profile["split"],
                  "method": "bm25", "recommendations": ranked}
        validate_record(record, profile, catalog, config["top_k"])
        records.append(record)
        timings.append(elapsed_ms)
        recommended.update(item["movie_id"] for item in ranked)
        zero_queries += not any(item["score"] > 0 for item in ranked)
        shortfalls += len(ranked) < config["top_k"]
        if relevant is not None:
            metrics = diagnostic_metrics([item["movie_id"] for item in ranked],
                                         relevant[profile["user_id"]])
            per_user.append({"user_id": profile["user_id"], **metrics})
    summary = {
        "users": len(profiles), "catalog_movies": len(catalog),
        "candidate_pool_size": config["top_k"], "invalid_predictions": 0,
        "missing_users": 0, "user_coverage": 1.0,
        "recommendation_catalog_coverage": len(recommended) / len(catalog),
        "zero_match_queries": zero_queries, "short_candidate_pools": shortfalls,
        "latency_ms": {"mean": statistics.mean(timings), "p50": percentile(timings, 0.5),
                       "p95": percentile(timings, 0.95), "max": max(timings)},
        "latency_protocol": "Single-process CPU, three warmups; tokenization, scoring, watched filtering "
                            "and top-k selection included; index load/preparation, I/O and validation excluded.",
    }
    if per_user:
        summary["diagnostic_metrics"] = {
            metric: statistics.mean(row[metric] for row in per_user)
            for metric in per_user[0] if metric != "user_id"
        }
    return records, summary, per_user


def provenance(data_dir: Path, filenames: list[str]) -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "git_commit": revision,
        "source_sha256": {path.name: sha256(path) for path in
                          (Path(__file__), Path(__file__).with_name("bm25.py"))},
        "input_sha256": {name: sha256(data_dir / name) for name in ["manifest.json", *filenames]},
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(),
        "compute_cost": "Local CPU; no paid compute or API calls.",
        "evaluation_status": "B2 local diagnostics only; shared Member 2 evaluator is authoritative.",
    }


def check_config(config: dict, manifest: dict, require_frozen: bool = False) -> None:
    if config.get("tokenizer") != TOKENIZER_VERSION:
        raise ValueError("Configuration tokenizer differs from this implementation")
    if (config.get("dataset_version") != manifest["version"]
            or config.get("documents_sha256") != manifest["file_sha256"]["movie_documents.jsonl"]
            or config.get("catalog_sha256") != manifest["file_sha256"]["catalog.csv"]):
        raise ValueError("Configuration belongs to a different dataset/catalog; retune on validation")
    if require_frozen and config.get("selection_split") != "validation":
        raise ValueError("Test prediction requires a configuration selected and frozen on validation")
    if (type(config["top_k"]) is not int or config["top_k"] < 100):
        raise ValueError("Export at least 100 candidates to measure candidate recall at 20/50/100")


def qualitative_examples(data_dir: Path, profiles: list[dict], records: list[dict],
                         relevant: dict, per_user: list[dict], fields: list[str]) -> list[dict]:
    documents = {row["movie_id"]: row for row in read_jsonl(data_dir / "movie_documents.jsonl")}
    by_user = {row["user_id"]: row for row in records}
    profile_map = {row["user_id"]: row for row in profiles}
    ordered = sorted(per_user, key=lambda row: (row["ndcg@10"], row["user_id"]))
    selected = [("low_ndcg", row) for row in ordered[:3]]
    selected += [("high_ndcg", row) for row in ordered[-3:]]
    result = []
    for label, row in selected:
        user_id = row["user_id"]
        query_terms = set(tokenize(profile_map[user_id]["text"]))
        recommendations = by_user[user_id]["recommendations"]
        ranked_ids = {item["movie_id"] for item in recommendations}

        def describe(movie_id):
            document = documents[movie_id]
            retrieval_text = document_text(document, fields)
            return {"movie_id": movie_id, "title": document["title"], "text": document["text"],
                    "retrieval_text": retrieval_text,
                    "query_overlap": sorted(query_terms & set(tokenize(retrieval_text))),
                    "relevant": movie_id in relevant[user_id]}

        result.append({"selection": label, **row, "profile": profile_map[user_id]["text"],
                       "top_5": [{**item, **describe(item["movie_id"])} for item in recommendations[:5]],
                       "missed_at_100": [describe(movie_id) for movie_id in
                                         sorted(relevant[user_id] - ranked_ids)[:5]]})
    return result


def save_report(report_dir: Path, summary: dict, per_user: list[dict]) -> None:
    write_json(report_dir / "summary.json", summary)
    if per_user:
        with (report_dir / "per_user_metrics.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(per_user[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(per_user)
        metrics = summary["diagnostic_metrics"]
        table = ("# B2 diagnostic results\n\n"
                 "Pending verification with Member 2's shared evaluator. This is not the final shared table.\n\n"
                 "| Method | Recall@10 | Candidate Recall@50 | NDCG@10 | Users | User coverage |\n"
                 "|---|---:|---:|---:|---:|---:|\n"
                 f"| BM25 (B2) | {metrics['recall@10']:.6f} | {metrics['recall@50']:.6f} | "
                 f"{metrics['ndcg@10']:.6f} | {summary['users']} | {summary['user_coverage']:.2%} |\n")
        (report_dir / "results.md").write_text(table, encoding="utf-8")


def tune(args) -> None:
    started = time.perf_counter()
    filenames = ["movie_documents.jsonl", "catalog.csv", "validation_profiles.jsonl", "validation_targets.csv"]
    manifest = checked_manifest(args.data, filenames)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if grid["objective"] != "ndcg@10" or type(grid["top_k"]) is not int or grid["top_k"] < 100:
        raise ValueError("Tune NDCG@10 and retain at least 100 candidates")
    profiles = load_profiles(args.data, "validation", args.user_ids)
    if args.user_ids is None:
        if type(grid["validation_users"]) is not int or grid["validation_users"] < 1:
            raise ValueError("validation_users must be a positive integer")
        profiles = sorted(random.Random(grid["seed"]).sample(
            profiles, min(grid["validation_users"], len(profiles))), key=lambda row: row["user_id"])
    catalog = catalog_ids(args.data)
    relevant, target_coverage = load_validation_targets(args.data, profiles, catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = [row["user_id"] for row in profiles]
    write_json(args.output_dir / "validation_user_ids.json", selected_ids)
    trials = []
    for fields in grid["fields"]:
        index_start = time.perf_counter()
        index = build_index(args.data, fields)
        indexing_seconds = time.perf_counter() - index_start
        for k1, b in itertools.product(grid["k1"], grid["b"]):
            config = {"fields": fields, "k1": k1, "b": b, "top_k": grid["top_k"]}
            _, summary, _ = run_predictions(index, profiles, config, catalog, relevant)
            trial = {"config": config, "indexing_seconds": indexing_seconds, **summary}
            trials.append(trial)
            print(json.dumps({"trial": len(trials), **config,
                              **summary["diagnostic_metrics"]}), flush=True)
    if not trials:
        raise ValueError("Empty BM25 grid")
    # Stable grid order resolves exact ties; no extra unregistered selection objectives.
    best = max(trials, key=lambda row: row["diagnostic_metrics"][grid["objective"]])
    config = {
        "version": "bm25-frozen-v1", **best["config"], "seed": grid["seed"],
        "tokenizer": TOKENIZER_VERSION, "dataset_version": manifest["version"],
        "documents_sha256": manifest["file_sha256"]["movie_documents.jsonl"],
        "catalog_sha256": manifest["file_sha256"]["catalog.csv"],
        "selection_split": "validation", "selection_objective": grid["objective"],
        "selection_users": selected_ids, "grid_sha256": sha256(args.grid),
        "validation_profiles_sha256": manifest["file_sha256"]["validation_profiles.jsonl"],
        "validation_targets_sha256": manifest["file_sha256"]["validation_targets.csv"],
    }
    write_json(args.frozen_config, config)
    report = {"configuration": config, "trials": trials, "selected_trial": trials.index(best) + 1,
              "selection_tie_break": "First trial in grid order", "target_coverage": target_coverage,
              "elapsed_seconds": time.perf_counter() - started, **provenance(args.data, filenames)}
    report["seed"] = grid["seed"]
    write_json(args.output_dir / "tuning.json", report)
    print(f"Frozen validation-selected configuration: {args.frozen_config}")


def index_command(args) -> None:
    started = time.perf_counter()
    manifest = checked_manifest(args.data, ["movie_documents.jsonl", "catalog.csv"])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_config(config, manifest)
    index = build_index(args.data, config["fields"])
    if set(index.movie_ids) != catalog_ids(args.data):
        raise ValueError("Canonical documents and catalog have different movie IDs")
    index.save(args.output)
    write_json(args.output.with_suffix(".report.json"), {
        "indexing_seconds": time.perf_counter() - started, "catalog_movies": len(index.movie_ids),
        "vocabulary_size": len(index.postings), "index_bytes": args.output.stat().st_size,
        "index_sha256": sha256(args.output), "configuration": config,
        "seed": config["seed"],
        **provenance(args.data, ["movie_documents.jsonl", "catalog.csv"]),
    })
    print(f"Saved {len(index.movie_ids)} movie documents to {args.output}")


def predict(args) -> None:
    started = time.perf_counter()
    filenames = ["movie_documents.jsonl", "catalog.csv", f"{args.split}_profiles.jsonl"]
    if args.split == "validation":
        filenames.append("validation_targets.csv")
    manifest = checked_manifest(args.data, filenames)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_config(config, manifest, require_frozen=args.split == "test")
    load_start = time.perf_counter()
    index = BM25Index.load(args.index)
    if (index.metadata["fields"] != config["fields"]
            or index.metadata["documents_sha256"] != config["documents_sha256"]
            or index.metadata["dataset_version"] != config["dataset_version"]):
        raise ValueError("Index and frozen configuration differ; rebuild the index")
    load_seconds = time.perf_counter() - load_start
    prepare_start = time.perf_counter()
    index.prepare(config["k1"], config["b"])
    prepare_seconds = time.perf_counter() - prepare_start
    profiles = load_profiles(args.data, args.split, args.user_ids)
    catalog = catalog_ids(args.data)
    relevant, target_coverage = (load_validation_targets(args.data, profiles, catalog)
                                 if args.split == "validation" else (None, None))
    records, summary, per_user = run_predictions(index, profiles, config, catalog, relevant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    summary.update({"split": args.split, "configuration": config, "index_load_seconds": load_seconds,
                    "index_prepare_seconds": prepare_seconds, "index_sha256": sha256(args.index),
                    "predictions_sha256": sha256(args.output), "configuration_sha256": sha256(args.config),
                    "target_coverage": target_coverage, "elapsed_seconds": time.perf_counter() - started,
                    **provenance(args.data, filenames)})
    summary["seed"] = config["seed"]
    save_report(args.report_dir, summary, per_user)
    if relevant is not None:
        write_json(args.report_dir / "qualitative_examples.json",
                   qualitative_examples(args.data, profiles, records, relevant, per_user, config["fields"]))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tune_parser = commands.add_parser("tune", help="Validation-only parameter and document-field sweep")
    tune_parser.add_argument("--grid", type=Path, default=Path("configs/bm25_grid.json"))
    tune_parser.add_argument("--frozen-config", type=Path, default=Path("configs/bm25_frozen.json"))
    tune_parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/bm25-tuning"))
    tune_parser.add_argument("--user-ids", type=Path, help="Shared validation user IDs as a JSON array")
    tune_parser.set_defaults(function=tune)
    index_parser = commands.add_parser("index", help="Build and save a catalog-only BM25 index")
    index_parser.add_argument("--output", type=Path, default=Path("data/processed/bm25/index.json"))
    index_parser.set_defaults(function=index_command)
    predict_parser = commands.add_parser("predict", help="Export and validate B2 prediction JSONL")
    predict_parser.add_argument("--index", type=Path, default=Path("data/processed/bm25/index.json"))
    predict_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--report-dir", type=Path, required=True)
    predict_parser.add_argument("--user-ids", type=Path, help="Shared evaluation user IDs as a JSON array")
    predict_parser.set_defaults(function=predict)
    for command in (tune_parser, index_parser, predict_parser):
        command.add_argument("--data", type=Path, default=DEFAULT_DATA)
    for command in (index_parser, predict_parser):
        command.add_argument("--config", type=Path, default=Path("configs/bm25_frozen.json"))
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
