"""Prepare a grounded manual explanation audit and summarize completed reviews."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from src.baselines.common import DATA, CONFIG, check_inputs, jsonl, provenance, read_config, save_json
from src.data.build import sha256, write_jsonl


def prepare(data, predictions, output, config, split="validation"):
    if split not in ("validation", "test"):
        raise ValueError("Invalid split")
    check_inputs(data, ["movie_documents.jsonl", f"{split}_profiles.jsonl"])
    rng = random.Random(config["seed"])
    sample, counts = [], Counter()
    seen_users = set()
    for record in jsonl(predictions):
        uid = record["user_id"]
        if record["split"] != split:
            raise ValueError("Prediction split differs from audit split")
        if uid in seen_users:
            raise ValueError(f"Duplicate prediction user: {uid}")
        seen_users.add(uid)
        seen_movies = set()
        for rec in record["recommendations"]:
            mid = rec["movie_id"]
            if mid in seen_movies:
                raise ValueError(f"Duplicate recommended movie for {uid}: {mid}")
            seen_movies.add(mid)
            if not isinstance(rec.get("explanation"), str) or not rec["explanation"].strip():
                counts["missing_explanations"] += 1
                continue
            counts["explanations"] += 1
            row = {"user_id": uid, "movie_id": mid, "rank": rec["rank"], "explanation": rec["explanation"]}
            if len(sample) < config["audit_sample_size"]:
                sample.append(row)
            else:
                index = rng.randrange(counts["explanations"])
                if index < config["audit_sample_size"]:
                    sample[index] = row
    if not sample:
        raise ValueError("No explanations available for manual audit")
    selected_users = {r["user_id"] for r in sample}
    profiles = {r["user_id"]: r for r in jsonl(data / f"{split}_profiles.jsonl") if r["user_id"] in selected_users}
    docs = {r["movie_id"]: r for r in jsonl(data / "movie_documents.jsonl")}
    for row in sample:
        row.update({"profile": profiles[row["user_id"]]["text"],
                    "candidate_document": docs[row["movie_id"]]["text"],
                    "verdict": None, "unsupported_claims": [], "reviewer_notes": ""})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, sample)
    report = {
        **provenance(data, config), "split": split, **counts, "sample_size": len(sample),
        "predictions_sha256": sha256(predictions), "packet_sha256": sha256(output),
        "status": "pending_manual_review",
        "instructions": "Fill verdict (supported/unsupported/unclear), unsupported_claims, and reviewer_notes. Review only against supplied input evidence; add any extra model input before judging.",
    }
    save_json(output.with_suffix(".report.json"), report)
    return report


def summarize(packet, output):
    counts = Counter()
    examples = []
    seen = set()
    for row in jsonl(packet):
        key = (row["user_id"], row["movie_id"])
        if key in seen:
            raise ValueError(f"Duplicate audit item: {key}")
        seen.add(key)
        verdict = row["verdict"]
        if verdict is not None and verdict not in ("supported", "unsupported", "unclear"):
            raise ValueError(f"Invalid verdict: {verdict}")
        if verdict == "unsupported" and not row["unsupported_claims"]:
            raise ValueError("Unsupported verdicts must identify unsupported claims")
        if verdict == "supported" and row["unsupported_claims"]:
            raise ValueError("Supported verdict contradicts listed unsupported claims")
        counts[verdict or "pending"] += 1
        if verdict == "unsupported" and len(examples) < 5:
            examples.append({k: row[k] for k in ("user_id", "movie_id", "explanation", "unsupported_claims")})
    total = sum(counts.values())
    if not total:
        raise ValueError("Empty audit packet")
    reviewed = total - counts["pending"]
    report = {
        "packet_sha256": sha256(packet), "sample_size": total, "reviewed": reviewed,
        "status": "complete" if not counts["pending"] else "pending_manual_review",
        "verdict_counts": {k: counts[k] for k in ("supported", "unsupported", "unclear", "pending")},
        "unsupported_fraction_of_reviewed": counts["unsupported"] / reviewed if reviewed else None,
        "unclear_fraction_of_reviewed": counts["unclear"] / reviewed if reviewed else None,
        "unsupported_examples": examples,
        "limitation": "This measures explanation grounding in the sampled evidence, not recommendation accuracy.",
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=DATA)
    prep.add_argument("--config", type=Path, default=CONFIG)
    prep.add_argument("--split", choices=("validation", "test"), default="validation")
    prep.add_argument("--predictions", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("outputs/reports/p1-audit.jsonl"))
    summary = sub.add_parser("summarize")
    summary.add_argument("--packet", type=Path, default=Path("outputs/reports/p1-audit.jsonl"))
    summary.add_argument("--output", type=Path, default=Path("outputs/reports/p1-audit-summary.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.data, args.predictions, args.output, read_config(args.config), args.split)
    else:
        report = summarize(args.packet, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
