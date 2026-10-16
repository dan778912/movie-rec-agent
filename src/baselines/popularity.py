"""B0: rank unseen movies by positive training-rating count."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from src.baselines.common import DATA, CONFIG, check_inputs, jsonl, provenance, read_config, save_json
from src.data.build import csv_rows, sha256, write_jsonl


def predict(data, output, config, split="validation", user_ids=None):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    k = config["popularity_k"]
    if type(k) is not int or k < 1:
        raise ValueError("popularity_k must be positive")
    started = time.perf_counter()
    check_inputs(data, ["catalog.csv", f"{split}_profiles.jsonl"])
    scores = {int(r["movie_id"]): int(r["train_positives"]) for r in csv_rows(data / "catalog.csv")}
    order = sorted(scores, key=lambda mid: (-scores[mid], mid))
    counts = Counter()
    selected = set()
    def predictions():
        for profile in jsonl(data / f"{split}_profiles.jsonl"):
            uid = profile["user_id"]
            if user_ids is not None and uid not in user_ids:
                continue
            if uid in selected:
                raise ValueError(f"Duplicate profile user: {uid}")
            selected.add(uid)
            seen = set(profile["history_movie_ids"])
            recs = []
            for mid in order:
                if mid in seen:
                    continue
                recs.append({"movie_id": mid, "score": scores[mid], "rank": len(recs) + 1})
                if len(recs) == k:
                    break
            counts["users"] += 1
            counts["short_lists"] += len(recs) < k
            counts["recommendations"] += len(recs)
            yield {"user_id": uid, "split": split, "method": "popularity", "recommendations": recs}
        if user_ids is not None and selected != user_ids:
            raise ValueError(f"Requested users missing from {split} profiles: {sorted(user_ids - selected)[:10]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, predictions())
    if not selected:
        raise ValueError("No evaluation users were selected")
    report = {
        **provenance(data, config), "split": split, "method": "popularity", **counts,
        "user_ids": sorted(selected), "error_count": 0,
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": sha256(output),
        "scoring": "train_positives descending; movie_id ascending on ties",
        "validation_sweep": "No model hyperparameters: export top-100 once; shared evaluator scores each cutoff.",
        "limitations": ["Popularity is global; user-specific behavior is watched-item exclusion only.",
                        "Popularity counts remain training-only for both evaluation splits.",
                        "Shared validation, metric results, and measured failure cases await Member 2's evaluator."],
    }
    save_json(output.with_suffix(".report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, help="Default: outputs/predictions/b0-<split>.jsonl")
    parser.add_argument("--user-ids", type=Path, help="Member 2's fixed subset: JSON array of integer user IDs")
    args = parser.parse_args()
    subset = None
    if args.user_ids:
        ids = json.loads(args.user_ids.read_text())
        if not isinstance(ids, list) or any(type(uid) is not int for uid in ids) or len(set(ids)) != len(ids):
            parser.error("--user-ids must contain a JSON array of unique integer IDs")
        subset = set(ids)
    output = args.output or Path(f"outputs/predictions/b0-{args.split}.jsonl")
    report = predict(args.data, output, read_config(args.config), args.split, subset)
    print(json.dumps({k: v for k, v in report.items() if k not in ("user_ids", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
