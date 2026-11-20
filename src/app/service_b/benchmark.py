"""Measure retrieval on the real catalog; this is not P2 recommendation evaluation."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from src.evaluation.p2_run_manifest import _percentile
from src.retrieval.online import OnlineRetriever, SentenceEncoder, canonical, chroma_client, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--documents", default="data/processed/ml25m-global-v1/movie_documents.jsonl")
    p.add_argument("--profiles", default="data/processed/ml25m-global-v1/validation_profiles.jsonl")
    p.add_argument("--chroma-path", default="data/indexes/chroma")
    p.add_argument("--output", required=True)
    p.add_argument("--users", type=int, default=20)
    args = p.parse_args()
    if args.users <= 0:
        p.error("--users must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(args.manifest).read_text())
    started = time.perf_counter()
    encoder = SentenceEncoder(manifest["embedding"])
    retriever = OnlineRetriever(chroma_client(args.chroma_path), args.manifest, args.documents, encoder)
    startup = (time.perf_counter() - started) * 1000
    profiles = []
    with Path(args.profiles).open() as source:
        for line in source:
            profiles.append(json.loads(line))
            if len(profiles) == args.users:
                break
    retriever.initial_pool(profile_text=profiles[0]["text"], clarifications=[],
                           history_movie_ids=profiles[0]["history_movie_ids"])
    measurements = []
    for profile in profiles:
        start = time.perf_counter()
        pool = retriever.initial_pool(profile_text=profile["text"], clarifications=[],
                                      history_movie_ids=profile["history_movie_ids"])
        initial_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        expanded = retriever.search(query="Comedy with friendship themes", max_new=50,
            exclude_movie_ids=list(set(profile["history_movie_ids"]) | set(pool.movie_ids)))
        expansion_ms = (time.perf_counter() - start) * 1000
        assert not set(pool.movie_ids) & set(profile["history_movie_ids"])
        assert not set(expanded.movie_ids) & (set(profile["history_movie_ids"]) | set(pool.movie_ids))
        measurements.append({"user_id": profile["user_id"], "initial_count": len(pool.movie_ids),
            "expansion_count": len(expanded.movie_ids), "initial_ms": initial_ms, "expansion_ms": expansion_ms})
    report = {"measurement_version": "service-b-retrieval-benchmark-v1", "hardware": platform.platform(),
        "processor": platform.machine(), "device": "cpu", "concurrency": 1,
        "scope": "Real MiniLM and Chroma retrieval only; no P2 checkpoint, no recommendation metrics",
        "population": "First N canonical validation profiles in file order; not the official 300-user run",
        "embedding": manifest["embedding"], "index_version": manifest["index_version"],
        "index_manifest_sha256": file_hash(args.manifest), "profiles_sha256": file_hash(args.profiles),
        "catalog_count": manifest["count"], "startup_ms": startup, "warmup_calls": 1,
        "users": len(measurements), "measurements": measurements,
        "latency_ms": {key: {"p50": _percentile([r[key] for r in measurements], .5),
                             "p95": _percentile([r[key] for r in measurements], .95)}
                       for key in ("initial_ms", "expansion_ms")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as out:
        out.write(canonical(report))
    print(json.dumps({k: v for k, v in report.items() if k != "measurements"}, indent=2))


if __name__ == "__main__":
    main()
