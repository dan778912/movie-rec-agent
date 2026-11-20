"""Use exactly Member 1's metadata universe for offline clarification answers."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.p2_samples import check_documents, load_metadata
from src.retrieval.online import file_hash

RAW_METADATA_FILES = ("movies.csv", "genome-scores.csv", "genome-tags.csv", "tags.csv")


def load_simulator_metadata(data, raw):
    data, raw = Path(data), Path(raw)
    manifest = json.loads((data / "manifest.json").read_text())
    for name in RAW_METADATA_FILES:
        if file_hash(raw / name) != manifest["raw_sha256"][name]:
            raise ValueError(f"Simulator raw metadata checksum mismatch: {name}")
    for name in ("catalog.csv", "movie_documents.jsonl", "statistics.json"):
        if file_hash(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Simulator processed metadata checksum mismatch: {name}")
    statistics = json.loads((data / "statistics.json").read_text())
    metadata, catalog_ids, _ = load_metadata(raw, data, manifest["config"], statistics["validation_cutoff_unix"])
    check_documents(data, metadata, catalog_ids)
    # This mapping deliberately includes watched movies outside the recommendation
    # catalog, just as training does. It never expands the retrieval catalog.
    return metadata
