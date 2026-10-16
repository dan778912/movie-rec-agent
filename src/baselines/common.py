"""Small standard-library helpers shared by Member 3's commands."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from src.data.build import sha256

DATA = Path("data/processed/ml25m-global-v1")
CONFIG = Path("configs/member3_v1.json")


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("training_user_limit", "examples_per_user", "minimum_history", "popularity_k", "audit_sample_size"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["positive_rating_at_least"] != 4.0 or config["negative_rating_at_most"] != 2.5:
        raise ValueError("The shared relevance contract requires positive >=4 and negative <=2.5")
    return config


def check_inputs(data, names):
    """Check consumed artifacts against Member 1's manifest, without opening test labels."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in names:
        if sha256(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    return manifest


def provenance(data, config):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    sources = sorted(Path("src/baselines").glob("*.py")) + [Path("src/profiles/text.py")]
    return {
        "config": config,
        "split_version": json.loads((data / "manifest.json").read_text())["version"],
        "split_manifest_sha256": sha256(data / "manifest.json"),
        "git_commit": revision.stdout.strip() or "unknown",
        "worktree_dirty": bool(status.stdout),
        "source_sha256": {str(p): sha256(p) for p in sources},
    }


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
