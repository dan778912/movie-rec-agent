"""One-command MovieLens 25M download, build, and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.build import build
from src.data.download import download
from src.data.verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ml25m-global-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_v1.json"))
    args = parser.parse_args()
    source = download(args.raw_root)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stats = build(args.raw_root / "ml-25m", args.output, config)
    checks = verify(args.output)
    print(json.dumps({"source": source, "statistics": stats, "verification": checks}, indent=2))


if __name__ == "__main__":
    main()

