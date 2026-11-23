"""Package Member 1's Phase 2 data as one ZIP for teammates.

The generated files are Git-ignored, so they travel as a ZIP. The package
refuses to build unless the quality report shows zero errors and every packaged
data file still matches the hash recorded by that report. It contains the data
files, the generator/check/seed reports, the field guide
(``docs/member1_p2_handoff.md`` as ``README.md``), committed examples, and a
``SHA256SUMS.txt`` that ``shasum -a 256 -c`` can verify after unzipping.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.data.build import sha256

OUTPUT = Path("data/processed/p2-v2")
PACKAGE = Path("data/processed/handoff/member1_p2_v2_handoff.zip")
DATA_FILES = (
    "p2_samples_train.jsonl", "p2_samples_dev.jsonl",
    "p2_score_samples_train.jsonl", "p2_score_samples_dev.jsonl",
    "p2_episodes_train.jsonl", "p2_episodes_dev.jsonl",
    "catalog_seed.jsonl",
)
REPORTS = ("p2_quality_report.json", "p2_samples.report.json", "catalog_seed.report.json")
DOCS = {
    "README.md": Path("docs/member1_p2_handoff.md"),
    "docs/member1_p2_report.md": Path("docs/member1_p2_report.md"),
    "examples/p2_sample_examples.jsonl": Path("docs/p2_sample_examples.jsonl"),
    "examples/p2_score_sample_examples.jsonl": Path("docs/p2_score_sample_examples.jsonl"),
}


def package(output: Path, target: Path) -> dict:
    quality = json.loads((output / "p2_quality_report.json").read_text(encoding="utf-8"))
    if quality["error_count"]:
        raise SystemExit(f"Quality report has {quality['error_count']} errors; refusing to package")
    recorded = quality["artifact_sha256"]
    members: dict[str, Path] = {}
    for name in DATA_FILES:
        path = output / name
        if recorded.get(name) != sha256(path):
            raise SystemExit(f"{name} differs from the hash in p2_quality_report.json; rerun p2_checks")
        members[f"data/{name}"] = path
    for name in REPORTS:
        members[f"reports/{name}"] = output / name
    for arcname, path in DOCS.items():
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        members[arcname] = path
    sums = {arcname: sha256(path) for arcname, path in sorted(members.items())}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for arcname, path in sorted(members.items()):
            archive.write(path, arcname)
        archive.writestr("SHA256SUMS.txt", "".join(f"{digest}  {name}\n" for name, digest in sums.items()))
    return {"package": str(target), "package_sha256": sha256(target), "files": sums}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Directory with generated P2 files")
    parser.add_argument("--package", type=Path, default=PACKAGE, help="ZIP file to write")
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.package), indent=2))


if __name__ == "__main__":
    main()
