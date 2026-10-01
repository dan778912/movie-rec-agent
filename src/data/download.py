"""Download and verify the official MovieLens 25M archive."""

from __future__ import annotations

import hashlib
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://files.grouplens.org/datasets/movielens/"
ARCHIVE = "ml-25m.zip"
REQUIRED = {
    "ratings.csv", "movies.csv", "tags.csv", "genome-tags.csv",
    "genome-scores.csv", "links.csv", "README.txt",
}


def _fetch(url: str, destination: Path) -> None:
    # Windows curl uses the system certificate store; some bundled Python
    # distributions have no usable CA bundle.
    if os.name == "nt":
        subprocess.run(["curl.exe", "--fail", "--location", "--show-error", "--silent",
                        "--output", str(destination), url], check=True)
    else:
        urllib.request.urlretrieve(url, destination)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher's archive integrity checksum
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw_dir: Path) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE
    checksum_file = raw_dir / f"{ARCHIVE}.md5"
    if not checksum_file.exists():
        _fetch(BASE + checksum_file.name, checksum_file)
    expected = checksum_file.read_text(encoding="ascii").split()[0].lower()
    if len(expected) != 32 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid publisher MD5 file")
    if not archive.exists() or _md5(archive) != expected:
        partial = raw_dir / f"{ARCHIVE}.part"
        if partial.exists():
            partial.unlink()
        _fetch(BASE + ARCHIVE, partial)
        if _md5(partial) != expected:
            partial.unlink()
            raise ValueError("Downloaded archive does not match GroupLens MD5")
        partial.replace(archive)
    extracted = raw_dir / "ml-25m"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = {Path(name).name: name for name in bundle.namelist() if name.startswith("ml-25m/")}
        missing = REQUIRED - names.keys()
        if missing:
            raise ValueError(f"Archive missing required files: {sorted(missing)}")
        for name in sorted(REQUIRED):
            destination = extracted / name
            if destination.exists() and destination.stat().st_size == bundle.getinfo(names[name]).file_size:
                continue
            partial = destination.with_suffix(destination.suffix + ".part")
            with bundle.open(names[name]) as source, partial.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
            partial.replace(destination)
    return {"archive_md5": expected, "source_url": BASE + ARCHIVE}
