"""Prepare a reproducible, label-blind manual audit of P2 explanations.

This command validates a complete P2 export, samples user episodes without
reading held-out targets, and writes an immutable worksheet. It does not judge
whether explanation claims are true. Reviewers fill the blank judgment fields.

Run from the repository root::

    python -m src.evaluation.p2_grounding_audit --catalog CATALOG.csv \
        --movie-documents MOVIE_DOCUMENTS.jsonl --profiles PROFILES.jsonl \
        --users USERS.json --split validation \
        --candidate-pool FROZEN_B3.jsonl --predictions P2.jsonl \
        --trace P2_TRACE.jsonl --sample-size 30 --seed 20261002 \
        --output-dir outputs/reports/p2-grounding-audit-validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .evaluate import _canonical_bytes
from .validate import (
    P2_EXPLANATION_COUNT, SPLITS, PredictionValidationError, _parse_json,
    _sha256, load_context, validate_predictions,
)


AUDIT_VERSION = "p2-grounding-audit-v1"
OUTPUT_FILES = ("manifest.json", "review.csv")
REVIEW_COLUMNS = (
    "split", "user_id", "rank", "movie_id", "explanation_text",
    "catalog_title", "catalog_genres", "movie_document_text",
    "document_genres_json", "genome_tags_json", "user_tags_json",
    "movie_document_record_sha256", "visible_profile_text",
    "clarification_turns_json", "termination_reason", "initial_pool_member",
    "retrieved_pool_member", "final_pool_member", "profile_record_sha256",
    "trace_record_sha256", "claims_reviewed", "catalog_or_document_evidence",
    "profile_or_answer_evidence", "unsupported_or_unverifiable_claims",
    "grounding_label", "reviewer", "review_notes", "adjudication",
)


@dataclass(frozen=True)
class GroundingAudit:
    manifest: dict
    rows: list[dict]


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rank(seed: int, split: str, user_id: int) -> bytes:
    return hashlib.sha256(f"{seed}|{split}|{user_id}".encode("ascii")).digest()


def _selected_users(expected: set[int], sample_size: int, seed: int, split: str) -> list[int]:
    if type(sample_size) is not int or not 0 < sample_size <= len(expected):
        raise ValueError(f"sample_size must be an integer from 1 to {len(expected)}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer, excluding booleans")
    return sorted(expected, key=lambda user_id: (_rank(seed, split, user_id), user_id))[:sample_size]


def _profile_records(path: Path, selected: set[int]) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open("rb") as source:
        for raw in source:
            record = _parse_json(raw)
            user_id = record["user_id"]
            if user_id in selected:
                records[user_id] = record
    if set(records) != selected:
        raise ValueError("selected profiles changed after validation")
    for user_id, record in records.items():
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"profile for user {user_id} lacks visible text for grounding audit")
    return records


def _catalog_records(path: Path, movie_ids: set[int]) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for record in reader:
            movie_id = int(record["movie_id"])
            if movie_id in movie_ids:
                records[movie_id] = record
    if set(records) != movie_ids:
        raise ValueError("selected catalog rows changed after validation")
    return records


def _movie_document_records(
    path: Path, catalog_ids: set[int], selected_catalog: dict[int, dict[str, str]],
) -> dict[int, dict]:
    """Read the versioned documents and bind sampled evidence to the catalog."""
    selected: dict[int, dict] = {}
    seen: set[int] = set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            record = _parse_json(raw)
            if not isinstance(record, dict) or type(record.get("movie_id")) is not int:
                raise ValueError(f"{path}, line {line_number}: document requires integer movie_id")
            movie_id = record["movie_id"]
            if movie_id not in catalog_ids or movie_id in seen:
                raise ValueError(f"{path}, line {line_number}: duplicate or out-of-catalog movie_id {movie_id}")
            seen.add(movie_id)
            if movie_id not in selected_catalog:
                continue
            for field in ("title", "text"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{path}, line {line_number}: document {movie_id} lacks {field}")
            for field in ("genres", "genome_tags", "user_tags"):
                value = record.get(field)
                if not isinstance(value, list) or any(not isinstance(tag, str) or not tag.strip() for tag in value):
                    raise ValueError(f"{path}, line {line_number}: document {movie_id} has invalid {field}")
            catalog_row = selected_catalog[movie_id]
            expected_genres = (
                [] if catalog_row.get("genres") == "(no genres listed)"
                else catalog_row.get("genres", "").split("|")
            )
            if record["title"] != catalog_row.get("title") or record["genres"] != expected_genres:
                raise ValueError(f"{path}, line {line_number}: document {movie_id} disagrees with catalog")
            selected[movie_id] = record
    if seen != catalog_ids or set(selected) != set(selected_catalog):
        raise ValueError("movie document IDs do not exactly cover the catalog and sampled movies")
    return selected


def prepare_audit(
    catalog_path: Path, movie_documents_path: Path, profiles_path: Path, users_path: Path, split: str,
    candidate_pool_path: Path, predictions_path: Path, trace_path: Path,
    *, sample_size: int, seed: int, required_count: int = 50,
) -> GroundingAudit:
    """Validate all users, then sample complete episodes using source-blind hashes.

    The selector uses only the saved user IDs and a declared seed. It does not
    inspect targets, scores, outcomes, explanation text, or action paths. The
    worksheet covers every top-ten explanation for each sampled episode.
    """
    paths = {
        "catalog": Path(catalog_path).resolve(),
        "movie_documents": Path(movie_documents_path).resolve(),
        "profiles": Path(profiles_path).resolve(),
        "users": Path(users_path).resolve(),
        "candidate_pool": Path(candidate_pool_path).resolve(),
        "predictions": Path(predictions_path).resolve(),
        "trace": Path(trace_path).resolve(),
    }
    movie_documents_sha256 = _sha256(paths["movie_documents"])
    context = load_context(paths["catalog"], paths["profiles"], paths["users"], split)
    selected = _selected_users(context.expected_user_ids, sample_size, seed, split)
    validated = validate_predictions(
        paths["predictions"], context, method="p2", required_count=required_count,
        candidate_pool_path=paths["candidate_pool"], trace_path=paths["trace"],
    )
    selected_profiles = _profile_records(paths["profiles"], set(selected))
    selected_movie_ids = {
        item["movie_id"]
        for user_id in selected
        for item in validated.records[user_id]["recommendations"][:P2_EXPLANATION_COUNT]
    }
    catalog = _catalog_records(paths["catalog"], selected_movie_ids)
    documents = _movie_document_records(paths["movie_documents"], context.catalog_ids, catalog)
    source_sha256 = {**validated.report["source_sha256"], "movie_documents": movie_documents_sha256}
    if any(_sha256(path) != source_sha256[name] for name, path in paths.items()):
        raise ValueError("an input changed during audit preparation; restart with immutable inputs")

    rows: list[dict] = []
    for user_id in selected:
        profile = selected_profiles[user_id]
        trace = validated.traces[user_id]
        explanation_event = next(event for event in trace["events"] if event["action"] == "explain_recommendations")
        clarification_turns = [
            {key: event[key] for key in ("question", "answer", "answer_source")}
            for event in trace["events"] if event["action"] == "ask_clarification"
        ]
        added = {
            movie_id for event in trace["events"] if event["action"] == "retrieve_candidates"
            for movie_id in event["added_movie_ids"]
        }
        initial = set(trace["initial_candidate_movie_ids"])
        final = set(trace["final_candidate_movie_ids"])
        for rank, explanation in enumerate(explanation_event["explanations"], start=1):
            movie_id = explanation["movie_id"]
            metadata = catalog[movie_id]
            document = documents[movie_id]
            rows.append({
                "split": split, "user_id": user_id, "rank": rank, "movie_id": movie_id,
                "explanation_text": explanation["text"],
                "catalog_title": metadata.get("title", ""),
                "catalog_genres": metadata.get("genres", ""),
                "movie_document_text": document["text"],
                "document_genres_json": _canonical_bytes(document["genres"]).decode("utf-8").strip(),
                "genome_tags_json": _canonical_bytes(document["genome_tags"]).decode("utf-8").strip(),
                "user_tags_json": _canonical_bytes(document["user_tags"]).decode("utf-8").strip(),
                "movie_document_record_sha256": _digest(_canonical_bytes(document)),
                "visible_profile_text": profile["text"],
                "clarification_turns_json": _canonical_bytes(clarification_turns).decode("utf-8").strip(),
                "termination_reason": trace["termination_reason"],
                "initial_pool_member": int(movie_id in initial),
                "retrieved_pool_member": int(movie_id in added),
                "final_pool_member": int(movie_id in final),
                "profile_record_sha256": _digest(_canonical_bytes(profile)),
                "trace_record_sha256": _digest(_canonical_bytes(trace)),
                "claims_reviewed": "", "catalog_or_document_evidence": "",
                "profile_or_answer_evidence": "", "unsupported_or_unverifiable_claims": "",
                "grounding_label": "", "reviewer": "", "review_notes": "", "adjudication": "",
            })
    review_bytes = _review_bytes(rows)
    manifest = {
        "status": "prepared_for_manual_review", "audit_version": AUDIT_VERSION,
        "split": split, "population_user_count": len(context.expected_user_ids),
        "sample_size": sample_size, "seed": seed,
        "selection_rule": "first sample_size users sorted by SHA-256(seed|split|user_id), then user_id; labels and traces are not used",
        "selected_user_ids_in_selection_order": selected,
        "explanations_per_user": min(P2_EXPLANATION_COUNT, required_count),
        "review_row_count": len(rows),
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": dict(sorted(source_sha256.items())),
        "review_csv_sha256": _digest(review_bytes),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "judgment_status": "unreviewed",
    }
    return GroundingAudit(manifest=manifest, rows=rows)


def _review_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def write_audit(output_dir: Path, audit: GroundingAudit) -> None:
    """Create a two-file audit package without replacing previous work."""
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    sources = {name: Path(path).resolve() for name, path in audit.manifest["source_paths"].items()}
    if any(resolved == source for source in sources.values()):
        raise ValueError("output directory must be outside input files")
    for source in sources.values():
        for ancestor in source.parents:
            if ancestor.name == "processed" and ancestor.parent.name == "data" and (
                resolved == ancestor or ancestor in resolved.parents
            ):
                raise ValueError("output directory must be outside processed data")
    payloads = {
        "manifest.json": _canonical_bytes(audit.manifest),
        "review.csv": _review_bytes(audit.rows),
    }
    if _digest(payloads["review.csv"]) != audit.manifest["review_csv_sha256"]:
        raise ValueError("review CSV changed after manifest creation")
    if output_dir.exists():
        if not output_dir.is_dir() or {item.name for item in output_dir.iterdir()} != set(OUTPUT_FILES) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw
            for name, raw in payloads.items()
        ):
            raise ValueError("existing audit differs; preserve it and choose a new output directory")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, raw in payloads.items():
            (staged / name).write_bytes(raw)
        if output_dir.exists():
            raise ValueError("output directory appeared during staging")
        os.rename(staged, output_dir)
    finally:
        if staged.exists():
            if staged.resolve().parent != output_dir.parent.resolve():
                raise ValueError("temporary audit path escaped output parent")
            for name in OUTPUT_FILES:
                file = staged / name
                if file.is_file():
                    file.unlink()
            staged.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--movie-documents", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--required-count", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        audit = prepare_audit(
            args.catalog, args.movie_documents, args.profiles, args.users, args.split, args.candidate_pool,
            args.predictions, args.trace, sample_size=args.sample_size,
            seed=args.seed, required_count=args.required_count,
        )
        write_audit(args.output_dir, audit)
    except PredictionValidationError as error:
        print(str(error), file=sys.stderr)
        print(_canonical_bytes(error.report).decode("utf-8"), end="")
        return 1
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"P2 grounding audit preparation failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(audit.manifest).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
