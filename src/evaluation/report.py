"""Build portable, verified comparison tables from saved evaluation bundles.

Only supplied methods and saved bootstrap comparisons appear. Synthetic tables
are explicitly labelled and are implementation checks, not project findings.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import bootstrap
from .evaluate import _canonical_bytes, _json_object, _sha256

METHOD_ORDER = ("popularity", "bm25", "embedding", "p1")
SYSTEMS = {"popularity": "B0 (popularity)", "bm25": "B2 (BM25)",
           "embedding": "B3 (embedding)", "p1": "P1 (fine-tuned reranker)"}
COVERAGE_FIELDS = ("positive_targets", "in_catalog_positive_targets",
                   "out_of_catalog_positive_targets", "catalog_positive_coverage")
DIFFERENCE_COLUMNS = ("baseline_method", "challenger_method", "metric", "baseline_mean",
                      "challenger_mean", "mean_difference", "ci_lower", "ci_upper",
                      "confidence", "resamples", "seed")
OUTPUT_FILES = ("table.csv", "table.md", "differences.csv", "report.json")


@dataclass(frozen=True)
class TableResult:
    summary: dict
    rows: list[dict]
    comparisons: list[dict]
    markdown: str


def _same(value, expected, label: str) -> None:
    if _canonical_bytes(value) != _canonical_bytes(expected):
        raise ValueError(f"{label} disagrees between reports")


def _compatible(first: tuple, other: tuple) -> None:
    left, left_users = first[:2]
    right, right_users = other[:2]
    for key in ("dataset_version", "evaluation_version", "split", "config_status",
                "user_count", "settings", "coverage"):
        _same(left[key], right[key], key)
    _same(left["implementation"]["source_sha256"], right["implementation"]["source_sha256"],
          "scoring implementation")
    for key in bootstrap.SHARED_SOURCES:
        _same(left["source_sha256"][key], right["source_sha256"][key], f"{key} source hash")
    if set(left["metrics"]) != set(right["metrics"]):
        raise ValueError("runs have different metric sets")
    if left_users.keys() != right_users.keys():
        raise ValueError("runs must have exactly the same user IDs")
    for user in left_users:
        for key in bootstrap.COUNTS:
            if left_users[user][key] != right_users[user][key]:
                raise ValueError(f"runs have different {key} for user {user}")


def _check_differences(path: Path, expected: list[dict]) -> None:
    """Check saved user differences, including reordered rows, against scores."""
    by_user = {row["user_id"]: row for row in expected}
    seen = set()
    columns = set(expected[0])
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != columns:
            raise ValueError("comparison per_user_differences.csv has invalid columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("comparison per_user_differences.csv has a malformed row")
            if re.fullmatch(r"[1-9][0-9]*", row["user_id"]) is None:
                raise ValueError("comparison user_id must be a positive integer")
            user = int(row["user_id"])
            if user not in by_user or user in seen:
                raise ValueError("comparison contains unexpected or duplicate users")
            seen.add(user)
            for key, value in by_user[user].items():
                if key == "user_id":
                    continue
                if isinstance(value, str):
                    if row[key] != value:
                        raise ValueError(f"comparison user {user} has inconsistent {key}")
                else:
                    number = float(row[key])
                    if not math.isfinite(number) or not math.isclose(number, value, rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"comparison user {user} has inconsistent {key}")
    if seen != by_user.keys():
        raise ValueError("comparison is missing users")


def _load_comparison(directory: Path, runs: dict, run_dirs: dict) -> tuple[dict, dict, dict]:
    directory = Path(directory).resolve()
    paths = {"comparison": directory / "comparison.json",
             "per_user_differences": directory / "per_user_differences.csv"}
    saved = _json_object(paths["comparison"])
    baseline, challenger = saved.get("baseline_method"), saved.get("challenger_method")
    if baseline not in runs or challenger not in runs or baseline == challenger:
        raise ValueError("comparison requires two distinct methods present in the table")
    # Bind report bytes before recomputing. Historical absolute paths may differ
    # when a complete bundle is copied; hashes identify the actual inputs.
    expected_hashes = {}
    for prefix, method in (("baseline", baseline), ("challenger", challenger)):
        expected_hashes.update({f"{prefix}_{name.split('.')[0]}": value
                               for name, value in runs[method][3].items()})
    _same(saved.get("source_sha256"), expected_hashes, "comparison source hashes")
    expected = bootstrap.compare_runs(run_dirs[baseline], run_dirs[challenger])
    if set(saved) != set(expected.summary):
        raise ValueError("comparison summary has missing or unexpected fields")
    for key, value in expected.summary.items():
        if key != "source_paths":
            _same(saved[key], value, f"comparison {key}")
    old_paths = saved.get("source_paths")
    if not isinstance(old_paths, dict) or set(old_paths) != set(expected.summary["source_paths"]):
        raise ValueError("comparison source_paths is incomplete")
    if any(not isinstance(value, str) or not value or not Path(value).is_absolute() for value in old_paths.values()):
        raise ValueError("comparison source_paths must contain absolute paths")
    _check_differences(paths["per_user_differences"], expected.per_user)
    return saved, paths, {key: _sha256(path) for key, path in paths.items()}


def _csv_bytes(rows: list[dict], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _escape(value) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _markdown(summary: dict, rows: list[dict], comparisons: list[dict]) -> str:
    experiment = summary["shared_experiment"]
    synthetic = summary["kind"] == "synthetic"
    lines = ["# Synthetic evaluation check" if synthetic else "# Recommendation experiment results", ""]
    if synthetic:
        lines += ["Synthetic examples verify the implementation; these are not model performance findings.", ""]
    lines += [f"Dataset: `{_escape(experiment['dataset_version'])}`. "
              f"Evaluation: `{_escape(experiment['evaluation_version'])}`. "
              f"Split: `{_escape(experiment['split'])}`. Users: {rows[0]['user_count']}.", "",
              f"Configuration status: `{_escape(summary['config_status'])}`. "
              "A generated table does not establish team approval of this configuration.", "",
              "## Mean recommendation metrics", "",
              "Scores are macro means over the same users. Values below are rounded to six decimal places; CSV/JSON retain full precision.", ""]
    metrics = summary["metric_names"]
    lines += ["| System | " + " | ".join(_escape(name) for name in metrics) + " |",
              "|---|" + "---:|" * len(metrics)]
    for row in rows:
        lines.append("| " + row["system"] + " | " + " | ".join(f"{row[name]:.6f}" for name in metrics) + " |")
    lines += ["", "Recall at the full candidate-pool size measures candidate coverage. P1 reorders its frozen B3 pool, so it cannot improve recall at that full size.", "",
              "## Data and prediction coverage", "",
              "Catalog coverage is in-catalog positive targets divided by all positive targets. Full-split coverage includes users outside the selected subset; prediction-user coverage describes submitted records.", "",
              "| System | Full-split catalog coverage | Selected-user catalog coverage | Full-split positive / in-catalog / outside | Selected positive / in-catalog / outside | Prediction-user coverage |",
              "|---|---:|---:|---|---|---:|"]
    for row in rows:
        counts = [" / ".join(str(row[f"{scope}_{field}"]) for field in COVERAGE_FIELDS[:3])
                  for scope in ("full_split", "selected_users")]
        lines.append(f"| {row['system']} | {row['full_split_catalog_positive_coverage']:.6f} | "
                     f"{row['selected_users_catalog_positive_coverage']:.6f} | {counts[0]} | {counts[1]} | "
                     f"{row['prediction_user_coverage']:.6f} |")
    if comparisons:
        lines += ["", "## Supplied paired bootstrap comparisons", "",
                  "Differences mean challenger minus baseline. Intervals are paired-user percentile intervals; no p-value or significance decision is produced.", "",
                  "| Baseline | Challenger | Metric | Difference | Interval | Confidence | Resamples | Seed |",
                  "|---|---|---|---:|---|---:|---:|---:|"]
        for row in comparisons:
            lines.append(f"| {row['baseline_method']} | {row['challenger_method']} | {row['metric']} | "
                         f"{row['mean_difference']:+.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] | "
                         f"{row['confidence']:.6f} | {row['resamples']} | {row['seed']} |")
    else:
        lines += ["", "No comparison bundle was supplied; no confidence intervals were generated."]
    missing = summary["missing_methods"]
    lines += ["", "Methods without supplied results: " + (", ".join(missing) if missing else "none") + ".",
              "Missing methods have no placeholder scores. These tables report supplied results only.", ""]
    return "\n".join(lines)


def build_table(run_dirs: Sequence[Path], *, comparison_dirs: Sequence[Path] = (), kind: str) -> TableResult:
    """Verify compatible saved bundles and construct deterministic report tables."""
    if kind not in ("synthetic", "experiment"):
        raise ValueError("kind must be synthetic or experiment")
    if isinstance(run_dirs, (str, bytes, Path)) or not 1 <= len(run_dirs) <= len(METHOD_ORDER):
        raise ValueError("provide between one and four scored run directories")
    if isinstance(comparison_dirs, (str, bytes, Path)):
        raise ValueError("comparison_dirs must be a sequence of directories")
    runs, directories = {}, {}
    for directory in run_dirs:
        loaded = bootstrap._load_run(Path(directory))
        method = loaded[0]["method"]
        if method in runs:
            raise ValueError(f"duplicate method: {method}")
        runs[method], directories[method] = loaded, Path(directory).resolve()
    methods = [method for method in METHOD_ORDER if method in runs]
    first = runs[methods[0]]
    for method in methods[1:]:
        _compatible(first, runs[method])
    if kind == "experiment" and first[0]["dataset_version"].lower().startswith("synthetic"):
        raise ValueError("synthetic dataset reports require --kind synthetic")
    if "p1" in runs:
        if "embedding" not in runs:
            raise ValueError("a P1 table requires its frozen embedding counterpart")
        if runs["p1"][0]["source_sha256"]["candidate_pool"] != runs["embedding"][0]["source_sha256"]["predictions"]:
            raise ValueError("P1 candidate pool hash does not match embedding predictions")
    metrics = sorted(first[0]["metrics"])
    rows, hashes, paths = [], {}, {}
    for method in methods:
        summary, _, report_paths, report_hashes = runs[method]
        row = {"method": method, "system": SYSTEMS[method], "split": summary["split"],
               "user_count": summary["user_count"], **{name: summary["metrics"][name] for name in metrics}}
        for scope in ("full_split", "selected_users"):
            row.update({f"{scope}_{key}": summary["coverage"][scope][key] for key in COVERAGE_FIELDS})
        row["prediction_user_coverage"] = summary["prediction_user_coverage"]
        rows.append(row)
        hashes.update({f"run_{method}_{name.split('.')[0]}": value for name, value in report_hashes.items()})
        paths.update({f"run_{method}_{name.split('.')[0]}": str(path) for name, path in report_paths.items()})
        paths.update({f"run_{method}_original_{key}": value for key, value in summary["source_paths"].items()})
    comparisons, pairs = [], set()
    loaded_comparisons = []
    for directory in comparison_dirs:
        saved, report_paths, report_hashes = _load_comparison(Path(directory), runs, directories)
        pair = (saved["baseline_method"], saved["challenger_method"])
        if pair in pairs:
            raise ValueError("duplicate comparison pair")
        pairs.add(pair)
        loaded_comparisons.append((pair, saved, report_paths, report_hashes))
    loaded_comparisons.sort(key=lambda item: tuple(METHOD_ORDER.index(method) for method in item[0]))
    for (baseline, challenger), saved, report_paths, report_hashes in loaded_comparisons:
        prefix = f"comparison_{baseline}_to_{challenger}"
        hashes.update({f"{prefix}_{name}": value for name, value in report_hashes.items()})
        paths.update({f"{prefix}_{name}": str(path) for name, path in report_paths.items()})
        paths.update({f"{prefix}_original_{key}": value for key, value in saved["source_paths"].items()})
        for metric in metrics:
            values = saved["metrics"][metric]
            comparisons.append({"baseline_method": baseline, "challenger_method": challenger,
                                "metric": metric, **{key: values[key] for key in DIFFERENCE_COLUMNS[3:]}})
    reference = first[0]
    summary = {"status": "table_generated", "kind": kind, "run_count": len(runs),
               "comparison_count": len(pairs), "methods": methods,
               "missing_methods": [method for method in METHOD_ORDER if method not in runs],
               "metric_names": metrics, "config_status": reference["config_status"],
               "shared_experiment": {key: reference[key] for key in
                                     ("dataset_version", "evaluation_version", "split", "settings")},
               "shared_source_sha256": {key: reference["source_sha256"][key] for key in bootstrap.SHARED_SOURCES},
               "rows": rows, "comparisons": comparisons, "source_sha256": hashes,
               "implementation": {"source_sha256": {"report": _sha256(Path(__file__).resolve()),
                                                        "bootstrap": _sha256(Path(bootstrap.__file__).resolve()),
                                                        **reference["implementation"]["source_sha256"]}},
               "source_paths": paths}
    return TableResult(summary, rows, comparisons, _markdown(summary, rows, comparisons))


def write_table(output_dir: Path, result: TableResult) -> None:
    """Atomically save four immutable output files; identical reruns succeed."""
    if not result.rows or result.summary.get("status") != "table_generated":
        raise ValueError("cannot write a table without verified rows")
    _same(result.rows, result.summary.get("rows"), "table rows")
    _same(result.comparisons, result.summary.get("comparisons"), "table comparisons")
    if result.markdown != _markdown(result.summary, result.rows, result.comparisons):
        raise ValueError("table Markdown disagrees with verified values")
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    sources = result.summary.get("source_paths")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("table requires source_paths for safe output handling")
    for key, value in sources.items():
        source = Path(value).resolve()
        # All scored/comparison input directories and original data directories
        # are protected, including symbolic-link aliases and nested outputs.
        if key.endswith("data_dir"):
            protected = source
        elif "_original_" not in key:
            protected = source.parent
        else:
            protected = None
        if protected is not None and (resolved == protected or protected in resolved.parents):
            raise ValueError("output directory must be outside input report and original data directories")
        if resolved == source:
            raise ValueError("output directory must not overwrite an input")
        for name in OUTPUT_FILES:
            destination = output_dir / name
            if destination.resolve() == source or (destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError("output must not overwrite or alias an input")
    columns = ["method", "system", "split", "user_count", *result.summary["metric_names"]]
    columns += [f"{scope}_{key}" for scope in ("full_split", "selected_users") for key in COVERAGE_FIELDS]
    columns.append("prediction_user_coverage")
    payloads = {"table.csv": _csv_bytes(result.rows, columns),
                "table.md": result.markdown.encode("utf-8"),
                "differences.csv": _csv_bytes(result.comparisons, DIFFERENCE_COLUMNS),
                "report.json": _canonical_bytes(result.summary)}
    if output_dir.exists():
        if not output_dir.is_dir() or {item.name for item in output_dir.iterdir()} != set(payloads) or any(
            not (output_dir / name).is_file() or (output_dir / name).read_bytes() != raw for name, raw in payloads.items()
        ):
            raise ValueError("existing results differ; preserve them and choose a new output directory")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, raw in payloads.items():
            (staged / name).write_bytes(raw)
        if output_dir.exists():
            raise ValueError("output directory appeared during staging; choose a new output directory")
        os.rename(staged, output_dir)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--comparisons", nargs="*", type=Path, default=[])
    parser.add_argument("--kind", choices=("synthetic", "experiment"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_table(args.runs, comparison_dirs=args.comparisons, kind=args.kind)
        write_table(args.output_dir, result)
    except (OSError, ValueError, TypeError, UnicodeError, csv.Error) as error:
        print(f"Table generation failed: {error}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result.summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
