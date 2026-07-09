#!/usr/bin/env python3
"""Normalize baseline memory-accounting evidence into one schema.

The normalized schema keeps the *semantics* fixed across methods:

- ``loaded_index_bytes``: raw/base vectors plus resident index/search state
  at the point just before query execution.
- ``loaded_vector_excluded_index_bytes``: diagnostic structure/search-state
  overhead with the raw/base vector payload excluded.

Most baselines already emit these two byte fields in
``results/accounting/*_accounting_latest.tsv``.  DET-LSH has no separate
load-index/accounting-only mode, so its adapter reads the build-probe JSONL and
maps the native indexing-completion RSS to the same raw-inclusive semantic.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
ACCOUNTING_DIR = ROOT / "results" / "accounting"
DETLSH_SUMMARY = (
    ROOT
    / "docs"
    / "evaluation"
    / "raw_baselines"
    / "det-lsh-build"
    / "detlsh_build_probe_20260520_summary.jsonl"
)
RSS_INVERSION_TOLERANCE_KB = 1024


@dataclass(frozen=True)
class TsvSource:
    method: str
    path: Path
    variant: str = ""


TSV_SOURCES: tuple[TsvSource, ...] = (
    TsvSource("SCORE", ACCOUNTING_DIR / "score_accounting_latest.tsv"),
    TsvSource("HNSW", ACCOUNTING_DIR / "hnsw_accounting_latest.tsv"),
    TsvSource("MIRAGE", ACCOUNTING_DIR / "mirage_accounting_latest.tsv"),
    TsvSource("CRISP", ACCOUNTING_DIR / "crisp_accounting_latest.tsv"),
    TsvSource("TaCo", ACCOUNTING_DIR / "taco_fixed8x10_accounting_latest.tsv", "fixed8x10"),
    TsvSource("SuCo", ACCOUNTING_DIR / "suco_accounting_latest.tsv"),
    TsvSource("alpha-CNG", ACCOUNTING_DIR / "alphacng_accounting_latest.tsv"),
    TsvSource("SHG", ACCOUNTING_DIR / "shg_accounting_latest.tsv"),
    TsvSource("IVF-RaBitQ", ACCOUNTING_DIR / "ivfrabitq_accounting_latest.tsv"),
)


FIELDNAMES = [
    "method",
    "variant",
    "dataset",
    "status",
    "loaded_index_bytes",
    "loaded_vector_excluded_index_bytes",
    "raw_payload_bytes",
    "resident_after_kb",
    "resident_peak_kb",
    "index_path",
    "source_kind",
    "source_path",
    "semantic",
    "validation_status",
    "validation_note",
    "note",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_status(row: dict[str, str]) -> str:
    status = row.get("status", "").strip()
    return status if status else "ok"


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def normalize_tsv_source(source: TsvSource) -> list[dict[str, str]]:
    if not source.path.exists():
        raise FileNotFoundError(source.path)
    rows: list[dict[str, str]] = []
    with source.path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "method": source.method,
                    "variant": source.variant,
                    "dataset": row.get("dataset", "").strip(),
                    "status": normalize_status(row),
                    "loaded_index_bytes": row.get("loaded_index_bytes", "").strip(),
                    "loaded_vector_excluded_index_bytes": row.get(
                        "loaded_vector_excluded_index_bytes", ""
                    ).strip(),
                    "raw_payload_bytes": "",
                    "resident_after_kb": row.get("load_after_vmrss_kb", "").strip(),
                    "resident_peak_kb": row.get("load_after_vmhwm_kb", "").strip(),
                    "index_path": first_present(row, ("index_path", "index_dir")),
                    "source_kind": "loader_accounting_tsv",
                    "source_path": rel(source.path),
                    "semantic": "raw_inclusive_loaded_index",
                    "validation_status": "",
                    "validation_note": "",
                    "note": row.get("note", "").strip(),
                }
            )
    return rows


def mb_to_bytes(value: int) -> int:
    """Convert native DET-LSH decimal MB to bytes.

    DET-LSH prints ``getCurrentRSS() / 1000000`` as MB, so the inverse is a
    decimal MB conversion rather than MiB.
    """

    return value * 1_000_000


def bytes_to_kb_floor(value: int) -> int:
    return value // 1024


def json_int(item: Mapping[str, object], key: str) -> int:
    value = item[key]
    if not isinstance(value, (int, str)):
        raise TypeError(f"expected integer-like JSON field {key}, got {type(value).__name__}")
    return int(value)


def normalize_detlsh(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows: list[dict[str, str]] = []
    with summary_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = cast(Mapping[str, object], json.loads(line))
            after_bytes = mb_to_bytes(json_int(item, "memory_after_indexing_mb"))
            structure_bytes = mb_to_bytes(json_int(item, "native_index_footprint_mb"))
            rows.append(
                {
                    "method": "DET-LSH",
                    "variant": "build_probe_20260520",
                    "dataset": str(item["dataset"]),
                    "status": str(item.get("status", "OK")).lower(),
                    "loaded_index_bytes": str(after_bytes),
                    "loaded_vector_excluded_index_bytes": str(structure_bytes),
                    "raw_payload_bytes": "",
                    "resident_after_kb": str(bytes_to_kb_floor(after_bytes)),
                    "resident_peak_kb": str(json_int(item, "peak_build_rss_kb")),
                    "index_path": str(item.get("index_path", "")),
                    "source_kind": "native_build_probe_jsonl",
                    "source_path": rel(summary_path),
                    "semantic": "raw_inclusive_indexing_complete_before_query",
                    "validation_status": "",
                    "validation_note": "",
                    "note": str(item.get("note", "")),
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(detlsh_summary: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in TSV_SOURCES:
        rows.extend(normalize_tsv_source(source))
    rows.extend(normalize_detlsh(detlsh_summary))
    annotate_validation(rows)
    return rows


def parse_int_or_none(text: str) -> int | None:
    if not text:
        return None
    return int(text)


def median_int(values: list[int]) -> int:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def append_note(existing: str, note: str) -> str:
    return note if not existing else f"{existing}; {note}"


def annotate_validation(rows: list[dict[str, str]]) -> None:
    """Attach cross-method validation notes without mutating core metrics.

    The raw payload implied by a row is ``loaded_index_bytes -
    loaded_vector_excluded_index_bytes``.  Within a dataset this should be close
    across methods; large deviations usually mean the row covered a different
    cardinality/dimensionality or used a non-equivalent accounting path.
    """

    dataset_payloads: dict[str, list[int]] = {}
    for row in rows:
        loaded = parse_int_or_none(row["loaded_index_bytes"])
        structure = parse_int_or_none(row["loaded_vector_excluded_index_bytes"])
        if loaded is None or structure is None:
            if row["status"] not in ("ok", "OK"):
                row["validation_status"] = "not_available"
                continue
            row["validation_status"] = "missing"
            row["validation_note"] = append_note(
                row["validation_note"], "missing unified footprint field"
            )
            continue
        payload = loaded - structure
        row["raw_payload_bytes"] = str(payload)
        if payload < 0:
            row["validation_status"] = "warn"
            row["validation_note"] = append_note(
                row["validation_note"], "loaded_index_bytes < loaded_vector_excluded_index_bytes"
            )
        else:
            dataset_payloads.setdefault(row["dataset"], []).append(payload)

        resident_after = parse_int_or_none(row["resident_after_kb"])
        resident_peak = parse_int_or_none(row["resident_peak_kb"])
        if resident_after is not None and resident_peak is not None and resident_peak < resident_after:
            delta_kb = resident_after - resident_peak
            if delta_kb <= RSS_INVERSION_TOLERANCE_KB:
                row["validation_note"] = append_note(
                    row["validation_note"],
                    f"rss_cross_interface_inversion_tolerated_kb={delta_kb}",
                )
            else:
                row["validation_status"] = "warn"
                row["validation_note"] = append_note(
                    row["validation_note"],
                    f"resident_peak_kb < resident_after_kb by {delta_kb} KB",
                )

    consensus_by_dataset = {
        dataset: median_int(payloads)
        for dataset, payloads in dataset_payloads.items()
        if len(payloads) >= 3
    }
    for row in rows:
        payload = parse_int_or_none(row["raw_payload_bytes"])
        consensus = consensus_by_dataset.get(row["dataset"])
        if payload is None or consensus is None or consensus <= 0:
            continue
        rel_error = abs(payload - consensus) / consensus
        if rel_error > 0.05:
            row["validation_status"] = "warn"
            row["validation_note"] = append_note(
                row["validation_note"],
                f"raw_payload_bytes deviates from dataset consensus by {rel_error:.1%}",
            )

    for row in rows:
        if not row["validation_status"]:
            row["validation_status"] = "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--output",
        default=str(ACCOUNTING_DIR / "unified_accounting_latest.tsv"),
        help="Unified normalized TSV output path.",
    )
    _ = parser.add_argument(
        "--detlsh-output",
        default=str(ACCOUNTING_DIR / "detlsh_accounting_latest.tsv"),
        help="Normalized DET-LSH-only TSV output path.",
    )
    _ = parser.add_argument(
        "--detlsh-summary",
        default=str(DETLSH_SUMMARY),
        help="DET-LSH build-probe JSONL summary input.",
    )
    args = parser.parse_args()

    output = cast(str, args.output)
    detlsh_output = cast(str, args.detlsh_output)
    detlsh_summary = Path(cast(str, args.detlsh_summary))
    rows = build_rows(detlsh_summary)
    detlsh_rows = [row for row in rows if row["method"] == "DET-LSH"]
    write_tsv(Path(output), rows)
    write_tsv(Path(detlsh_output), detlsh_rows)
    print(f"wrote {output} rows={len(rows)}")
    print(f"wrote {detlsh_output} rows={len(detlsh_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
