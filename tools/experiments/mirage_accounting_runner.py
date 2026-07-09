#!/usr/bin/env python3
"""MIRAGE/Faiss loader-side accounting runner."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
MIRAGE_BIN = ROOT / "baselines" / "mirage" / "build" / "demos" / "demo_mirage_runner"


PATTERNS = {
    "loaded_index_bytes": r"MIRAGE_ACCOUNTING total_loaded_index_bytes=(\d+)",
    "loaded_vector_excluded_index_bytes": r"MIRAGE_ACCOUNTING total_vector_excluded_index_bytes=(\d+)",
    "raw_vector_bytes": r"MIRAGE_ACCOUNTING component=raw_vectors bytes=(\d+)",
    "hnsw_graph_bytes": r"MIRAGE_ACCOUNTING component=hnsw_graph bytes=(\d+)",
    "hnsw_levels_bytes": r"MIRAGE_ACCOUNTING component=hnsw_levels bytes=(\d+)",
    "hnsw_offsets_bytes": r"MIRAGE_ACCOUNTING component=hnsw_offsets bytes=(\d+)",
    "hnsw_neighbors_bytes": r"MIRAGE_ACCOUNTING component=hnsw_neighbors bytes=(\d+)",
    "hnsw_assign_probas_bytes": r"MIRAGE_ACCOUNTING component=hnsw_assign_probas bytes=(\d+)",
    "hnsw_cum_nneighbor_bytes": r"MIRAGE_ACCOUNTING component=hnsw_cum_nneighbor bytes=(\d+)",
    "load_after_vmrss_kb": r"MIRAGE_ACCOUNTING load_after_vmrss_kb=(-?\d+)",
    "load_after_vmhwm_kb": r"MIRAGE_ACCOUNTING load_after_vmhwm_kb=(-?\d+)",
}


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_metrics(output: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for key, pattern in PATTERNS.items():
        matches = re.findall(pattern, output)
        if not matches:
            raise RuntimeError(f"missing MIRAGE accounting metric: {key}")
        metrics[key] = int(matches[-1])
    return metrics


def run_accounting(args: argparse.Namespace) -> None:
    if not MIRAGE_BIN.exists():
        raise FileNotFoundError(f"MIRAGE runner missing: {MIRAGE_BIN}")
    index_path = resolve_path(args.index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"MIRAGE index missing: {index_path}")

    cmd = [str(MIRAGE_BIN), "search", "--index", str(index_path), "--accounting-only"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"MIRAGE accounting failed (exit {proc.returncode})")

    result = {
        "mode": "accounting_only",
        "index_path": str(index_path),
        "wall_seconds": round(elapsed, 2),
    }
    result.update(parse_metrics(output))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()
    run_accounting(args)


if __name__ == "__main__":
    main()
