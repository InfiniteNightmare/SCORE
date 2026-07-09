#!/usr/bin/env python3
"""Alpha-CNG build/search runner with loader-side accounting support."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
ALPHACNG_ROOT = (
    ROOT
    / "baselines"
    / "Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search"
)
BUILD_BIN = ALPHACNG_ROOT / "build" / "tests" / "test_alphacng_index"
SEARCH_BIN = ALPHACNG_ROOT / "build" / "tests" / "test_alphacng_search"
DEFAULT_TOPK = 10


def parse_accounting(output: str, *, required: bool = False) -> dict[str, int]:
    patterns = {
        "loaded_index_bytes": r"ALPHACNG_ACCOUNTING total_loaded_index_bytes=(\d+)",
        "loaded_vector_excluded_index_bytes": r"ALPHACNG_ACCOUNTING total_vector_excluded_index_bytes=(\d+)",
        "raw_vector_bytes": r"ALPHACNG_ACCOUNTING component=raw_vectors bytes=(\d+)",
        "base_vector_allocation_bytes": r"ALPHACNG_ACCOUNTING component=base_vector_allocation bytes=(\d+)",
        "graph_resident_bytes": r"ALPHACNG_ACCOUNTING component=graph_resident bytes=(\d+)",
        "graph_neighbor_bytes": r"ALPHACNG_ACCOUNTING component=graph_neighbors bytes=(\d+)",
        "graph_container_bytes": r"ALPHACNG_ACCOUNTING component=graph_containers bytes=(\d+)",
        "graph_edge_count": r"ALPHACNG_ACCOUNTING graph_edge_count=(\d+)",
        "load_after_vmrss_kb": r"ALPHACNG_ACCOUNTING load_after_vmrss_kb=(-?\d+)",
        "load_after_vmhwm_kb": r"ALPHACNG_ACCOUNTING load_after_vmhwm_kb=(-?\d+)",
    }
    metrics: dict[str, int] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[key] = int(matches[-1])
        elif required:
            raise RuntimeError(f"missing Alpha-CNG accounting metric: {key}")
    return metrics


def parse_optional_float(pattern: str, output: str) -> float | None:
    matches = re.findall(pattern, output)
    return float(matches[-1]) if matches else None


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_exists(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def fvecs_row_count(path: pathlib.Path) -> int:
    with path.open("rb") as handle:
        raw_dim = handle.read(4)
    if len(raw_dim) != 4:
        raise ValueError(f"empty or truncated fvecs file: {path}")
    dim = int.from_bytes(raw_dim, byteorder="little", signed=True)
    if dim <= 0:
        raise ValueError(f"invalid fvecs dimension {dim}: {path}")
    row_bytes = (dim + 1) * 4
    size = path.stat().st_size
    if size % row_bytes != 0:
        raise ValueError(f"fvecs size is not divisible by row size: {path}")
    return size // row_bytes


def run_build(args: argparse.Namespace) -> None:
    ensure_exists(BUILD_BIN, "Alpha-CNG build binary")
    base_path = resolve_path(args.base)
    knn_graph_path = resolve_path(args.knn_graph)
    index_path = resolve_path(args.index_path)
    ensure_exists(base_path, "base vectors")
    ensure_exists(knn_graph_path, "KNN graph")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(BUILD_BIN),
        str(base_path),
        str(knn_graph_path),
        str(args.L),
        str(args.R),
        str(args.C),
        str(args.alpha_start),
        str(args.alpha_step),
        str(args.alpha_max),
        str(args.tau),
        str(args.T),
        str(index_path),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    print(f"Running Alpha-CNG build: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"Alpha-CNG build failed (exit {proc.returncode})")
    result = {
        "base_path": str(base_path),
        "knn_graph_path": str(knn_graph_path),
        "index_path": str(index_path),
        "threads": args.threads,
        "wall_seconds": round(elapsed, 2),
    }
    print(json.dumps(result, indent=2))


def run_search(args: argparse.Namespace) -> None:
    ensure_exists(SEARCH_BIN, "Alpha-CNG search binary")
    base_path = resolve_path(args.base)
    index_path = resolve_path(args.index_path)
    ensure_exists(base_path, "base vectors")
    ensure_exists(index_path, "Alpha-CNG index")

    accounting_only = bool(args.accounting_only)
    query_path: pathlib.Path | None = None
    gt_path: pathlib.Path | None = None

    if accounting_only:
        cmd = [str(SEARCH_BIN), str(base_path), str(index_path), "--accounting-only"]
    else:
        if args.query is None or args.gt is None:
            raise ValueError("--query and --gt are required unless --accounting-only is set")
        query_path = resolve_path(args.query)
        gt_path = resolve_path(args.gt)
        ensure_exists(query_path, "query vectors")
        ensure_exists(gt_path, "ground truth")
        cmd = [
            str(SEARCH_BIN),
            str(base_path),
            str(query_path),
            str(gt_path),
            str(index_path),
            args.search_L,
            str(args.topk),
        ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    print(f"Running Alpha-CNG search: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"Alpha-CNG search failed (exit {proc.returncode})")

    result = {
        "mode": "accounting_only" if args.accounting_only else "search",
        "base_path": str(base_path),
        "index_path": str(index_path),
        "topk": args.topk,
        "threads": args.threads,
        "wall_seconds": round(elapsed, 2),
    }
    if not accounting_only:
        if query_path is None or gt_path is None:
            raise RuntimeError("internal error: query/ground-truth paths were not resolved")
        native_elapsed = parse_optional_float(r"Elapsed time:\s*([\d.]+)\s*seconds", output)
        native_qps = parse_optional_float(r"qps=([\d.]+)", output)
        if native_qps is None and native_elapsed is not None and native_elapsed > 0:
            native_qps = fvecs_row_count(query_path) / native_elapsed
        result.update(
            {
                "query_path": str(query_path),
                "gt_path": str(gt_path),
                "search_L": args.search_L,
                "qps": native_qps,
                "p50_ms": parse_optional_float(r"p50_ms=([\d.]+)", output),
                "recall_at_k": parse_optional_float(r"recall ([\d.]+)", output),
            }
        )
    result.update(parse_accounting(output, required=accounting_only))
    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha-CNG build/search runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build Alpha-CNG graph index")
    build_p.add_argument("--base", required=True)
    build_p.add_argument("--knn-graph", required=True)
    build_p.add_argument("--index-path", required=True)
    build_p.add_argument("--L", type=int, required=True)
    build_p.add_argument("--R", type=int, required=True)
    build_p.add_argument("--C", type=int, required=True)
    build_p.add_argument("--alpha-start", type=float, default=0.9)
    build_p.add_argument("--alpha-step", type=float, default=0.05)
    build_p.add_argument("--alpha-max", type=float, default=1.6)
    build_p.add_argument("--tau", type=float, required=True)
    build_p.add_argument("--T", type=int, required=True)
    build_p.add_argument("--threads", type=int, default=48)
    build_p.set_defaults(func=run_build)

    search_p = subparsers.add_parser("search", help="Search or account an Alpha-CNG index")
    search_p.add_argument("--base", required=True)
    search_p.add_argument("--index-path", required=True)
    search_p.add_argument("--query")
    search_p.add_argument("--gt")
    search_p.add_argument("--search-L", default="300")
    search_p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    search_p.add_argument("--threads", type=int, default=48)
    search_p.add_argument("--accounting-only", action="store_true")
    search_p.set_defaults(func=run_search)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
