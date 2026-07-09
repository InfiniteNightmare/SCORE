#!/usr/bin/env python3
"""SuCo build/search runner using native --build-only / --search-only flags."""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THREADS = 48
DEFAULT_TOPK = 10
SUCO_BIN = ROOT / "baselines" / "SuCo" / "suco"


def suco_actual_index_path(index_path: str, use_srht: bool, dim: int) -> str:
    """SuCo appends _srht<dim>_seed42 to index path when --use-srht is set."""
    if use_srht:
        return f"{index_path}_srht{dim}_seed42"
    return index_path


def run_build(args: argparse.Namespace) -> None:
    base_path = pathlib.Path(args.base).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"base not found: {base_path}")

    cmd = [
        str(SUCO_BIN),
        "--build-only",
        "--dataset-path", str(base_path),
        "--query-path", str(base_path),  # dummy (not loaded in build-only)
        "--groundtruth-path", str(base_path),  # dummy
        "--dataset-size", str(args.dataset_size),
        "--query-size", "1",
        "--k-size", str(args.topk),
        "--data-dimensionality", str(args.data_dim),
        "--subspace-num", str(args.subspace_num),
        "--subspace-dimensionality", str(args.subspace_dim),
        "--kmeans-num-centroid", str(args.kmeans_centroids),
        "--kmeans-num-iters", str(args.kmeans_iters),
        "--collision-ratio", str(args.collision_ratio),
        "--candidate-ratio", str(args.candidate_ratio),
        "--index-path", args.index_path,
    ]
    if args.use_srht:
        cmd.append("--use-srht")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)

    print(f"Running SuCo build: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"SuCo build failed (exit {proc.returncode})")

    # Parse metrics
    build_seconds = 0.0
    m = re.search(r"build_seconds=([\d.]+)", output)
    if m:
        build_seconds = float(m.group(1))

    footprint_mb = 0.0
    m = re.search(r"index_footprint_mb=([\d.]+)", output)
    if m:
        footprint_mb = float(m.group(1))

    actual_index = suco_actual_index_path(args.index_path, args.use_srht, args.data_dim)
    idx_path = pathlib.Path(actual_index)
    index_size_bytes = idx_path.stat().st_size if idx_path.exists() else 0

    result = {
        "base_path": str(base_path),
        "index_path": actual_index,
        "dataset_size": args.dataset_size,
        "data_dim": args.data_dim,
        "subspace_num": args.subspace_num,
        "subspace_dim": args.subspace_dim,
        "use_srht": args.use_srht,
        "threads": args.threads,
        "wall_seconds": round(elapsed, 2),
        "build_seconds": round(build_seconds, 6),
        "index_footprint_mb": footprint_mb,
        "index_size_bytes": index_size_bytes,
    }
    print(json.dumps(result, indent=2))


def run_search(args: argparse.Namespace) -> None:
    query_path = pathlib.Path(args.query).resolve()
    gt_path = pathlib.Path(args.gt).resolve() if args.gt else None

    if not query_path.exists():
        raise FileNotFoundError(f"query not found: {query_path}")

    cmd = [
        str(SUCO_BIN),
        "--search-only",
        "--dataset-path", args.base,
        "--query-path", str(query_path),
        "--groundtruth-path", str(gt_path) if gt_path else str(query_path),
        "--dataset-size", str(args.dataset_size),
        "--query-size", str(args.query_size),
        "--k-size", str(args.topk),
        "--data-dimensionality", str(args.data_dim),
        "--subspace-num", str(args.subspace_num),
        "--subspace-dimensionality", str(args.subspace_dim),
        "--kmeans-num-centroid", str(args.kmeans_centroids),
        "--kmeans-num-iters", str(args.kmeans_iters),
        "--collision-ratio", str(args.collision_ratio),
        "--candidate-ratio", str(args.candidate_ratio),
        "--index-path", args.index_path,
    ]
    if args.use_srht:
        cmd.append("--use-srht")
    if args.threads > 1:
        cmd.append("--parallel-query")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)

    print(f"Running SuCo search ({args.threads}T): {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"SuCo search failed (exit {proc.returncode})")

    # Parse metrics
    qps = 0.0
    m = re.search(r"qps=([\d.]+)", output)
    if m:
        qps = float(m.group(1))

    p50_ms = float("nan")
    m = re.search(r"p50_ms=([\d.]+)", output)
    if m:
        p50_ms = float(m.group(1))

    recall = float("nan")
    reported_ks: list[int] = []
    for m_iter in re.finditer(r"When k = (\d+), \(recall, ratio\) = \(([\d.]+),", output):
        reported_k = int(m_iter.group(1))
        reported_ks.append(reported_k)
        if reported_k == args.topk:
            recall = float(m_iter.group(2))

    if not reported_ks:
        raise RuntimeError("SuCo search output did not report any recall lines")
    if recall != recall:
        raise RuntimeError(
            f"SuCo search output did not report Recall@{args.topk}; "
            f"highest reported k was {max(reported_ks)}"
        )

    actual_index = suco_actual_index_path(args.index_path, args.use_srht, args.data_dim)
    result: dict[str, object] = {
        "index_path": actual_index,
        "query_path": str(query_path),
        "gt_path": str(gt_path) if gt_path else "",
        "queries": args.query_size,
        "topk": args.topk,
        "threads": args.threads,
        "wall_seconds": round(elapsed, 2),
        "qps": qps,
        "p50_ms": p50_ms,
        "recall_k": args.topk,
        "recall_at_k": recall,
    }
    if args.topk == 10:
        result["recall_at_10"] = recall
    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SuCo build/search runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_common_args(p):
        p.add_argument("--base", required=True)
        p.add_argument("--index-path", required=True)
        p.add_argument("--dataset-size", type=int, required=True)
        p.add_argument("--data-dim", type=int, required=True)
        p.add_argument("--subspace-num", type=int, default=8)
        p.add_argument("--subspace-dim", type=int, required=True)
        p.add_argument("--kmeans-centroids", type=int, default=50)
        p.add_argument("--kmeans-iters", type=int, default=2)
        p.add_argument("--collision-ratio", type=float, default=0.05)
        p.add_argument("--candidate-ratio", type=float, default=0.005)
        p.add_argument("--use-srht", action="store_true")
        p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        p.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    build_p = subparsers.add_parser("build", help="Build SuCo index (no query)")
    add_common_args(build_p)
    build_p.set_defaults(func=run_build)

    search_p = subparsers.add_parser("search", help="Search SuCo index")
    add_common_args(search_p)
    search_p.add_argument("--query", required=True)
    search_p.add_argument("--gt", default=None)
    search_p.add_argument("--query-size", type=int, required=True)
    search_p.set_defaults(func=run_search)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
