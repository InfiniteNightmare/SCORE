#!/usr/bin/env python3
"""Runner for the official DET-LSH baseline binary.

The upstream DET-LSH program performs indexing and querying in one process; it
does not expose separate build-only/search-only modes.  This wrapper keeps that
behavior explicit and emits the JSON metrics expected by the local benchmark
parsers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import time
from typing import cast


ROOT = pathlib.Path(__file__).resolve().parents[2]
DETLSH_BIN = ROOT / "baselines" / "DET-LSH" / "bin" / "DETLSH"
DEFAULT_TOPK = 10
DEFAULT_THREADS = 1
DEFAULT_LSH_DIM = 16
DEFAULT_L_SIZE = 4
DEFAULT_SAMPLE_TYPE = 2


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_exists(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_size(path: pathlib.Path, expected_bytes: int, label: str) -> None:
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"{label} has {actual} bytes, expected {expected_bytes}. "
            "DET-LSH expects raw row-major binary inputs, not fvecs/ivecs headers."
        )


def cpu_type_for_threads(threads: int) -> int:
    """Map a simple thread count to DET-LSH's CPU-affinity code."""

    mapping = {
        1: 1,
        2: 21,
        4: 41,
        6: 61,
        8: 81,
        10: 101,
        12: 121,
        24: 242,
    }
    if threads not in mapping:
        supported = ", ".join(str(value) for value in sorted(mapping))
        raise ValueError(
            f"unsupported DET-LSH thread count {threads}; supported counts: {supported}. "
            "Use --cpu-type to pass an upstream CPU-affinity code explicitly."
        )
    return mapping[threads]


def effective_sample_size(sample_size: int | None, dataset_size: int, sample_ratio: float) -> int:
    if sample_size is not None:
        return sample_size
    return max(1, int(dataset_size * sample_ratio))


def effective_max_candidate_size(
    max_candidate_size: int | None,
    topk: int,
    dataset_size: int,
    candidate_ratio: float,
) -> int:
    if max_candidate_size is not None:
        return max_candidate_size
    return max(
        topk,
        int(dataset_size * candidate_ratio) + topk,
    )


def parse_required_float(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing DET-LSH metric: {name}")
    return float(match.group(1))


def parse_native_metrics(output: str, query_size: int, topk: int) -> dict[str, object]:
    index_ms = parse_required_float(
        r"The total time of encoding and indexing phase is:\s*([0-9.]+)ms\.",
        output,
        "index_ms",
    )
    avg_query_ms = parse_required_float(
        r"The average query time is\s*([0-9.]+)ms\.",
        output,
        "avg_query_ms",
    )
    recall = parse_required_float(
        r"The average recall value is:\s*([0-9.]+)",
        output,
        "recall",
    )
    ratio = parse_required_float(
        r"The overall ratio is:\s*([0-9.]+)",
        output,
        "overall_ratio",
    )
    footprint_mb = parse_required_float(
        r"The indexing footprint is:\s*(-?[0-9.]+)MB",
        output,
        "index_footprint_mb",
    )
    qps = 1000.0 / avg_query_ms if avg_query_ms > 0 else math.inf
    return {
        "build_seconds": index_ms / 1000.0,
        "index_ms": index_ms,
        "avg_latency_ms": avg_query_ms,
        "qps": qps,
        "recall_at_k": recall,
        "recall_k": topk,
        "overall_ratio": ratio,
        "index_footprint_mb": footprint_mb,
        "queries": float(query_size),
        "latency_note": "DET-LSH reports mean query latency only; no p50 is available.",
    }


def make_native_command(args: argparse.Namespace, *, validate_inputs: bool) -> list[str]:
    base_path = resolve_path(args.base)
    query_path = resolve_path(args.query)
    gt_path = resolve_path(args.gt)
    index_path = resolve_path(args.index_path)

    ensure_exists(base_path, "DET-LSH raw-f32 base vectors")
    ensure_exists(query_path, "DET-LSH raw-f32 query vectors")
    ensure_exists(gt_path, "DET-LSH int64 ground truth")

    if validate_inputs:
        validate_size(base_path, args.dataset_size * args.data_dim * 4, "base vectors")
        validate_size(query_path, args.query_size * args.data_dim * 4, "query vectors")
        validate_size(gt_path, args.query_size * args.topk * 8, "ground truth")

    if args.cpu_type is None:
        cpu_type = cpu_type_for_threads(args.threads)
    else:
        cpu_type = args.cpu_type

    sample_size = effective_sample_size(
        cast(int | None, args.sample_size),
        cast(int, args.dataset_size),
        cast(float, args.sample_ratio),
    )
    max_candidate_size = effective_max_candidate_size(
        cast(int | None, args.max_candidate_size),
        cast(int, args.topk),
        cast(int, args.dataset_size),
        cast(float, args.candidate_ratio),
    )

    return [
        str(DETLSH_BIN),
        "--dataset", str(base_path),
        "--queries", str(query_path),
        "--groundtruth", str(gt_path),
        "--index-path", str(index_path),
        # Upstream subtracts 100 from --dataset-size before loading data because
        # its paper workflow removes 100 queries from the original dataset.
        # Our repo stores base/query separately, so pass n+100 by default.
        "--dataset-size", str(args.dataset_size + args.native_dataset_size_offset),
        "--queries-size", str(args.query_size),
        "--data-dimensionality", str(args.data_dim),
        "--k-size", str(args.topk),
        "--LSH-dimensionality", str(args.lsh_dim),
        "--l-size", str(args.l_size),
        "--search-radius", str(args.search_radius),
        "--leaf-size", str(args.leaf_size),
        "--sample-size", str(sample_size),
        "--sample-type", str(args.sample_type),
        "--max-candidate-size", str(max_candidate_size),
        "--cpu-type", str(cpu_type),
        "--queue-number", str(args.queue_number),
    ]


def run_search(args: argparse.Namespace) -> None:
    if args.input_format != "raw-f32":
        raise ValueError("DET-LSH upstream binary expects raw float32 vectors without fvecs headers")
    if not DETLSH_BIN.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"DET-LSH binary not found: {DETLSH_BIN}. Build upstream first with "
            "`cd baselines/DET-LSH && ./configure && make`."
        )

    cmd = make_native_command(args, validate_inputs=not args.dry_run)
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "command": cmd}, indent=2))
        return

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    resolved_index_path = resolve_path(args.index_path)
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"DET-LSH failed (exit {proc.returncode})")

    metrics = parse_native_metrics(output, args.query_size, args.topk)
    result: dict[str, object] = {
        "mode": "combined_native_run",
        "method": "det-lsh",
        "base_path": str(resolve_path(args.base)),
        "query_path": str(resolve_path(args.query)),
        "gt_path": str(resolve_path(args.gt)),
        "index_path": str(resolved_index_path),
        "dataset_size": args.dataset_size,
        "native_dataset_size_arg": args.dataset_size + args.native_dataset_size_offset,
        "native_dataset_size_offset": args.native_dataset_size_offset,
        "data_dim": args.data_dim,
        "topk": args.topk,
        "threads": args.threads,
        "lsh_dim": args.lsh_dim,
        "l_size": args.l_size,
        "candidate_ratio": args.candidate_ratio,
        "search_radius": args.search_radius,
        "leaf_size": args.leaf_size,
        "sample_ratio": args.sample_ratio,
        "sample_size": effective_sample_size(
            cast(int | None, args.sample_size),
            cast(int, args.dataset_size),
            cast(float, args.sample_ratio),
        ),
        "sample_type": args.sample_type,
        "max_candidate_size": effective_max_candidate_size(
            cast(int | None, args.max_candidate_size),
            cast(int, args.topk),
            cast(int, args.dataset_size),
            cast(float, args.candidate_ratio),
        ),
        "wall_seconds": round(elapsed, 6),
        "semantics_note": (
            "DET-LSH executable builds and queries in one process. Local native "
            "code may include memory-lifetime fixes; see DET-LSH source diff for provenance."
        ),
    }
    result.update(metrics)
    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official DET-LSH binary and emit local benchmark JSON"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_search_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base", required=True, help="Raw float32 base vectors, no fvecs headers")
        p.add_argument("--query", required=True, help="Raw float32 query vectors, no fvecs headers")
        p.add_argument("--gt", required=True, help="int64 ground-truth ids, row-major")
        p.add_argument("--index-path", required=True, help="DET-LSH temporary index directory")
        p.add_argument("--dataset-size", type=int, required=True)
        p.add_argument("--native-dataset-size-offset", type=int, default=100)
        p.add_argument("--query-size", type=int, required=True)
        p.add_argument("--data-dim", type=int, required=True)
        p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
        p.add_argument("--cpu-type", type=int, default=None)
        p.add_argument("--queue-number", type=int, default=1)
        p.add_argument("--lsh-dim", type=int, default=DEFAULT_LSH_DIM)
        p.add_argument("--l-size", type=int, default=DEFAULT_L_SIZE)
        p.add_argument("--search-radius", type=float, required=True)
        p.add_argument("--candidate-ratio", type=float, required=True)
        p.add_argument("--max-candidate-size", type=int, default=None)
        p.add_argument("--leaf-size", type=int, default=1000)
        p.add_argument("--sample-size", type=int, default=None)
        p.add_argument("--sample-ratio", type=float, default=0.1)
        p.add_argument("--sample-type", type=int, default=DEFAULT_SAMPLE_TYPE)
        p.add_argument("--input-format", choices=["raw-f32"], default="raw-f32")
        p.add_argument("--dry-run", action="store_true")

    search_p = subparsers.add_parser(
        "search", help="Run DET-LSH's combined index+query executable"
    )
    add_search_args(search_p)
    search_p.set_defaults(func=run_search)

    run_p = subparsers.add_parser("run", help="Alias for search")
    add_search_args(run_p)
    run_p.set_defaults(func=run_search)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
