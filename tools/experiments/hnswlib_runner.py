#!/usr/bin/env python3
"""Minimal local hnswlib runner for fvecs/ivecs datasets."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import hnswlib
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THREADS = 48
DEFAULT_TOPK = 10
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 50
DEFAULT_BATCH_SIZE = 100_000


def ivecs_mmap(path: pathlib.Path) -> np.ndarray:
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:]


def fvecs_mmap(path: pathlib.Path) -> np.ndarray:
    return ivecs_mmap(path).view(np.float32)


def ensure_exists(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def overlap_recall(labels: np.ndarray, gt: np.ndarray, topk: int) -> float:
    gt_topk = np.asarray(gt[:, :topk])
    hits = [
        len(set(np.asarray(pred_row).tolist()) & set(np.asarray(gt_row).tolist()))
        / topk
        for pred_row, gt_row in zip(labels, gt_topk)
    ]
    return float(np.mean(hits)) if hits else math.nan


def build(args: argparse.Namespace) -> None:
    base_path = resolve_path(args.base)
    out_dir = resolve_path(args.output_dir)
    ensure_exists(base_path, "base vectors")
    out_dir.mkdir(parents=True, exist_ok=True)

    xb = fvecs_mmap(base_path)
    if args.max_base is not None:
        xb = xb[: args.max_base]
    count, dim = xb.shape

    index = hnswlib.Index(space=args.space, dim=dim)
    index.init_index(
        max_elements=count,
        M=args.M,
        ef_construction=args.ef_construction,
        random_seed=args.random_seed,
    )
    index.set_num_threads(args.threads)

    ids = np.arange(count)
    t0 = time.perf_counter()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        index.add_items(
            np.asarray(xb[start:end]), ids[start:end], num_threads=args.threads
        )
    build_seconds = time.perf_counter() - t0

    index_path = out_dir / "index.bin"
    meta_path = out_dir / "meta.json"
    index.save_index(str(index_path))
    meta = {
        "base_path": str(base_path),
        "index_path": str(index_path),
        "space": args.space,
        "dim": int(dim),
        "count": int(count),
        "M": args.M,
        "ef_construction": args.ef_construction,
        "default_ef_search": DEFAULT_EF_SEARCH,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "build_seconds": build_seconds,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def search(args: argparse.Namespace) -> None:
    index_dir = resolve_path(args.index_dir)
    query_path = resolve_path(args.query)
    ensure_exists(index_dir / "index.bin", "hnsw index")
    ensure_exists(index_dir / "meta.json", "hnsw metadata")
    ensure_exists(query_path, "query vectors")

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    topk = args.topk
    ef_search = (
        args.ef_search if args.ef_search is not None else meta["default_ef_search"]
    )

    index = hnswlib.Index(space=meta["space"], dim=meta["dim"])
    index.load_index(str(index_dir / "index.bin"), max_elements=meta["count"])
    index.set_num_threads(args.threads)
    index.set_ef(ef_search)

    xq = fvecs_mmap(query_path)
    if args.max_query is not None:
        xq = xq[: args.max_query]

    # HNSW only scales across a batch of queries. For 1-thread runs we keep
    # per-query timing so p50 is truthful; for multi-thread runs we switch to
    # batch search so QPS is truthful and leave p50 as NaN.
    if args.threads == 1:
        per_query_ms = []
        labels = np.empty((xq.shape[0], topk), dtype=np.uint64)
        distances = np.empty((xq.shape[0], topk), dtype=np.float32)
        t0 = time.perf_counter()
        for qi in range(xq.shape[0]):
            qs = time.perf_counter()
            query_labels, query_distances = index.knn_query(
                np.asarray(xq[qi : qi + 1]), k=topk, num_threads=1
            )
            qe = time.perf_counter()
            labels[qi] = query_labels[0]
            distances[qi] = query_distances[0]
            per_query_ms.append((qe - qs) * 1000.0)
        elapsed = time.perf_counter() - t0
        p50_ms = float(np.percentile(per_query_ms, 50)) if per_query_ms else math.nan
        search_mode = "serial_per_query"
    else:
        t0 = time.perf_counter()
        labels, distances = index.knn_query(
            np.asarray(xq), k=topk, num_threads=args.threads
        )
        elapsed = time.perf_counter() - t0
        p50_ms = math.nan
        search_mode = "batch_parallel"

    result = {
        "index_dir": str(index_dir),
        "query_path": str(query_path),
        "queries": int(xq.shape[0]),
        "topk": topk,
        "threads": args.threads,
        "ef_search": ef_search,
        "search_mode": search_mode,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": (elapsed / xq.shape[0]) * 1000.0,
        "p50_ms": p50_ms,
        "qps": xq.shape[0] / elapsed if elapsed > 0 else math.inf,
    }

    if args.gt:
        gt_path = resolve_path(args.gt)
        ensure_exists(gt_path, "ground truth")
        gt = ivecs_mmap(gt_path)
        if args.max_query is not None:
            gt = gt[: args.max_query]
        result["recall_at_k"] = overlap_recall(labels, gt, topk)
        result["gt_path"] = str(gt_path)

    if args.save_results:
        results_path = resolve_path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(results_path, labels=labels, distances=distances)
        result["results_path"] = str(results_path)

    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local hnswlib build/search runner for repo fvecs/ivecs datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build an HNSW index from .fvecs base vectors"
    )
    build_parser.add_argument("--base", required=True, help="Path to base .fvecs file")
    build_parser.add_argument(
        "--output-dir", required=True, help="Output directory for index.bin/meta.json"
    )
    build_parser.add_argument("--space", choices=["l2", "ip", "cosine"], default="l2")
    build_parser.add_argument("--M", type=int, default=DEFAULT_M)
    build_parser.add_argument(
        "--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION
    )
    build_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    build_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_parser.add_argument("--random-seed", type=int, default=100)
    build_parser.add_argument("--max-base", type=int)
    build_parser.set_defaults(func=build)

    search_parser = subparsers.add_parser(
        "search", help="Search an existing HNSW index with .fvecs queries"
    )
    search_parser.add_argument(
        "--index-dir", required=True, help="Directory containing index.bin/meta.json"
    )
    search_parser.add_argument(
        "--query", required=True, help="Path to query .fvecs file"
    )
    search_parser.add_argument(
        "--gt", help="Optional ground-truth .ivecs for recall computation"
    )
    search_parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    search_parser.add_argument("--ef-search", type=int)
    search_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    search_parser.add_argument("--max-query", type=int)
    search_parser.add_argument(
        "--save-results", help="Optional .npz path for labels/distances"
    )
    search_parser.set_defaults(func=search)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
