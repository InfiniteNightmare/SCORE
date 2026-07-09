#!/usr/bin/env python3
"""FAISS IMI-OPQ baseline runner for repo fvecs/ivecs datasets."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import faiss
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TOPK = 10
DEFAULT_THREADS = 1
DEFAULT_BATCH_SIZE = 100_000
DEFAULT_FACTORY = "OPQ16,IMI2x12,PQ16"


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_exists(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ivecs_mmap(path: pathlib.Path) -> np.ndarray:
    raw = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:]


def fvecs_mmap(path: pathlib.Path) -> np.ndarray:
    return ivecs_mmap(path).view(np.float32)


def as_float32_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def overlap_recall(labels: np.ndarray, gt: np.ndarray, topk: int) -> float:
    gt_topk = np.asarray(gt[:, :topk])
    hits = [
        len(set(np.asarray(pred_row).tolist()) & set(np.asarray(gt_row).tolist())) / topk
        for pred_row, gt_row in zip(labels, gt_topk)
    ]
    return float(np.mean(hits)) if hits else math.nan


def tree_size_bytes(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def set_faiss_threads(threads: int) -> None:
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(threads)


def set_index_parameter(index: faiss.Index, name: str, value: int) -> bool:
    try:
        faiss.ParameterSpace().set_index_parameter(index, name, value)
        return True
    except Exception:
        return False


def build(args: argparse.Namespace) -> None:
    base_path = resolve_path(args.base)
    index_path = resolve_path(args.index_path)
    ensure_exists(base_path, "base fvecs")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    xb = fvecs_mmap(base_path)
    if args.max_base is not None:
        xb = xb[: args.max_base]
    count, dim = xb.shape
    if args.data_dim is not None and dim != args.data_dim:
        raise ValueError(f"base dim {dim} does not match --data-dim {args.data_dim}")

    train_count = min(count, args.train_size or count)
    if train_count <= 0:
        raise ValueError("training set must be non-empty")

    set_faiss_threads(args.threads)
    index = faiss.index_factory(dim, args.factory, faiss.METRIC_L2)

    t0 = time.perf_counter()
    train = as_float32_contiguous(xb[:train_count])
    index.train(train)
    train_seconds = time.perf_counter() - t0

    add_t0 = time.perf_counter()
    for start in range(0, count, args.add_batch_size):
        end = min(start + args.add_batch_size, count)
        index.add(as_float32_contiguous(xb[start:end]))
    add_seconds = time.perf_counter() - add_t0


    faiss.write_index(index, str(index_path))
    result = {
        "mode": "build",
        "method": "imi-opq",
        "base_path": str(base_path),
        "index_path": str(index_path),
        "factory": args.factory,
        "dataset_size": int(count),
        "data_dim": int(dim),
        "train_size": int(train_count),
        "threads": args.threads,
        "train_seconds": train_seconds,
        "add_seconds": add_seconds,
        "build_seconds": train_seconds + add_seconds,
        "index_size_bytes": index_path.stat().st_size,
    }
    print(json.dumps(result, indent=2))


def search(args: argparse.Namespace) -> None:
    index_path = resolve_path(args.index_path)
    query_path = resolve_path(args.query)
    gt_path = resolve_path(args.gt) if args.gt else None
    ensure_exists(index_path, "FAISS index")
    ensure_exists(query_path, "query fvecs")
    if gt_path is not None:
        ensure_exists(gt_path, "ground truth ivecs")

    set_faiss_threads(args.threads)
    index = faiss.read_index(str(index_path))
    applied_params: dict[str, int] = {}
    skipped_params: dict[str, int] = {}
    for name, value in [
        ("nprobe", args.nprobe),
        ("max_codes", args.max_codes),
        ("polysemous_ht", args.polysemous_ht),
    ]:
        if value is None:
            continue
        if set_index_parameter(index, name, value):
            applied_params[name] = value
        else:
            skipped_params[name] = value

    xq = fvecs_mmap(query_path)
    if args.max_query is not None:
        xq = xq[: args.max_query]

    labels = np.empty((xq.shape[0], args.topk), dtype=np.int64)
    distances = np.empty((xq.shape[0], args.topk), dtype=np.float32)
    per_query_ms: list[float] = []
    t0 = time.perf_counter()
    if args.batch_query:
        distances, labels = index.search(as_float32_contiguous(xq), args.topk)
        elapsed = time.perf_counter() - t0
        p50_ms = math.nan
        search_mode = "batch"
    else:
        for qi in range(xq.shape[0]):
            qs = time.perf_counter()
            query_distances, query_labels = index.search(
                as_float32_contiguous(xq[qi : qi + 1]), args.topk
            )
            qe = time.perf_counter()
            distances[qi] = query_distances[0]
            labels[qi] = query_labels[0]
            per_query_ms.append((qe - qs) * 1000.0)
        elapsed = time.perf_counter() - t0
        p50_ms = float(np.percentile(per_query_ms, 50)) if per_query_ms else math.nan
        search_mode = "serial_per_query"

    result: dict[str, object] = {
        "mode": "search",
        "method": "imi-opq",
        "index_path": str(index_path),
        "query_path": str(query_path),
        "queries": int(xq.shape[0]),
        "topk": args.topk,
        "threads": args.threads,
        "search_mode": search_mode,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": (elapsed / xq.shape[0]) * 1000.0 if xq.shape[0] else math.nan,
        "p50_ms": p50_ms,
        "qps": xq.shape[0] / elapsed if elapsed > 0 else math.inf,
        "applied_params": applied_params,
        "skipped_params": skipped_params,
        "index_size_bytes": tree_size_bytes(index_path),
    }
    if gt_path is not None:
        gt = ivecs_mmap(gt_path)
        if args.max_query is not None:
            gt = gt[: args.max_query]
        result["recall_at_k"] = overlap_recall(labels, gt, args.topk)
        if args.topk == 10:
            result["recall_at_10"] = result["recall_at_k"]
        result["gt_path"] = str(gt_path)
    if args.save_results:
        results_path = resolve_path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(results_path, labels=labels, distances=distances)
        result["results_path"] = str(results_path)

    print(json.dumps(result, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAISS IMI-OPQ build/search runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build a FAISS IMI-OPQ index")
    build_p.add_argument("--base", required=True)
    build_p.add_argument("--index-path", required=True)
    build_p.add_argument("--factory", default=DEFAULT_FACTORY)
    build_p.add_argument("--data-dim", type=int, default=None)
    build_p.add_argument("--train-size", type=int, default=None)
    build_p.add_argument("--max-base", type=int, default=None)
    build_p.add_argument("--add-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    build_p.set_defaults(func=build)

    search_p = subparsers.add_parser("search", help="Search a FAISS IMI-OPQ index")
    search_p.add_argument("--index-path", required=True)
    search_p.add_argument("--query", required=True)
    search_p.add_argument("--gt", default=None)
    search_p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    search_p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    search_p.add_argument("--nprobe", type=int, default=None)
    search_p.add_argument("--max-codes", type=int, default=None)
    search_p.add_argument("--polysemous-ht", type=int, default=None)
    search_p.add_argument("--max-query", type=int, default=None)
    search_p.add_argument("--batch-query", action="store_true")
    search_p.add_argument("--save-results", default=None)
    search_p.set_defaults(func=search)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
