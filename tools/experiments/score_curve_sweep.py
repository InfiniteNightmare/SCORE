#!/usr/bin/env python3
"""Run SCORE build/search points and emit raw logs for Recall-QPS curves.

The script intentionally stays at the experiment-automation layer: it invokes the
existing SCORE build/search binaries without changing index or query semantics.
Use ``--build-backend stream-reorder`` to build with the optimized streaming
reorder path (``score_stream_build_reorder``), or ``stream-no-reorder`` for the
matching streaming builder without PID reorder. Both consume ``.fbin`` base
vectors and expose streaming worker controls. Summary search logs are written
as JSON files under
``docs/evaluation/raw_baselines/score/<dataset>/`` so they can be consumed by
``parse_curve_raw_logs.py`` together with baseline raw logs.  The ``--index-dir``
is the phase boundary: if it does not exist the script builds it first and records
build metrics; if it already exists the script treats it as a frozen index and
runs search curves directly.  Use ``--force-rebuild`` for an explicit rebuild.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "docs" / "evaluation" / "raw_baselines" / "score"
RESULTS_TSV = ROOT / "docs" / "evaluation" / "baseline_results.tsv"
DEFAULT_BATCH_BUILD_BIN = "build/score_build"
DEFAULT_STREAM_REORDER_BUILD_BIN = "build/score_stream_build_reorder"
DEFAULT_STREAM_NO_REORDER_BUILD_BIN = "build/score_stream_build_no_reorder"
DEFAULT_IMI_FROM_INDEX_BUILD_BIN = "build/score_imi_from_index"
RESULT_FIELD_ORDER = [
    "timestamp",
    "mode",
    "method",
    "label",
    "dataset",
    "runs",
    "build_seconds",
    "max_rss_kb",
    "index_size_bytes",
    "recall_trimmed",
    "qps_trimmed",
    "lat_p50_trimmed",
    "avg_latency_ms_trimmed",
    "adapter",
    "raw_log",
]
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_REBUILD_ROOT = (ROOT / "results").resolve()


@dataclass(frozen=True)
class DatasetConfig:
    base: str
    query: str
    gt: str


DATASETS: dict[str, DatasetConfig] = {
    "gist1m": DatasetConfig(
        base="dataset/gist/gist_base.fvecs",
        query="dataset/gist/gist_query.fvecs",
        gt="dataset/gist/gist_groundtruth.ivecs",
    ),
    "laion1m": DatasetConfig(
        base="dataset/laion1m/laion1m_base.fvecs",
        query="dataset/laion1m/laion1m_query.fvecs",
        gt="dataset/laion1m/laion1m_groundtruth.ivecs",
    ),
    "cohere1m": DatasetConfig(
        base="dataset/cohere1m/cohere1m_base.fvecs",
        query="dataset/cohere1m/cohere1m_query.fvecs",
        gt="dataset/cohere1m/cohere1m_groundtruth.ivecs",
    ),
    "imagenet1m_2048": DatasetConfig(
        base="dataset/imagenet1m_2048/imagenet1m_2048_base.fvecs",
        query="dataset/imagenet1m_2048/imagenet1m_2048_query.fvecs",
        gt="dataset/imagenet1m_2048/imagenet1m_2048_groundtruth.ivecs",
    ),
    "cohere10m": DatasetConfig(
        base="dataset/cohere10m/cohere10m_base.fvecs",
        query="dataset/cohere10m/cohere10m_query.fvecs",
        gt="dataset/cohere10m/cohere10m_groundtruth.ivecs",
    ),
    "wikipedia": DatasetConfig(
        base="dataset/wikipedia_bge_m3/wikipedia_bge_m3_base.fvecs",
        query="dataset/wikipedia_bge_m3/wikipedia_bge_m3_query.fvecs",
        gt="dataset/wikipedia_bge_m3/wikipedia_bge_m3_groundtruth.ivecs",
    ),
}


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_float_csv(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_int_csv(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated int")
    return values


def validate_label(text: str) -> str:
    if not SAFE_LABEL_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("label must match [A-Za-z0-9_-]+")
    return text


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def default_stream_base(dataset: DatasetConfig) -> Path:
    base = Path(dataset.base)
    if base.suffix != ".fvecs":
        raise ValueError(f"cannot infer .fbin path from dataset base: {dataset.base}")
    return base.with_suffix(".fbin")


def tree_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def ensure_results_header() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "timestamp\tmode\tmethod\tlabel\tdataset\truns\tbuild_seconds\tmax_rss_kb\t"
        "index_size_bytes\trecall_trimmed\tqps_trimmed\tlat_p50_trimmed\t"
        "avg_latency_ms_trimmed\tadapter\traw_log\n",
        encoding="utf-8",
    )


def write_result(row: Mapping[str, object]) -> None:
    ensure_results_header()
    upsert_key = (
        str(row.get("mode", "")),
        str(row.get("method", "")),
        str(row.get("label", "")),
        str(row.get("dataset", "")),
    )
    rows: list[dict[str, str]] = []
    with RESULTS_TSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for existing in reader:
            existing_key = (
                existing.get("mode", ""),
                existing.get("method", ""),
                existing.get("label", ""),
                existing.get("dataset", ""),
            )
            if existing_key != upsert_key:
                rows.append({key: existing.get(key, "") for key in RESULT_FIELD_ORDER})
    rows.append({key: str(row.get(key, "")) for key in RESULT_FIELD_ORDER})
    with RESULTS_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELD_ORDER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_can_write(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact without --overwrite-logs: {path.relative_to(ROOT)}")
    if dry_run and path.exists() and overwrite:
        print(f"DRY-RUN would overwrite {path.relative_to(ROOT)}")


def ensure_safe_rebuild_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved == SAFE_REBUILD_ROOT or not resolved.is_relative_to(SAFE_REBUILD_ROOT):
        raise RuntimeError(f"refusing to remove index directory outside {SAFE_REBUILD_ROOT}: {resolved}")


def trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    core = ordered[1:-1] if len(ordered) > 2 else ordered
    return sum(core) / len(core)


def parse_elapsed_seconds(text: str) -> float:
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([^\n]+)", text)
    if not match:
        raise RuntimeError("missing wall-clock time in /usr/bin/time output")
    parts = match.group(1).strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])


def parse_max_rss_kb(text: str) -> int:
    match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    if not match:
        raise RuntimeError("missing Maximum resident set size (kbytes)")
    return int(match.group(1))


def parse_search_output(text: str, topk: int) -> dict[str, float]:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    patterns = {
        "recall_at_k": rf"Recall@{topk}:\s+({number})",
        "qps": rf"QPS:\s+({number})",
        "total_ms": rf"Total time:\s+({number})\s+ms",
    }
    metrics: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
        elif key != "total_ms":
            raise RuntimeError(f"missing SCORE search metric: {key}")
    if "qps" in metrics:
        metrics["avg_latency_ms"] = 1000.0 / metrics["qps"]
    return metrics


def run_command(command: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        print("DRY-RUN " + json.dumps(command))
        return None
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def build_input_path(args: argparse.Namespace, dataset: DatasetConfig) -> Path:
    if args.build_backend == "imi-from-index":
        if args.source_index_dir is None:
            raise ValueError("--source-index-dir is required for --build-backend imi-from-index")
        return resolve(args.source_index_dir)
    if args.build_backend in ("stream-reorder", "stream-no-reorder"):
        return resolve(args.base_fbin) if args.base_fbin else resolve(default_stream_base(dataset))
    return resolve(dataset.base)


def method_name(args: argparse.Namespace) -> str:
    search_name = Path(args.search_bin).name if args.search_bin else ""
    if args.build_backend == "imi-from-index" or search_name == "score_imi_search":
        return "score_imi"
    return "score"


def raw_dir_for(args: argparse.Namespace) -> Path:
    method = method_name(args)
    if method == "score":
        return RAW_ROOT / args.dataset
    return RAW_ROOT.parent / method / args.dataset


def batch_build_command(args: argparse.Namespace, dataset: DatasetConfig, index_dir: Path) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        str(resolve(args.build_bin)),
        str(build_input_path(args, dataset)),
        str(index_dir),
        "-M",
        str(args.M),
        "-rabitq_k",
        str(args.rabitq_k),
        "-rabitq_seed",
        str(args.rabitq_seed),
        "-ivf_k",
        str(args.ivf_k),
    ]
    if args.pid_reorder:
        command.append("-pid_reorder")
    else:
        command.append("-no_pid_reorder")
    command.extend(args.build_extra)
    return command


def stream_reorder_build_command(args: argparse.Namespace, dataset: DatasetConfig, index_dir: Path) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        str(resolve(args.build_bin)),
        str(build_input_path(args, dataset)),
        str(index_dir),
        "-M",
        str(args.M),
        "-rabitq_k",
        str(args.rabitq_k),
        "-rabitq_seed",
        str(args.rabitq_seed),
        "-ivf_k",
        str(args.ivf_k),
        "-sample_rows",
        str(args.sample_rows),
        "-block_mib",
        str(args.block_mib),
        "-train_iters",
        str(args.train_iters),
        "-subspace_workers",
        str(args.subspace_workers),
        "-io_rows",
        str(args.io_rows),
        "-bucket_buffer_rows",
        str(args.bucket_buffer_rows),
        "-rewrite_workers",
        str(args.rewrite_workers),
    ]
    if args.clustering_mode != "hierarchical":
        command.extend(["-clustering_mode", args.clustering_mode])
    if args.subspace_schedule != "dynamic":
        command.extend(["-subspace_schedule", args.subspace_schedule])
    if args.rotation_type != "fhtkac":
        command.extend(["-rotation_type", args.rotation_type])
    if args.subspace_input != "shared":
        command.extend(["-subspace_input", args.subspace_input])
    command.extend(args.build_extra)
    return command


def stream_no_reorder_build_command(args: argparse.Namespace, dataset: DatasetConfig, index_dir: Path) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        str(resolve(args.build_bin)),
        str(build_input_path(args, dataset)),
        str(index_dir),
        "-M",
        str(args.M),
        "-rabitq_k",
        str(args.rabitq_k),
        "-rabitq_seed",
        str(args.rabitq_seed),
        "-ivf_k",
        str(args.ivf_k),
        "-sample_rows",
        str(args.sample_rows),
        "-block_mib",
        str(args.block_mib),
        "-train_iters",
        str(args.train_iters),
        "-subspace_workers",
        str(args.subspace_workers),
    ]
    command.extend(args.build_extra)
    return command


def imi_from_index_build_command(args: argparse.Namespace, dataset: DatasetConfig, index_dir: Path) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        str(resolve(args.build_bin)),
        str(build_input_path(args, dataset)),
        str(index_dir),
        "-M",
        str(args.M),
        "-imi_k_half",
        str(args.imi_k_half),
        "-sample_rows",
        str(args.sample_rows),
        "-train_iters",
        str(args.train_iters),
    ]
    command.extend(args.build_extra)
    return command


def run_build(args: argparse.Namespace, dataset: DatasetConfig, index_dir: Path) -> None:
    raw_dir = raw_dir_for(args)
    build_label = f"{args.label}_build"
    raw_log = raw_dir / f"{build_label}.log"
    ensure_can_write(raw_log, overwrite=args.overwrite_logs, dry_run=args.dry_run)

    if index_dir.exists():
        if not args.force_rebuild:
            raise RuntimeError(f"index directory already exists: {index_dir}")
        ensure_safe_rebuild_dir(index_dir)
        if args.dry_run:
            print(f"DRY-RUN remove existing index: {index_dir}")
        else:
            shutil.rmtree(index_dir)

    input_path = build_input_path(args, dataset)
    if not args.dry_run and not input_path.exists():
        raise FileNotFoundError(f"missing SCORE build input: {input_path.relative_to(ROOT)}")

    if args.build_backend == "stream-reorder":
        command = stream_reorder_build_command(args, dataset, index_dir)
    elif args.build_backend == "stream-no-reorder":
        command = stream_no_reorder_build_command(args, dataset, index_dir)
    elif args.build_backend == "imi-from-index":
        command = imi_from_index_build_command(args, dataset, index_dir)
    else:
        command = batch_build_command(args, dataset, index_dir)

    start = time.perf_counter()
    proc = run_command(command, dry_run=args.dry_run)
    if proc is None:
        return
    output = proc.stdout + proc.stderr
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"SCORE build failed; see {raw_log.relative_to(ROOT)}")

    row = {
        "timestamp": iso_timestamp(),
        "mode": "build",
        "method": method_name(args),
        "label": args.label,
        "dataset": args.dataset,
        "runs": 1,
        "build_seconds": f"{time.perf_counter() - start:.6f}",
        "max_rss_kb": parse_max_rss_kb(output),
        "index_size_bytes": tree_size_bytes(index_dir),
        "adapter": method_name(args),
        "raw_log": raw_log.relative_to(ROOT).as_posix(),
    }
    write_result(row)
    print(json.dumps(row, indent=2))


def search_label(args: argparse.Namespace, alpha: float, beta: float, max_w: int) -> str:
    prefix = f"{args.label}_" if args.label else ""
    topk_suffix = "" if args.topk == 10 else f"_k{args.topk}"
    thread_suffix = "1t" if args.threads == 1 else f"{args.threads}t"
    index_part = f"imik{args.imi_k_half}" if method_name(args) == "score_imi" else f"ivfk{args.ivf_k}"
    return f"{prefix}m{args.M}_{index_part}_a{alpha:g}_b{beta:g}_w{max_w}{topk_suffix}_{thread_suffix}"


def run_search_point(
    args: argparse.Namespace,
    dataset: DatasetConfig,
    index_dir: Path,
    alpha: float,
    beta: float,
    max_w: int,
) -> None:
    label = search_label(args, alpha, beta, max_w)
    raw_dir = raw_dir_for(args)
    summary_log = raw_dir / f"{label}.log"
    run_metrics: list[dict[str, float]] = []
    ensure_can_write(summary_log, overwrite=args.overwrite_logs, dry_run=args.dry_run)

    for run_id in range(1, args.runs + 1):
        detail_log = raw_dir / f"{label}_run{run_id}.detail.log"
        ensure_can_write(detail_log, overwrite=args.overwrite_logs, dry_run=args.dry_run)
        command = [
            str(resolve(args.search_bin)),
            str(index_dir),
            str(resolve(dataset.query)),
            str(resolve(args.gt) if args.gt else resolve(dataset.gt)),
            "-topk",
            str(args.topk),
            "-M",
            str(args.M),
            "-alpha",
            str(alpha),
            "-beta",
            str(beta),
            "-threads",
            str(args.threads),
            "-max_w",
            str(max_w),
        ]
        command.extend(args.search_extra)
        proc = run_command(command, dry_run=args.dry_run)
        if proc is None:
            continue
        output = proc.stdout + proc.stderr
        detail_log.parent.mkdir(parents=True, exist_ok=True)
        detail_log.write_text(output, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"SCORE search failed; see {detail_log.relative_to(ROOT)}")
        run_metrics.append(parse_search_output(output, args.topk))

    if args.dry_run:
        return
    recall = trimmed_mean([item["recall_at_k"] for item in run_metrics])
    qps = trimmed_mean([item["qps"] for item in run_metrics])
    avg_latency_ms = trimmed_mean([item["avg_latency_ms"] for item in run_metrics])
    summary = {
        "method": method_name(args),
        "dataset": args.dataset,
        "label": label,
        "runs": args.runs,
        "M": args.M,
        "ivf_k": args.ivf_k,
        "imi_k_half": args.imi_k_half,
        "alpha": alpha,
        "beta": beta,
        "max_w": max_w,
        "threads": args.threads,
        "recall_k": args.topk,
        "recall_at_k": recall,
        "qps": qps,
        "avg_latency_ms": avg_latency_ms,
        "run_metrics": run_metrics,
    }
    if args.topk == 10:
        summary["recall_at_10"] = recall
    summary_log.parent.mkdir(parents=True, exist_ok=True)
    summary_log.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    row = {
        "timestamp": iso_timestamp(),
        "mode": "search",
        "method": method_name(args),
        "label": label,
        "dataset": args.dataset,
        "runs": args.runs,
        "recall_trimmed": f"{recall:.6f}",
        "qps_trimmed": f"{qps:.6f}",
        "avg_latency_ms_trimmed": f"{avg_latency_ms:.6f}",
        "adapter": method_name(args),
        "raw_log": summary_log.relative_to(ROOT).as_posix(),
    }
    write_result(row)
    print(json.dumps({"summary": row, "metrics": summary}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--label", type=validate_label, default="default")
    parser.add_argument("--build-backend", choices=("batch", "stream-reorder", "stream-no-reorder", "imi-from-index"), default="batch")
    parser.add_argument("--build-bin", default=None)
    parser.add_argument("--search-bin", default="build/score_search")
    parser.add_argument("--source-index-dir", type=Path, default=None, help="existing SCORE+IVF index used by --build-backend imi-from-index")
    parser.add_argument("--gt", type=Path, default=None, help="override ground-truth ivecs path for search")
    parser.add_argument(
        "--base-fbin",
        type=Path,
        default=None,
        help="base vectors for --build-backend stream-reorder; defaults to the dataset _base.fbin path",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--overwrite-logs", action="store_true", help="allow replacing existing raw SCORE logs and upserting TSV rows")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--rabitq-k", type=int, default=200)
    parser.add_argument("--rabitq-seed", type=int, default=42)
    parser.add_argument("--ivf-k", type=int, default=2500)
    parser.add_argument("--imi-k-half", type=int, default=50)
    parser.add_argument("--alphas", type=parse_float_csv, default=parse_float_csv("0.03,0.04,0.05,0.06"))
    parser.add_argument("--betas", type=parse_float_csv, default=parse_float_csv("0.001,0.002,0.005,0.01,0.02"))
    parser.add_argument("--max-ws", type=parse_int_csv, default=parse_int_csv("15"))
    parser.add_argument("--pid-reorder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-rows", type=int, default=65536)
    parser.add_argument("--block-mib", type=int, default=256)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--subspace-workers", type=int, default=0)
    parser.add_argument("--clustering-mode", choices=("hierarchical", "flat"), default="hierarchical")
    parser.add_argument("--subspace-schedule", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--rotation-type", choices=("fhtkac", "hadamard", "qr", "identity", "none"), default="fhtkac")
    parser.add_argument("--subspace-input", choices=("shared", "raw", "separate", "separate_fhtkac"), default="shared")
    parser.add_argument("--io-rows", type=int, default=65536)
    parser.add_argument("--bucket-buffer-rows", type=int, default=64)
    parser.add_argument("--rewrite-workers", type=int, default=4)
    parser.add_argument("--build-extra", nargs="*", default=[])
    parser.add_argument("--search-extra", nargs="*", default=[])
    args = parser.parse_args()
    build_bin_was_provided = args.build_bin is not None
    search_bin_was_provided = args.search_bin != "build/score_search"

    if args.build_bin is None:
        if args.build_backend == "stream-reorder":
            args.build_bin = DEFAULT_STREAM_REORDER_BUILD_BIN
        elif args.build_backend == "stream-no-reorder":
            args.build_bin = DEFAULT_STREAM_NO_REORDER_BUILD_BIN
        elif args.build_backend == "imi-from-index":
            args.build_bin = DEFAULT_IMI_FROM_INDEX_BUILD_BIN
        else:
            args.build_bin = DEFAULT_BATCH_BUILD_BIN
    if args.build_backend == "imi-from-index" and not search_bin_was_provided:
        args.search_bin = "build/score_imi_search"
    if (
        args.build_backend == "batch"
        and build_bin_was_provided
        and Path(args.build_bin).name == "score_stream_build_reorder"
    ):
        raise SystemExit("use --build-backend stream-reorder with score_stream_build_reorder")
    if args.build_backend == "imi-from-index" and args.source_index_dir is None:
        raise SystemExit("--source-index-dir is required with --build-backend imi-from-index")

    if args.runs <= 0 or args.M <= 0 or args.rabitq_k <= 0 or args.ivf_k <= 0 or args.imi_k_half <= 0 or args.topk <= 0:
        raise SystemExit("runs, M, rabitq-k, ivf-k, imi-k-half, and topk must be positive")
    streaming_positive = (
        args.sample_rows,
        args.block_mib,
        args.train_iters,
        args.io_rows,
        args.bucket_buffer_rows,
        args.rewrite_workers,
    )
    if any(value <= 0 for value in streaming_positive) or args.subspace_workers < 0:
        raise SystemExit("streaming row/worker options must be positive except subspace-workers, which may be 0")
    if args.build_backend == "stream-reorder" and not args.pid_reorder:
        raise SystemExit("--build-backend stream-reorder always uses PID reorder; remove --no-pid-reorder")
    if args.build_backend == "stream-no-reorder" and args.pid_reorder:
        args.pid_reorder = False

    dataset = DATASETS[args.dataset]
    index_dir = resolve(args.index_dir)
    if args.force_rebuild or not index_dir.exists():
        run_build(args, dataset, index_dir)
    else:
        print(f"Using existing SCORE index: {index_dir}")

    for max_w in args.max_ws:
        if max_w <= 0:
            raise SystemExit("max-w values must be positive")
        for alpha in args.alphas:
            for beta in args.betas:
                run_search_point(args, dataset, index_dir, alpha, beta, max_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
