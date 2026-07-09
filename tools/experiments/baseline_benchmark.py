#!/usr/bin/env python3
"""Unified local wrapper for baseline build/search metric capture."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "docs" / "evaluation" / "raw_baselines"
RESULTS_TSV = ROOT / "docs" / "evaluation" / "baseline_results.tsv"


def trimmed_mean(values: list[float]) -> float:
    if len(values) <= 2:
        return sum(values) / len(values)
    ordered = sorted(values)
    trimmed = ordered[1:-1]
    return sum(trimmed) / len(trimmed)


def iso_timestamp() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def ensure_results_header() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "timestamp\tmode\tmethod\tlabel\tdataset\truns\tbuild_seconds\tmax_rss_kb\t"
        "index_size_bytes\trecall_trimmed\tqps_trimmed\tlat_p50_trimmed\tadapter\traw_log\n",
        encoding="utf-8",
    )


def append_result(row: Mapping[str, object]) -> None:
    ensure_results_header()
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(row[key]) for key in row) + "\n")


def resolve_path(text: str) -> pathlib.Path:
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def tree_size_bytes(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def sum_index_sizes(paths: list[str]) -> int:
    return sum(tree_size_bytes(resolve_path(path)) for path in paths)


def run_shell(
    command: str, cwd: pathlib.Path, raw_path: pathlib.Path, with_time: bool
) -> str:
    if with_time:
        cmd = ["/usr/bin/time", "-v", "bash", "-lc", command]
    else:
        cmd = ["bash", "-lc", command]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {command}\nsee {raw_path}")
    return output


def parse_required(
    pattern: str, text: str, transform: Callable[[str], float] | None = None
) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing metric pattern: {pattern}")
    value = match.group(1)
    if value.lower() == "nan":
        return float("nan")
    return transform(value) if transform else float(value)


def parse_json_metric(key: str, text: str) -> float:
    data = json.loads(text)
    return float(data[key])


def parse_search_metrics(method: str, text: str) -> dict[str, float]:
    if method == "hnsw":
        return {
            "recall": parse_json_metric("recall_at_k", text),
            "qps": parse_json_metric("qps", text),
            "lat_p50": parse_json_metric("p50_ms", text),
        }
    adapters: dict[str, dict[str, tuple[str, Callable[[str], float] | None]]] = {
        "mirage": {
            "recall": (r"recall_at_10=([\d.]+)", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=(nan|[\d.]+)", None),
        },
        "shg": {
            "recall": (r"avg_recall=([\d.]+)", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
        "ivf-rabitq": {
            "recall": (r"Recall = ([\d.]+)%", lambda v: float(v) / 100.0),
            "qps": (r"QPS = ([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
        "alphacng": {
            "recall": (r"recall ([\d.]+)", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
        "suco": {
            "recall": (r"When k = 10, \(recall, ratio\) = \(([\d.]+),", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
        "taco": {
            "recall": (r"When k = 10, \(recall, ratio\) = \(([\d.]+),", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
        "crisp": {
            "recall": (r"For k = 10, \(Recall@k, Ratio@k\) = \(([\d.]+),", None),
            "qps": (r"qps=([\d.]+)", None),
            "lat_p50": (r"p50_ms=([\d.]+)", None),
        },
    }
    if method not in adapters:
        raise RuntimeError(f"unsupported method adapter: {method}")
    spec = adapters[method]
    return {
        key: parse_required(pattern, text, transform)
        for key, (pattern, transform) in spec.items()
    }


def parse_max_rss_kb(text: str) -> int:
    match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    if not match:
        raise RuntimeError("missing Maximum resident set size (kbytes) in timed output")
    return int(match.group(1))


def run_build(args: argparse.Namespace) -> None:
    raw_path = RAW_DIR / args.method / args.dataset / f"{args.label}_build.log"
    start = time.perf_counter()
    output = run_shell(args.command, ROOT, raw_path, with_time=True)
    build_seconds = time.perf_counter() - start
    row = {
        "timestamp": iso_timestamp(),
        "mode": "build",
        "method": args.method,
        "label": args.label,
        "dataset": args.dataset,
        "runs": 1,
        "build_seconds": f"{build_seconds:.6f}",
        "max_rss_kb": parse_max_rss_kb(output),
        "index_size_bytes": sum_index_sizes(args.index_path),
        "recall_trimmed": "",
        "qps_trimmed": "",
        "lat_p50_trimmed": "",
        "adapter": args.method,
        "raw_log": str(raw_path.relative_to(ROOT)),
    }
    append_result(row)
    print(json.dumps(row, indent=2))


def run_search(args: argparse.Namespace) -> None:
    runs: list[dict[str, float]] = []
    raw_paths: list[str] = []
    for run_id in range(1, args.runs + 1):
        raw_path = (
            RAW_DIR
            / args.method
            / args.dataset
            / f"{args.label}_search_run{run_id}.log"
        )
        output = run_shell(args.command, ROOT, raw_path, with_time=False)
        runs.append(parse_search_metrics(args.method, output))
        raw_paths.append(str(raw_path.relative_to(ROOT)))
    summary = {
        "recall_trimmed": trimmed_mean([run["recall"] for run in runs]),
        "qps_trimmed": trimmed_mean([run["qps"] for run in runs]),
        "lat_p50_trimmed": trimmed_mean([run["lat_p50"] for run in runs]),
    }
    row = {
        "timestamp": iso_timestamp(),
        "mode": "search",
        "method": args.method,
        "label": args.label,
        "dataset": args.dataset,
        "runs": args.runs,
        "build_seconds": "",
        "max_rss_kb": "",
        "index_size_bytes": "",
        "recall_trimmed": f"{summary['recall_trimmed']:.6f}",
        "qps_trimmed": f"{summary['qps_trimmed']:.6f}",
        "lat_p50_trimmed": f"{summary['lat_p50_trimmed']:.6f}",
        "adapter": args.method,
        "raw_log": raw_paths[-1],
    }
    append_result(row)
    print(json.dumps({"summary": row, "raw_logs": raw_paths}, indent=2))


def run_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "entrypoints": {
            "score": "tools/experiments/benchmark.py",
            "baselines": "tools/experiments/baseline_benchmark.py",
        },
        "methods": [
            "hnsw",
            "mirage",
            "shg",
            "ivf-rabitq",
            "alphacng",
            "suco",
            "taco",
            "crisp",
        ],
        "search_policy": "5 runs, drop min/max, average middle 3",
        "build_metrics": ["build_seconds", "max_rss_kb", "index_size_bytes"],
        "search_metrics": ["recall_trimmed", "qps_trimmed", "lat_p50_trimmed"],
    }
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified local baseline benchmark wrapper"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Capture one baseline build/index run"
    )
    build_parser.add_argument("--method", required=True)
    build_parser.add_argument("--label", required=True)
    build_parser.add_argument("--dataset", required=True)
    build_parser.add_argument(
        "--command", required=True, help="Canonical local build/index command"
    )
    build_parser.add_argument(
        "--index-path",
        action="append",
        required=True,
        help="Index file/dir to size; repeatable",
    )
    build_parser.set_defaults(func=run_build)

    search_parser = subparsers.add_parser(
        "search", help="Capture trimmed-mean baseline search metrics"
    )
    search_parser.add_argument("--method", required=True)
    search_parser.add_argument("--label", required=True)
    search_parser.add_argument("--dataset", required=True)
    search_parser.add_argument(
        "--command", required=True, help="Canonical local search command"
    )
    search_parser.add_argument("--runs", type=int, default=5)
    search_parser.set_defaults(func=run_search)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Print wrapper policy manifest"
    )
    manifest_parser.set_defaults(func=run_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
