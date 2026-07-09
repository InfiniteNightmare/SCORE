#!/usr/bin/env python3
"""Local SHG runner with explicit build/search phases."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import TypedDict


ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "docs" / "evaluation" / "raw_baselines" / "shg"
RESULTS_TSV = ROOT / "docs" / "evaluation" / "baseline_results.tsv"
SHG_ROOT = ROOT / "baselines" / "SHG-Index"
PREPARE = SHG_ROOT / "build" / "preparedata"
SEARCH = SHG_ROOT / "build" / "example_search"

CONSTRUCTION_PATTERNS = {
    "build_index_seconds": re.compile(r"Build Index Time\s*:\s*([\d.]+)"),
    "preprocess_seconds": re.compile(r"Preprocessing Time\s*:\s*([\d.]+)"),
    "shortcut_seconds": re.compile(r"Build shortcut time\s*:\s*([\d.]+)"),
}
SEARCH_PATTERNS = {
    "recall": re.compile(r"avg_recall=([\d.]+)"),
    "qps": re.compile(r"qps=([\d.]+)"),
    "lat_p50": re.compile(r"p50_ms=([\d.]+)"),
}
MAX_RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")


class SearchRun(TypedDict):
    raw_log: pathlib.Path
    recall: float
    qps: float
    lat_p50: float


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) <= 2:
        return sum(ordered) / len(ordered)
    core = ordered[1:-1]
    return sum(core) / len(core)


def ensure_results_header() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "timestamp\tmode\tmethod\tlabel\tdataset\truns\tbuild_seconds\tmax_rss_kb\t"
        "index_size_bytes\trecall_trimmed\tqps_trimmed\tlat_p50_trimmed\tadapter\traw_log\n",
        encoding="utf-8",
    )


def append_result(row: dict[str, str]) -> None:
    ensure_results_header()
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(row[key]) for key in row) + "\n")


def parse_required(pattern: re.Pattern[str], text: str, label: str) -> float:
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"missing {label}")
    return float(match.group(1))


def parse_construction_metrics(text: str) -> dict[str, float]:
    return {
        key: parse_required(pattern, text, key)
        for key, pattern in CONSTRUCTION_PATTERNS.items()
    }


def parse_search_metrics(text: str) -> dict[str, float]:
    return {
        key: parse_required(pattern, text, key)
        for key, pattern in SEARCH_PATTERNS.items()
    }


def parse_max_rss_kb(text: str) -> int:
    match = MAX_RSS_PATTERN.search(text)
    if not match:
        raise RuntimeError("missing Maximum resident set size (kbytes)")
    return int(match.group(1))


def dataset_paths(dataset: str) -> dict[str, pathlib.Path]:
    dataset_dir = ROOT / "dataset" / dataset
    return {
        "dir": dataset_dir,
        "base": dataset_dir / f"{dataset}_base.fvecs",
        "query": dataset_dir / f"{dataset}_query.fvecs",
        "gt": dataset_dir / f"{dataset}_groundtruth.ivecs",
        "data_new": dataset_dir / f"{dataset}.data_new",
    }


def default_index_path(dataset: str, label: str) -> pathlib.Path:
    return ROOT / "results" / "baselines" / "shg" / f"{dataset}_{label}.index"


def ensure_data_new(dataset: str) -> pathlib.Path:
    paths = dataset_paths(dataset)
    if paths["data_new"].exists():
        return paths["data_new"]
    if not PREPARE.exists():
        raise FileNotFoundError(f"preparedata missing: {PREPARE}")
    if not paths["base"].exists():
        raise FileNotFoundError(f"missing base fvecs: {paths['base']}")
    paths["dir"].mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PREPARE), str(paths["base"]), str(paths["data_new"])],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return paths["data_new"]


def run_timed(command: list[str], raw_path: pathlib.Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["/usr/bin/time", "-v", *command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"SHG command failed: see {raw_path}")
    return output


def build_results_dir() -> pathlib.Path:
    return SHG_ROOT / "build" / "results"


def build_logs(dataset: str, k: int) -> tuple[pathlib.Path, pathlib.Path]:
    return (
        build_results_dir() / f"{dataset}_heds_construction.txt",
        build_results_dir() / f"{dataset}_heds_{k}_search.txt",
    )


def run_build(dataset: str, label: str, index_path: pathlib.Path) -> dict[str, object]:
    ensure_data_new(dataset)
    if not SEARCH.exists():
        raise FileNotFoundError(f"example_search missing: {SEARCH}")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = RAW_DIR / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{label}_build.log"
    output = run_timed(
        [str(SEARCH), "build", "heds", dataset, str(index_path)],
        raw_path,
        os.environ.copy(),
    )

    construction_log, _ = build_logs(dataset, 10)
    if not construction_log.exists():
        raise RuntimeError("SHG construction log was not generated")
    construction_copy = raw_dir / f"{label}_construction.txt"
    shutil.copy2(construction_log, construction_copy)

    construction = parse_construction_metrics(construction_copy.read_text(encoding="utf-8"))
    max_rss_kb = parse_max_rss_kb(output)
    build_seconds = (
        construction["build_index_seconds"]
        + construction["preprocess_seconds"]
        + construction["shortcut_seconds"]
    )
    return {
        "raw_log": raw_path,
        "construction_log": construction_copy,
        "build_seconds": build_seconds,
        "max_rss_kb": max_rss_kb,
        "index_size_bytes": index_path.stat().st_size,
    }


def run_search_once(
    dataset: str,
    label: str,
    index_path: pathlib.Path,
    k: int,
    ef_search: int,
    threads: int,
    run_id: int,
) -> SearchRun:
    if not SEARCH.exists():
        raise FileNotFoundError(f"example_search missing: {SEARCH}")
    if not index_path.exists():
        raise FileNotFoundError(f"missing SHG index: {index_path}")

    raw_dir = RAW_DIR / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{label}_search_run{run_id}.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    output = run_timed(
        [
            str(SEARCH),
            "search",
            "heds",
            dataset,
            str(k),
            str(ef_search),
            str(index_path),
            "shortcut-prune",
        ],
        raw_path,
        env,
    )

    search = parse_search_metrics(output)
    return {
        "raw_log": raw_path,
        "recall": search["recall"],
        "qps": search["qps"],
        "lat_p50": search["lat_p50"],
    }


def summarize_search_runs(runs: list[SearchRun]) -> dict[str, float]:
    return {
        "recall": trimmed_mean([float(run["recall"]) for run in runs]),
        "qps": trimmed_mean([float(run["qps"]) for run in runs]),
        "lat_p50": trimmed_mean([float(run["lat_p50"]) for run in runs]),
    }


def build_row(dataset: str, label: str, result: dict[str, object]) -> dict[str, str]:
    construction_log = result["construction_log"]
    if not isinstance(construction_log, pathlib.Path):
        construction_log = pathlib.Path(str(construction_log))
    build_seconds = float(str(result["build_seconds"]))
    max_rss_kb = int(str(result["max_rss_kb"]))
    index_size_bytes = int(str(result["index_size_bytes"]))
    return {
        "timestamp": iso_timestamp(),
        "mode": "build",
        "method": "shg",
        "label": label,
        "dataset": dataset,
        "runs": "1",
        "build_seconds": f"{build_seconds:.6f}",
        "max_rss_kb": str(max_rss_kb),
        "index_size_bytes": str(index_size_bytes),
        "recall_trimmed": "",
        "qps_trimmed": "",
        "lat_p50_trimmed": "",
        "adapter": "shg",
        "raw_log": str(construction_log.relative_to(ROOT)),
    }


def search_row(dataset: str, label: str, runs: int, result: dict[str, float], raw_log: pathlib.Path) -> dict[str, str]:
    return {
        "timestamp": iso_timestamp(),
        "mode": "search",
        "method": "shg",
        "label": label,
        "dataset": dataset,
        "runs": str(runs),
        "build_seconds": "",
        "max_rss_kb": "",
        "index_size_bytes": "",
        "recall_trimmed": f"{result['recall']:.6f}",
        "qps_trimmed": f"{result['qps']:.6f}",
        "lat_p50_trimmed": f"{result['lat_p50']:.6f}",
        "adapter": "shg",
        "raw_log": str(raw_log.relative_to(ROOT)),
    }


def cmd_build(args: argparse.Namespace) -> None:
    index_path = pathlib.Path(args.index_path).resolve() if args.index_path else default_index_path(args.dataset, args.label)
    result = run_build(args.dataset, args.label, index_path)
    row = build_row(args.dataset, args.label, result)
    if args.append_results:
        append_result(row)
    print(json.dumps({"build": row, "index_path": str(index_path)}, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    index_path = pathlib.Path(args.index_path).resolve() if args.index_path else default_index_path(args.dataset, args.label)
    runs = [
        run_search_once(
            args.dataset,
            args.label,
            index_path,
            args.k,
            args.ef_search,
            args.threads,
            run_id,
        )
        for run_id in range(1, args.runs + 1)
    ]
    summary = summarize_search_runs(runs)
    row = search_row(args.dataset, args.label, args.runs, summary, runs[-1]["raw_log"])
    if args.append_results:
        append_result(row)
    print(json.dumps({"search": row, "index_path": str(index_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Run SHG build only")
    build_parser.add_argument("--dataset", required=True)
    build_parser.add_argument("--label", required=True)
    build_parser.add_argument("--index-path")
    build_parser.add_argument("--append-results", action="store_true")
    build_parser.set_defaults(func=cmd_build)

    search_parser = subparsers.add_parser("search", help="Run SHG search only")
    search_parser.add_argument("--dataset", required=True)
    search_parser.add_argument("--label", required=True)
    search_parser.add_argument("--index-path")
    search_parser.add_argument("--k", type=int, default=10)
    search_parser.add_argument("--ef-search", type=int, default=10)
    search_parser.add_argument("--threads", type=int, default=1)
    search_parser.add_argument("--runs", type=int, default=5)
    search_parser.add_argument("--append-results", action="store_true")
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
