#!/usr/bin/env python3
"""Parse raw 1-thread method curve logs into a TSV summary.

This script is intentionally read-only with respect to raw experiment logs.  It
discovers per-parameter ``*_1t.log`` files under ``docs/evaluation/raw_baselines``
and regenerates:

* ``docs/evaluation/curve_1t_raw_results.tsv``
* ``docs/evaluation/baseline_data_status.md``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "docs" / "evaluation"
RAW_ROOT = EVAL_DIR / "raw_baselines"
OUT_TSV = EVAL_DIR / "curve_1t_raw_results.tsv"
OUT_STATUS = EVAL_DIR / "baseline_data_status.md"
REPORT_RECALL_K = 10

DATASETS = [
    "gist1m",
    "laion1m",
    "cohere1m",
    "imagenet1m_2048",
    "cohere10m",
    "wikipedia",
]

METHOD_DIRS = [
    "hnsw",
    "mirage",
    "crisp",
    "taco",
    "taco_fixed8x10",
    "suco",
    "shg",
    "alphacng",
    "ivfrabitq",
    "det-lsh",
    "score",
]

METHOD_NAMES = {"ivfrabitq": "ivf-rabitq", "taco_fixed8x10": "taco-fixed8x10"}

FIELDNAMES = [
    "method",
    "dataset",
    "label",
    "parameter_key",
    "collision_ratio",
    "min_score_factor",
    "candidate_ratio",
    "ef_search",
    "k_size",
    "recall_k",
    "recall_at_k",
    "recall_at_10",
    "qps",
    "p50_ms",
    "pareto_1t_recall_qps",
    "raw_log",
    "note",
]


@dataclass
class ParsedPoint:
    method: str
    dataset: str
    label: str
    parameter_key: str
    raw_log: Path
    recall_k: int
    recall_at_k: float
    qps: float
    p50_ms: float | None = None
    collision_ratio: str = ""
    min_score_factor: str = ""
    candidate_ratio: str = ""
    ef_search: str = ""
    k_size: str = ""
    pareto: bool = False
    note: str = ""

    def row(self) -> dict[str, str]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "label": self.label,
            "parameter_key": self.parameter_key,
            "collision_ratio": self.collision_ratio,
            "min_score_factor": self.min_score_factor,
            "candidate_ratio": self.candidate_ratio,
            "ef_search": self.ef_search,
            "k_size": self.k_size,
            "recall_k": str(self.recall_k),
            "recall_at_k": format_float(self.recall_at_k),
            "recall_at_10": format_float(self.recall_at_k) if self.recall_k == 10 else "",
            "qps": format_float(self.qps),
            "p50_ms": "" if self.p50_ms is None else format_float(self.p50_ms),
            "pareto_1t_recall_qps": "yes" if self.pareto else "no",
            "raw_log": self.raw_log.relative_to(REPO_ROOT).as_posix(),
            "note": self.note,
        }


def format_float(value: float) -> str:
    return f"{value:.12g}"


def parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {text}")
    return value


def parse_decimal_token(text: str) -> str:
    return format_float(parse_float(text.replace("p", ".")))


def get_nested(metrics: dict[str, object], key: str) -> object | None:
    if key in metrics:
        return metrics[key]
    search = metrics.get("search")
    if isinstance(search, dict):
        return search.get(key)
    return None


def first_present(*values: object | None) -> object | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def parse_metrics(path: Path) -> tuple[float, int | None, float, float | None, str]:
    text = path.read_text(errors="replace")
    stripped = text.strip()
    note = ""

    json_text = ""
    if stripped.startswith("{"):
        json_text = stripped
    elif "{" in text and "}" in text:
        json_text = text[text.find("{") : text.rfind("}") + 1]

    if json_text:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_text)
        if not isinstance(data, dict):
            raise ValueError("JSON log root is not an object")
        recall_obj = first_present(
            get_nested(data, "recall_at_k"),
            get_nested(data, "recall_at_10"),
            get_nested(data, "recall_trimmed"),
        )
        recall_k_obj = first_present(
            get_nested(data, "recall_k"),
            get_nested(data, "topk"),
            get_nested(data, "k"),
        )
        qps_obj = first_present(get_nested(data, "qps"), get_nested(data, "qps_trimmed"))
        p50_obj = first_present(
            get_nested(data, "p50_ms"),
            get_nested(data, "lat_p50_trimmed"),
            get_nested(data, "p50_trimmed"),
        )
        if recall_obj is None or qps_obj is None:
            raise ValueError("missing recall/qps in JSON log")
        recall = parse_float(str(recall_obj))
        recall_k = None if recall_k_obj in (None, "") else int(str(recall_k_obj))
        qps = parse_float(str(qps_obj))
        p50 = None if p50_obj in (None, "") else parse_float(str(p50_obj))
        return recall, recall_k, qps, p50, note

    per_query = re.findall(
        r"^query=\d+\s+time_s=([0-9]+(?:\.[0-9]+)?)\s+recall=([0-9]+(?:\.[0-9]+)?)\s*$",
        text,
        re.MULTILINE,
    )
    if per_query:
        times = [parse_float(item[0]) for item in per_query]
        recalls = [parse_float(item[1]) for item in per_query]
        total_time = sum(times)
        if total_time <= 0:
            raise ValueError("non-positive per-query total time")
        return (
            sum(recalls) / len(recalls),
            None,
            len(times) / total_time,
            statistics.median(times) * 1000.0,
            "parsed per-query SHG log",
        )

    qps_match = re.search(r"\bqps\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not qps_match:
        qps_match = re.search(r"\bQPS\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not qps_match:
        raise ValueError("missing qps")
    qps = parse_float(qps_match.group(1))

    p50_match = re.search(r"\bp50_ms\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    p50 = parse_float(p50_match.group(1)) if p50_match else None

    recall: float | None = None
    recall_k: int | None = None

    # Paper figures use Recall@10.  Some native baselines (notably CRISP) must
    # run with a wider top-k/groundtruth width but print per-k metrics including
    # k=10; prefer that line instead of accidentally taking the final k=100 line.
    recall_at_10_patterns = [
        r"recall_at_10\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"Recall@10:\s*([0-9]+(?:\.[0-9]+)?)",
        r"For\s+k\s*=\s*10,\s*\(Recall@k,\s*Ratio@k\)\s*=\s*\(([0-9]+(?:\.[0-9]+)?),",
        r"When\s+k\s*=\s*10,\s*\(recall,\s*ratio\)\s*=\s*\(([0-9]+(?:\.[0-9]+)?),",
    ]
    for pattern in recall_at_10_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            recall_k = 10
            recall = parse_float(matches[-1])
            break

    if recall is None:
        recall_patterns = [
            r"recall_at_(\d+)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            r"Recall@(\d+):\s*([0-9]+(?:\.[0-9]+)?)",
            r"For\s+k\s*=\s*(\d+),\s*\(Recall@k,\s*Ratio@k\)\s*=\s*\(([0-9]+(?:\.[0-9]+)?),",
            r"When\s+k\s*=\s*(\d+),\s*\(recall,\s*ratio\)\s*=\s*\(([0-9]+(?:\.[0-9]+)?),",
            r"\brecall\s+([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pattern in recall_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                last = matches[-1]
                if isinstance(last, tuple):
                    recall_k = int(last[0])
                    recall = parse_float(last[1])
                else:
                    recall = parse_float(last)
                break

    if recall is None:
        percent_match = re.search(r"\bRecall\s*=\s*([0-9]+(?:\.[0-9]+)?)%", text)
        if percent_match:
            recall = parse_float(percent_match.group(1)) / 100.0

    if recall is None:
        raise ValueError("missing recall")
    return recall, recall_k, qps, p50, note


def parse_label(method_dir: str, stem: str) -> dict[str, str] | None:
    if method_dir in {"hnsw", "mirage"}:
        match = re.fullmatch(r"ef(\d+)(?:_k(\d+))?_1t", stem)
        if not match:
            return None
        ef, recall_k = match.groups()
        params = {"parameter_key": f"ef={ef}", "ef_search": ef}
        if recall_k:
            params["parameter_key"] += f",k={recall_k}"
            params["recall_k"] = recall_k
        return params

    if method_dir == "crisp":
        decimal = r"\d+(?:[.p]\d+)?"
        match = re.fullmatch(rf"a({decimal})_msf({decimal})_k(\d+)_1t", stem)
        if not match:
            return None
        alpha_token, msf_token, k = match.groups()
        alpha = parse_decimal_token(alpha_token)
        msf = parse_decimal_token(msf_token)
        return {
            "parameter_key": f"a={alpha},msf={msf},k={k}",
            "collision_ratio": alpha,
            "min_score_factor": msf,
            "k_size": k,
        }

    if method_dir in {"taco", "taco_fixed8x10", "suco"}:
        decimal = r"\d+(?:[.p]\d+)?"
        match = re.fullmatch(rf"a({decimal})_b({decimal})(?:_k(\d+))?_1t", stem)
        if not match:
            return None
        alpha_token, beta_token, recall_k = match.groups()
        alpha = parse_decimal_token(alpha_token)
        beta = parse_decimal_token(beta_token)
        params = {
            "parameter_key": f"a={alpha},b={beta}" + (f",k={recall_k}" if recall_k else ""),
            "collision_ratio": alpha,
            "candidate_ratio": beta,
        }
        if recall_k:
            params["recall_k"] = recall_k
        return params

    if method_dir == "shg":
        match = re.fullmatch(r"ef(\d+)_k(\d+)_1t", stem)
        if not match:
            return None
        ef, k = match.groups()
        return {"parameter_key": f"ef={ef},k={k}", "ef_search": ef, "k_size": k, "recall_k": k}

    if method_dir == "alphacng":
        match = re.fullmatch(r"L(\d+)_k(\d+)_1t", stem)
        if not match:
            return None
        level, k = match.groups()
        return {"parameter_key": f"L={level},k={k}", "candidate_ratio": level, "k_size": k, "recall_k": k}

    if method_dir == "ivfrabitq":
        match = re.fullmatch(r"nprobe(\d+)(?:_bq(\d+))?_k(\d+)_1t", stem)
        if not match:
            return None
        nprobe, bq, k = match.groups()
        parameter_key = f"nprobe={nprobe},k={k}"
        if bq:
            parameter_key = f"nprobe={nprobe},bq={bq},k={k}"
        return {"parameter_key": parameter_key, "candidate_ratio": nprobe, "k_size": k, "recall_k": k}

    if method_dir == "det-lsh":
        decimal = r"\d+(?:[.p]\d+)?"
        match = re.fullmatch(rf"L(\d+)_K(\d+)_b({decimal})_r({decimal})_1t", stem)
        if not match:
            return None
        l_size, lsh_dim, beta_token, radius_token = match.groups()
        beta = parse_decimal_token(beta_token)
        radius = parse_decimal_token(radius_token)
        return {
            "parameter_key": f"L={l_size},K={lsh_dim},b={beta},r={radius}",
            "candidate_ratio": beta,
            "ef_search": radius,
            "k_size": str(REPORT_RECALL_K),
            "recall_k": str(REPORT_RECALL_K),
        }

    if method_dir == "score":
        match = re.fullmatch(
            r"(?:(?P<variant>[A-Za-z0-9_-]+)_)?m(?P<m>\d+)_ivfk(?P<ivfk>\d+)_"
            r"a(?P<alpha>[0-9.]+)_b(?P<beta>[0-9.]+)_w(?P<w>\d+)"
            r"(?:_k(?P<k>\d+))?_1t",
            stem,
        )
        if not match:
            return None
        parts = match.groupdict()
        variant = parts.get("variant") or "default"
        k = parts.get("k") or "10"
        return {
            "parameter_key": (
                f"variant={variant},M={parts['m']},ivf_k={parts['ivfk']},"
                f"a={parts['alpha']},b={parts['beta']},w={parts['w']},k={k}"
            ),
            "collision_ratio": parts["alpha"],
            "candidate_ratio": parts["beta"],
            "k_size": k,
            "recall_k": k,
        }

    return None


def discover_points() -> tuple[list[ParsedPoint], list[str]]:
    points: list[ParsedPoint] = []
    skipped: list[str] = []
    for method_dir in METHOD_DIRS:
        for dataset in DATASETS:
            root = RAW_ROOT / method_dir / dataset
            if not root.exists():
                continue
            for log_path in sorted(root.glob("*_1t.log")):
                params = parse_label(method_dir, log_path.stem)
                if params is None:
                    skipped.append(f"{log_path.relative_to(REPO_ROOT)}: unmatched label")
                    continue
                try:
                    recall, metric_recall_k, qps, p50, note = parse_metrics(log_path)
                except Exception as exc:  # noqa: BLE001 - status should report all bad raw logs.
                    skipped.append(f"{log_path.relative_to(REPO_ROOT)}: {exc}")
                    continue
                recall_k = metric_recall_k or int(params.get("recall_k", str(REPORT_RECALL_K)))
                if recall_k != REPORT_RECALL_K:
                    continue
                point = ParsedPoint(
                    method=METHOD_NAMES.get(method_dir, method_dir),
                    dataset=dataset,
                    label=log_path.stem,
                    parameter_key=params["parameter_key"],
                    raw_log=log_path,
                    recall_k=recall_k,
                    recall_at_k=recall,
                    qps=qps,
                    p50_ms=p50,
                    collision_ratio=params.get("collision_ratio", ""),
                    min_score_factor=params.get("min_score_factor", ""),
                    candidate_ratio=params.get("candidate_ratio", ""),
                    ef_search=params.get("ef_search", ""),
                    k_size=params.get("k_size", ""),
                    note=note,
                )
                points.append(point)
    points.sort(key=lambda p: (DATASETS.index(p.dataset), p.method, p.label))
    mark_pareto(points)
    return points, skipped


def mark_pareto(points: list[ParsedPoint]) -> None:
    by_group: dict[tuple[str, str, int], list[ParsedPoint]] = defaultdict(list)
    for point in points:
        by_group[(point.dataset, point.method, point.recall_k)].append(point)

    for group_points in by_group.values():
        for point in group_points:
            dominated = False
            for other in group_points:
                if other is point:
                    continue
                no_worse = other.recall_at_k >= point.recall_at_k and other.qps >= point.qps
                strictly_better = other.recall_at_k > point.recall_at_k or other.qps > point.qps
                if no_worse and strictly_better:
                    dominated = True
                    break
            point.pareto = not dominated


def write_tsv(points: list[ParsedPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for point in points:
            writer.writerow(point.row())


def write_status(points: list[ParsedPoint], skipped: list[str], path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_dataset_method: dict[tuple[str, str, int], list[ParsedPoint]] = defaultdict(list)
    for point in points:
        by_dataset_method[(point.dataset, point.method, point.recall_k)].append(point)

    lines = [
        "# Curve raw-log data status",
        "",
        f"Generated by `tools/experiments/parse_curve_raw_logs.py` at {now}.",
        "",
        f"Parsed raw 1T search logs: **{len(points)}**.",
        f"Skipped/unparseable logs: **{len(skipped)}**.",
        "",
        "## Coverage by dataset/method",
        "",
        "| dataset | method | recall_k | points | pareto | recall range | max recall | max qps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        groups = sorted((method, recall_k) for (ds, method, recall_k) in by_dataset_method if ds == dataset)
        for method, recall_k in groups:
            group = by_dataset_method[(dataset, method, recall_k)]
            recalls = [p.recall_at_k for p in group]
            qps_values = [p.qps for p in group]
            pareto_count = sum(1 for p in group if p.pareto)
            lines.append(
                "| "
                f"{dataset} | {method} | {recall_k} | {len(group)} | {pareto_count} | "
                f"{format_float(min(recalls))}-{format_float(max(recalls))} | "
                f"{format_float(max(recalls))} | {format_float(max(qps_values))} |"
            )

    if skipped:
        lines.extend(["", "## Skipped logs", ""])
        lines.extend(f"- `{item}`" for item in skipped[:200])
        if len(skipped) > 200:
            lines.append(f"- ... {len(skipped) - 200} more")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_TSV)
    parser.add_argument("--status", type=Path, default=OUT_STATUS)
    parser.add_argument("--check-only", action="store_true", help="parse and report without writing files")
    args = parser.parse_args()

    points, skipped = discover_points()
    if not args.check_only:
        write_tsv(points, args.output)
        write_status(points, skipped, args.status)
    print(f"parsed={len(points)}")
    print(f"skipped={len(skipped)}")
    if skipped:
        print("first_skipped=" + skipped[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
