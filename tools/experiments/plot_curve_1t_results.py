#!/usr/bin/env python3
"""Plot paper-ready 1-thread Recall@k/QPS frontier curves."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "docs" / "evaluation"
DEFAULT_INPUT = EVAL_DIR / "curve_1t_raw_results.tsv"
DEFAULT_OUTPUT_DIR = EVAL_DIR / "figures"
FIGURE_PREFIX = "fig3_recall_qps_frontier"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DATASETS = [
    "gist1m",
    "laion1m",
    "cohere1m",
    "imagenet1m_2048",
    "cohere10m",
    "wikipedia",
]
DATASET_LABELS = {
    "gist1m": "GIST",
    "laion1m": "LAION",
    "cohere1m": "Cohere1M",
    "imagenet1m_2048": "ImageNet",
    "cohere10m": "Cohere10M",
    "wikipedia": "Wikipedia",
}

METHOD_ORDER = [
    "score",
    "suco",
    "taco",
    "crisp",
    "hnsw",
    "alphacng",
    "ivf-rabitq",
    "det-lsh",
]
METHOD_LABELS = {
    "score": "SCORE",
    "suco": "SuCo",
    "taco": "TaCo",
    "crisp": "CRISP",
    "hnsw": "HNSW",
    "alphacng": "α-CNG",
    "ivf-rabitq": "IVF-RaBitQ",
    "det-lsh": "DET-LSH",
}
METHOD_COLORS = {
    # Colorblind-safe, publication-oriented palette based on Okabe-Ito.
    "score": "#0072B2",      # blue
    "suco": "#E69F00",       # orange
    "taco": "#56B4E9",       # sky blue
    "crisp": "#009E73",      # green
    "hnsw": "#D55E00",       # vermilion
    "alphacng": "#CC79A7",   # purple
    "ivf-rabitq": "#B8860B", # dark goldenrod, more legible than bright yellow
    "det-lsh": "#6B6B6B",    # neutral gray
}
METHOD_MARKERS = {
    "score": "h",
    "suco": "*",
    "taco": "v",
    "crisp": "X",
    "hnsw": "o",
    "alphacng": "D",
    "ivf-rabitq": "P",
    "det-lsh": "d",
}


@dataclass(frozen=True)
class Point:
    dataset: str
    method: str
    label: str
    recall: float
    qps: float
    recall_k: int
    pareto: bool


def load_points(path: Path, recall_k: int) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                row_recall_k = int(row.get("recall_k") or "10")
                if row_recall_k != recall_k:
                    continue
                recall_text = row.get("recall_at_k") or row.get("recall_at_10")
                if recall_text is None:
                    continue
                recall = float(recall_text)
                qps = float(row["qps"])
            except (KeyError, ValueError):
                continue
            points.append(
                Point(
                    dataset=row.get("dataset", ""),
                    method=row.get("method", ""),
                    label=row.get("label", ""),
                    recall=recall,
                    qps=qps,
                    recall_k=row_recall_k,
                    pareto=row.get("pareto_1t_recall_qps") == "yes",
                )
            )
    return points


def grouped_pareto(points: list[Point]) -> dict[tuple[str, str], list[Point]]:
    groups: dict[tuple[str, str], list[Point]] = defaultdict(list)
    for point in points:
        if point.pareto:
            groups[(point.dataset, point.method)].append(point)
    for group in groups.values():
        group.sort(key=lambda p: (p.recall, p.qps))
    return groups


def plot_dataset(ax: plt.Axes, dataset: str, groups: dict[tuple[str, str], list[Point]], *, high_recall: bool, recall_k: int) -> None:
    for method in METHOD_ORDER:
        group = groups.get((dataset, method), [])
        if not group:
            continue
        xs = [p.recall for p in group]
        ys = [p.qps for p in group]
        ax.plot(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=5.0 if method == "score" else 4.2,
            linewidth=1.8 if method == "score" else 1.25,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_title(DATASET_LABELS.get(dataset, dataset))
    ax.set_xlabel(f"Recall@{recall_k}")
    ax.set_ylabel("QPS")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.set_yscale("log")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", colors="#000000", width=0.8)
    if high_recall:
        ax.set_xlim(0.90, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)


def legend_handles() -> tuple[list[plt.Line2D], list[str]]:
    handles = []
    labels = []
    for method in METHOD_ORDER:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=METHOD_COLORS.get(method),
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=1.8 if method == "score" else 1.25,
                markersize=5.5 if method == "score" else 4.8,
            )
        )
        labels.append(METHOD_LABELS.get(method, method))
    return handles, labels


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(f"refusing to overwrite existing figure(s) without --overwrite: {joined}")
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")


def plot_overview(points: list[Point], output_dir: Path, *, high_recall: bool, overwrite: bool, recall_k: int) -> None:
    groups = grouped_pareto(points)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.9), sharex=False, sharey=False)
    for ax, dataset in zip(axes.ravel(), DATASETS, strict=True):
        plot_dataset(ax, dataset, groups, high_recall=high_recall, recall_k=recall_k)
    handles, labels = legend_handles()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    suffix = "highrecall" if high_recall else "full"
    k_suffix = "" if recall_k == 10 else f"_k{recall_k}"
    save_figure(fig, output_dir, f"{FIGURE_PREFIX}{k_suffix}_overview_{suffix}", overwrite=overwrite)
    plt.close(fig)


def plot_individual(points: list[Point], output_dir: Path, *, high_recall: bool, overwrite: bool, recall_k: int) -> None:
    groups = grouped_pareto(points)
    suffix = "highrecall" if high_recall else "full"
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(4.8, 3.35))
        plot_dataset(ax, dataset, groups, high_recall=high_recall, recall_k=recall_k)
        if groups.get((dataset, "score")) or any(groups.get((dataset, method)) for method in METHOD_ORDER):
            ax.legend(loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        k_suffix = "" if recall_k == 10 else f"_k{recall_k}"
        save_figure(fig, output_dir, f"{FIGURE_PREFIX}{k_suffix}_{suffix}_{dataset}", overwrite=overwrite)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-full", action="store_true", help="also generate full-range figures")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing figure files")
    parser.add_argument("--recall-k", type=int, default=10, help="plot only points with this Recall@k")
    args = parser.parse_args()

    points = load_points(args.input, args.recall_k)
    plot_overview(points, args.output_dir, high_recall=True, overwrite=args.overwrite, recall_k=args.recall_k)
    plot_individual(points, args.output_dir, high_recall=True, overwrite=args.overwrite, recall_k=args.recall_k)
    if args.include_full:
        plot_overview(points, args.output_dir, high_recall=False, overwrite=args.overwrite, recall_k=args.recall_k)
        plot_individual(points, args.output_dir, high_recall=False, overwrite=args.overwrite, recall_k=args.recall_k)
    print(f"plotted_points={len(points)}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
