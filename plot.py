"""Plot evaluation results following Nature figure guidelines.

Reproduces figures/tables from:
  - Snell et al. 2017: Figure 2 (distance/way), Figure 4 (way ablation)
  - Bateni et al. 2022: Figure 6 (refinement steps), Table 2 (10-way)

Nature style: Helvetica/Arial, 5-7pt text, 8pt bold panel labels,
no gridlines, Wong colour-blind palette (no orange), axis ticks+labels,
single column = 89 mm, double column = 183 mm, max height = 170 mm.

Usage:
    python plot.py --results-dir results/ --output-dir plots/
    python plot.py --results-dir results/ --plot omniglot_fewshot
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------- Nature style constants ----------

MM_TO_INCHES = 1.0 / 25.4
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MAX_HEIGHT_MM = 170

SINGLE_COL = SINGLE_COL_MM * MM_TO_INCHES
DOUBLE_COL = DOUBLE_COL_MM * MM_TO_INCHES
MAX_HEIGHT = MAX_HEIGHT_MM * MM_TO_INCHES

# Wong colour-blind palette (no orange per user request)
BLACK = "#000000"
SKY_BLUE = "#56b4e9"
BLUISH_GREEN = "#009e73"
YELLOW = "#f0e442"
BLUE = "#0072b2"
VERMILLION = "#d55e00"
REDDISH_PURPLE = "#cc79a7"

PALETTE = [BLUE, VERMILLION, BLUISH_GREEN, SKY_BLUE, REDDISH_PURPLE, YELLOW, BLACK]


def _setup_nature_style() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------- Data loading ----------

@dataclass(frozen=True)
class ResultEntry:
    benchmark: str
    accuracy_mean: float
    accuracy_ci95: float
    n_way_test: int
    n_shot_test: int
    distance: str
    train_way: int
    transductive: bool
    n_refinement_steps: int


def load_results(results_dir: Path, benchmark_prefix: str) -> list[ResultEntry]:
    entries: list[ResultEntry] = []
    for path in sorted(results_dir.glob(f"{benchmark_prefix}*.json")):
        data = json.loads(path.read_text())
        cfg = data["config"]
        entries.append(ResultEntry(
            benchmark=data["benchmark"],
            accuracy_mean=data["accuracy_mean"],
            accuracy_ci95=data["accuracy_ci95"],
            n_way_test=cfg["n_way_test"],
            n_shot_test=cfg["n_shot_test"],
            distance=cfg["distance"],
            train_way=cfg["train_way"],
            transductive=cfg["transductive"],
            n_refinement_steps=cfg.get("n_refinement_steps", 4),
        ))
    return entries


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"Saved: {output_dir / name}.{{pdf,png}}")


# ---------- P1: Omniglot few-shot (Table 1) ----------

def plot_omniglot_fewshot(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "omniglot_fewshot")
    if not entries:
        print("No omniglot_fewshot results found, skipping.")
        return

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.65))

    entries_sorted = sorted(entries, key=lambda x: (x.n_way_test, x.n_shot_test))
    labels = [f"{e.n_way_test}-way {e.n_shot_test}-shot" for e in entries_sorted]
    means = [e.accuracy_mean * 100 for e in entries_sorted]

    x = np.arange(len(labels))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(x, means, width=0.55, color=colors, edgecolor=BLACK, linewidth=0.3)

    for bar, val in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Omniglot few-shot classification", fontweight="bold")
    ax.set_ylim(90, 101)

    fig.tight_layout()
    _save_fig(fig, output_dir, "omniglot_fewshot")


# ---------- P2: miniImageNet few-shot (Table 2) ----------

def plot_miniimagenet_fewshot(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_fewshot")
    if not entries:
        print("No miniimagenet_fewshot results found, skipping.")
        return

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.7, SINGLE_COL * 0.65))

    entries_sorted = sorted(entries, key=lambda x: x.n_shot_test)
    labels = [f"{e.n_shot_test}-shot" for e in entries_sorted]
    means = [e.accuracy_mean * 100 for e in entries_sorted]
    ci95s = [e.accuracy_ci95 * 100 for e in entries_sorted]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=ci95s, width=0.45, capsize=2,
                  color=[BLUE, BLUISH_GREEN], edgecolor=BLACK, linewidth=0.3,
                  error_kw={"linewidth": 0.5})

    for bar, val, ci in zip(bars, means, ci95s, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("miniImageNet 5-way classification", fontweight="bold")
    ax.set_ylim(30, 85)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_fewshot")


# ---------- P3: Distance & way comparison (Figure 2) ----------

def plot_distance_way_comparison(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_distance_way")
    if not entries:
        print("No miniimagenet_distance_way results found, skipping.")
        return

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6), sharey=True)

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")

        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: (x.distance, x.train_way),
        )

        group_labels = [f"{e.train_way}-way\n{e.distance}" for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]
        colors = [REDDISH_PURPLE if e.distance == "cosine" else BLUE for e in shot_entries]

        x = np.arange(len(group_labels))
        bars = ax.bar(x, means, yerr=ci95s, width=0.55, capsize=2,
                      color=colors, edgecolor=BLACK, linewidth=0.3,
                      error_kw={"linewidth": 0.5})

        for bar, val, ci in zip(bars, means, ci95s, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_title(f"{shot}-shot", fontweight="bold")
        ax.set_ylim(20, 80)

    # legend with coloured boxes, not coloured text
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=REDDISH_PURPLE, edgecolor=BLACK, linewidth=0.3, label="Cosine"),
                       Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=0.3, label="Euclidean")]
    axes[1].legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_distance_way")


# ---------- P4: Training way ablation (Figure 4) ----------

def plot_way_ablation(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_way_ablation")
    if not entries:
        print("No miniimagenet_way_ablation results found, skipping.")
        return

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6))

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")

        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: x.train_way,
        )
        ways = [e.train_way for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]

        ax.errorbar(ways, means, yerr=ci95s, marker="o", markersize=3,
                    linewidth=1, capsize=2, color=BLUE,
                    markeredgecolor=BLACK, markeredgewidth=0.3,
                    elinewidth=0.5)
        ax.set_xlabel("Training classes per episode")
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_title(f"{shot}-shot", fontweight="bold")
        ax.set_xticks(ways)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_way_ablation")


# ---------- P5: Transductive vs inductive (5-way) ----------

def plot_transductive(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_transductive")
    if not entries:
        print("No miniimagenet_transductive results found, skipping.")
        return

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.65))

    inductive = sorted([e for e in entries if not e.transductive], key=lambda x: x.n_shot_test)
    transductive = sorted([e for e in entries if e.transductive], key=lambda x: x.n_shot_test)

    labels = [f"{e.n_shot_test}-shot" for e in inductive]
    x = np.arange(len(labels))
    width = 0.3

    bars1 = ax.bar(x - width / 2, [e.accuracy_mean * 100 for e in inductive], width,
                   yerr=[e.accuracy_ci95 * 100 for e in inductive], capsize=2,
                   color=BLUE, edgecolor=BLACK, linewidth=0.3,
                   error_kw={"linewidth": 0.5})
    bars2 = ax.bar(x + width / 2, [e.accuracy_mean * 100 for e in transductive], width,
                   yerr=[e.accuracy_ci95 * 100 for e in transductive], capsize=2,
                   color=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3,
                   error_kw={"linewidth": 0.5})

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("5-way accuracy (%)")
    ax.set_title("Inductive vs transductive, miniImageNet", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=0.3, label="Inductive"),
                       Patch(facecolor=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3, label="Transductive")]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_ylim(30, 85)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_transductive")


# ---------- P6: 10-way results (Bateni Table 2) ----------

def plot_10way(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_10way")
    if not entries:
        print("No miniimagenet_10way results found, skipping.")
        return

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.65))

    inductive = sorted([e for e in entries if not e.transductive], key=lambda x: x.n_shot_test)
    transductive = sorted([e for e in entries if e.transductive], key=lambda x: x.n_shot_test)

    labels = [f"{e.n_shot_test}-shot" for e in inductive]
    x = np.arange(len(labels))
    width = 0.3

    bars1 = ax.bar(x - width / 2, [e.accuracy_mean * 100 for e in inductive], width,
                   yerr=[e.accuracy_ci95 * 100 for e in inductive], capsize=2,
                   color=BLUE, edgecolor=BLACK, linewidth=0.3,
                   error_kw={"linewidth": 0.5})
    bars2 = ax.bar(x + width / 2, [e.accuracy_mean * 100 for e in transductive], width,
                   yerr=[e.accuracy_ci95 * 100 for e in transductive], capsize=2,
                   color=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3,
                   error_kw={"linewidth": 0.5})

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("10-way accuracy (%)")
    ax.set_title("Inductive vs transductive, miniImageNet 10-way", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=0.3, label="Inductive"),
                       Patch(facecolor=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3, label="Transductive")]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_ylim(20, 75)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_10way")


# ---------- P7: Refinement steps sweep (Bateni Figure 6) ----------

def plot_refinement_steps(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "refinement_steps")
    if not entries:
        print("No refinement_steps results found, skipping.")
        return

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6))

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")

        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: x.n_refinement_steps,
        )
        steps = [e.n_refinement_steps for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]

        ax.errorbar(steps, means, yerr=ci95s, marker="o", markersize=3,
                    linewidth=1, capsize=2, color=BLUISH_GREEN,
                    markeredgecolor=BLACK, markeredgewidth=0.3,
                    elinewidth=0.5)

        # mark step 0 (inductive baseline) distinctly
        ax.axhline(y=means[0], color=BLUE, linewidth=0.5, linestyle="--", alpha=0.7)

        ax.set_xlabel("Max refinement steps")
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_title(f"{shot}-shot", fontweight="bold")
        ax.set_xticks(steps)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=BLUISH_GREEN, marker="o", markersize=3, linewidth=1, label="Transductive"),
        Line2D([0], [0], color=BLUE, linewidth=0.5, linestyle="--", label="Inductive baseline"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, output_dir, "refinement_steps")


# ---------- P8: CUB zero-shot (Snell Table 3) ----------

def plot_cub_zeroshot(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "cub_zeroshot")
    if not entries:
        print("No cub_zeroshot results found, skipping.")
        return

    _setup_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

    e = entries[0]

    # paper reference baselines (Table 3, Snell 2017)
    baselines = [
        ("ALE", 26.9),
        ("SJE\n(AlexNet)", 40.3),
        ("SJE\n(GoogLeNet)", 50.1),
        ("DS-SJE", 50.4),
        ("DA-SJE", 50.9),
    ]
    all_labels = [b[0] for b in baselines] + ["ProtoNet\n(ours)"]
    all_vals = [b[1] for b in baselines] + [e.accuracy_mean * 100]
    all_ci = [0.0] * len(baselines) + [e.accuracy_ci95 * 100]

    n_baselines = len(baselines)
    colors = [SKY_BLUE] * n_baselines + [VERMILLION]

    x = np.arange(len(all_labels))
    bars = ax.bar(x, all_vals, yerr=all_ci, width=0.6, capsize=2,
                  color=colors, edgecolor=BLACK, linewidth=0.3,
                  error_kw={"linewidth": 0.5})

    for bar, val in zip(bars, all_vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("50-way 0-shot accuracy (%)")
    ax.set_title("CUB-200 zero-shot classification", fontweight="bold")
    ax.set_ylim(0, 65)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SKY_BLUE, edgecolor=BLACK, linewidth=0.3, label="Prior work"),
        Patch(facecolor=VERMILLION, edgecolor=BLACK, linewidth=0.3, label="ProtoNet (ours)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    _save_fig(fig, output_dir, "cub_zeroshot")


# ---------- Dispatch ----------

class PlotType(Enum):
    OMNIGLOT_FEWSHOT = "omniglot_fewshot"
    MINIIMAGENET_FEWSHOT = "miniimagenet_fewshot"
    MINIIMAGENET_DISTANCE_WAY = "miniimagenet_distance_way"
    MINIIMAGENET_WAY_ABLATION = "miniimagenet_way_ablation"
    MINIIMAGENET_TRANSDUCTIVE = "miniimagenet_transductive"
    MINIIMAGENET_10WAY = "miniimagenet_10way"
    REFINEMENT_STEPS = "refinement_steps"
    CUB_ZEROSHOT = "cub_zeroshot"


PLOT_FUNCTIONS: dict[PlotType, Callable[[Path, Path], None]] = {
    PlotType.OMNIGLOT_FEWSHOT: plot_omniglot_fewshot,
    PlotType.MINIIMAGENET_FEWSHOT: plot_miniimagenet_fewshot,
    PlotType.MINIIMAGENET_DISTANCE_WAY: plot_distance_way_comparison,
    PlotType.MINIIMAGENET_WAY_ABLATION: plot_way_ablation,
    PlotType.MINIIMAGENET_TRANSDUCTIVE: plot_transductive,
    PlotType.MINIIMAGENET_10WAY: plot_10way,
    PlotType.REFINEMENT_STEPS: plot_refinement_steps,
    PlotType.CUB_ZEROSHOT: plot_cub_zeroshot,
}


def plot_all(results_dir: Path, output_dir: Path) -> None:
    for plot_type, plot_fn in PLOT_FUNCTIONS.items():
        print(f"\n--- {plot_type.value} ---")
        plot_fn(results_dir, output_dir)


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation results (Nature style)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with JSON results")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for plot images")
    parser.add_argument("--plot", type=str, default=None, help="Specific plot to generate (default: all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if args.plot:
        plot_type = PlotType(args.plot)
        PLOT_FUNCTIONS[plot_type](results_dir, output_dir)
    else:
        plot_all(results_dir, output_dir)


if __name__ == "__main__":
    main()
