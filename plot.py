from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM_TO_INCHES = 1.0 / 25.4
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MAX_HEIGHT_MM = 170

SINGLE_COL = SINGLE_COL_MM * MM_TO_INCHES
DOUBLE_COL = DOUBLE_COL_MM * MM_TO_INCHES
MAX_HEIGHT = MAX_HEIGHT_MM * MM_TO_INCHES

BLACK = "#000000"
SKY_BLUE = "#56b4e9"
BLUISH_GREEN = "#009e73"
YELLOW = "#f0e442"
BLUE = "#0072b2"
VERMILLION = "#d55e00"
REDDISH_PURPLE = "#cc79a7"

PALETTE = [BLUE, VERMILLION, BLUISH_GREEN, SKY_BLUE, REDDISH_PURPLE, YELLOW, BLACK]

P_SKY_BLUE = "#56b4e9"
P_VERMILLION = "#d55e00"
P_GREEN = "#009e73"
P_BLUE = "#0072b2"


def _setup_nature_style() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
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
        },
    )


def _setup_poster_style() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        },
    )


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
        entries.append(
            ResultEntry(
                benchmark=data["benchmark"],
                accuracy_mean=data["accuracy_mean"],
                accuracy_ci95=data["accuracy_ci95"],
                n_way_test=cfg["n_way_test"],
                n_shot_test=cfg["n_shot_test"],
                distance=cfg["distance"],
                train_way=cfg["train_way"],
                transductive=cfg["transductive"],
                n_refinement_steps=cfg.get("n_refinement_steps", 4),
            ),
        )
    return entries


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"Saved: {output_dir / name}.{{pdf,png}}")


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
    ci95s = [e.accuracy_ci95 * 100 for e in entries_sorted]

    x = np.arange(len(labels))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(
        x,
        means,
        yerr=ci95s,
        width=0.55,
        capsize=2,
        color=colors,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    for bar, val, ci in zip(bars, means, ci95s, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 0.15,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Omniglot few-shot classification", fontweight="bold")
    ax.set_ylim(90, 101)

    fig.tight_layout()
    _save_fig(fig, output_dir, "omniglot_fewshot")


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
    bars = ax.bar(
        x,
        means,
        yerr=ci95s,
        width=0.45,
        capsize=2,
        color=[BLUE, BLUISH_GREEN],
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    for bar, val, ci in zip(bars, means, ci95s, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("miniImageNet 5-way classification", fontweight="bold")
    ax.set_ylim(30, 85)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_fewshot")


def plot_poster_n_way_improves(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_distance_way")
    if not entries:
        print("No miniimagenet_distance_way results found, skipping.")
        return

    _setup_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

    edge_lw = 1.0
    err_lw = 1.5
    cap = 4
    ann_fs = 16
    width = 0.35

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.set_title(f"{label}   {shot}-shot", fontweight="bold", loc="left")

        shot_entries = [e for e in entries if e.n_shot_test == shot]
        train_ways = sorted({e.train_way for e in shot_entries})
        cosine_by_way = {e.train_way: e for e in shot_entries if e.distance == "cosine"}
        euclidean_by_way = {e.train_way: e for e in shot_entries if e.distance == "euclidean"}

        group_labels = [f"{w}-way" for w in train_ways]
        x = np.arange(len(train_ways))

        for offset, by_way, color in [
            (-width / 2, cosine_by_way, P_VERMILLION),
            (width / 2, euclidean_by_way, P_BLUE),
        ]:
            means = [by_way[w].accuracy_mean * 100 for w in train_ways]
            ci95s = [by_way[w].accuracy_ci95 * 100 for w in train_ways]
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=ci95s,
                capsize=cap,
                color=color,
                edgecolor=BLACK,
                linewidth=edge_lw,
                error_kw={"linewidth": err_lw},
            )
            for bar, val, ci in zip(bars, means, ci95s, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ci + 0.3,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=ann_fs,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_ylim(20, 80)

    legend_elements = [
        Patch(facecolor=P_VERMILLION, edgecolor=BLACK, linewidth=edge_lw, label="Cosine"),
        Patch(facecolor=P_BLUE, edgecolor=BLACK, linewidth=edge_lw, label="Euclidean"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save_fig(fig, output_dir, "poster_n_way_improves")


def plot_way_ablation(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_way_ablation")
    if not entries:
        print("No miniimagenet_way_ablation results found, skipping.")
        return

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6))

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: x.train_way,
        )
        ways = [e.train_way for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]

        ax.errorbar(
            ways,
            means,
            yerr=ci95s,
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            color=BLUE,
            markeredgecolor=BLACK,
            markeredgewidth=0.3,
            elinewidth=0.5,
        )
        ax.set_xlabel("Training classes per episode")
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_title(f"{shot}-shot", fontweight="bold")
        ax.set_xticks(ways)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_way_ablation")


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

    ind_means = [e.accuracy_mean * 100 for e in inductive]
    ind_ci = [e.accuracy_ci95 * 100 for e in inductive]
    trans_means = [e.accuracy_mean * 100 for e in transductive]
    trans_ci = [e.accuracy_ci95 * 100 for e in transductive]

    bars1 = ax.bar(
        x - width / 2,
        ind_means,
        width,
        yerr=ind_ci,
        capsize=2,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )
    bars2 = ax.bar(
        x + width / 2,
        trans_means,
        width,
        yerr=trans_ci,
        capsize=2,
        color=BLUISH_GREEN,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    for bars, ci_vals in [(bars1, ind_ci), (bars2, trans_ci)]:
        for bar, ci in zip(bars, ci_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.3,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("5-way accuracy (%)")
    ax.set_title("Inductive vs transductive, miniImageNet", fontweight="bold")

    legend_elements = [
        Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=0.3, label="Inductive"),
        Patch(facecolor=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3, label="Transductive"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_ylim(30, 85)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_transductive")


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

    ind_means = [e.accuracy_mean * 100 for e in inductive]
    ind_ci = [e.accuracy_ci95 * 100 for e in inductive]
    trans_means = [e.accuracy_mean * 100 for e in transductive]
    trans_ci = [e.accuracy_ci95 * 100 for e in transductive]

    bars1 = ax.bar(
        x - width / 2,
        ind_means,
        width,
        yerr=ind_ci,
        capsize=2,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )
    bars2 = ax.bar(
        x + width / 2,
        trans_means,
        width,
        yerr=trans_ci,
        capsize=2,
        color=BLUISH_GREEN,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    for bars, ci_vals in [(bars1, ind_ci), (bars2, trans_ci)]:
        for bar, ci in zip(bars, ci_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.3,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("10-way accuracy (%)")
    ax.set_title("Inductive vs transductive, miniImageNet 10-way", fontweight="bold")

    legend_elements = [
        Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=0.3, label="Inductive"),
        Patch(facecolor=BLUISH_GREEN, edgecolor=BLACK, linewidth=0.3, label="Transductive"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_ylim(20, 75)

    fig.tight_layout()
    _save_fig(fig, output_dir, "miniimagenet_10way")


def plot_refinement_steps(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "refinement_steps")
    if not entries:
        print("No refinement_steps results found, skipping.")
        return

    _setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.6))

    for ax, shot, label in zip(axes, [1, 5], ["a", "b"], strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: x.n_refinement_steps,
        )
        steps = [e.n_refinement_steps for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]

        ax.errorbar(
            steps,
            means,
            yerr=ci95s,
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            color=BLUISH_GREEN,
            markeredgecolor=BLACK,
            markeredgewidth=0.3,
            elinewidth=0.5,
        )

        ax.axhline(y=means[0], color=BLUE, linewidth=0.5, linestyle="--", alpha=0.7)

        ax.set_xlabel("Max refinement steps")
        ax.set_ylabel("5-way test accuracy (%)")
        ax.set_title(f"{shot}-shot", fontweight="bold")
        ax.set_xticks(steps)

    legend_elements = [
        Line2D([0], [0], color=BLUISH_GREEN, marker="o", markersize=3, linewidth=1, label="Transductive"),
        Line2D([0], [0], color=BLUE, linewidth=0.5, linestyle="--", label="Inductive baseline"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, output_dir, "refinement_steps")


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

    n_baselines = len(baselines)

    x = np.arange(len(all_labels))

    ax.bar(x[:n_baselines], all_vals[:n_baselines], width=0.6, color=SKY_BLUE, edgecolor=BLACK, linewidth=0.3)
    protonet_ci = e.accuracy_ci95 * 100
    ax.bar(
        x[n_baselines:],
        all_vals[n_baselines:],
        width=0.6,
        yerr=[protonet_ci],
        capsize=2,
        color=VERMILLION,
        edgecolor=BLACK,
        linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    for i, val in enumerate(all_vals):
        ci = protonet_ci if i >= n_baselines else 0.0
        ax.text(x[i], val + ci + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("50-way 0-shot accuracy (%)")
    ax.set_title("CUB-200 zero-shot classification", fontweight="bold")
    ax.set_ylim(0, 65)

    legend_elements = [
        Patch(facecolor=SKY_BLUE, edgecolor=BLACK, linewidth=0.3, label="Prior work"),
        Patch(facecolor=VERMILLION, edgecolor=BLACK, linewidth=0.3, label="ProtoNet (ours)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    _save_fig(fig, output_dir, "cub_zeroshot")


def plot_poster_reproduction(results_dir: Path, output_dir: Path) -> None:
    omniglot = load_results(results_dir, "omniglot_fewshot")
    mini = load_results(results_dir, "miniimagenet_fewshot")
    cub = load_results(results_dir, "cub_zeroshot")

    if not (omniglot and mini and cub):
        print("Missing results for poster reproduction plot, skipping.")
        return

    _setup_poster_style()

    our_omni = {(e.n_way_test, e.n_shot_test): e for e in omniglot}
    our_mini = {(e.n_way_test, e.n_shot_test): e for e in mini}
    our_cub_entry = cub[0]

    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 2.2], wspace=0.5)
    bar_w = 0.25
    ann_fs = 16
    edge_lw = 1.0
    err_lw = 1.5
    cap = 4

    def _draw_grouped_bars(
        ax: plt.Axes,
        x: np.ndarray,
        baseline: list[float],
        paper: list[float],
        ours: list[float],
        ours_ci: list[float],
        ann_offset: float,
    ) -> None:
        ax.bar(x - bar_w, baseline, bar_w, color=P_SKY_BLUE, edgecolor=BLACK, linewidth=edge_lw)
        ax.bar(x, paper, bar_w, color=P_VERMILLION, edgecolor=BLACK, linewidth=edge_lw)
        ax.bar(
            x + bar_w,
            ours,
            bar_w,
            yerr=ours_ci,
            capsize=cap,
            color=P_GREEN,
            edgecolor=BLACK,
            linewidth=edge_lw,
            error_kw={"linewidth": err_lw},
        )
        for i in range(len(x)):
            ax.text(
                x[i] - bar_w,
                baseline[i] + ann_offset,
                f"{baseline[i]:.1f}",
                ha="center",
                va="bottom",
                fontsize=ann_fs,
            )
            ax.text(x[i], paper[i] + ann_offset, f"{paper[i]:.1f}", ha="center", va="bottom", fontsize=ann_fs)
            ax.text(
                x[i] + bar_w,
                ours[i] + ours_ci[i] + ann_offset,
                f"{ours[i]:.1f}",
                ha="center",
                va="bottom",
                fontsize=ann_fs,
            )

    ax = fig.add_subplot(gs[0])
    ax.set_title("a   Omniglot", fontweight="bold", loc="left")

    omni_settings = [(5, 1), (5, 5)]
    x = np.arange(len(omni_settings))
    _draw_grouped_bars(
        ax,
        x,
        baseline=[98.1, 98.9],
        paper=[98.8, 99.7],
        ours=[our_omni[s].accuracy_mean * 100 for s in omni_settings],
        ours_ci=[our_omni[s].accuracy_ci95 * 100 for s in omni_settings],
        ann_offset=0.15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["1-shot", "5-shot"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(95, 100)

    ax = fig.add_subplot(gs[1])
    ax.set_title("b   miniImageNet", fontweight="bold", loc="left")

    mini_settings = [(5, 1), (5, 5)]
    xm = np.arange(len(mini_settings))
    _draw_grouped_bars(
        ax,
        xm,
        baseline=[43.4, 51.1],
        paper=[49.4, 68.2],
        ours=[our_mini[s].accuracy_mean * 100 for s in mini_settings],
        ours_ci=[our_mini[s].accuracy_ci95 * 100 for s in mini_settings],
        ann_offset=0.8,
    )
    ax.set_xticks(xm)
    ax.set_xticklabels(["1-shot", "5-shot"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 80)

    ax = fig.add_subplot(gs[2])
    ax.set_title("c   CUB zero-shot", fontweight="bold", loc="left")

    cub_labels = ["SJE\n(GNet)", "DA-\nSJE", "Paper", "Ours"]
    cub_vals = [50.1, 50.9, 54.6, our_cub_entry.accuracy_mean * 100]
    cub_colors = [P_SKY_BLUE, P_SKY_BLUE, P_VERMILLION, P_GREEN]
    cub_ci = our_cub_entry.accuracy_ci95 * 100

    xc = np.arange(len(cub_labels))
    ax.bar(xc[:3], cub_vals[:3], 0.55, color=cub_colors[:3], edgecolor=BLACK, linewidth=edge_lw)
    ax.bar(
        xc[3:],
        cub_vals[3:],
        0.55,
        yerr=[cub_ci],
        capsize=cap,
        color=[cub_colors[3]],
        edgecolor=BLACK,
        linewidth=edge_lw,
        error_kw={"linewidth": err_lw},
    )
    for i, val in enumerate(cub_vals):
        ci_off = cub_ci if i == 3 else 0.0
        ax.text(xc[i], val + ci_off + 0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=ann_fs)

    ax.set_xticks(xc)
    ax.set_xticklabels(cub_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 62)

    legend_elements = [
        Patch(facecolor=P_SKY_BLUE, edgecolor=BLACK, linewidth=edge_lw, label="Prior work"),
        Patch(facecolor=P_VERMILLION, edgecolor=BLACK, linewidth=edge_lw, label="ProtoNet (paper)"),
        Patch(facecolor=P_GREEN, edgecolor=BLACK, linewidth=edge_lw, label="ProtoNet (ours)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))

    fig.subplots_adjust(bottom=0.18)
    _save_fig(fig, output_dir, "poster_reproduction")


def _get_shot_series(
    entries: list[ResultEntry],
    shot: int,
) -> tuple[list[int], list[float], list[float]]:
    shot_entries = sorted(
        [e for e in entries if e.n_shot_test == shot],
        key=lambda e: e.n_refinement_steps,
    )
    steps = [e.n_refinement_steps for e in shot_entries]
    means = [e.accuracy_mean * 100 for e in shot_entries]
    ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]
    return steps, means, ci95s


def _align_baselines(ax_left: plt.Axes, v_left: float, ax_right: plt.Axes, v_right: float) -> None:
    lo, hi = ax_left.get_ylim()
    frac = (v_left - lo) / (hi - lo)
    r = ax_right.get_ylim()
    span = r[1] - r[0]
    new_lo = v_right - frac * span
    ax_right.set_ylim(new_lo, new_lo + span)


def plot_poster_refinement_steps(results_dir: Path, output_dir: Path) -> None:
    mini_entries = load_results(results_dir, "refinement_steps")
    omni_entries = load_results(results_dir, "omniglot_refinement_steps")

    if not (mini_entries and omni_entries):
        print("Missing refinement_steps results, skipping.")
        return

    _setup_poster_style()
    fig, (ax_omni, ax_mini) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

    dataset_specs: list[tuple[plt.Axes, str, list[ResultEntry], str]] = [
        (ax_omni, "Omniglot", omni_entries, "a"),
        (ax_mini, "miniImageNet", mini_entries, "b"),
    ]
    shot_styles: list[tuple[int, str, str]] = [
        (1, "o", P_GREEN),
        (5, "s", P_VERMILLION),
    ]

    for ax, dataset_name, entries, label in dataset_specs:
        for shot, marker, color in shot_styles:
            steps, means, ci95s = _get_shot_series(entries, shot)
            baseline = means[0]
            deltas = [m - baseline for m in means]

            ax.errorbar(
                steps,
                deltas,
                yerr=ci95s,
                marker=marker,
                markersize=10,
                linewidth=4,
                capsize=5,
                color=color,
                markeredgecolor=BLACK,
                markeredgewidth=1.0,
                elinewidth=2,
                label=f"{shot}-shot",
            )

        ax.axhline(y=0, color="grey", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("Refinement steps")
        ax.set_title(f"{label}   {dataset_name}", fontweight="bold", loc="left")
        ax.set_xticks(steps)

    ax_omni.set_ylabel("Accuracy gain (pp)")

    legend_elements = [
        Line2D([0], [0], color=P_GREEN, marker="o", markersize=10, linewidth=4, label="1-shot"),
        Line2D([0], [0], color=P_VERMILLION, marker="s", markersize=10, linewidth=4, label="5-shot"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--", alpha=0.5, label="Inductive baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save_fig(fig, output_dir, "poster_refinement_steps")


def plot_poster_embeddings(results_dir: Path, output_dir: Path) -> None:
    data_path = output_dir / "embedding_data.npz"
    if not data_path.exists():
        print(f"Missing {data_path}, skipping. Run embedding extraction first.")
        return

    _setup_poster_style()

    data = np.load(data_path, allow_pickle=True)
    tsne_2d = data["tsne_2d"]
    pca_2d = data["pca_2d"]
    labels = data["labels"].astype(int)
    point_types = data["point_types"]
    pca_var = data["pca_var"]

    class_colors = [P_BLUE, P_VERMILLION, P_GREEN, P_SKY_BLUE, REDDISH_PURPLE]

    fig, (ax_tsne, ax_pca) = plt.subplots(1, 2, figsize=(22, 9))

    for ax, coords, title in [
        (ax_tsne, tsne_2d, "t-SNE"),
        (ax_pca, pca_2d, f"PCA ({pca_var[0] + pca_var[1]:.0%} var.)"),
    ]:
        for c in range(5):
            s_mask = (labels == c) & (point_types == "support")
            q_mask = (labels == c) & (point_types == "query")
            p_mask = (labels == c) & (point_types == "prototype")

            ax.scatter(
                coords[q_mask, 0],
                coords[q_mask, 1],
                c=class_colors[c],
                marker="o",
                s=80,
                alpha=0.5,
                edgecolors="none",
            )
            ax.scatter(
                coords[s_mask, 0],
                coords[s_mask, 1],
                c=class_colors[c],
                marker="o",
                s=120,
                alpha=0.9,
                edgecolors=BLACK,
                linewidths=1.5,
            )
            ax.scatter(
                coords[p_mask, 0],
                coords[p_mask, 1],
                c=class_colors[c],
                marker="*",
                s=600,
                zorder=5,
                edgecolors=BLACK,
                linewidths=1.5,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markersize=10,
            linestyle="",
            alpha=0.9,
            markeredgecolor=BLACK,
            markeredgewidth=1.5,
            label="Support",
        ),
        Line2D([0], [0], marker="o", color="grey", markersize=8, linestyle="", alpha=0.5, label="Query"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="grey",
            markersize=18,
            linestyle="",
            markeredgecolor=BLACK,
            markeredgewidth=1.5,
            label="Prototype",
        ),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    _save_fig(fig, output_dir, "poster_embeddings")


def plot_poster_way_ablation(results_dir: Path, output_dir: Path) -> None:
    entries = load_results(results_dir, "miniimagenet_way_ablation")
    if not entries:
        print("No miniimagenet_way_ablation results found, skipping.")
        return

    _setup_poster_style()
    fig, ax = plt.subplots(figsize=(11, 9))

    shot_styles: list[tuple[int, str, str]] = [
        (1, "o", P_BLUE),
        (5, "s", P_VERMILLION),
    ]

    for shot, marker, color in shot_styles:
        shot_entries = sorted(
            [e for e in entries if e.n_shot_test == shot],
            key=lambda x: x.train_way,
        )
        ways = [e.train_way for e in shot_entries]
        means = [e.accuracy_mean * 100 for e in shot_entries]
        ci95s = [e.accuracy_ci95 * 100 for e in shot_entries]

        ax.errorbar(
            ways,
            means,
            yerr=ci95s,
            marker=marker,
            markersize=10,
            linewidth=4,
            capsize=5,
            color=color,
            markeredgecolor=BLACK,
            markeredgewidth=1.0,
            elinewidth=2,
            label=f"{shot}-shot",
        )

    ax.axvline(x=5, color="grey", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.text(5.5, ax.get_ylim()[0] + 1, "test ways", fontsize=20, color="grey", alpha=0.7)

    ax.set_xlabel("Training classes per episode")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Training way ablation", fontweight="bold")
    ax.set_xticks(ways)
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save_fig(fig, output_dir, "poster_way_ablation")


def plot_poster_scaling_ablation(results_dir: Path, output_dir: Path) -> None:
    _setup_poster_style()
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        },
    )

    distances = ["Euclidean", "Cosine", "L1", r"L$\infty$"]
    scalings = [r"$1$", r"$d^{-1/2}$", r"$d^{-1}$"]

    # (run1, run2) per cell — from grid training
    raw: list[list[tuple[float, float]]] = [
        # Euclidean
        [(60.97, 60.93), (66.65, 65.91), (65.16, 64.93)],
        # Cosine
        [(52.76, 52.13), (46.65, 49.35), (47.95, 48.24)],
        # L1
        [(59.09, 59.28), (66.80, 65.57), (55.07, 56.92)],
        # L∞
        [(48.47, 48.05), (37.08, 36.64), (36.52, 35.27)],
    ]

    means = np.array([[(a + b) / 2 for a, b in row] for row in raw])
    spreads = np.array([[abs(a - b) / 2 for a, b in row] for row in raw])

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(means, cmap="RdYlGn", aspect="auto", vmin=35, vmax=70)

    ax.set_xticks(range(len(scalings)))
    ax.set_xticklabels(scalings)
    ax.set_yticks(range(len(distances)))
    ax.set_yticklabels(distances)
    ax.set_xlabel("Scaling factor")

    for i in range(len(distances)):
        for j in range(len(scalings)):
            mean = means[i, j]
            spread = spreads[i, j]
            text_color = "white" if mean < 45 else "black"
            ax.text(
                j,
                i,
                f"{mean:.1f} ± {spread:.1f}%",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title("Distance scaling ablation", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    fig.tight_layout()
    _save_fig(fig, output_dir, "poster_scaling_ablation")


class PlotType(Enum):
    OMNIGLOT_FEWSHOT = "omniglot_fewshot"
    MINIIMAGENET_FEWSHOT = "miniimagenet_fewshot"
    POSTER_N_WAY_IMPROVES = "poster_n_way_improves"
    MINIIMAGENET_WAY_ABLATION = "miniimagenet_way_ablation"
    MINIIMAGENET_TRANSDUCTIVE = "miniimagenet_transductive"
    MINIIMAGENET_10WAY = "miniimagenet_10way"
    REFINEMENT_STEPS = "refinement_steps"
    CUB_ZEROSHOT = "cub_zeroshot"
    POSTER_REPRODUCTION = "poster_reproduction"
    POSTER_REFINEMENT_STEPS = "poster_refinement_steps"
    POSTER_EMBEDDINGS = "poster_embeddings"
    POSTER_WAY_ABLATION = "poster_way_ablation"
    POSTER_SCALING_ABLATION = "poster_scaling_ablation"


PLOT_FUNCTIONS: dict[PlotType, Callable[[Path, Path], None]] = {
    PlotType.OMNIGLOT_FEWSHOT: plot_omniglot_fewshot,
    PlotType.MINIIMAGENET_FEWSHOT: plot_miniimagenet_fewshot,
    PlotType.POSTER_N_WAY_IMPROVES: plot_poster_n_way_improves,
    PlotType.MINIIMAGENET_WAY_ABLATION: plot_way_ablation,
    PlotType.MINIIMAGENET_TRANSDUCTIVE: plot_transductive,
    PlotType.MINIIMAGENET_10WAY: plot_10way,
    PlotType.REFINEMENT_STEPS: plot_refinement_steps,
    PlotType.CUB_ZEROSHOT: plot_cub_zeroshot,
    PlotType.POSTER_REPRODUCTION: plot_poster_reproduction,
    PlotType.POSTER_REFINEMENT_STEPS: plot_poster_refinement_steps,
    PlotType.POSTER_EMBEDDINGS: plot_poster_embeddings,
    PlotType.POSTER_WAY_ABLATION: plot_poster_way_ablation,
    PlotType.POSTER_SCALING_ABLATION: plot_poster_scaling_ablation,
}


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
        for plot_type, plot_fn in PLOT_FUNCTIONS.items():
            print(f"\n--- {plot_type.value} ---")
            plot_fn(results_dir, output_dir)


if __name__ == "__main__":
    main()
