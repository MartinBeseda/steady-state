#!/usr/bin/env python3
"""
Publication-quality figures for the hybrid cluster-signature SSD recommender.

Purpose
-------
Run this script AFTER the hybrid recommender script has produced:

    tsfresh_targeted_clustering_results/
      HYBRID_CLUSTER_SIGNATURE_SSD_RECOMMENDATION/
        *_hybrid_model_summary.csv
        *_hybrid_best_unique_series_predictions.csv
        *_hybrid_best_model_feature_importances.csv
        *_hybrid_signature_feature_importance_share.csv
        *_separate_wins_counts.csv
        *_separate_wins_reason_counts.csv

and the clustering sweep has produced:

    tsfresh_targeted_clustering_results/
      tables/
        *_cluster_method_eval.csv

This script does NOT re-run the expensive ML experiment.
It recomputes final presentation/paper metrics from the saved outputs and creates
clean figures for a manuscript or meeting.

Default story:
--------------
The default comparison deliberately avoids emphasizing the tiny hybrid-vs-raw-RF
gain. It compares:

    Random baseline
    Majority KB baseline
    Hybrid TSFresh + cluster-signature recommender

If you later want the raw TSFresh RF ablation plot too, set:

    INCLUDE_TSFRESH_ONLY_IF_AVAILABLE = True
"""

from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)



# Needed for loading cached clustering objects saved from scripts where
# ClusteringResult lived in __main__. Without this, pd.read_pickle may fail with:
# AttributeError: Can't get attribute 'ClusteringResult' on <module '__main__' ...>
@dataclass
class ClusteringResult:
    variant: str
    config_name: str
    method: str
    labels: object
    embedding: object
    features: object
    series_keys: object
    data_normalized: object
    metrics: object


# ============================================================
# USER CONFIG
# ============================================================

BASE_OUTDIR = Path("tsfresh_targeted_clustering_results")
HYBRID_SUBDIR = "HYBRID_CLUSTER_SIGNATURE_SSD_RECOMMENDATION"
TABLES_SUBDIR = "tables"

# If None, the script auto-detects the prefix from *_hybrid_model_summary.csv.
PREFIX = None

PAPER_FIG_DIRNAME = "PAPER_FIGURES"

# Keep False for the main paper story if the raw RF improvement is too small.
INCLUDE_TSFRESH_ONLY_IF_AVAILABLE = False

# UMAP plotting from the cached clustering object.
# This does NOT re-run UMAP/HDBSCAN. It loads the existing cached result.
CONFIG_CACHE_SUBDIR = "configuration_cache"
UMAP_POINT_SIZE = 7
UMAP_ALPHA = 0.75

# Candidate locations of the older TSFresh-only recommender output, used only if
# INCLUDE_TSFRESH_ONLY_IF_AVAILABLE = True.
TSFRESH_ONLY_SUBDIR_CANDIDATES = [
    "SUPERVISED_SSD_RECOMMENDATION_FAST_SEPARATE_WINS",
    "SUPERVISED_SSD_RECOMMENDATION_FAST",
    "SUPERVISED_SSD_RECOMMENDATION_TWO_STAGE_SEARCH_FIXED",
]

DPI = 350
SAVE_PDF = True
SAVE_SVG = True
SAVE_EPS = True


# ============================================================
# STYLE HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, outpath: Path):
    fig.savefig(outpath.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    if SAVE_EPS:
        fig.savefig(outpath.with_suffix(".eps"), bbox_inches="tight", format="eps")
    plt.close(fig)


def prettify_axes(ax):
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_bars(ax, bars, fmt="{:.0f}", dy=3, fontsize=9):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def label_barh(ax, bars, fmt="{:.0f}", dx=3, fontsize=9):
    for b in bars:
        w = b.get_width()
        ax.annotate(
            fmt.format(w),
            xy=(w, b.get_y() + b.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
        )


def short_feature_name(s: str, max_len: int = 72) -> str:
    s = str(s)
    s = s.replace("value__", "")
    s = s.replace("__", " | ")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


# ============================================================
# FILE LOADING
# ============================================================

def auto_detect_prefix(results_dir: Path) -> str:
    files = sorted(results_dir.glob("*_hybrid_model_summary.csv"))
    if not files:
        raise FileNotFoundError(
            f"No '*_hybrid_model_summary.csv' found in {results_dir}. "
            "Run the hybrid recommender script first."
        )
    name = files[0].name
    return name.replace("_hybrid_model_summary.csv", "")


def load_inputs():
    results_dir = BASE_OUTDIR / HYBRID_SUBDIR
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing hybrid results directory: {results_dir}")

    prefix = PREFIX or auto_detect_prefix(results_dir)
    tables_dir = BASE_OUTDIR / TABLES_SUBDIR

    paths = {
        "summary": results_dir / f"{prefix}_hybrid_model_summary.csv",
        "unique": results_dir / f"{prefix}_hybrid_best_unique_series_predictions.csv",
        "importance": results_dir / f"{prefix}_hybrid_best_model_feature_importances.csv",
        "sig_share": results_dir / f"{prefix}_hybrid_signature_feature_importance_share.csv",
        "wins": results_dir / f"{prefix}_separate_wins_counts.csv",
        "reasons": results_dir / f"{prefix}_separate_wins_reason_counts.csv",
        "cluster_eval": tables_dir / f"{prefix}_cluster_method_eval.csv",
        "signature_defs": results_dir / f"{prefix}_hybrid_cluster_signature_definitions_by_split.csv",
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print("Warning: some optional files are missing:")
        for k in missing:
            print(f"  {k}: {paths[k]}")

    data = {}
    for k, p in paths.items():
        if p.exists():
            data[k] = pd.read_csv(p)
        else:
            data[k] = pd.DataFrame()

    return prefix, results_dir, data



def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def configuration_cache_path(prefix: str) -> Path:
    """
    Prefix is usually:
        all_timeseries_pca50_umap5_nn30_md0.0_mcs50
    Cached filename is:
        all_timeseries__pca50_umap5_nn30_md0.0_mcs50.pkl
    """
    parts = prefix.split("_", 2)
    if len(parts) >= 3 and parts[0] == "all" and parts[1] == "timeseries":
        variant = "all_timeseries"
        config = parts[2]
    elif len(parts) >= 4 and parts[0] == "steady" and parts[1] == "jagt" and parts[2] == "only":
        variant = "steady_jagt_only"
        config = parts[3]
    else:
        # Fallback: assume first token is variant and the rest is config.
        variant = parts[0]
        config = "_".join(parts[1:])

    return BASE_OUTDIR / CONFIG_CACHE_SUBDIR / f"{safe_name(variant)}__{safe_name(config)}.pkl"


def load_cached_clustering(prefix: str):
    path = configuration_cache_path(prefix)
    if not path.exists():
        print(f"Warning: cached clustering file not found: {path}")
        return None
    payload = pd.read_pickle(path)
    if "result" in payload:
        return payload["result"]
    print(f"Warning: cached clustering file has no 'result' key: {path}")
    return None

def find_best_hybrid(summary: pd.DataFrame) -> pd.Series:
    hybrids = summary[summary["model"].astype(str).str.startswith("hybrid_")].copy()
    if hybrids.empty:
        raise RuntimeError("No hybrid_* row found in hybrid_model_summary.csv")
    hybrids = hybrids.sort_values(
        ["accuracy", "balanced_accuracy", "cp_f1"],
        ascending=[False, False, False],
    )
    return hybrids.iloc[0]


def try_load_tsfresh_only_accuracy(prefix: str):
    """
    Optional: load the older TSFresh-only unique-series result if available.
    Returns (accuracy_percent, correct_count) or (None, None).
    """
    if not INCLUDE_TSFRESH_ONLY_IF_AVAILABLE:
        return None, None

    for subdir in TSFRESH_ONLY_SUBDIR_CANDIDATES:
        d = BASE_OUTDIR / subdir
        if not d.exists():
            continue

        candidates = list(d.glob("*unique*predictions*.csv")) + list(d.glob("*best_unique_series_predictions.csv"))
        for p in candidates:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if "correct_prediction" in df.columns:
                acc = 100.0 * float(df["correct_prediction"].mean())
                correct = int(df["correct_prediction"].sum())
                return acc, correct

    return None, None


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_unique_metrics(unique: pd.DataFrame) -> dict:
    y_true = unique["true_cp_win"].astype(int).values
    y_pred = unique["pred_cp_win"].astype(int).values

    return {
        "n": int(len(y_true)),
        "true_cp": int(y_true.sum()),
        "true_kb": int((y_true == 0).sum()),
        "pred_cp": int(y_pred.sum()),
        "pred_kb": int((y_pred == 0).sum()),
        "correct": int((y_true == y_pred).sum()),
        "incorrect": int((y_true != y_pred).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cp_precision": precision_score(y_true, y_pred, zero_division=0),
        "cp_recall": recall_score(y_true, y_pred, zero_division=0),
        "cp_f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_confusion(unique: pd.DataFrame) -> pd.DataFrame:
    y_true = unique["true_cp_win"].astype(int).values
    y_pred = unique["pred_cp_win"].astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # rows: actual KB, actual CP; columns: pred KB, pred CP
    return pd.DataFrame(
        cm,
        index=["Actual KB-KSSD", "Actual CP-SSD"],
        columns=["Predicted KB-KSSD", "Predicted CP-SSD"],
    )


# ============================================================
# FIGURES
# ============================================================

def fig_winner_distribution(unique: pd.DataFrame, outdir: Path):
    counts = pd.DataFrame([
        {"category": "CP-SSD", "count": int((unique["true_cp_win"] == 1).sum())},
        {"category": "KB-KSSD", "count": int((unique["true_cp_win"] == 0).sum())},
    ])
    counts.to_csv(outdir / "fig01_winner_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(counts["category"], counts["count"])
    label_bars(ax, bars)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title("Actual SSD winners")
    ax.set_ylim(0, max(counts["count"]) * 1.18)
    prettify_axes(ax)
    savefig(fig, outdir / "fig01_winner_distribution")


def fig_recommendation_summary(unique: pd.DataFrame, outdir: Path):
    m = compute_unique_metrics(unique)

    # Preserve the numerical outputs in a CSV file.
    counts = pd.DataFrame([
        {"category": "Actual CP-SSD wins", "count": m["true_cp"]},
        {"category": "Actual KB-KSSD wins", "count": m["true_kb"]},
        {"category": "Predicted CP-SSD", "count": m["pred_cp"]},
        {"category": "Predicted KB-KSSD", "count": m["pred_kb"]},
        {"category": "Correct predictions", "count": m["correct"]},
        {"category": "Wrong predictions", "count": m["incorrect"]},
    ])
    counts.to_csv(
        outdir / "fig02_recommendation_summary.csv",
        index=False,
    )

    methods = ["CP-SSD", "KB-KSSD"]

    actual = np.array([
        m["true_cp"],
        m["true_kb"],
    ])

    predicted = np.array([
        m["pred_cp"],
        m["pred_kb"],
    ])

    x = np.arange(len(methods))
    width = 0.25

    # Muted publication-style palette.
    cp_actual_color = "#496F9E"
    cp_predicted_color = "#C9D9EE"

    kb_actual_color = "#B84A55"
    kb_predicted_color = "#E9BCC1"

    line_color = "#A52A2A"  # dim red / brown-red

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # Draw bars separately so that CP-SSD and KB-KSSD use their own colors.
    cp_actual_bar = ax.bar(
        x[0] - width / 2,
        actual[0],
        width,
        color=cp_actual_color,
        edgecolor="#1F1F1F",
        linewidth=0.8,
        label="Actual winner",
        zorder=3,
    )

    cp_predicted_bar = ax.bar(
        x[0] + width / 2,
        predicted[0],
        width,
        color=cp_predicted_color,
        edgecolor=cp_actual_color,
        linewidth=0.8,
        hatch="//",
        label="Predicted winner",
        zorder=3,
    )

    kb_actual_bar = ax.bar(
        x[1] - width / 2,
        actual[1],
        width,
        color=kb_actual_color,
        edgecolor="#1F1F1F",
        linewidth=0.8,
        zorder=3,
    )

    kb_predicted_bar = ax.bar(
        x[1] + width / 2,
        predicted[1],
        width,
        color=kb_predicted_color,
        edgecolor=kb_actual_color,
        linewidth=0.8,
        hatch="//",
        zorder=3,
    )

    # Horizontal lines for global prediction outcomes.
    correct_line = ax.axhline(
        y=m["correct"],
        color=line_color,
        linestyle=(0, (5, 2)),
        linewidth=1.8,
        label=f'Correct predictions ({m["correct"]})',
        zorder=2,
    )

    incorrect_line = ax.axhline(
        y=m["incorrect"],
        color=line_color,
        linestyle=(0, (1.2, 2.2)),
        linewidth=1.8,
        label=f'Wrong predictions ({m["incorrect"]})',
        zorder=2,
    )

    # Bar value labels, matched to each method's color.
    bar_groups = [
        (cp_actual_bar, cp_actual_color),
        (cp_predicted_bar, cp_actual_color),
        (kb_actual_bar, "#9F1722"),
        (kb_predicted_bar, "#9F1722"),
    ]

    for bars, text_color in bar_groups:
        ax.bar_label(
            bars,
            labels=[
                f"{int(bar.get_height())}"
                for bar in bars
            ],
            padding=5,
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

    # Value labels at the right side of the horizontal lines.
    xmax = x[-1] + 0.47

    ax.text(
        xmax,
        m["correct"],
        f'{m["correct"]}',
        color=line_color,
        fontsize=10,
        fontweight="semibold",
        va="center",
        ha="left",
    )

    ax.text(
        xmax,
        m["incorrect"],
        f'{m["incorrect"]}',
        color=line_color,
        fontsize=10,
        fontweight="semibold",
        va="center",
        ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        methods,
        fontsize=11,
    )

    ax.set_ylabel(
        "Number of JAGT time series",
        fontsize=11,
    )

    ymax = max(
        actual.max(),
        predicted.max(),
        m["correct"],
        m["incorrect"],
    )

    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlim(-0.48, xmax + 0.12)

    # Subtle horizontal grid.
    ax.grid(
        axis="y",
        linestyle=(0, (4, 3)),
        linewidth=0.7,
        color="#B8B8B8",
        alpha=0.55,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    # Minimal axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        labelsize=10,
        width=0.8,
    )

    # Custom legend patches preserve the intended appearance.
    from matplotlib.patches import Patch

    actual_patch = Patch(
        facecolor=cp_actual_color,
        edgecolor="#1F1F1F",
        linewidth=0.8,
        label="Actual winner",
    )

    predicted_patch = Patch(
        facecolor=cp_predicted_color,
        edgecolor=cp_actual_color,
        linewidth=0.8,
        hatch="//",
        label="Predicted winner",
    )

    handles = [
        actual_patch,
        predicted_patch,
        correct_line,
        incorrect_line,
    ]

    labels = [
        "Actual winner",
        "Predicted winner",
        f'Correct predictions ({m["correct"]})',
        f'Wrong predictions ({m["incorrect"]})',
    ]

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        facecolor="white",
        framealpha=1.0,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.5,
        borderpad=0.7,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.96,
        top=0.96,
        bottom=0.25,
    )

    savefig(
        fig,
        outdir / "fig02_recommendation_summary",
    )


def fig_correct_false_by_method(unique: pd.DataFrame, outdir: Path):
    df = unique.copy()
    rows = [
        {
            "category": "Correct CP-SSD",
            "count": int(((df["true_cp_win"] == 1) & (df["pred_cp_win"] == 1)).sum()),
        },
        {
            "category": "False CP-SSD",
            "count": int(((df["true_cp_win"] == 0) & (df["pred_cp_win"] == 1)).sum()),
        },
        {
            "category": "Correct KB-KSSD",
            "count": int(((df["true_cp_win"] == 0) & (df["pred_cp_win"] == 0)).sum()),
        },
        {
            "category": "False KB-KSSD",
            "count": int(((df["true_cp_win"] == 1) & (df["pred_cp_win"] == 0)).sum()),
        },
    ]
    counts = pd.DataFrame(rows)
    counts.to_csv(outdir / "fig03_correct_false_by_method.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(counts["category"], counts["count"])
    label_bars(ax, bars)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title("Correct and false SSD recommendations")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts["category"], rotation=20, ha="right")
    ax.set_ylim(0, max(counts["count"]) * 1.20)
    prettify_axes(ax)
    savefig(fig, outdir / "fig03_correct_false_by_method")


def fig_confusion_matrix(unique: pd.DataFrame, outdir: Path):
    cm = compute_confusion(unique)
    cm.to_csv(outdir / "fig04_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm.values)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(cm.columns, rotation=20, ha="right")
    ax.set_yticklabels(cm.index)

    for i in range(2):
        row_sum = cm.values[i].sum()
        for j in range(2):
            value = int(cm.values[i, j])
            pct = 100.0 * value / row_sum if row_sum else 0.0
            ax.text(
                j,
                i,
                f"{value}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
            )

    ax.set_title("Hybrid recommender confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, outdir / "fig04_confusion_matrix")


def fig_accuracy_vs_baselines(
    unique: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    prefix: str,
):
    metrics = compute_unique_metrics(unique)
    best = find_best_hybrid(summary)

    random_acc = 50.0
    majority_acc = (
        100.0
        * max(metrics["true_cp"], metrics["true_kb"])
        / metrics["n"]
    )
    hybrid_holdout = 100.0 * float(best["accuracy"])
    hybrid_unique = 100.0 * metrics["accuracy"]

    # Preserve all numerical values in the CSV output.
    df = pd.DataFrame([
        {
            "method": "Random baseline",
            "accuracy": random_acc,
            "display_type": "reference line",
        },
        {
            "method": "Majority baseline",
            "accuracy": majority_acc,
            "display_type": "reference line",
        },
        {
            "method": "Mean holdout accuracy",
            "accuracy": hybrid_holdout,
            "display_type": "bar",
        },
        {
            "method": "Aggregated unique-series accuracy",
            "accuracy": hybrid_unique,
            "display_type": "bar",
        },
    ])

    df.to_csv(
        outdir / "fig05_accuracy_vs_baselines.csv",
        index=False,
    )

    # Muted publication-style palette.
    holdout_color = "#496F9E"
    unique_color = "#B84A55"

    holdout_edge = "#294E78"
    unique_edge = "#8E2F39"

    random_line_color = "#777777"
    majority_line_color = "#8B3A3A"

    labels = [
        "Mean holdout\naccuracy",
        "Aggregated unique-series\naccuracy",
    ]
    values = [
        hybrid_holdout,
        hybrid_unique,
    ]

    # Keep the two recommender results visually grouped.
    x = np.array([0.82, 1.38])
    width = 0.42

    fig, ax = plt.subplots(figsize=(7.4, 4.7))

    holdout_bar = ax.bar(
        x[0],
        values[0],
        width=width,
        color=holdout_color,
        edgecolor=holdout_edge,
        linewidth=0.9,
        label="Mean holdout accuracy",
        zorder=3,
    )

    unique_bar = ax.bar(
        x[1],
        values[1],
        width=width,
        color=unique_color,
        edgecolor=unique_edge,
        linewidth=0.9,
        hatch="//",
        label="Aggregated unique-series accuracy",
        zorder=3,
    )

    # Baselines as reference lines rather than bars.
    random_line = ax.axhline(
        random_acc,
        color=random_line_color,
        linestyle=(0, (2, 2)),
        linewidth=1.5,
        label=f"Random baseline ({random_acc:.1f}%)",
        zorder=1,
    )

    majority_line = ax.axhline(
        majority_acc,
        color=majority_line_color,
        linestyle=(0, (6, 2)),
        linewidth=1.7,
        label=f"Majority baseline ({majority_acc:.1f}%)",
        zorder=2,
    )

    # Numerical labels above bars.
    ax.bar_label(
        holdout_bar,
        labels=[f"{hybrid_holdout:.1f}%"],
        padding=5,
        fontsize=10,
        fontweight="bold",
        color=holdout_edge,
    )

    ax.bar_label(
        unique_bar,
        labels=[f"{hybrid_unique:.1f}%"],
        padding=5,
        fontsize=10,
        fontweight="bold",
        color=unique_edge,
    )

    # Baseline values at the right edge.
    xmax = 1.88

    ax.text(
        xmax,
        random_acc,
        f"{random_acc:.1f}%",
        color=random_line_color,
        fontsize=9,
        fontweight="semibold",
        va="center",
        ha="left",
    )

    ax.text(
        xmax,
        majority_acc,
        f"{majority_acc:.1f}%",
        color=majority_line_color,
        fontsize=9,
        fontweight="semibold",
        va="center",
        ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=11,
    )

    ax.set_ylim(0, 100)
    ax.set_xlim(0.25, 2.08)

    ax.grid(
        axis="y",
        linestyle=(0, (4, 3)),
        linewidth=0.7,
        color="#B8B8B8",
        alpha=0.5,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        labelsize=10,
        width=0.8,
    )

    # Legend only explains the two baseline lines.
    ax.legend(
        handles=[
            random_line,
            majority_line,
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=2.8,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.95,
        top=0.96,
        bottom=0.22,
    )

    savefig(
        fig,
        outdir / "fig05_accuracy_vs_baselines",
    )


def fig_metrics_summary(unique: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    best = find_best_hybrid(summary)
    rows = pd.DataFrame([
        {"metric": "Accuracy", "score": 100.0 * float(best["accuracy"])},
        {"metric": "Balanced accuracy", "score": 100.0 * float(best["balanced_accuracy"])},
        {"metric": "CP precision", "score": 100.0 * float(best["cp_precision"])},
        {"metric": "CP recall", "score": 100.0 * float(best["cp_recall"])},
        {"metric": "CP F1", "score": 100.0 * float(best["cp_f1"])},
    ])
    rows.to_csv(outdir / "fig06_metric_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(rows["metric"], rows["score"])
    label_bars(ax, bars, fmt="{:.1f}%")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hybrid recommender holdout metrics")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows["metric"], rotation=18, ha="right")
    prettify_axes(ax)
    savefig(fig, outdir / "fig06_metric_summary")


def fig_feature_importance(importance: pd.DataFrame, outdir: Path, top_n: int = 20):
    if importance.empty:
        print("Skipping feature importance figure: missing input.")
        return

    df = importance.head(top_n).copy()
    df["short_feature"] = df["feature"].apply(short_feature_name)
    df["display_feature"] = np.where(
        df["is_signature_feature"].astype(bool),
        df["short_feature"] + " [signature]",
        df["short_feature"],
    )
    df = df.sort_values("mean_importance", ascending=True)
    df.to_csv(outdir / "fig07_top_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    bars = ax.barh(df["display_feature"], df["mean_importance"])
    label_barh(ax, bars, fmt="{:.3f}", fontsize=7)
    ax.set_xlabel("Mean feature importance")
    ax.set_title(f"Top {top_n} features of the hybrid recommender")
    prettify_axes(ax)
    savefig(fig, outdir / "fig07_top_feature_importance")


def fig_signature_importance_share(importance: pd.DataFrame, outdir: Path):
    if importance.empty or "is_signature_feature" not in importance.columns:
        print("Skipping signature importance share: missing input.")
        return

    df = importance.copy()
    df["feature_type"] = np.where(
        df["is_signature_feature"].astype(bool),
        "Cluster-signature features",
        "Direct TSFresh features",
    )
    share = (
        df.groupby("feature_type", as_index=False)
        .agg(total_importance=("mean_importance", "sum"))
    )
    share["percentage"] = 100.0 * share["total_importance"] / share["total_importance"].sum()
    share.to_csv(outdir / "fig08_signature_importance_share.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(share["feature_type"], share["percentage"])
    label_bars(ax, bars, fmt="{:.1f}%")
    ax.set_ylabel("Share of total importance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Contribution of cluster-signature features")
    ax.set_xticks(np.arange(len(share)))
    ax.set_xticklabels(share["feature_type"], rotation=16, ha="right")
    prettify_axes(ax)
    savefig(fig, outdir / "fig08_signature_importance_share")


def fig_cluster_specialization(cluster_eval: pd.DataFrame, outdir: Path, top_n: int = 16):
    if cluster_eval.empty:
        print("Skipping cluster specialization: missing cluster_eval.")
        return

    required = {"cluster", "n_jagt", "cp_best_wins_vs_kb", "kb_wins_vs_cp_best"}
    if not required.issubset(set(cluster_eval.columns)):
        print("Skipping cluster specialization: cluster_eval lacks required columns.")
        return

    df = cluster_eval.copy()
    df = df[df["cluster"] != -1].copy()
    df = df[df["n_jagt"] > 0].copy()
    df = df.sort_values("n_jagt", ascending=False).head(top_n)

    df["cluster_label"] = "C" + df["cluster"].astype(int).astype(str)
    df = df.sort_values("n_jagt", ascending=True)
    df.to_csv(outdir / "fig09_cluster_specialization.csv", index=False)

    y = np.arange(len(df))
    cp = df["cp_best_wins_vs_kb"].astype(float).values
    kb = df["kb_wins_vs_cp_best"].astype(float).values

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.barh(y, kb, label="KB-KSSD wins")
    ax.barh(y, cp, left=kb, label="CP-SSD wins")
    ax.set_yticks(y)
    ax.set_yticklabels(df["cluster_label"])
    ax.set_xlabel("Number of labelled JAGT series")
    ax.set_title("SSD preference by cluster")
    ax.legend(frameon=False)
    prettify_axes(ax)
    savefig(fig, outdir / "fig09_cluster_specialization")


def fig_signature_model_ranking(summary: pd.DataFrame, outdir: Path, top_n: int = 12):
    hybrids = summary[summary["model"].astype(str).str.startswith("hybrid_")].copy()
    if hybrids.empty:
        print("Skipping model ranking: no hybrid rows.")
        return

    hybrids["label"] = (
        hybrids["model"].astype(str).str.replace("hybrid_", "", regex=False)
        + " | "
        + hybrids["feature_selection"].astype(str)
        + " | tsf="
        + hybrids["base_top_n"].astype(int).astype(str)
        + " | sig="
        + hybrids["signature_top_k"].astype(int).astype(str)
    )

    df = hybrids.sort_values("accuracy", ascending=True).tail(top_n)
    df.to_csv(outdir / "fig10_hybrid_model_ranking.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bars = ax.barh(df["label"], 100.0 * df["accuracy"])
    label_barh(ax, bars, fmt="{:.1f}%", fontsize=8)
    ax.set_xlabel("Mean holdout accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Best hybrid recommender configurations")
    prettify_axes(ax)
    savefig(fig, outdir / "fig10_hybrid_model_ranking")


def fig_roc_pr_curves(unique: pd.DataFrame, outdir: Path):
    if "pred_cp_rate" not in unique.columns:
        print("Skipping ROC/PR curves: pred_cp_rate missing.")
        return

    y_true = unique["true_cp_win"].astype(int).values
    score = unique["pred_cp_rate"].astype(float).values

    # ROC
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(outdir / "fig11_roc_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve from repeated-split vote score")
    ax.legend(frameon=False)
    prettify_axes(ax)
    savefig(fig, outdir / "fig11_roc_curve")

    # Precision-recall
    precision, recall, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)

    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_df.to_csv(outdir / "fig12_precision_recall_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision--recall curve for CP-SSD recommendation")
    ax.legend(frameon=False)
    prettify_axes(ax)
    savefig(fig, outdir / "fig12_precision_recall_curve")



# ============================================================
# UMAP AND TIKZ WORKFLOW EXPORTS
# ============================================================

def fig_umap_clusters(prefix: str, outdir: Path, unique: pd.DataFrame | None = None):
    """
    Recreate the UMAP cluster plot from the cached clustering object.

    This does not re-run PCA/UMAP/HDBSCAN. It uses result.embedding and result.labels
    from configuration_cache/*.pkl.
    """
    result = load_cached_clustering(prefix)
    if result is None:
        return

    emb = np.asarray(result.embedding)
    labels = np.asarray(result.labels)
    if emb.ndim != 2 or emb.shape[1] < 2:
        print("Skipping UMAP plot: cached embedding does not have at least 2 columns.")
        return

    plot_df = pd.DataFrame({
        "umap1": emb[:, 0],
        "umap2": emb[:, 1],
        "cluster": labels.astype(int),
        "key": list(result.series_keys),
    })

    plot_df.to_csv(outdir / "fig_umap_clusters_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    noise = plot_df[plot_df["cluster"] == -1]
    clustered = plot_df[plot_df["cluster"] != -1]

    if not clustered.empty:
        sc = ax.scatter(
            clustered["umap1"],
            clustered["umap2"],
            c=clustered["cluster"],
            s=UMAP_POINT_SIZE,
            alpha=UMAP_ALPHA,
            linewidths=0,
            cmap="tab20",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cluster")
    if not noise.empty:
        ax.scatter(
            noise["umap1"],
            noise["umap2"],
            s=max(UMAP_POINT_SIZE - 2, 2),
            alpha=0.25,
            linewidths=0,
            marker=".",
            label="Noise",
        )
        ax.legend(frameon=False, loc="best")

    # Optional overlay of labelled JAGT points if unique predictions are available.
    if unique is not None and not unique.empty and "key" in unique.columns:
        jkeys = set(unique["key"].astype(str))
        jagt = plot_df[plot_df["key"].astype(str).isin(jkeys)]
        if not jagt.empty:
            ax.scatter(
                jagt["umap1"],
                jagt["umap2"],
                s=22,
                facecolors="none",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.55,
                label="JAGT",
            )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP projection of TSFresh feature space")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, outdir / "fig_umap_clusters")


def export_tikz_workflow_figures(outdir: Path):
    """
    Export TikZ snippets for the two 'manual' workflow diagrams:
      1. TSFresh feature extraction.
      2. Hybrid SSD recommendation.

    These are meant to be pasted directly into the manuscript.
    They are generated as .tex snippets, not compiled into EPS.
    """
    tsfresh = r"""% Requires:
% \usepackage{tikz}
% \usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit}

\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=0.9cm,
    box/.style={rectangle, rounded corners, draw, align=center, minimum width=2.7cm, minimum height=0.75cm, font=\small},
    arrow/.style={-{Latex[length=2mm]}, thick}
]
\node[box] (raw) {Raw benchmark\\time series};
\node[box, right=of raw] (tsfresh) {TSFresh\\descriptors};
\node[box, right=of tsfresh] (matrix) {Feature\\matrix};
\node[box, below=of tsfresh] (families) {Autocorrelation\\Entropy\\Peaks\\Quantiles\\FFT\\Change statistics};

\draw[arrow] (raw) -- (tsfresh);
\draw[arrow] (tsfresh) -- (matrix);
\draw[arrow] (families) -- (tsfresh);
\end{tikzpicture}
\caption{Overview of the TSFresh feature extraction process.}
\label{fig:tsfresh-extraction}
\end{figure}
"""
    workflow = r"""% Requires:
% \usepackage{tikz}
% \usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit}

\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=0.95cm and 1.05cm,
    box/.style={rectangle, rounded corners, draw, align=center, minimum width=3.05cm, minimum height=0.75cm, font=\small},
    widebox/.style={rectangle, rounded corners, draw, align=center, minimum width=5.6cm, minimum height=0.85cm, font=\small},
    arrow/.style={-{Latex[length=2mm]}, thick}
]
\node[box] (input) {Benchmark\\time series};
\node[box, below=of input] (tsfresh) {TSFresh\\feature extraction};

\node[box, below left=of tsfresh] (pca) {PCA};
\node[box, below=of pca] (umap) {UMAP};
\node[box, below=of umap] (hdbscan) {HDBSCAN\\clusters};
\node[box, below=of hdbscan] (signatures) {Cluster signatures\\prominent features};

\node[box, below right=of tsfresh] (selection) {Feature selection\\mutual information};
\node[widebox, below=3.7cm of tsfresh] (hybrid) {Hybrid feature vector\\selected TSFresh features + signature similarities};
\node[box, below=of hybrid] (rf) {Random Forest\\recommender};
\node[box, below left=of rf] (cp) {Recommend\\CP-SSD};
\node[box, below right=of rf] (kb) {Recommend\\KB-KSSD};

\draw[arrow] (input) -- (tsfresh);
\draw[arrow] (tsfresh) -- (pca);
\draw[arrow] (pca) -- (umap);
\draw[arrow] (umap) -- (hdbscan);
\draw[arrow] (hdbscan) -- (signatures);
\draw[arrow] (tsfresh) -- (selection);
\draw[arrow] (selection) -- (hybrid);
\draw[arrow] (signatures) -- (hybrid);
\draw[arrow] (hybrid) -- (rf);
\draw[arrow] (rf) -- (cp);
\draw[arrow] (rf) -- (kb);
\end{tikzpicture}
\caption{Overview of the hybrid SSD recommendation framework. Existing clustering results are used to derive cluster-signature similarity features, which are combined with selected TSFresh descriptors and passed to a supervised recommender.}
\label{fig:workflow}
\end{figure*}
"""
    (outdir / "TIKZ_fig_tsfresh_extraction.tex").write_text(tsfresh)
    (outdir / "TIKZ_fig_hybrid_recommender_workflow.tex").write_text(workflow)


# ============================================================
# TABLE EXPORTS
# ============================================================

def export_summary_tables(prefix: str, data: dict, outdir: Path):
    unique = data["unique"]
    summary = data["summary"]
    importance = data["importance"]

    m = compute_unique_metrics(unique)
    best = find_best_hybrid(summary)

    perf = pd.DataFrame([
        {
            "metric": "Unique-series accuracy",
            "value": m["accuracy"],
            "percentage": 100.0 * m["accuracy"],
        },
        {
            "metric": "Mean holdout accuracy",
            "value": float(best["accuracy"]),
            "percentage": 100.0 * float(best["accuracy"]),
        },
        {
            "metric": "Mean holdout balanced accuracy",
            "value": float(best["balanced_accuracy"]),
            "percentage": 100.0 * float(best["balanced_accuracy"]),
        },
        {
            "metric": "CP precision",
            "value": float(best["cp_precision"]),
            "percentage": 100.0 * float(best["cp_precision"]),
        },
        {
            "metric": "CP recall",
            "value": float(best["cp_recall"]),
            "percentage": 100.0 * float(best["cp_recall"]),
        },
        {
            "metric": "CP F1",
            "value": float(best["cp_f1"]),
            "percentage": 100.0 * float(best["cp_f1"]),
        },
    ])
    perf.to_csv(outdir / "table_main_performance_metrics.csv", index=False)

    model_line = pd.DataFrame([{
        "best_model": best["model"],
        "feature_selection": best["feature_selection"],
        "base_top_n": int(best["base_top_n"]),
        "signature_top_k": int(best["signature_top_k"]),
        "mean_holdout_accuracy": float(best["accuracy"]),
        "unique_accuracy": m["accuracy"],
        "n_correct_unique": m["correct"],
        "n_total_unique": m["n"],
    }])
    model_line.to_csv(outdir / "table_best_model_configuration.csv", index=False)

    if not importance.empty:
        importance.head(30).to_csv(outdir / "table_top30_feature_importances.csv", index=False)

    latex = []
    latex.append("% Auto-generated figure includes for the paper")
    latex.append("% Adjust filenames/widths as needed.")
    latex.append("")
    for i in range(1, 13):
        matches = sorted(outdir.glob(f"fig{i:02d}_*.pdf"))
        for p in matches:
            stem = p.stem
            latex.append(r"\begin{figure}[t]")
            latex.append(r"\centering")
            latex.append(rf"\includegraphics[width=\linewidth]{{{p.as_posix()}}}")
            latex.append(rf"\caption{{TODO caption for {stem}.}}")
            latex.append(rf"\label{{fig:{stem}}}")
            latex.append(r"\end{figure}")
            latex.append("")
    (outdir / "latex_figure_placeholders.tex").write_text("\n".join(latex))



def export_manuscript_figure_map(outdir: Path):
    """
    Writes a direct mapping between manuscript placeholders and generated figure names.
    Use .eps for Elsevier submission if you prefer EPS, or .pdf/.png during drafting.
    """
    rows = [
        {
            "manuscript_label": "fig:tsfresh-extraction",
            "section": "Feature Extraction",
            "replace_placeholder_with": "fig_workflow_tsfresh_extraction",
            "recommended_file": "TIKZ_fig_tsfresh_extraction.tex",
            "status": "tikz snippet",
            "note": "Generated as TikZ snippet; paste it directly into the manuscript, or compile separately if you need EPS.",
        },
        {
            "manuscript_label": "fig:umap-clusters",
            "section": "Clustering Analysis",
            "replace_placeholder_with": "fig_umap_clusters",
            "recommended_file": "fig_umap_clusters.eps",
            "status": "generated",
            "note": "UMAP plot recreated from cached clustering result; clustering is not re-run.",
        },
        {
            "manuscript_label": "fig:cluster-signatures",
            "section": "Cluster signatures",
            "replace_placeholder_with": "fig09_cluster_specialization",
            "recommended_file": "fig09_cluster_specialization.eps",
            "status": "generated",
            "note": "This is the best currently generated proxy for cluster signatures: cluster-wise CP vs KB specialization.",
        },
        {
            "manuscript_label": "fig:workflow",
            "section": "SSD Recommendation Framework",
            "replace_placeholder_with": "fig_workflow_hybrid_recommender",
            "recommended_file": "TIKZ_fig_hybrid_recommender_workflow.tex",
            "status": "tikz snippet",
            "note": "Generated as TikZ snippet; paste it directly into the manuscript, or compile separately if you need EPS.",
        },
        {
            "manuscript_label": "fig:recommendation-detail",
            "section": "Cluster-signature similarity features",
            "replace_placeholder_with": "fig08_signature_importance_share",
            "recommended_file": "fig08_signature_importance_share.eps",
            "status": "generated",
            "note": "Shows how much cluster-signature features contribute to the hybrid model.",
        },
        {
            "manuscript_label": "fig:wins",
            "section": "RQ1",
            "replace_placeholder_with": "fig01_winner_distribution",
            "recommended_file": "fig01_winner_distribution.eps",
            "status": "generated",
            "note": "Actual CP-SSD vs KB-KSSD wins.",
        },
        {
            "manuscript_label": "fig:accuracy",
            "section": "RQ2",
            "replace_placeholder_with": "fig05_accuracy_vs_baselines",
            "recommended_file": "fig05_accuracy_vs_baselines.eps",
            "status": "generated",
            "note": "Use this instead of raw-RF ablation if you want to emphasize random/majority baselines and hybrid performance.",
        },
        {
            "manuscript_label": "fig:actual-vs-predicted",
            "section": "RQ3",
            "replace_placeholder_with": "fig02_recommendation_summary",
            "recommended_file": "fig02_recommendation_summary.eps",
            "status": "generated",
            "note": "Actual winners, predicted winners, correct and wrong predictions.",
        },
        {
            "manuscript_label": "fig:correct-false",
            "section": "RQ3",
            "replace_placeholder_with": "fig04_confusion_matrix",
            "recommended_file": "fig04_confusion_matrix.eps",
            "status": "generated",
            "note": "Prefer this for paper; it is clearer than the correct/false barplot. Alternative: fig03_correct_false_by_method.eps.",
        },
        {
            "manuscript_label": "fig:importance",
            "section": "RQ5",
            "replace_placeholder_with": "fig07_top_feature_importance",
            "recommended_file": "fig07_top_feature_importance.eps",
            "status": "generated",
            "note": "Top feature importances of the hybrid recommender.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "MANUSCRIPT_FIGURE_MAPPING.csv", index=False)

    lines = []
    lines.append("% Direct figure replacements for the current manuscript")
    lines.append("% Generated by make_hybrid_ssd_paper_figures_WITH_EPS.py")
    lines.append("")
    for r in rows:
        if r["status"] == "generated":
            lines.append(f"% Replace placeholder {r['manuscript_label']} with:")
            lines.append(r"\begin{figure}[t]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=\linewidth]{{{r['recommended_file']}}}")
            lines.append(rf"\caption{{TODO: {r['note']}}}")
            lines.append(rf"\label{{{r['manuscript_label']}}}")
            lines.append(r"\end{figure}")
            lines.append("")
        elif r["status"] == "tikz snippet":
            lines.append(f"% Replace placeholder {r['manuscript_label']} by pasting the content of:")
            lines.append(f"% {r['recommended_file']}")
            lines.append(f"% Note: {r['note']}")
            lines.append("")
        else:
            lines.append(f"% {r['manuscript_label']}: {r['recommended_file']} ({r['status']})")
            lines.append(f"% Note: {r['note']}")
            lines.append("")
    (outdir / "MANUSCRIPT_FIGURE_REPLACEMENTS.tex").write_text("\n".join(lines))

# ============================================================
# MAIN
# ============================================================

def main():
    prefix, results_dir, data = load_inputs()
    outdir = ensure_dir(results_dir / PAPER_FIG_DIRNAME)

    print("\nPublication figure generation for hybrid SSD recommender")
    print("=" * 72)
    print(f"Prefix:       {prefix}")
    print(f"Input dir:    {results_dir}")
    print(f"Output dir:   {outdir}")
    print(f"Include raw TSFresh-only ablation if available: {INCLUDE_TSFRESH_ONLY_IF_AVAILABLE}")

    unique = data["unique"]
    summary = data["summary"]

    if unique.empty:
        raise RuntimeError("Missing unique-series predictions.")
    if summary.empty:
        raise RuntimeError("Missing hybrid model summary.")

    metrics = compute_unique_metrics(unique)
    best = find_best_hybrid(summary)

    print("\nBest hybrid configuration:")
    print(best)

    print("\nUnique-series metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Main figures
    fig_winner_distribution(unique, outdir)
    fig_recommendation_summary(unique, outdir)
    fig_correct_false_by_method(unique, outdir)
    fig_confusion_matrix(unique, outdir)
    fig_accuracy_vs_baselines(unique, summary, outdir, prefix)
    fig_metrics_summary(unique, summary, outdir)
    fig_feature_importance(data["importance"], outdir, top_n=20)
    fig_signature_importance_share(data["importance"], outdir)
    fig_cluster_specialization(data["cluster_eval"], outdir, top_n=16)
    fig_signature_model_ranking(summary, outdir, top_n=12)
    fig_roc_pr_curves(unique, outdir)
    fig_umap_clusters(prefix, outdir, unique=unique)
    export_tikz_workflow_figures(outdir)

    # Tables and LaTeX snippets
    export_summary_tables(prefix, data, outdir)
    export_manuscript_figure_map(outdir)

    print("\nDone. Generated paper-ready outputs in:")
    print(outdir)
    print("\nKey recommended plots:")
    print("  fig01_winner_distribution")
    print("  fig02_recommendation_summary")
    print("  fig04_confusion_matrix")
    print("  fig05_accuracy_vs_baselines")
    print("  fig07_top_feature_importance")
    print("  fig09_cluster_specialization")
    print("  fig_umap_clusters")
    print("  TIKZ_fig_tsfresh_extraction.tex")
    print("  TIKZ_fig_hybrid_recommender_workflow.tex")
    print("\nFor the main paper story, I recommend using fig05 WITHOUT raw RF ablation.")
    print("Also generated EPS files and manuscript mapping files:")
    print("  MANUSCRIPT_FIGURE_MAPPING.csv")
    print("  MANUSCRIPT_FIGURE_REPLACEMENTS.tex")


if __name__ == "__main__":
    main()
