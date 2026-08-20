#!/usr/bin/env python3

"""
Select best CP-SSD configuration from cpssd-new-results/classification_*
and compare it against:
    - original CP-SSD
    - KB-KSSD

Ground truth:
    - steady JAGT from full_classification.json
    - all timeseries are steady
    - therefore prediction == -1 is wrong
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# ======================================================
# CONFIG
# ======================================================

JAGT_PATH = "full_classification.json"
CPSSD_RESULTS_GLOB = "cpssd-new-results/classification_*"

OUTDIR = "ssd_best_cpssd_comparison_results"
os.makedirs(OUTDIR, exist_ok=True)

FALSE_UNSTEADY_PENALTY = 10_000

METHODS = {
    "CP-SSD best": "cp_best",
    "CP-SSD original": "cp_orig",
    "KB-KSSD": "kb",
}


# ======================================================
# PLOTTING HELPERS
# ======================================================

def prettify(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{name}.png", dpi=300)
    plt.close()


# ======================================================
# JAGT REFERENCE
# ======================================================

def get_jagt_steady_idx(e):
    man_labels = np.array([
        e["idxs"]["vittorio"],
        e["idxs"]["michele"],
        e["idxs"]["daniele"],
        e["idxs"]["luca"],
        e["idxs"]["martin"],
    ])

    labels = DBSCAN(eps=50, min_samples=3).fit(
        man_labels.reshape(-1, 1)
    ).labels_

    mask = labels > -1

    if not mask.any():
        return int(sorted(man_labels)[-3])

    vals = man_labels[mask]

    if np.sum(mask) == 4:
        return int(sorted(vals)[-2])

    return int(np.median(vals))


# ======================================================
# BEST CP-SSD CONFIG SELECTION
# ======================================================

def evaluate_cpssd_config(config_dir, jagt):
    errors_penalized = []
    errors_valid = []

    false_unsteady = 0
    missing = 0

    for key, e in jagt.items():
        fork_name, fork_idx = key.rsplit("_", 1)
        fork_idx = int(fork_idx)

        jagt_idx = get_jagt_steady_idx(e)

        fp = os.path.join(config_dir, fork_name)

        if not os.path.exists(fp):
            missing += 1
            continue

        try:
            pred = json.load(open(fp))["steady_state_starts"][fork_idx]
        except Exception:
            missing += 1
            continue

        if pred == -1:
            false_unsteady += 1
            errors_penalized.append(FALSE_UNSTEADY_PENALTY)
        else:
            err = abs(pred - jagt_idx)
            errors_penalized.append(err)
            errors_valid.append(err)

    if len(errors_penalized) == 0:
        return None

    return {
        "config_dir": config_dir,
        "config_name": os.path.basename(config_dir),
        "n_eval": len(errors_penalized),
        "missing": missing,
        "false_unsteady": false_unsteady,
        "false_unsteady_frac": false_unsteady / len(errors_penalized),
        "mae_valid_only": float(np.mean(errors_valid)) if errors_valid else np.nan,
        "median_valid_only": float(np.median(errors_valid)) if errors_valid else np.nan,
        "mae_penalized": float(np.mean(errors_penalized)),
        "median_penalized": float(np.median(errors_penalized)),
    }


def find_best_cpssd_config(jagt):
    rows = []

    for config_dir in sorted(glob.glob(CPSSD_RESULTS_GLOB)):
        if not os.path.isdir(config_dir):
            continue

        res = evaluate_cpssd_config(config_dir, jagt)

        if res is not None:
            rows.append(res)

    if not rows:
        raise RuntimeError(f"No valid CP-SSD configs found in {CPSSD_RESULTS_GLOB}")

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=[
            "false_unsteady",
            "mae_valid_only",
            "mae_penalized",
            "missing",
        ],
        ascending=True,
    )

    df.to_csv(f"{OUTDIR}/cpssd_config_ranking.csv", index=False)

    print("\n" + "=" * 80)
    print("TOP CP-SSD CONFIGURATIONS")
    print("=" * 80)
    print(df.head(20))

    best = df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST CP-SSD CONFIGURATION")
    print("=" * 80)
    print(best)

    return df, best


# ======================================================
# METHOD COMPARISON
# ======================================================

def compare_method_single(pred_idx, jagt_idx, prefix):
    steady = pred_idx >= 0

    return {
        f"{prefix}_steady": steady,
        f"{prefix}_false_unsteady": not steady,
        f"{prefix}_abs_err": abs(pred_idx - jagt_idx) if steady else np.nan,
        f"{prefix}_signed_err": pred_idx - jagt_idx if steady else np.nan,
    }


def build_results_table(jagt, best_config_dir):
    rows = []

    for key, e in jagt.items():
        fork_name, fork_idx = key.rsplit("_", 1)
        fork_idx = int(fork_idx)

        jagt_idx = get_jagt_steady_idx(e)

        kb_idx = e["idxs"]["kbkssd"]
        cp_orig_idx = e["idxs"]["cpssd"]

        cp_best_fp = os.path.join(best_config_dir, fork_name)

        if not os.path.exists(cp_best_fp):
            cp_best_idx = -1
        else:
            try:
                cp_best_idx = json.load(open(cp_best_fp))["steady_state_starts"][fork_idx]
            except Exception:
                cp_best_idx = -1

        row = {
            "key": key,
            "fork_name": fork_name,
            "fork_idx": fork_idx,
            "jagt_idx": jagt_idx,
            "cp_best_idx": cp_best_idx,
            "cp_orig_idx": cp_orig_idx,
            "kb_idx": kb_idx,
        }

        row.update(compare_method_single(cp_best_idx, jagt_idx, "cp_best"))
        row.update(compare_method_single(cp_orig_idx, jagt_idx, "cp_orig"))
        row.update(compare_method_single(kb_idx, jagt_idx, "kb"))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/ssd_best_config_comparison_table.csv", index=False)

    return df


# ======================================================
# PAIRWISE COMPARISON
# ======================================================

def winner_pair(a_err, b_err):
    if np.isnan(a_err) and np.isnan(b_err):
        return "both_false_unsteady"
    if np.isnan(a_err):
        return "B"
    if np.isnan(b_err):
        return "A"
    if a_err < b_err:
        return "A"
    if b_err < a_err:
        return "B"
    return "tie"


def pairwise_winner_counts(df, a_prefix, b_prefix, a_name, b_name):
    counts = {
        f"{a_name} better": 0,
        f"{b_name} better": 0,
        "tie": 0,
        "both false unsteady": 0,
    }

    for _, r in df.iterrows():
        w = winner_pair(r[f"{a_prefix}_abs_err"], r[f"{b_prefix}_abs_err"])

        if w == "A":
            counts[f"{a_name} better"] += 1
        elif w == "B":
            counts[f"{b_name} better"] += 1
        elif w == "tie":
            counts["tie"] += 1
        else:
            counts["both false unsteady"] += 1

    return counts


# ======================================================
# SUMMARIES
# ======================================================

def print_summary(df):
    n = len(df)

    print("\n" + "=" * 80)
    print("BINARY STEADY/UNSTEADY SUMMARY")
    print("=" * 80)

    print(f"Total steady JAGT timeseries: {n}")

    for method_name, prefix in METHODS.items():
        n_steady = int(df[f"{prefix}_steady"].sum())
        n_false = int(df[f"{prefix}_false_unsteady"].sum())

        print(f"\n{method_name}")
        print(f"  predicts steady:      {n_steady} / {n} ({n_steady / n:.1%})")
        print(f"  false unsteady (-1):  {n_false} / {n} ({n_false / n:.1%})")

    print("\n" + "=" * 80)
    print("DEVIATION SUMMARY")
    print("=" * 80)

    rows = []

    for method_name, prefix in METHODS.items():
        vals = df[f"{prefix}_abs_err"].dropna()

        row = {
            "method": method_name,
            "n_valid": len(vals),
            "false_unsteady": int(df[f"{prefix}_false_unsteady"].sum()),
            "mae": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
        }
        rows.append(row)

        print(f"\n{method_name}")
        print(f"  valid steady predictions: {len(vals)}")
        print(f"  MAE:     {vals.mean():.2f}")
        print(f"  Median:  {vals.median():.2f}")
        print(f"  Std:     {vals.std():.2f}")
        print(f"  Min:     {vals.min():.2f}")
        print(f"  Max:     {vals.max():.2f}")

    pd.DataFrame(rows).to_csv(f"{OUTDIR}/method_error_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("PAIRWISE WINNERS")
    print("=" * 80)

    comparisons = [
        ("cp_best", "cp_orig", "CP-SSD best", "CP-SSD original"),
        ("cp_best", "kb", "CP-SSD best", "KB-KSSD"),
        ("cp_orig", "kb", "CP-SSD original", "KB-KSSD"),
    ]

    winner_rows = []

    for a_prefix, b_prefix, a_name, b_name in comparisons:
        counts = pairwise_winner_counts(df, a_prefix, b_prefix, a_name, b_name)

        print(f"\n{a_name} vs {b_name}")
        for k, v in counts.items():
            print(f"  {k}: {v}")

        for k, v in counts.items():
            winner_rows.append({
                "comparison": f"{a_name} vs {b_name}",
                "result": k,
                "count": v,
            })

    pd.DataFrame(winner_rows).to_csv(
        f"{OUTDIR}/pairwise_winner_counts.csv",
        index=False,
    )


# ======================================================
# PLOTS
# ======================================================

def plot_best_config_ranking(config_df):
    top = config_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top["config_name"], top["mae_valid_only"])

    ax.set_ylabel("MAE on valid steady predictions")
    ax.set_title("Top 20 CP-SSD Configurations")
    plt.xticks(rotation=75, ha="right", fontsize=8)

    prettify(ax)
    savefig("top20_cpssd_config_mae")


def plot_config_false_unsteady(config_df):
    top = config_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top["config_name"], top["false_unsteady"])

    ax.set_ylabel("False unsteady predictions")
    ax.set_title("Top 20 CP-SSD Configurations: False Unsteady Counts")
    plt.xticks(rotation=75, ha="right", fontsize=8)

    prettify(ax)
    savefig("top20_cpssd_config_false_unsteady")


def plot_binary_detection(df):
    labels = []
    vals = []

    for method_name, prefix in METHODS.items():
        labels.extend([
            f"{method_name}\nsteady",
            f"{method_name}\nfalse unsteady",
        ])
        vals.extend([
            int(df[f"{prefix}_steady"].sum()),
            int(df[f"{prefix}_false_unsteady"].sum()),
        ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, vals)

    ax.set_ylabel("Number of timeseries")
    ax.set_title("Binary Detection on Steady JAGT\n(-1 is wrong here)")

    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")

    prettify(ax)
    savefig("binary_detection_three_methods")


def plot_three_method_agreement(df):
    b = {
        "best": df["cp_best_steady"],
        "orig": df["cp_orig_steady"],
        "kb": df["kb_steady"],
    }

    categories = {
        "all steady": b["best"] & b["orig"] & b["kb"],
        "all false\nunsteady": (~b["best"]) & (~b["orig"]) & (~b["kb"]),
        "only best\nsteady": b["best"] & (~b["orig"]) & (~b["kb"]),
        "only orig\nsteady": (~b["best"]) & b["orig"] & (~b["kb"]),
        "only KB\nsteady": (~b["best"]) & (~b["orig"]) & b["kb"],
        "best+orig\nsteady": b["best"] & b["orig"] & (~b["kb"]),
        "best+KB\nsteady": b["best"] & (~b["orig"]) & b["kb"],
        "orig+KB\nsteady": (~b["best"]) & b["orig"] & b["kb"],
    }

    labels = list(categories.keys())
    vals = [int(mask.sum()) for mask in categories.values()]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, vals)

    ax.set_ylabel("Number of timeseries")
    ax.set_title("Three-Method Binary Agreement")

    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")

    plt.xticks(rotation=30, ha="right")
    prettify(ax)
    savefig("three_method_binary_agreement")


def plot_abs_error_boxplot(df):
    data = [
        df["cp_best_abs_err"].dropna(),
        df["cp_orig_abs_err"].dropna(),
        df["kb_abs_err"].dropna(),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(
        data,
        labels=["CP best", "CP original", "KB-KSSD"],
        showfliers=False,
    )

    ax.set_ylabel("|predicted idx - JAGT idx|")
    ax.set_title("Absolute Deviation from JAGT\n(steady predictions only)")

    prettify(ax)
    savefig("absolute_error_boxplot_three_methods")


def plot_abs_error_hist(df):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["cp_best_abs_err"].dropna(), bins=30, alpha=0.55, label="CP best")
    ax.hist(df["cp_orig_abs_err"].dropna(), bins=30, alpha=0.55, label="CP original")
    ax.hist(df["kb_abs_err"].dropna(), bins=30, alpha=0.55, label="KB-KSSD")

    ax.set_xlabel("|predicted idx - JAGT idx|")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution")
    ax.legend(frameon=False)

    prettify(ax)
    savefig("absolute_error_histogram_three_methods")


def plot_signed_error_hist(df):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["cp_best_signed_err"].dropna(), bins=30, alpha=0.55, label="CP best")
    ax.hist(df["cp_orig_signed_err"].dropna(), bins=30, alpha=0.55, label="CP original")
    ax.hist(df["kb_signed_err"].dropna(), bins=30, alpha=0.55, label="KB-KSSD")

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("predicted idx - JAGT idx")
    ax.set_ylabel("Count")
    ax.set_title("Signed Error Distribution\nnegative = earlier, positive = later than JAGT")
    ax.legend(frameon=False)

    prettify(ax)
    savefig("signed_error_histogram_three_methods")


def plot_pairwise_winners(df):
    comparisons = [
        ("cp_best", "cp_orig", "CP best", "CP original"),
        ("cp_best", "kb", "CP best", "KB-KSSD"),
        ("cp_orig", "kb", "CP original", "KB-KSSD"),
    ]

    for a, b, an, bn in comparisons:
        counts = pairwise_winner_counts(df, a, b, an, bn)

        labels = list(counts.keys())
        vals = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)

        ax.set_title(f"{an} vs {bn}")
        ax.set_ylabel("Count")

        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")

        plt.xticks(rotation=30, ha="right")
        prettify(ax)

        fname = (
            "pairwise_"
            + an.replace(" ", "_")
            + "_vs_"
            + bn.replace(" ", "_")
        )
        savefig(fname)


def plot_error_difference(df, a_prefix, b_prefix, a_name, b_name):
    mask = df[f"{a_prefix}_steady"] & df[f"{b_prefix}_steady"]

    diff = df.loc[mask, f"{b_prefix}_abs_err"] - df.loc[mask, f"{a_prefix}_abs_err"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diff, bins=30, alpha=0.75)

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(f"{b_name} abs error - {a_name} abs error")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Error Difference: {a_name} vs {b_name}\n"
        f"negative = {b_name} better, positive = {a_name} better"
    )

    prettify(ax)

    fname = (
        "error_difference_"
        + a_name.replace(" ", "_")
        + "_vs_"
        + b_name.replace(" ", "_")
    )
    savefig(fname)


def plot_prediction_scatter(df, x_prefix, y_prefix, x_name, y_name):
    mask = df[f"{x_prefix}_steady"] & df[f"{y_prefix}_steady"]

    if mask.sum() == 0:
        return

    xvals = df.loc[mask, f"{x_prefix}_idx"]
    yvals = df.loc[mask, f"{y_prefix}_idx"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(xvals, yvals, alpha=0.6, s=20)

    lo = min(xvals.min(), yvals.min())
    hi = max(xvals.max(), yvals.max())

    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel(f"{x_name} predicted idx")
    ax.set_ylabel(f"{y_name} predicted idx")
    ax.set_title(f"{x_name} vs {y_name}\n(steady predictions only)")

    prettify(ax)

    fname = (
        "prediction_scatter_"
        + x_name.replace(" ", "_")
        + "_vs_"
        + y_name.replace(" ", "_")
    )
    savefig(fname)


def make_all_plots(df, config_df):
    plot_best_config_ranking(config_df)
    plot_config_false_unsteady(config_df)

    plot_binary_detection(df)
    plot_three_method_agreement(df)

    plot_abs_error_boxplot(df)
    plot_abs_error_hist(df)
    plot_signed_error_hist(df)

    plot_pairwise_winners(df)

    plot_error_difference(df, "cp_best", "cp_orig", "CP best", "CP original")
    plot_error_difference(df, "cp_best", "kb", "CP best", "KB-KSSD")
    plot_error_difference(df, "cp_orig", "kb", "CP original", "KB-KSSD")

    plot_prediction_scatter(df, "cp_best", "cp_orig", "CP best", "CP original")
    plot_prediction_scatter(df, "cp_best", "kb", "CP best", "KB-KSSD")
    plot_prediction_scatter(df, "cp_orig", "kb", "CP original", "KB-KSSD")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    jagt = json.load(open(JAGT_PATH))

    config_df, best = find_best_cpssd_config(jagt)

    best_config_dir = best["config_dir"]

    df = build_results_table(jagt, best_config_dir)

    with open(f"{OUTDIR}/best_cpssd_config.txt", "w") as f:
        f.write(str(best))

    print_summary(df)
    make_all_plots(df, config_df)

    print("\nDone.")
    print(f"Best CP-SSD config: {best_config_dir}")
    print(f"Results saved in: {OUTDIR}/")