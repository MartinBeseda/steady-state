#!/usr/bin/env python3
"""
Error analysis for the supervised TSFresh SSD recommendation model.

Purpose
-------
Analyze where the supervised SSD recommender fails, especially:
    - CP-SSD wins predicted as KB-KSSD,
    - KB-KSSD wins predicted as CP-SSD,
    - error concentration by cluster,
    - CP recall by cluster,
    - features distinguishing correct vs wrong predictions.

Expected inputs
---------------
This script expects outputs produced by the supervised recommendation script, especially:

    *_best_model_unique_series_predictions.csv
    *_JAGT_prediction_dataset_meta.csv
    *_JAGT_feature_matrix.csv
    *_best_model_feature_importances.csv

It searches automatically inside:

    tsfresh_targeted_clustering_results/SUPERVISED_SSD_RECOMMENDATION_FAST/

or another folder specified below.

Outputs
-------
Created in:

    <SUPERVISED_RESULTS_DIR>/ERROR_ANALYSIS/

Main outputs:
    confusion_breakdown.csv
    cluster_error_summary.csv
    cp_cluster_recall.csv
    feature_error_contrast.csv
    top_importance_error_features.csv

Main plots:
    confusion_breakdown.png
    cluster_error_rate_barplot.png
    cp_recall_by_cluster.png
    feature_error_contrast.png
    prediction_outcome_by_cluster.png
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIG
# ============================================================

SUPERVISED_RESULTS_DIR = "tsfresh_targeted_clustering_results/SUPERVISED_SSD_RECOMMENDATION_FAST"

# If True, analyse only CP-related errors in feature contrast:
#   correctly predicted CP vs missed CP.
FOCUS_ON_MISSED_CP = True

TOP_N_FEATURES = 30


# ============================================================
# PATHS
# ============================================================

ERROR_DIR = os.path.join(SUPERVISED_RESULTS_DIR, "ERROR_ANALYSIS")
os.makedirs(ERROR_DIR, exist_ok=True)


def find_one(pattern, required=True):
    hits = sorted(glob.glob(os.path.join(SUPERVISED_RESULTS_DIR, pattern)))
    if not hits:
        if required:
            raise FileNotFoundError(
                f"Could not find file matching pattern:\n"
                f"{os.path.join(SUPERVISED_RESULTS_DIR, pattern)}"
            )
        return None
    return hits[0]


def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs():
    pred_path = find_one("*_best_model_unique_series_predictions.csv")
    meta_path = find_one("*_JAGT_prediction_dataset_meta.csv")
    x_path = find_one("*_JAGT_feature_matrix.csv")
    imp_path = find_one("*_best_model_feature_importances.csv", required=False)

    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)
    X = pd.read_csv(x_path)

    if imp_path is not None:
        importances = pd.read_csv(imp_path)
    else:
        importances = pd.DataFrame()

    print("Loaded:")
    print(f"- predictions: {pred_path}")
    print(f"- meta:        {meta_path}")
    print(f"- features:    {x_path}")
    if imp_path:
        print(f"- importances: {imp_path}")
    else:
        print("- importances: not found")

    return pred, meta, X, importances


def merge_predictions_with_meta_and_features(pred, meta, X):
    """
    Merge unique predictions with meta and feature matrix.

    The supervised script writes meta and X in the same row order.
    We attach features to meta through row order and then merge by key.
    """
    meta = meta.copy().reset_index(drop=True)
    X = X.copy().reset_index(drop=True)

    if len(meta) != len(X):
        raise RuntimeError(
            f"Meta and feature matrix length mismatch: meta={len(meta)}, X={len(X)}"
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(X.columns)

    meta_features = pd.concat([meta, X.add_prefix("feat__")], axis=1)

    df = pred.merge(
        meta_features,
        on="key",
        how="left",
        suffixes=("", "_meta"),
    )

    if df["cluster"].isna().any():
        missing = df[df["cluster"].isna()]["key"].tolist()
        raise RuntimeError(f"Some prediction keys missing from meta: {missing[:10]}")

    # Ensure labels.
    if "true_winner" not in df.columns:
        df["true_winner"] = np.where(df["true_cp_win"] == 1, "CP", "KB")
    if "pred_winner" not in df.columns:
        df["pred_winner"] = np.where(df["pred_cp_win"] == 1, "CP", "KB")

    df["is_correct"] = df["true_winner"] == df["pred_winner"]

    def outcome(row):
        if row["true_winner"] == "KB" and row["pred_winner"] == "KB":
            return "KB_correct"
        if row["true_winner"] == "CP" and row["pred_winner"] == "CP":
            return "CP_correct"
        if row["true_winner"] == "CP" and row["pred_winner"] == "KB":
            return "CP_missed"
        if row["true_winner"] == "KB" and row["pred_winner"] == "CP":
            return "KB_false_CP"
        return "unknown"

    df["prediction_outcome"] = df.apply(outcome, axis=1)

    return df, feature_cols


# ============================================================
# CONFUSION / BASIC ERROR ANALYSIS
# ============================================================

def confusion_breakdown(df):
    out = (
        df.groupby(["true_winner", "pred_winner"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["true_winner", "pred_winner"])
    )

    out.to_csv(os.path.join(ERROR_DIR, "confusion_breakdown.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = out["true_winner"] + " → " + out["pred_winner"]
    ax.bar(labels, out["count"])
    ax.set_ylabel("Number of unique JAGT series")
    ax.set_title("Prediction confusion breakdown")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "confusion_breakdown.png"), dpi=300)
    plt.close()

    return out


def cluster_error_summary(df):
    rows = []

    for cid, g in df.groupby("cluster"):
        n = len(g)
        n_correct = int(g["is_correct"].sum())
        n_error = n - n_correct

        n_cp = int((g["true_winner"] == "CP").sum())
        n_kb = int((g["true_winner"] == "KB").sum())

        pred_cp = int((g["pred_winner"] == "CP").sum())
        pred_kb = int((g["pred_winner"] == "KB").sum())

        cp_correct = int(((g["true_winner"] == "CP") & (g["pred_winner"] == "CP")).sum())
        cp_missed = int(((g["true_winner"] == "CP") & (g["pred_winner"] == "KB")).sum())
        kb_correct = int(((g["true_winner"] == "KB") & (g["pred_winner"] == "KB")).sum())
        kb_false_cp = int(((g["true_winner"] == "KB") & (g["pred_winner"] == "CP")).sum())

        rows.append({
            "cluster": int(cid),
            "n": n,
            "n_correct": n_correct,
            "n_error": n_error,
            "error_rate": n_error / n if n else np.nan,
            "accuracy": n_correct / n if n else np.nan,
            "real_cp": n_cp,
            "real_kb": n_kb,
            "pred_cp": pred_cp,
            "pred_kb": pred_kb,
            "cp_correct": cp_correct,
            "cp_missed": cp_missed,
            "kb_correct": kb_correct,
            "kb_false_cp": kb_false_cp,
            "cp_recall": cp_correct / n_cp if n_cp else np.nan,
            "cp_precision": cp_correct / pred_cp if pred_cp else np.nan,
            "kb_recall": kb_correct / n_kb if n_kb else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("n_error", ascending=False)
    out.to_csv(os.path.join(ERROR_DIR, "cluster_error_summary.csv"), index=False)

    return out


def plot_cluster_error_summary(summary):
    # Error rate by cluster.
    sub = summary.sort_values("error_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"C{int(c)}\n(n={int(n)})" for c, n in zip(sub["cluster"], sub["n"])]
    ax.bar(labels, sub["error_rate"])
    ax.set_ylabel("Error rate")
    ax.set_title("Prediction error rate by cluster")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "cluster_error_rate_barplot.png"), dpi=300)
    plt.close()

    # CP recall by cluster.
    cp = summary[summary["real_cp"] > 0].copy().sort_values("cp_recall", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"C{int(c)}\n(CP={int(n)})" for c, n in zip(cp["cluster"], cp["real_cp"])]
    ax.bar(labels, cp["cp_recall"])
    ax.set_ylabel("CP recall")
    ax.set_title("CP-favorable series recovered by cluster")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "cp_recall_by_cluster.png"), dpi=300)
    plt.close()

    # Stacked-style grouped outcome counts.
    counts = summary.sort_values("cluster").copy()
    x = np.arange(len(counts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5 * width, counts["kb_correct"], width, label="KB correct")
    ax.bar(x - 0.5 * width, counts["cp_correct"], width, label="CP correct")
    ax.bar(x + 0.5 * width, counts["cp_missed"], width, label="CP missed")
    ax.bar(x + 1.5 * width, counts["kb_false_cp"], width, label="KB false CP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(c)}" for c in counts["cluster"]])
    ax.set_ylabel("Number of unique JAGT series")
    ax.set_title("Prediction outcomes by cluster")
    ax.legend(frameon=False, ncol=2)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "prediction_outcome_by_cluster.png"), dpi=300)
    plt.close()


# ============================================================
# FEATURE CONTRAST
# ============================================================

def feature_error_contrast(df, feature_cols):
    """
    Compare feature values between groups.

    Default focus:
      CP_correct vs CP_missed

    This answers:
      Which TSFresh features distinguish CP cases that we find
      from CP cases that we miss?
    """
    feat_cols_prefixed = [f"feat__{c}" for c in feature_cols if f"feat__{c}" in df.columns]

    rows = []

    if FOCUS_ON_MISSED_CP:
        a_name = "CP_correct"
        b_name = "CP_missed"

        a = df[df["prediction_outcome"] == a_name]
        b = df[df["prediction_outcome"] == b_name]
    else:
        a_name = "correct_all"
        b_name = "wrong_all"

        a = df[df["is_correct"]]
        b = df[~df["is_correct"]]

    if len(a) == 0 or len(b) == 0:
        raise RuntimeError(
            f"Cannot compute feature contrast: {a_name} n={len(a)}, {b_name} n={len(b)}"
        )

    for pref in feat_cols_prefixed:
        feat = pref.replace("feat__", "", 1)

        av = pd.to_numeric(a[pref], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        bv = pd.to_numeric(b[pref], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if len(av) < 3 or len(bv) < 3:
            continue

        mean_a = float(av.mean())
        mean_b = float(bv.mean())
        std_pooled = float(np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2.0))

        if std_pooled == 0 or np.isnan(std_pooled):
            effect = 0.0
        else:
            effect = (mean_a - mean_b) / std_pooled

        rows.append({
            "feature": feat,
            "group_A": a_name,
            "group_B": b_name,
            "n_A": len(av),
            "n_B": len(bv),
            "mean_A": mean_a,
            "mean_B": mean_b,
            "mean_diff_A_minus_B": mean_a - mean_b,
            "abs_mean_diff": abs(mean_a - mean_b),
            "cohens_d_A_minus_B": effect,
            "abs_cohens_d": abs(effect),
        })

    out = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)
    out.to_csv(os.path.join(ERROR_DIR, "feature_error_contrast.csv"), index=False)

    # Plot top features.
    sub = out.head(TOP_N_FEATURES).copy()
    sub = sub.sort_values("abs_cohens_d", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sub))
    ax.barh(y, sub["abs_cohens_d"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=7)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title(f"Top features distinguishing {a_name} from {b_name}")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "feature_error_contrast.png"), dpi=300)
    plt.close()

    return out


def intersect_with_importances(contrast, importances):
    if importances.empty:
        return pd.DataFrame()

    imp = importances.copy()

    if "mean_importance" in imp.columns:
        score_col = "mean_importance"
    elif "importance" in imp.columns:
        score_col = "importance"
    else:
        return pd.DataFrame()

    imp_agg = (
        imp.groupby("feature", as_index=False)
        .agg(
            model_importance=(score_col, "mean"),
            importance_splits=("feature", "count"),
        )
    )

    out = contrast.merge(imp_agg, on="feature", how="inner")
    out["combined_rank_score"] = out["abs_cohens_d"] * out["model_importance"]
    out = out.sort_values("combined_rank_score", ascending=False)

    out.to_csv(os.path.join(ERROR_DIR, "top_importance_error_features.csv"), index=False)

    sub = out.head(TOP_N_FEATURES).copy()
    sub = sub.sort_values("combined_rank_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sub))
    ax.barh(y, sub["combined_rank_score"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=7)
    ax.set_xlabel("|effect size| × model importance")
    ax.set_title("Important features also associated with missed CP errors")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(ERROR_DIR, "top_importance_error_features.png"), dpi=300)
    plt.close()

    return out


# ============================================================
# README
# ============================================================

def write_readme(confusion, summary, contrast):
    n = int(confusion["count"].sum())

    correct = int(confusion.loc[confusion["true_winner"] == confusion["pred_winner"], "count"].sum())
    acc = correct / n if n else np.nan

    cp_real = int(confusion.loc[confusion["true_winner"] == "CP", "count"].sum())
    cp_correct = int(
        confusion[
            (confusion["true_winner"] == "CP")
            & (confusion["pred_winner"] == "CP")
        ]["count"].sum()
    )
    cp_recall = cp_correct / cp_real if cp_real else np.nan

    kb_real = int(confusion.loc[confusion["true_winner"] == "KB", "count"].sum())
    kb_correct = int(
        confusion[
            (confusion["true_winner"] == "KB")
            & (confusion["pred_winner"] == "KB")
        ]["count"].sum()
    )
    kb_recall = kb_correct / kb_real if kb_real else np.nan

    worst_clusters = summary.sort_values("n_error", ascending=False).head(5)

    lines = []
    lines.append("SSD recommender error analysis")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Unique JAGT series analysed: {n}")
    lines.append(f"Correct predictions: {correct}")
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append("")
    lines.append(f"Real CP wins: {cp_real}")
    lines.append(f"Correct CP predictions: {cp_correct}")
    lines.append(f"CP recall: {cp_recall:.4f}")
    lines.append("")
    lines.append(f"Real KB wins: {kb_real}")
    lines.append(f"Correct KB predictions: {kb_correct}")
    lines.append(f"KB recall: {kb_recall:.4f}")
    lines.append("")
    lines.append("Worst clusters by number of errors:")
    for _, r in worst_clusters.iterrows():
        lines.append(
            f"- C{int(r['cluster'])}: errors={int(r['n_error'])}/{int(r['n'])}, "
            f"error_rate={r['error_rate']:.3f}, "
            f"CP_missed={int(r['cp_missed'])}, KB_false_CP={int(r['kb_false_cp'])}"
        )
    lines.append("")
    lines.append("Top features distinguishing correctly found CP cases from missed CP cases:")
    for _, r in contrast.head(10).iterrows():
        lines.append(
            f"- {r['feature']}: |d|={r['abs_cohens_d']:.3f}, "
            f"mean_diff={r['mean_diff_A_minus_B']:.4g}"
        )
    lines.append("")
    lines.append("Key interpretation:")
    lines.append("- If errors are concentrated in few clusters, improve those regimes specifically.")
    lines.append("- If missed CP cases have distinct features, add targeted CP-recovery rules or train a secondary model on CP-vs-missed-CP.")
    lines.append("- If errors are diffuse and feature contrast is weak, TSFresh-only accuracy may be close to its ceiling.")

    with open(os.path.join(ERROR_DIR, "README_ERROR_ANALYSIS.txt"), "w") as f:
        f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    pred, meta, X, importances = load_inputs()
    df, feature_cols = merge_predictions_with_meta_and_features(pred, meta, X)

    df.to_csv(os.path.join(ERROR_DIR, "prediction_with_meta_and_features.csv"), index=False)

    conf = confusion_breakdown(df)
    summary = cluster_error_summary(df)
    plot_cluster_error_summary(summary)

    contrast = feature_error_contrast(df, feature_cols)
    intersect_with_importances(contrast, importances)

    write_readme(conf, summary, contrast)

    print("\nConfusion breakdown:")
    print(conf)

    print("\nWorst clusters by number of errors:")
    print(summary.sort_values("n_error", ascending=False).head(10))

    print("\nTop feature contrasts:")
    print(contrast.head(15)[["feature", "abs_cohens_d", "mean_A", "mean_B"]])

    print(f"\nDone. Outputs saved in:\n{ERROR_DIR}/")


if __name__ == "__main__":
    main()
