#!/usr/bin/env python3
"""
CP-signature based SSD recommendation experiment.

Goal
----
1) Extract prominent TSFresh feature signatures from clusters where CP-SSD beats KB-KSSD.
2) For JAGT time series, compute whether each series is close to one or more CP-winning signatures.
3) Predict whether CP-SSD or KB-KSSD should be preferred.
4) Compare the prediction against computational ground truth:
       CP wins if cp_best_abs_err < kb_abs_err
       KB wins otherwise

The script is intentionally simple and interpretable:
- no black-box model,
- signatures are feature combinations extracted from CP-winning clusters,
- prediction is based on distance to CP signatures,
- threshold is tuned only on the train split and evaluated on a held-out test split.

Works with either:
    ANALYSIS_MODE = "all_timeseries"
or:
    ANALYSIS_MODE = "steady_jagt_only"

For "all_timeseries", clustering may have been built on all time series, but prediction/evaluation is
performed only on JAGT series.

For "steady_jagt_only", clustering and prediction/evaluation are both restricted to JAGT series.
"""

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline


# ============================================================
# USER CONFIG
# ============================================================

# Choose one:
#   "all_timeseries"
#   "steady_jagt_only"
ANALYSIS_MODE = "all_timeseries"

OUTDIR_ALL = "tsfresh_targeted_clustering_results"
OUTDIR_STEADY = "tsfresh_targeted_clustering_results_steady_jagt_only"

JAGT_METHOD_TABLE_PATH = "../man_steady_comparison/ssd_best_cpssd_comparison_results/ssd_best_config_comparison_table.csv"

ALPHA = 0.05
MIN_JAGT = 5
TOP_FEATURES_PER_CP_CLUSTER = 12

TEST_SIZE = 0.30
N_REPEATS = 20
RANDOM_STATE = 42
MIN_ABS_ADVANTAGE = 0.0
THRESHOLD_OBJECTIVE = "balanced_accuracy"  # or "cp_f1"

# Winner semantics:
# - if CP returns a valid index and KB returns -1, CP gets +1 win;
# - if KB returns a valid index and CP returns -1, KB gets +1 win;
# - if both fail, both get +0;
# - if both return valid indices, lower abs error gets +1;
# - if both valid errors are equal, both get +0.
#
# The supervised binary recommender is trained/evaluated only on decisive cases:
# exactly one of CP/KB has a win. Ties and both-fail cases are retained in audit outputs.
EXCLUDE_NON_DECISIVE_FROM_SUPERVISED = True

# Supervised predictor settings.
# "topN" feature sets are selected only from the training split.
SUPERVISED_TOP_FEATURES = [20, 50, 100]
FEATURE_SELECTION_METHODS = ["mutual_info", "anova"]

SIGNATURE_TOP_FEATURES = [5, 8, 12, 20]
MIN_SIGNATURE_TRAIN_JAGT = 5
SIGNATURE_SCORE_MODE = "weighted_max"

# Hybrid recommender:
# TSFresh features + cluster-signature similarity features -> supervised model.
HYBRID_SIGNATURE_TOP_K = [5, 8, 12, 20]
HYBRID_BASE_TOP_FEATURES = [20, 50, 100]
HYBRID_FEATURE_SELECTION_METHODS = ["mutual_info", "anova"]
HYBRID_MODELS = ["random_forest", "extra_trees", "hist_gradient_boosting"]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")

RESULTS_SUBDIR = "HYBRID_CLUSTER_SIGNATURE_SSD_RECOMMENDATION"


@dataclass
class ClusteringResult:
    variant: str
    config_name: str
    method: str
    labels: np.ndarray
    embedding: np.ndarray
    features: pd.DataFrame
    series_keys: list
    data_normalized: list
    metrics: dict


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def get_outdir():
    if ANALYSIS_MODE == "all_timeseries":
        return OUTDIR_ALL
    if ANALYSIS_MODE == "steady_jagt_only":
        return OUTDIR_STEADY
    raise ValueError("ANALYSIS_MODE must be 'all_timeseries' or 'steady_jagt_only'.")


OUTDIR = get_outdir()
TABLES_DIR = os.path.join(OUTDIR, "tables")
CONFIG_CACHE_DIR = os.path.join(OUTDIR, "configuration_cache")
RESULTS_DIR = os.path.join(OUTDIR, RESULTS_SUBDIR)
os.makedirs(RESULTS_DIR, exist_ok=True)


def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_jagt_method_table(path=JAGT_METHOD_TABLE_PATH):
    df = pd.read_csv(path)
    if "key" not in df.columns:
        raise ValueError(f"Expected column 'key' in {path}")
    return df.set_index("key", drop=False)


def configuration_cache_path(variant, config_name):
    return os.path.join(CONFIG_CACHE_DIR, f"{safe_name(variant)}__{safe_name(config_name)}.pkl")


def load_cached_result(variant, config_name):
    path = configuration_cache_path(variant, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find cached clustering result:\n{path}\nRun the clustering sweep first.")
    payload = pd.read_pickle(path)
    return payload["result"], payload["cluster_eval"]


def choose_selected_clustering():
    summary_path = os.path.join(OUTDIR, "top_configurations_interpretable_cluster_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing {summary_path}. Run the clustering sweep first.")

    df = pd.read_csv(summary_path)
    if "variant" not in df.columns:
        raise ValueError("Expected column 'variant' in top_configurations_interpretable_cluster_summary.csv")

    df = df[df["variant"] == ANALYSIS_MODE].copy()
    if df.empty:
        raise RuntimeError(f"No interpretable cluster rows found for ANALYSIS_MODE={ANALYSIS_MODE!r}.")

    df = df[df["n_jagt"] >= MIN_JAGT].copy()
    df = df[df["sign_q_vs_cp_best"].notna()].copy()

    df["margin"] = df["kb_wins_vs_cp_best"] - df["cp_best_wins_vs_kb"]
    df["is_sig"] = df["sign_q_vs_cp_best"] < ALPHA
    df["is_kb"] = df["is_sig"] & (df["margin"] > 0)
    df["is_cp"] = df["is_sig"] & (df["margin"] < 0)

    rows = []
    for (variant, config), g in df.groupby(["variant", "config_name"]):
        kb = g[g["is_kb"]]
        cp = g[g["is_cp"]]
        kb_jagt = int(kb["n_jagt"].sum())
        cp_jagt = int(cp["n_jagt"].sum())
        rows.append({
            "variant": variant,
            "config_name": config,
            "n_kb_sig_clusters": len(kb),
            "n_cp_sig_clusters": len(cp),
            "kb_jagt": kb_jagt,
            "cp_jagt": cp_jagt,
            "total_sig_jagt": kb_jagt + cp_jagt,
            "balanced_specialization_jagt": min(kb_jagt, cp_jagt),
            "specialization_product": kb_jagt * cp_jagt,
            "best_cp_q": float(cp["sign_q_vs_cp_best"].min()) if len(cp) else np.nan,
            "best_kb_q": float(kb["sign_q_vs_cp_best"].min()) if len(kb) else np.nan,
        })

    rank = pd.DataFrame(rows)
    if rank.empty:
        raise RuntimeError("No clustering candidates found.")

    rank["has_both"] = (rank["n_kb_sig_clusters"] > 0) & (rank["n_cp_sig_clusters"] > 0)
    rank = rank.sort_values(
        by=["has_both", "balanced_specialization_jagt", "specialization_product", "total_sig_jagt", "n_cp_sig_clusters", "n_kb_sig_clusters", "cp_jagt", "kb_jagt"],
        ascending=[False, False, False, False, False, False, False, False],
    )
    rank.to_csv(os.path.join(RESULTS_DIR, "selected_clustering_candidates_ranked.csv"), index=False)
    return rank.iloc[0]


def feature_columns(zscores):
    return [c for c in zscores.columns if c not in {"cluster", "cluster_size", "size"} and pd.api.types.is_numeric_dtype(zscores[c])]


def load_selected_tables(selected):
    prefix = safe_name(f"{selected['variant']}_{selected['config_name']}")
    cluster_eval_path = os.path.join(TABLES_DIR, f"{prefix}_cluster_method_eval.csv")
    zscore_path = os.path.join(TABLES_DIR, f"{prefix}_cluster_feature_zscores.csv")
    if not os.path.exists(cluster_eval_path):
        raise FileNotFoundError(cluster_eval_path)
    if not os.path.exists(zscore_path):
        raise FileNotFoundError(zscore_path)
    return prefix, pd.read_csv(cluster_eval_path), pd.read_csv(zscore_path)


def identify_cp_clusters(cluster_eval):
    df = cluster_eval.copy()
    df = df[(df["cluster"] != -1) & (df["n_jagt"] >= MIN_JAGT)].copy()
    df["margin"] = df["kb_wins_vs_cp_best"] - df["cp_best_wins_vs_kb"]
    df["is_cp_sig"] = df["sign_q_vs_cp_best"].notna() & (df["sign_q_vs_cp_best"] < ALPHA) & (df["margin"] < 0)

    cp = df[df["is_cp_sig"]].copy()
    cp = cp.sort_values(["sign_q_vs_cp_best", "n_jagt"], ascending=[True, False])
    if cp.empty:
        cp = df[df["margin"] < 0].copy()
        cp = cp.sort_values(["margin", "n_jagt"], ascending=[True, False])
        if cp.empty:
            raise RuntimeError("No CP-winning or CP-leaning clusters found.")
        cp["is_fallback_non_significant"] = True
    else:
        cp["is_fallback_non_significant"] = False
    return cp


def extract_cp_signature_features(cp_clusters, zscores, top_k=TOP_FEATURES_PER_CP_CLUSTER):
    rows = []
    cols = feature_columns(zscores)
    for _, c_row in cp_clusters.iterrows():
        cid = c_row["cluster"]
        zrow = zscores[zscores["cluster"] == cid]
        if zrow.empty:
            zrow = zscores[zscores["cluster"] == float(cid)]
        if zrow.empty:
            continue
        z = zrow.iloc[0][cols].astype(float)
        tmp = pd.DataFrame({
            "cluster": cid,
            "feature": cols,
            "zscore": z.values,
            "abs_zscore": np.abs(z.values),
            "cluster_size": c_row["cluster_size"],
            "n_jagt": c_row["n_jagt"],
            "kb_wins_vs_cp_best": c_row["kb_wins_vs_cp_best"],
            "cp_best_wins_vs_kb": c_row["cp_best_wins_vs_kb"],
            "sign_q_vs_cp_best": c_row["sign_q_vs_cp_best"],
            "is_fallback_non_significant": c_row.get("is_fallback_non_significant", False),
        }).sort_values("abs_zscore", ascending=False).head(top_k)
        rows.append(tmp)
    if not rows:
        raise RuntimeError("Could not extract CP signature features from z-score table.")
    return pd.concat(rows, ignore_index=True)


def method_returned_valid_index(idx):
    """
    In this dataset, -1 means that the method explicitly decided
    that the time series is not steady. That is a valid method outcome,
    not a missing computation.
    """
    if pd.isna(idx):
        return False
    return int(idx) != -1


def build_jagt_dataset(result, labels, jagt_methods, return_audit=False):
    """
    Build the supervised JAGT dataset using separate CP/KB win indicators.

    Rules:
    1. CP valid, KB failed (-1): CP gets +1, KB gets +0.
    2. KB valid, CP failed (-1): KB gets +1, CP gets +0.
    3. Both failed (-1): CP gets +0, KB gets +0; non-decisive.
    4. Both valid: compare absolute errors.
       - lower error gets +1;
       - equal error gives +0 to both; non-decisive.

    The classifier is binary, so it can only train/evaluate on decisive rows:
        true_cp_win + true_kb_win == 1

    Non-decisive rows are retained in the audit table.
    """
    features = result.features.copy().reset_index(drop=True)
    series_keys = list(result.series_keys)
    labels = np.asarray(labels)

    if len(features) != len(series_keys) or len(labels) != len(series_keys):
        raise RuntimeError("Length mismatch among features, series_keys, and labels.")

    rows, feat_rows = [], []
    audit_rows = []

    for i, key in enumerate(series_keys):
        if key not in jagt_methods.index:
            continue

        r = jagt_methods.loc[key]

        jagt_idx = pd.to_numeric(r.get("jagt_idx", np.nan), errors="coerce")
        kb_idx = pd.to_numeric(r.get("kb_idx", np.nan), errors="coerce")
        cp_idx = pd.to_numeric(r.get("cp_best_idx", np.nan), errors="coerce")

        kb_err_raw = pd.to_numeric(r.get("kb_abs_err", np.nan), errors="coerce")
        cp_err_raw = pd.to_numeric(r.get("cp_best_abs_err", np.nan), errors="coerce")

        kb_valid = method_returned_valid_index(kb_idx)
        cp_valid = method_returned_valid_index(cp_idx)

        kb_win = 0
        cp_win = 0
        outcome_reason = ""

        kb_err_effective = kb_err_raw
        cp_err_effective = cp_err_raw

        if cp_valid and not kb_valid:
            cp_win = 1
            kb_win = 0
            outcome_reason = "cp_valid_kb_failed"

        elif kb_valid and not cp_valid:
            kb_win = 1
            cp_win = 0
            outcome_reason = "kb_valid_cp_failed"

        elif not kb_valid and not cp_valid:
            kb_win = 0
            cp_win = 0
            outcome_reason = "both_failed"

        else:
            if pd.isna(kb_err_effective):
                kb_err_effective = abs(kb_idx - jagt_idx)
            if pd.isna(cp_err_effective):
                cp_err_effective = abs(cp_idx - jagt_idx)

            if pd.isna(kb_err_effective) or pd.isna(cp_err_effective):
                kb_win = 0
                cp_win = 0
                outcome_reason = "missing_error_even_after_valid_indices"
            else:
                delta = cp_err_effective - kb_err_effective

                if abs(delta) < MIN_ABS_ADVANTAGE:
                    kb_win = 0
                    cp_win = 0
                    outcome_reason = "tie_or_below_MIN_ABS_ADVANTAGE"
                elif delta < 0:
                    cp_win = 1
                    kb_win = 0
                    outcome_reason = "cp_lower_abs_error"
                else:
                    kb_win = 1
                    cp_win = 0
                    outcome_reason = "kb_lower_abs_error"

        decisive = (cp_win + kb_win) == 1

        if cp_win == 1:
            true_winner = "CP"
        elif kb_win == 1:
            true_winner = "KB"
        else:
            true_winner = "NONE"

        audit_row = {
            "key": key,
            "cache_index": i,
            "cluster": int(labels[i]),
            "jagt_idx": jagt_idx,
            "kb_idx": kb_idx,
            "cp_best_idx": cp_idx,
            "kb_valid_index": kb_valid,
            "cp_best_valid_index": cp_valid,
            "kb_abs_err_raw": kb_err_raw,
            "cp_best_abs_err_raw": cp_err_raw,
            "kb_abs_err_effective": kb_err_effective,
            "cp_best_abs_err_effective": cp_err_effective,
            "kb_win": kb_win,
            "cp_win": cp_win,
            "true_winner": true_winner,
            "true_cp_win": cp_win,
            "true_kb_win": kb_win,
            "decisive_for_supervised_eval": decisive,
            "outcome_reason": outcome_reason,
        }
        audit_rows.append(audit_row)

        if EXCLUDE_NON_DECISIVE_FROM_SUPERVISED and not decisive:
            continue

        if not decisive:
            continue

        rows.append({
            "key": key,
            "cluster": int(labels[i]),
            "jagt_idx": jagt_idx,
            "kb_idx": kb_idx,
            "cp_best_idx": cp_idx,
            "kb_abs_err": kb_err_effective,
            "cp_best_abs_err": cp_err_effective,
            "kb_valid_index": kb_valid,
            "cp_best_valid_index": cp_valid,
            "kb_win": kb_win,
            "cp_win": cp_win,
            "true_winner": true_winner,
            "true_cp_win": int(cp_win == 1),
            "true_kb_win": int(kb_win == 1),
            "outcome_reason": outcome_reason,
        })
        feat_rows.append(features.iloc[i])

    meta = pd.DataFrame(rows)
    X = pd.DataFrame(feat_rows).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    if meta.empty:
        raise RuntimeError("No decisive JAGT rows for supervised CP-vs-KB evaluation.")

    if return_audit:
        return X, meta, audit

    return X, meta



def make_signature_objects(signatures, X_train, labels_train):
    objs = []
    for cid, g in signatures.groupby("cluster"):
        feats = [f for f in list(g["feature"].drop_duplicates()) if f in X_train.columns]
        if not feats:
            continue
        idx = np.where(labels_train == cid)[0]
        if len(idx) == 0:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train[feats].values)
        centroid = Xs[idx].mean(axis=0)
        objs.append({"cluster": cid, "features": feats, "scaler": scaler, "centroid": centroid, "n_train_cluster": len(idx), "n_features": len(feats)})
    if not objs:
        raise RuntimeError("No CP signature could be built on the training split. Try fewer top features or more train data.")
    return objs


def score_with_cp_signatures(X, signature_objs):
    all_scores = []
    for obj in signature_objs:
        feats = obj["features"]
        Xs = obj["scaler"].transform(X[feats].values)
        d = np.linalg.norm(Xs - obj["centroid"], axis=1)
        d_norm = d / np.sqrt(max(len(feats), 1))
        score = np.exp(-d_norm)
        all_scores.append(score)
    return np.vstack(all_scores).T.max(axis=1)


def choose_threshold(y_train, scores_train):
    thresholds = np.unique(np.quantile(scores_train, np.linspace(0.05, 0.95, 91)))
    if len(thresholds) == 0:
        thresholds = np.array([np.median(scores_train)])
    best_thr, best_val = thresholds[0], -np.inf
    for thr in thresholds:
        pred = (scores_train >= thr).astype(int)
        if THRESHOLD_OBJECTIVE == "balanced_accuracy":
            val = balanced_accuracy_score(y_train, pred)
        elif THRESHOLD_OBJECTIVE == "cp_f1":
            val = f1_score(y_train, pred, zero_division=0)
        else:
            raise ValueError("THRESHOLD_OBJECTIVE must be 'balanced_accuracy' or 'cp_f1'.")
        if val > best_val:
            best_val = val
            best_thr = thr
    return float(best_thr), float(best_val)


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cp_precision": precision_score(y_true, y_pred, zero_division=0),
        "cp_recall": recall_score(y_true, y_pred, zero_division=0),
        "cp_f1": f1_score(y_true, y_pred, zero_division=0),
        "n": len(y_true),
        "n_true_cp": int(np.sum(y_true == 1)),
        "n_pred_cp": int(np.sum(y_pred == 1)),
    }


def run_repeated_holdout(X, meta, signatures):
    y = meta["true_cp_win"].values.astype(int)
    labels = meta["cluster"].values
    if len(np.unique(y)) < 2:
        raise RuntimeError("Only one true winner class is present. Cannot evaluate CP-vs-KB prediction.")

    splitter = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    metrics_rows, pred_rows = [], []
    rng = np.random.default_rng(RANDOM_STATE)

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"Split {split_id + 1}/{N_REPEATS}")
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        labels_train = labels[train_idx]

        sig_objs = make_signature_objects(signatures, X_train, labels_train)
        train_scores = score_with_cp_signatures(X_train, sig_objs)
        test_scores = score_with_cp_signatures(X_test, sig_objs)
        thr, train_obj = choose_threshold(y_train, train_scores)
        y_pred = (test_scores >= thr).astype(int)

        majority_class = int(np.round(np.mean(y_train)))
        y_majority = np.full_like(y_test, majority_class)
        cp_prev = float(np.mean(y_train))
        y_random_prev = (rng.random(len(y_test)) < cp_prev).astype(int)
        y_random_5050 = (rng.random(len(y_test)) < 0.5).astype(int)

        for model_name, pred in [
            ("CP_signature_rule", y_pred),
            ("majority_baseline", y_majority),
            ("random_prevalence_baseline", y_random_prev),
            ("random_50_50_baseline", y_random_5050),
        ]:
            m = evaluate_predictions(y_test, pred)
            m.update({"split_id": split_id, "model": model_name, "threshold": thr if model_name == "CP_signature_rule" else np.nan,
                      "train_threshold_objective": train_obj if model_name == "CP_signature_rule" else np.nan,
                      "train_cp_prevalence": cp_prev})
            metrics_rows.append(m)

        out = meta_test.copy()
        out["split_id"] = split_id
        out["cp_signature_score"] = test_scores
        out["threshold"] = thr
        out["pred_cp_win"] = y_pred
        out["pred_winner"] = np.where(y_pred == 1, "CP", "KB")
        pred_rows.append(out)

    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def plot_metric_bars(metrics, prefix):
    agg = metrics.groupby("model", as_index=False).agg(
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        cp_precision=("cp_precision", "mean"),
        cp_recall=("cp_recall", "mean"),
        cp_f1=("cp_f1", "mean"),
    )
    agg.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_prediction_metrics_mean.csv"), index=False)

    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "cp_precision", "cp_recall", "cp_f1"]:
        sub = agg.sort_values(metric, ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(sub))
        ax.barh(y, sub[metric])
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"])
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.set_title(f"SSD winner prediction: {metric.replace('_', ' ')}")
        prettify_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_bar_{metric}.png"), dpi=300)
        plt.close()


def plot_predicted_vs_real_counts(predictions, prefix):
    df = predictions.copy()
    counts = pd.DataFrame({
        "category": ["Real CP wins", "Predicted CP wins", "Correct predicted CP wins"],
        "count": [int(df["true_cp_win"].sum()), int(df["pred_cp_win"].sum()), int(((df["true_cp_win"] == 1) & (df["pred_cp_win"] == 1)).sum())],
    })
    counts.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_predicted_vs_real_cp_counts.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts["category"], counts["count"])
    ax.set_ylabel("Count over all held-out appearances")
    ax.set_title("Predicted CP-SSD wins vs real CP-SSD wins")
    ax.set_xticklabels(counts["category"], rotation=20, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_predicted_vs_real_cp_counts.png"), dpi=300)
    plt.close()


def plot_confusion_bar(predictions, prefix):
    y_true = predictions["true_cp_win"].values.astype(int)
    y_pred = predictions["pred_cp_win"].values.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    df = pd.DataFrame([
        {"outcome": "KB correctly predicted", "count": int(cm[0, 0])},
        {"outcome": "KB predicted as CP", "count": int(cm[0, 1])},
        {"outcome": "CP predicted as KB", "count": int(cm[1, 0])},
        {"outcome": "CP correctly predicted", "count": int(cm[1, 1])},
    ])
    df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_confusion_counts.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["outcome"], df["count"])
    ax.set_ylabel("Count over all held-out appearances")
    ax.set_title("Confusion summary for CP-signature rule")
    ax.set_xticklabels(df["outcome"], rotation=25, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_confusion_counts.png"), dpi=300)
    plt.close()


def plot_score_distributions(predictions, prefix):
    df = predictions.copy()
    kb_scores = df[df["true_cp_win"] == 0]["cp_signature_score"].dropna().values
    cp_scores = df[df["true_cp_win"] == 1]["cp_signature_score"].dropna().values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([kb_scores, cp_scores], labels=["Real KB wins", "Real CP wins"], showfliers=False)
    ax.set_ylabel("CP signature score")
    ax.set_title("Are real CP wins closer to CP signatures?")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_cp_signature_score_distribution.png"), dpi=300)
    plt.close()


def plot_cp_features(signatures, prefix):
    agg = signatures.groupby("feature", as_index=False).agg(
        n_signatures=("cluster", "nunique"),
        mean_abs_zscore=("abs_zscore", "mean"),
        max_abs_zscore=("abs_zscore", "max"),
        total_jagt=("n_jagt", "sum"),
    ).sort_values(["n_signatures", "mean_abs_zscore", "total_jagt"], ascending=[False, False, False]).head(15)
    agg.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_aggregated_CP_signature_features.csv"), index=False)
    sub = agg.sort_values("mean_abs_zscore", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_abs_zscore"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=7)
    ax.set_xlabel("Mean |cluster z-score|")
    ax.set_title("Most recurring raw TSFresh features in CP-winning signatures")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_CP_signature_features.png"), dpi=300)
    plt.close()



# ============================================================
# SUPERVISED CLASSIFIERS
# ============================================================

def select_features_supervised(X_train, y_train, method, top_n):
    """
    Select features using training data only.
    """
    X_train = X_train.copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_features = X_train.shape[1]
    top_n = min(top_n, n_features)

    if method == "mutual_info":
        scores = mutual_info_classif(
            X_train.values,
            y_train,
            discrete_features=False,
            random_state=RANDOM_STATE,
        )
    elif method == "anova":
        scores, _ = f_classif(X_train.values, y_train)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError("Unknown feature-selection method.")

    order = np.argsort(scores)[::-1][:top_n]
    selected = list(X_train.columns[order])

    return selected, pd.DataFrame({
        "feature": X_train.columns,
        "score": scores,
        "selection_method": method,
    }).sort_values("score", ascending=False)


def make_classifier(model_name):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def remove_constant_columns(X_train, X_test):
    """
    Remove columns that are constant inside the training split.
    This prevents ANOVA division-by-zero warnings and avoids useless features.
    """
    nunique = X_train.nunique(dropna=False)
    keep = list(nunique[nunique > 1].index)

    if not keep:
        raise RuntimeError("All features are constant in this training split.")

    return X_train[keep].copy(), X_test[keep].copy()

def run_supervised_holdout(X, meta):
    """
    Repeated stratified holdout using supervised models.
    Prediction target:
        1 = CP-SSD wins
        0 = KB-KSSD wins
    """
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = meta["true_cp_win"].values.astype(int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Only one winner class is present.")

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model_names = [
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ]

    metric_rows = []
    pred_rows = []
    feature_rows = []

    rng = np.random.default_rng(RANDOM_STATE)

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"Split {split_id + 1}/{N_REPEATS}")
        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        X_test_full = X.iloc[test_idx].reset_index(drop=True)

        X_train_full, X_test_full = remove_constant_columns(X_train_full, X_test_full)

        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Baselines
        majority_class = int(np.round(np.mean(y_train)))
        y_majority = np.full_like(y_test, majority_class)

        cp_prev = float(np.mean(y_train))
        y_random_prev = (rng.random(len(y_test)) < cp_prev).astype(int)
        y_random_5050 = (rng.random(len(y_test)) < 0.5).astype(int)

        for model_name, pred in [
            ("majority_baseline", y_majority),
            ("random_prevalence_baseline", y_random_prev),
            ("random_50_50_baseline", y_random_5050),
        ]:
            m = evaluate_predictions(y_test, pred)
            m.update({
                "split_id": split_id,
                "model": model_name,
                "feature_selection": "none",
                "top_n": 0,
                "train_cp_prevalence": cp_prev,
            })
            metric_rows.append(m)

        for fs_method in FEATURE_SELECTION_METHODS:
            for top_n in SUPERVISED_TOP_FEATURES:
                selected, feature_scores = select_features_supervised(
                    X_train_full,
                    y_train,
                    method=fs_method,
                    top_n=top_n,
                )

                for model_name in model_names:
                    clf = make_classifier(model_name)

                    X_train = X_train_full[selected]
                    X_test = X_test_full[selected]

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    m = evaluate_predictions(y_test, y_pred)
                    m.update({
                        "split_id": split_id,
                        "model": model_name,
                        "feature_selection": fs_method,
                        "top_n": top_n,
                        "train_cp_prevalence": cp_prev,
                    })
                    metric_rows.append(m)

                    out = meta_test.copy()
                    out["split_id"] = split_id
                    out["model"] = model_name
                    out["feature_selection"] = fs_method
                    out["top_n"] = top_n
                    out["pred_cp_win"] = y_pred
                    out["pred_winner"] = np.where(y_pred == 1, "CP", "KB")
                    pred_rows.append(out)

                    # Model importances where available.
                    fitted = clf
                    if hasattr(fitted, "feature_importances_"):
                        importances = fitted.feature_importances_
                    else:
                        importances = None

                    if importances is not None:
                        for feat, imp in zip(selected, importances):
                            feature_rows.append({
                                "split_id": split_id,
                                "model": model_name,
                                "feature_selection": fs_method,
                                "top_n": top_n,
                                "feature": feat,
                                "importance": float(imp),
                            })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    feature_importances = pd.DataFrame(feature_rows)

    return metrics, predictions, feature_importances


def aggregate_unique_series_predictions_for_model(predictions, model, feature_selection, top_n):
    """
    Convert repeated test appearances into one prediction per unique JAGT series
    for one selected model configuration.
    """
    df = predictions[
        (predictions["model"] == model)
        & (predictions["feature_selection"] == feature_selection)
        & (predictions["top_n"] == top_n)
    ].copy()

    rows = []
    for key, g in df.groupby("key"):
        true_cp = int(g["true_cp_win"].iloc[0])
        pred_cp_rate = float(g["pred_cp_win"].mean())
        pred_cp = int(pred_cp_rate >= 0.5)

        rows.append({
            "key": key,
            "true_cp_win": true_cp,
            "true_winner": "CP" if true_cp else "KB",
            "pred_cp_win": pred_cp,
            "pred_winner": "CP" if pred_cp else "KB",
            "pred_cp_rate": pred_cp_rate,
            "n_test_appearances": int(len(g)),
            "correct_prediction": int(pred_cp == true_cp),
            "correct_cp_prediction": int((true_cp == 1) and (pred_cp == 1)),
            "correct_kb_prediction": int((true_cp == 0) and (pred_cp == 0)),
        })

    return pd.DataFrame(rows)


def plot_best_supervised_summary(unique_predictions, prefix, model_label):
    counts = pd.DataFrame([
        {"category": "CP-SSD wins", "count": int(unique_predictions["true_cp_win"].sum())},
        {"category": "KB-KSSD wins", "count": int((unique_predictions["true_cp_win"] == 0).sum())},
        {"category": "Predicted CP-SSD wins", "count": int(unique_predictions["pred_cp_win"].sum())},
        {"category": "Correct predictions", "count": int(unique_predictions["correct_prediction"].sum())},
    ])

    counts_path = os.path.join(RESULTS_DIR, f"{prefix}_best_model_unique_series_counts.csv")
    plot_path = os.path.join(RESULTS_DIR, f"{prefix}_best_model_unique_series_barplot.png")

    counts.to_csv(counts_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts["category"], counts["count"])
    ax.set_ylabel("Number of unique JAGT time series")
    ax.set_title(f"SSD recommendation on unique JAGT series\n{model_label}")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts["category"], rotation=20, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return counts, plot_path, counts_path


def plot_supervised_model_ranking(summary, prefix):
    sub = summary.copy()
    sub["label"] = (
        sub["model"].astype(str)
        + " | "
        + sub["feature_selection"].astype(str)
        + " | top"
        + sub["top_n"].astype(str)
    )
    sub = sub.sort_values("accuracy", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sub))
    ax.barh(y, sub["accuracy"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean holdout accuracy")
    ax.set_title("Supervised SSD recommendation: top configurations")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_supervised_model_ranking_accuracy.png"), dpi=300)
    plt.close()


def summarize_feature_importances(feature_importances, best_model, best_fs, best_top_n, prefix):
    if feature_importances.empty:
        return pd.DataFrame()

    df = feature_importances[
        (feature_importances["model"] == best_model)
        & (feature_importances["feature_selection"] == best_fs)
        & (feature_importances["top_n"] == best_top_n)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_splits=("split_id", "nunique"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    agg.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_best_model_feature_importances.csv"), index=False)

    sub = agg.head(20).sort_values("mean_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_importance"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=7)
    ax.set_xlabel("Mean feature importance")
    ax.set_title("Top features of the best supervised SSD recommender")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_best_model_feature_importances.png"), dpi=300)
    plt.close()

    return agg


# ============================================================
# SEPARATE-WINS AUDIT
# ============================================================

def summarize_separate_wins_audit(audit, prefix):
    """
    Save the full JAGT separate-wins audit and compact counts.
    """
    audit_path = os.path.join(RESULTS_DIR, f"{prefix}_FULL_JAGT_separate_wins_audit.csv")
    audit.to_csv(audit_path, index=False)

    reason_counts = (
        audit.groupby("outcome_reason", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    reason_counts.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_separate_wins_reason_counts.csv"), index=False)

    win_counts = pd.DataFrame([
        {"category": "CP wins", "count": int(audit["cp_win"].sum())},
        {"category": "KB wins", "count": int(audit["kb_win"].sum())},
        {"category": "Non-decisive", "count": int(((audit["cp_win"] + audit["kb_win"]) == 0).sum())},
    ])
    win_counts.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_separate_wins_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(win_counts["category"], win_counts["count"])
    ax.set_ylabel("Number of JAGT series")
    ax.set_title("Separate CP/KB wins under -1 failure semantics")
    ax.set_xticks(np.arange(len(win_counts)))
    ax.set_xticklabels(win_counts["category"], rotation=15, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_separate_wins_counts.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(reason_counts["outcome_reason"], reason_counts["count"])
    ax.set_ylabel("Number of JAGT series")
    ax.set_title("Outcome reasons under separate-wins semantics")
    ax.set_xticks(np.arange(len(reason_counts)))
    ax.set_xticklabels(reason_counts["outcome_reason"], rotation=25, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_separate_wins_reason_counts.png"), dpi=300)
    plt.close()

    return audit_path, reason_counts, win_counts


# ============================================================
# CLUSTER-SIGNATURE SSD RECOMMENDER
# ============================================================

def get_cluster_signature_features(zscores, cluster_id, top_k):
    cols = feature_columns(zscores)
    zrow = zscores[zscores["cluster"] == cluster_id]
    if zrow.empty:
        zrow = zscores[zscores["cluster"] == float(cluster_id)]
    if zrow.empty:
        return []
    vals = zrow.iloc[0][cols].astype(float)
    order = np.argsort(np.abs(vals.values))[::-1]
    selected = []
    for idx in order:
        f = cols[idx]
        if f not in selected:
            selected.append(f)
        if len(selected) >= top_k:
            break
    return selected


def build_train_cluster_signatures(X_train, meta_train, zscores, top_k):
    signatures = []
    X_train = X_train.copy()
    meta_train = meta_train.copy()

    for cluster_id, g in meta_train.groupby("cluster"):
        if cluster_id == -1:
            continue

        n = len(g)
        if n < MIN_SIGNATURE_TRAIN_JAGT:
            continue

        cp_wins = int((g["true_winner"] == "CP").sum())
        kb_wins = int((g["true_winner"] == "KB").sum())

        if cp_wins == kb_wins:
            continue

        winner = "CP" if cp_wins > kb_wins else "KB"
        loser_wins = min(cp_wins, kb_wins)
        winner_wins = max(cp_wins, kb_wins)
        strength = (winner_wins - loser_wins) / max(n, 1)

        features = get_cluster_signature_features(zscores, cluster_id, top_k)
        features = [f for f in features if f in X_train.columns]
        if not features:
            continue

        train_positions = list(g.index)
        centroid = X_train.loc[train_positions, features].mean(axis=0).values.astype(float)

        signatures.append({
            "cluster": int(cluster_id),
            "winner": winner,
            "n_train_jagt": int(n),
            "cp_wins_train": cp_wins,
            "kb_wins_train": kb_wins,
            "strength": float(strength),
            "features": features,
            "centroid": centroid,
            "top_k": int(top_k),
        })

    return signatures


def score_series_against_signatures(X_test, signatures):
    n = len(X_test)
    cp_scores_all = []
    kb_scores_all = []
    best_cp_cluster = np.full(n, fill_value=-999, dtype=int)
    best_kb_cluster = np.full(n, fill_value=-999, dtype=int)

    for sig in signatures:
        feats = sig["features"]
        if not feats:
            continue
        Xv = X_test[feats].values.astype(float)
        centroid = sig["centroid"].astype(float)
        d = np.linalg.norm(Xv - centroid, axis=1)
        d_norm = d / np.sqrt(max(len(feats), 1))
        sim = np.exp(-d_norm) * sig["strength"]

        if sig["winner"] == "CP":
            cp_scores_all.append((sim, sig["cluster"]))
        else:
            kb_scores_all.append((sim, sig["cluster"]))

    if cp_scores_all:
        cp_stack = np.vstack([x[0] for x in cp_scores_all])
        cp_score = cp_stack.mean(axis=0) if SIGNATURE_SCORE_MODE == "weighted_mean" else cp_stack.max(axis=0)
        best_cp_idx = np.argmax(cp_stack, axis=0)
        cp_clusters = [x[1] for x in cp_scores_all]
        best_cp_cluster = np.array([cp_clusters[i] for i in best_cp_idx], dtype=int)
    else:
        cp_score = np.zeros(n)

    if kb_scores_all:
        kb_stack = np.vstack([x[0] for x in kb_scores_all])
        kb_score = kb_stack.mean(axis=0) if SIGNATURE_SCORE_MODE == "weighted_mean" else kb_stack.max(axis=0)
        best_kb_idx = np.argmax(kb_stack, axis=0)
        kb_clusters = [x[1] for x in kb_scores_all]
        best_kb_cluster = np.array([kb_clusters[i] for i in best_kb_idx], dtype=int)
    else:
        kb_score = np.zeros(n)

    return cp_score, kb_score, best_cp_cluster, best_kb_cluster


def summarize_signatures_for_split(signatures, split_id):
    rows = []
    for sig in signatures:
        rows.append({
            "split_id": split_id,
            "cluster": sig["cluster"],
            "winner": sig["winner"],
            "n_train_jagt": sig["n_train_jagt"],
            "cp_wins_train": sig["cp_wins_train"],
            "kb_wins_train": sig["kb_wins_train"],
            "strength": sig["strength"],
            "top_k": sig["top_k"],
            "features": "; ".join(sig["features"]),
        })
    return rows


def run_cluster_signature_holdout(X, meta, zscores):
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = meta["true_cp_win"].values.astype(int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Only one winner class is present.")

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows = []
    pred_rows = []
    signature_rows = []
    rng = np.random.default_rng(RANDOM_STATE)

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"Cluster-signature split {split_id + 1}/{N_REPEATS}")

        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        X_test_full = X.iloc[test_idx].reset_index(drop=True)
        X_train_full, X_test_full = remove_constant_columns(X_train_full, X_test_full)

        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        majority_class = int(np.round(np.mean(y_train)))
        y_majority = np.full_like(y_test, majority_class)

        cp_prev = float(np.mean(y_train))
        y_random_prev = (rng.random(len(y_test)) < cp_prev).astype(int)
        y_random_5050 = (rng.random(len(y_test)) < 0.5).astype(int)

        for model_name, pred in [
            ("majority_baseline", y_majority),
            ("random_prevalence_baseline", y_random_prev),
            ("random_50_50_baseline", y_random_5050),
        ]:
            m = evaluate_predictions(y_test, pred)
            m.update({
                "split_id": split_id,
                "model": model_name,
                "top_k": 0,
                "n_signatures": 0,
                "n_cp_signatures": 0,
                "n_kb_signatures": 0,
                "train_cp_prevalence": cp_prev,
            })
            metric_rows.append(m)

        for top_k in SIGNATURE_TOP_FEATURES:
            signatures = build_train_cluster_signatures(
                X_train=X_train_full,
                meta_train=meta_train,
                zscores=zscores,
                top_k=top_k,
            )
            signature_rows.extend(summarize_signatures_for_split(signatures, split_id))

            n_cp_sig = sum(1 for s in signatures if s["winner"] == "CP")
            n_kb_sig = sum(1 for s in signatures if s["winner"] == "KB")

            if not signatures or n_cp_sig == 0 or n_kb_sig == 0:
                y_pred = y_majority.copy()
                cp_score = np.zeros(len(y_test))
                kb_score = np.zeros(len(y_test))
                best_cp_cluster = np.full(len(y_test), -999, dtype=int)
                best_kb_cluster = np.full(len(y_test), -999, dtype=int)
            else:
                cp_score, kb_score, best_cp_cluster, best_kb_cluster = score_series_against_signatures(
                    X_test_full,
                    signatures,
                )
                y_pred = (cp_score > kb_score).astype(int)

            m = evaluate_predictions(y_test, y_pred)
            m.update({
                "split_id": split_id,
                "model": "cluster_signature_rule",
                "top_k": top_k,
                "n_signatures": len(signatures),
                "n_cp_signatures": n_cp_sig,
                "n_kb_signatures": n_kb_sig,
                "train_cp_prevalence": cp_prev,
            })
            metric_rows.append(m)

            out = meta_test.copy()
            out["split_id"] = split_id
            out["model"] = "cluster_signature_rule"
            out["top_k"] = top_k
            out["pred_cp_win"] = y_pred
            out["pred_winner"] = np.where(y_pred == 1, "CP", "KB")
            out["cp_signature_score"] = cp_score
            out["kb_signature_score"] = kb_score
            out["signature_score_margin_cp_minus_kb"] = cp_score - kb_score
            out["best_cp_signature_cluster"] = best_cp_cluster
            out["best_kb_signature_cluster"] = best_kb_cluster
            pred_rows.append(out)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    signatures = pd.DataFrame(signature_rows)
    return metrics, predictions, signatures


def aggregate_unique_signature_predictions(predictions, top_k):
    df = predictions[
        (predictions["model"] == "cluster_signature_rule")
        & (predictions["top_k"] == top_k)
    ].copy()

    rows = []
    for key, g in df.groupby("key"):
        true_cp = int(g["true_cp_win"].iloc[0])
        pred_cp_rate = float(g["pred_cp_win"].mean())
        pred_cp = int(pred_cp_rate >= 0.5)

        rows.append({
            "key": key,
            "true_cp_win": true_cp,
            "true_winner": "CP" if true_cp else "KB",
            "pred_cp_win": pred_cp,
            "pred_winner": "CP" if pred_cp else "KB",
            "pred_cp_rate": pred_cp_rate,
            "mean_cp_signature_score": float(g["cp_signature_score"].mean()),
            "mean_kb_signature_score": float(g["kb_signature_score"].mean()),
            "mean_signature_margin_cp_minus_kb": float(g["signature_score_margin_cp_minus_kb"].mean()),
            "n_test_appearances": int(len(g)),
            "correct_prediction": int(pred_cp == true_cp),
            "correct_cp_prediction": int((true_cp == 1) and (pred_cp == 1)),
            "correct_kb_prediction": int((true_cp == 0) and (pred_cp == 0)),
        })

    return pd.DataFrame(rows)


def plot_signature_recommender_summary(unique_predictions, prefix, top_k):
    outdir = os.path.join(RESULTS_DIR, "cluster_signature_presentation_plots")
    os.makedirs(outdir, exist_ok=True)
    df = unique_predictions.copy()

    counts = pd.DataFrame([
        {"category": "CP-SSD wins", "count": int((df["true_winner"] == "CP").sum())},
        {"category": "KB-KSSD wins", "count": int((df["true_winner"] == "KB").sum())},
        {"category": "Predicted CP-SSD", "count": int((df["pred_winner"] == "CP").sum())},
        {"category": "Correct predictions", "count": int(df["correct_prediction"].sum())},
    ])
    counts.to_csv(os.path.join(outdir, f"{prefix}_signature_top{top_k}_summary_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(counts["category"], counts["count"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{int(h)}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title(f"Cluster-signature SSD recommendation, top-{top_k} features")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts["category"], rotation=18, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_signature_top{top_k}_summary_counts.png")
    plt.savefig(path, dpi=300)
    plt.close()

    correct_cp = int(((df["true_winner"] == "CP") & (df["pred_winner"] == "CP")).sum())
    false_cp = int(((df["true_winner"] == "KB") & (df["pred_winner"] == "CP")).sum())
    correct_kb = int(((df["true_winner"] == "KB") & (df["pred_winner"] == "KB")).sum())
    false_kb = int(((df["true_winner"] == "CP") & (df["pred_winner"] == "KB")).sum())

    cf = pd.DataFrame([
        {"category": "Correct CP", "count": correct_cp},
        {"category": "False CP", "count": false_cp},
        {"category": "Correct KB", "count": correct_kb},
        {"category": "False KB", "count": false_kb},
    ])
    cf.to_csv(os.path.join(outdir, f"{prefix}_signature_top{top_k}_correct_false_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(cf["category"], cf["count"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{int(h)}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title(f"Correct/false cluster-signature recommendations, top-{top_k}")
    ax.set_xticks(np.arange(len(cf)))
    ax.set_xticklabels(cf["category"], rotation=18, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    path2 = os.path.join(outdir, f"{prefix}_signature_top{top_k}_correct_false_counts.png")
    plt.savefig(path2, dpi=300)
    plt.close()

    return path, path2


def plot_signature_model_ranking(signature_summary, prefix):
    outdir = os.path.join(RESULTS_DIR, "cluster_signature_presentation_plots")
    os.makedirs(outdir, exist_ok=True)

    sub = signature_summary[
        signature_summary["model"] == "cluster_signature_rule"
    ].copy().sort_values("accuracy", ascending=True)

    if sub.empty:
        return None

    sub["label"] = "Top-" + sub["top_k"].astype(int).astype(str) + " features"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(sub["label"], 100 * sub["accuracy"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean holdout accuracy (%)")
    ax.set_title("Cluster-signature recommender: feature-combination size")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["label"], rotation=0, ha="center")
    prettify_axes(ax)
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_signature_model_ranking_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


# ============================================================
# HYBRID CLUSTER-SIGNATURE + TSFRESH RECOMMENDER
# ============================================================

def make_signature_meta_features(X, signatures):
    """
    Convert cluster-signature similarity scores into supervised meta-features.

    These features are derived from the existing clustering:
      - similarity to CP-favouring signatures,
      - similarity to KB-favouring signatures,
      - score margin,
      - IDs of closest CP/KB signature clusters,
      - number and strength of signatures.

    No UMAP/HDBSCAN is re-run here.
    """
    cp_score, kb_score, best_cp_cluster, best_kb_cluster = score_series_against_signatures(
        X,
        signatures,
    )

    n_cp_sig = sum(1 for s in signatures if s["winner"] == "CP")
    n_kb_sig = sum(1 for s in signatures if s["winner"] == "KB")
    cp_strength_sum = sum(s["strength"] for s in signatures if s["winner"] == "CP")
    kb_strength_sum = sum(s["strength"] for s in signatures if s["winner"] == "KB")

    eps = 1e-12

    meta = pd.DataFrame({
        "sig_cp_score": cp_score,
        "sig_kb_score": kb_score,
        "sig_margin_cp_minus_kb": cp_score - kb_score,
        "sig_abs_margin": np.abs(cp_score - kb_score),
        "sig_cp_over_kb_ratio": cp_score / (kb_score + eps),
        "sig_kb_over_cp_ratio": kb_score / (cp_score + eps),
        "sig_best_cp_cluster": best_cp_cluster.astype(float),
        "sig_best_kb_cluster": best_kb_cluster.astype(float),
        "sig_n_cp_signatures": float(n_cp_sig),
        "sig_n_kb_signatures": float(n_kb_sig),
        "sig_cp_strength_sum": float(cp_strength_sum),
        "sig_kb_strength_sum": float(kb_strength_sum),
        "sig_strength_margin_cp_minus_kb": float(cp_strength_sum - kb_strength_sum),
    })

    return meta


def run_hybrid_signature_supervised_holdout(X, meta, zscores):
    """
    Repeated holdout for the hybrid recommender.

    Training split:
      1. Build cluster signatures from training JAGT labels.
      2. Compute signature-similarity meta-features for train/test.
      3. Select top TSFresh features from training data only.
      4. Train model on [selected TSFresh features + signature meta-features].
      5. Evaluate on held-out test split.

    This makes clustering part of the recommender without requiring re-clustering.
    """
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = meta["true_cp_win"].values.astype(int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Only one winner class is present.")

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows = []
    pred_rows = []
    feature_rows = []
    signature_rows = []

    rng = np.random.default_rng(RANDOM_STATE)

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"Hybrid signature split {split_id + 1}/{N_REPEATS}")

        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        X_test_full = X.iloc[test_idx].reset_index(drop=True)
        X_train_full, X_test_full = remove_constant_columns(X_train_full, X_test_full)

        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        y_train = y[train_idx]
        y_test = y[test_idx]

        majority_class = int(np.round(np.mean(y_train)))
        y_majority = np.full_like(y_test, majority_class)

        cp_prev = float(np.mean(y_train))
        y_random_prev = (rng.random(len(y_test)) < cp_prev).astype(int)
        y_random_5050 = (rng.random(len(y_test)) < 0.5).astype(int)

        for model_name, pred in [
            ("majority_baseline", y_majority),
            ("random_prevalence_baseline", y_random_prev),
            ("random_50_50_baseline", y_random_5050),
        ]:
            m = evaluate_predictions(y_test, pred)
            m.update({
                "split_id": split_id,
                "model": model_name,
                "feature_selection": "none",
                "base_top_n": 0,
                "signature_top_k": 0,
                "n_signatures": 0,
                "n_cp_signatures": 0,
                "n_kb_signatures": 0,
                "train_cp_prevalence": cp_prev,
            })
            metric_rows.append(m)

        for signature_top_k in HYBRID_SIGNATURE_TOP_K:
            signatures = build_train_cluster_signatures(
                X_train=X_train_full,
                meta_train=meta_train,
                zscores=zscores,
                top_k=signature_top_k,
            )
            signature_rows.extend(summarize_signatures_for_split(signatures, split_id))

            n_cp_sig = sum(1 for s in signatures if s["winner"] == "CP")
            n_kb_sig = sum(1 for s in signatures if s["winner"] == "KB")

            sig_train = make_signature_meta_features(X_train_full, signatures)
            sig_test = make_signature_meta_features(X_test_full, signatures)

            for fs_method in HYBRID_FEATURE_SELECTION_METHODS:
                for base_top_n in HYBRID_BASE_TOP_FEATURES:
                    selected, feature_scores = select_features_supervised(
                        X_train_full,
                        y_train,
                        method=fs_method,
                        top_n=base_top_n,
                    )

                    X_train_hybrid = pd.concat(
                        [
                            X_train_full[selected].reset_index(drop=True),
                            sig_train.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    X_test_hybrid = pd.concat(
                        [
                            X_test_full[selected].reset_index(drop=True),
                            sig_test.reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    for model_name in HYBRID_MODELS:
                        clf = make_classifier(model_name)
                        clf.fit(X_train_hybrid, y_train)
                        y_pred = clf.predict(X_test_hybrid)

                        m = evaluate_predictions(y_test, y_pred)
                        m.update({
                            "split_id": split_id,
                            "model": f"hybrid_{model_name}",
                            "feature_selection": fs_method,
                            "base_top_n": base_top_n,
                            "signature_top_k": signature_top_k,
                            "n_signatures": len(signatures),
                            "n_cp_signatures": n_cp_sig,
                            "n_kb_signatures": n_kb_sig,
                            "train_cp_prevalence": cp_prev,
                        })
                        metric_rows.append(m)

                        out = meta_test.copy()
                        out["split_id"] = split_id
                        out["model"] = f"hybrid_{model_name}"
                        out["feature_selection"] = fs_method
                        out["base_top_n"] = base_top_n
                        out["signature_top_k"] = signature_top_k
                        out["pred_cp_win"] = y_pred
                        out["pred_winner"] = np.where(y_pred == 1, "CP", "KB")

                        for c in sig_test.columns:
                            out[c] = sig_test[c].values

                        pred_rows.append(out)

                        fitted = clf
                        if hasattr(fitted, "feature_importances_"):
                            importances = fitted.feature_importances_
                        else:
                            importances = None

                        if importances is not None:
                            for feat, imp in zip(X_train_hybrid.columns, importances):
                                feature_rows.append({
                                    "split_id": split_id,
                                    "model": f"hybrid_{model_name}",
                                    "feature_selection": fs_method,
                                    "base_top_n": base_top_n,
                                    "signature_top_k": signature_top_k,
                                    "feature": feat,
                                    "importance": float(imp),
                                    "is_signature_feature": str(feat).startswith("sig_"),
                                })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    feature_importances = pd.DataFrame(feature_rows)
    signatures = pd.DataFrame(signature_rows)

    return metrics, predictions, feature_importances, signatures


def aggregate_unique_hybrid_predictions(
    predictions,
    model,
    feature_selection,
    base_top_n,
    signature_top_k,
):
    df = predictions[
        (predictions["model"] == model)
        & (predictions["feature_selection"] == feature_selection)
        & (predictions["base_top_n"] == base_top_n)
        & (predictions["signature_top_k"] == signature_top_k)
    ].copy()

    rows = []
    for key, g in df.groupby("key"):
        true_cp = int(g["true_cp_win"].iloc[0])
        pred_cp_rate = float(g["pred_cp_win"].mean())
        pred_cp = int(pred_cp_rate >= 0.5)

        rows.append({
            "key": key,
            "true_cp_win": true_cp,
            "true_winner": "CP" if true_cp else "KB",
            "pred_cp_win": pred_cp,
            "pred_winner": "CP" if pred_cp else "KB",
            "pred_cp_rate": pred_cp_rate,
            "mean_sig_cp_score": float(g["sig_cp_score"].mean()) if "sig_cp_score" in g else np.nan,
            "mean_sig_kb_score": float(g["sig_kb_score"].mean()) if "sig_kb_score" in g else np.nan,
            "mean_sig_margin_cp_minus_kb": float(g["sig_margin_cp_minus_kb"].mean()) if "sig_margin_cp_minus_kb" in g else np.nan,
            "n_test_appearances": int(len(g)),
            "correct_prediction": int(pred_cp == true_cp),
            "correct_cp_prediction": int((true_cp == 1) and (pred_cp == 1)),
            "correct_kb_prediction": int((true_cp == 0) and (pred_cp == 0)),
        })

    return pd.DataFrame(rows)


def summarize_hybrid_feature_importances(feature_importances, best, prefix):
    if feature_importances.empty:
        return pd.DataFrame()

    df = feature_importances[
        (feature_importances["model"] == best["model"])
        & (feature_importances["feature_selection"] == best["feature_selection"])
        & (feature_importances["base_top_n"] == best["base_top_n"])
        & (feature_importances["signature_top_k"] == best["signature_top_k"])
    ].copy()

    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["feature", "is_signature_feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_splits=("split_id", "nunique"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    agg.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_best_model_feature_importances.csv"), index=False)

    sub = agg.head(20).sort_values("mean_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_importance"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=7)
    ax.set_xlabel("Mean feature importance")
    ax.set_title("Top features of the best hybrid SSD recommender")
    prettify_axes(ax)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{prefix}_hybrid_best_model_feature_importances.png")
    plt.savefig(path, dpi=300)
    plt.close()

    sig_share = (
        agg.groupby("is_signature_feature", as_index=False)
        .agg(total_importance=("mean_importance", "sum"))
    )
    sig_share.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_signature_feature_importance_share.csv"), index=False)

    return agg


def plot_hybrid_summary(unique_predictions, prefix, label):
    outdir = os.path.join(RESULTS_DIR, "hybrid_presentation_plots")
    os.makedirs(outdir, exist_ok=True)
    df = unique_predictions.copy()

    counts = pd.DataFrame([
        {"category": "CP-SSD wins", "count": int((df["true_winner"] == "CP").sum())},
        {"category": "KB-KSSD wins", "count": int((df["true_winner"] == "KB").sum())},
        {"category": "Predicted CP-SSD", "count": int((df["pred_winner"] == "CP").sum())},
        {"category": "Correct predictions", "count": int(df["correct_prediction"].sum())},
    ])
    counts.to_csv(os.path.join(outdir, f"{prefix}_hybrid_summary_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(counts["category"], counts["count"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{int(h)}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title(f"Hybrid cluster-signature SSD recommendation\\n{label}")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts["category"], rotation=18, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_hybrid_summary_counts.png")
    plt.savefig(path, dpi=300)
    plt.close()

    correct_cp = int(((df["true_winner"] == "CP") & (df["pred_winner"] == "CP")).sum())
    false_cp = int(((df["true_winner"] == "KB") & (df["pred_winner"] == "CP")).sum())
    correct_kb = int(((df["true_winner"] == "KB") & (df["pred_winner"] == "KB")).sum())
    false_kb = int(((df["true_winner"] == "CP") & (df["pred_winner"] == "KB")).sum())

    cf = pd.DataFrame([
        {"category": "Correct CP", "count": correct_cp},
        {"category": "False CP", "count": false_cp},
        {"category": "Correct KB", "count": correct_kb},
        {"category": "False KB", "count": false_kb},
    ])
    cf.to_csv(os.path.join(outdir, f"{prefix}_hybrid_correct_false_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(cf["category"], cf["count"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{int(h)}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of JAGT time series")
    ax.set_title("Correct and false hybrid recommendations")
    ax.set_xticks(np.arange(len(cf)))
    ax.set_xticklabels(cf["category"], rotation=18, ha="right")
    prettify_axes(ax)
    plt.tight_layout()
    path2 = os.path.join(outdir, f"{prefix}_hybrid_correct_false_counts.png")
    plt.savefig(path2, dpi=300)
    plt.close()

    return path, path2


def plot_hybrid_model_ranking(summary, prefix):
    outdir = os.path.join(RESULTS_DIR, "hybrid_presentation_plots")
    os.makedirs(outdir, exist_ok=True)

    sub = summary[summary["model"].astype(str).str.startswith("hybrid_")].copy()
    if sub.empty:
        return None

    sub["label"] = (
        sub["model"].astype(str).str.replace("hybrid_", "", regex=False)
        + " | "
        + sub["feature_selection"].astype(str)
        + " | tsf="
        + sub["base_top_n"].astype(int).astype(str)
        + " | sig="
        + sub["signature_top_k"].astype(int).astype(str)
    )

    sub = sub.sort_values("accuracy", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(sub))
    ax.barh(y, 100 * sub["accuracy"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean holdout accuracy (%)")
    ax.set_title("Hybrid SSD recommender: top configurations")
    prettify_axes(ax)
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_hybrid_model_ranking_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    print("\nHybrid cluster-signature SSD recommendation experiment")
    print("=" * 70)
    print(f"ANALYSIS_MODE = {ANALYSIS_MODE}")
    print(f"OUTDIR        = {OUTDIR}")
    print("Using existing clustering cache; clustering is NOT re-run.")

    selected = choose_selected_clustering()
    variant = selected["variant"]
    config = selected["config_name"]

    prefix, cluster_eval, zscores = load_selected_tables(selected)
    result, _ = load_cached_result(variant, config)
    jagt_methods = load_jagt_method_table()

    print(f"\nSelected existing clustering:")
    print(f"variant     = {variant}")
    print(f"config_name = {config}")
    print(f"n_series in cached clustering = {len(result.series_keys)}")

    if ANALYSIS_MODE == "steady_jagt_only":
        non_jagt = [k for k in result.series_keys if k not in jagt_methods.index]
        if non_jagt:
            raise RuntimeError(
                f"Expected only JAGT series, but found {len(non_jagt)} non-JAGT keys in cached result."
            )

    X, meta, jagt_audit = build_jagt_dataset(
        result,
        result.labels,
        jagt_methods,
        return_audit=True,
    )

    audit_path, audit_reason_counts, audit_win_counts = summarize_separate_wins_audit(jagt_audit, prefix)

    X.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_JAGT_feature_matrix_decisive_only.csv"), index=False)
    meta.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_JAGT_prediction_dataset_meta_decisive_only.csv"), index=False)

    print(f"\nFull JAGT separate-wins counts:")
    print(audit_win_counts)

    print("\nFull JAGT outcome reasons:")
    print(audit_reason_counts)

    print(f"\nDecisive JAGT prediction dataset used for evaluation: n={len(meta)}")
    print(meta["true_winner"].value_counts())

    metrics, predictions, feature_importances, signatures = run_hybrid_signature_supervised_holdout(
        X,
        meta,
        zscores,
    )

    metrics.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_metrics_all_splits.csv"), index=False)
    predictions.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_predictions_all_test_appearances.csv"), index=False)
    feature_importances.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_feature_importances_all.csv"), index=False)
    signatures.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_cluster_signature_definitions_by_split.csv"), index=False)

    summary = (
        metrics.groupby(["model", "feature_selection", "base_top_n", "signature_top_k"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            cp_precision=("cp_precision", "mean"),
            cp_recall=("cp_recall", "mean"),
            cp_f1=("cp_f1", "mean"),
            n=("n", "mean"),
            n_true_cp=("n_true_cp", "mean"),
            n_pred_cp=("n_pred_cp", "mean"),
            n_signatures=("n_signatures", "mean"),
            n_cp_signatures=("n_cp_signatures", "mean"),
            n_kb_signatures=("n_kb_signatures", "mean"),
        )
        .sort_values(["accuracy", "balanced_accuracy", "cp_f1"], ascending=[False, False, False])
    )

    summary.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_model_summary.csv"), index=False)

    non_baseline = summary[summary["model"].astype(str).str.startswith("hybrid_")].copy()
    if non_baseline.empty:
        raise RuntimeError("No hybrid model results were produced.")

    best = non_baseline.iloc[0]

    unique_predictions = aggregate_unique_hybrid_predictions(
        predictions,
        model=str(best["model"]),
        feature_selection=str(best["feature_selection"]),
        base_top_n=int(best["base_top_n"]),
        signature_top_k=int(best["signature_top_k"]),
    )

    unique_predictions.to_csv(
        os.path.join(RESULTS_DIR, f"{prefix}_hybrid_best_unique_series_predictions.csv"),
        index=False,
    )

    importances = summarize_hybrid_feature_importances(feature_importances, best, prefix)

    label = (
        f"{best['model']}, {best['feature_selection']}, "
        f"TSFresh top-{int(best['base_top_n'])}, signature top-{int(best['signature_top_k'])}"
    )
    p1, p2 = plot_hybrid_summary(unique_predictions, prefix, label)
    p3 = plot_hybrid_model_ranking(summary, prefix)

    print("\nBest hybrid cluster-signature configuration:")
    print(best)

    counts = pd.DataFrame([
        {"category": "CP-SSD wins", "count": int(unique_predictions["true_cp_win"].sum())},
        {"category": "KB-KSSD wins", "count": int((unique_predictions["true_cp_win"] == 0).sum())},
        {"category": "Predicted CP-SSD wins", "count": int(unique_predictions["pred_cp_win"].sum())},
        {"category": "Correct predictions", "count": int(unique_predictions["correct_prediction"].sum())},
    ])

    print("\nUnique-series summary counts:")
    print(counts)

    majority_acc = float((meta["true_winner"] == "KB").mean())
    best_acc = float(best["accuracy"])
    unique_acc = float(unique_predictions["correct_prediction"].mean())

    print(f"\nMajority-KB baseline on decisive JAGT: {majority_acc:.4f}")
    print(f"Best hybrid mean holdout accuracy: {best_acc:.4f}")
    print(f"Best hybrid unique-series accuracy: {unique_acc:.4f}")

    print("\nSaved main outputs:")
    print(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_model_summary.csv"))
    print(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_cluster_signature_definitions_by_split.csv"))
    print(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_best_unique_series_predictions.csv"))
    print(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_best_model_feature_importances.csv"))
    print(os.path.join(RESULTS_DIR, f"{prefix}_hybrid_signature_feature_importance_share.csv"))
    print(p1)
    print(p2)
    if p3:
        print(p3)

    print(f"\nDone. Outputs saved in:\n{RESULTS_DIR}/")

if __name__ == "__main__":
    main()
