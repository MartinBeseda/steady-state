#!/usr/bin/env python3
"""
Predict whether KB-KSSD or CP-SSD performs better from TSFresh features.

This script is meant as a post-processing analysis after the TSFresh clustering
and CP/Kb comparison scripts have already run.

Goal
----
For every steady JAGT timeseries, use TSFresh features as input X and define
classification targets such as:

    y = 1  -> KB-KSSD has lower absolute error than CP-SSD
    y = 0  -> CP-SSD has lower absolute error than KB-KSSD

Ties and cases where both methods fail can be excluded by default.

Outputs
-------
The script writes results to:

    tsfresh_testing_2/kb_vs_cp_feature_classifier/

including:
    - train/test metrics CSV
    - cross-validation metrics CSV
    - confusion matrices
    - ROC/PR curves
    - feature-importance tables and plots
    - prediction table for inspected JAGT series

Expected input files
--------------------
1) Cached TSFresh features:

    tsfresh_testing_2/feature_cache/drop_first1_normalized_efficient_features.pkl

2) Method-comparison table:

    tsfresh_testing_2/cluster_method_comparison/drop_first1_efficient_best_cp_kb_per_cluster.csv

   This table is useful for cluster-level summaries, but per-series labels are
   more reliably reconstructed from:

    ../man_steady_comparison/ssd_best_cpssd_comparison_results/ssd_best_config_comparison_table.csv

3) Series-key order:

    ../man_steady_comparison/series_kbkssd.json

The order of series_kbkssd.json must match cd.load_all_json(...), which is the
assumption already used in your clustering script.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================

OUTDIR = Path("tsfresh_testing_2/kb_vs_cp_feature_classifier")
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = Path(
    "tsfresh_testing_2/feature_cache/drop_first1_normalized_efficient_features.pkl"
)

SERIES_INDEX_PATH = Path("series_kbkssd.json")

METHOD_TABLE_PATH = Path(
    "../man_steady_comparison/ssd_best_cpssd_comparison_results/"
    "ssd_best_config_comparison_table.csv"
)

# Choose which CP-SSD variant to compare against KB-KSSD.
# Options: "cp_best" or "cp_orig"
CP_VARIANTS = ["cp_best", "cp_orig"]

# Drop first N points was used before feature extraction.
DROP_FIRST_N = 1

# If True, rows where CP and KB tie are removed from the binary classification.
DROP_TIES = True

# If True, rows where both methods falsely return unsteady are removed.
DROP_BOTH_WRONG = True

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5
TOP_N_FEATURES = 30


# ============================================================
# FEATURE FAMILY MAPPING
# ============================================================


def feature_family(feature: str) -> str:
    """Coarse human-readable grouping of TSFresh feature names."""
    f = feature.lower()

    if "cwt" in f or "fft" in f or "spkt_welch" in f:
        return "Frequency / wavelet structure"
    if "entropy" in f or "lempel" in f or "permutation" in f:
        return "Complexity / entropy"
    if "autocorrelation" in f or "c3" in f or "time_reversal" in f:
        return "Autocorrelation / temporal dependence"
    if "change_quantiles" in f or "mean_change" in f or "mean_abs_change" in f:
        return "Local variability"
    if "linear_trend" in f or "agg_linear_trend" in f:
        return "Trend / nonstationarity"
    if "standard_deviation" in f or "variance" in f or "kurtosis" in f or "skewness" in f:
        return "Distribution spread / shape"
    if "ratio_beyond" in f or "large_standard_deviation" in f or "range_count" in f:
        return "Bursts / heavy tails"
    if "number_peaks" in f or "number_cwt_peaks" in f:
        return "Peak structure"
    if "quantile" in f or "median" in f or "minimum" in f or "maximum" in f or "mean" in f:
        return "Level / distribution position"
    return "Other"


# ============================================================
# DATA LOADING
# ============================================================


def load_series_keys(path: Path, expected_len: int | None = None) -> list[str]:
    with open(path) as f:
        d = json.load(f)

    keys = list(d.keys())

    if expected_len is not None and len(keys) != expected_len:
        raise ValueError(
            f"Series-key count mismatch: {len(keys)} keys vs {expected_len} features"
        )

    return keys



def load_inputs():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature cache: {FEATURE_PATH}")
    if not SERIES_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing series index: {SERIES_INDEX_PATH}")
    if not METHOD_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing method table: {METHOD_TABLE_PATH}")

    print(f"Loading features: {FEATURE_PATH}")
    features = pd.read_pickle(FEATURE_PATH)

    print(f"Loading series keys: {SERIES_INDEX_PATH}")
    keys_full = load_series_keys(SERIES_INDEX_PATH)

    # Match the drop-first feature set. The previous clustering kept series with len > n.
    # Since we do not reload raw data here, we assume all original 5860 series survived drop_first1,
    # which was true in your logs. If lengths differ, fail loudly.
    if len(keys_full) != len(features):
        raise ValueError(
            f"Feature/key mismatch: {len(features)} feature rows vs {len(keys_full)} keys. "
            "If some series were removed by drop_first_n, create and save the key list in the clustering script."
        )

    features = features.copy()
    features.index = keys_full

    print(f"Loading method table: {METHOD_TABLE_PATH}")
    method_df = pd.read_csv(METHOD_TABLE_PATH)

    if "key" not in method_df.columns:
        raise ValueError("Method table must contain a 'key' column.")

    method_df = method_df.set_index("key", drop=False)

    common = features.index.intersection(method_df.index)
    print(f"Common steady JAGT series: {len(common)}")

    if len(common) == 0:
        raise ValueError("No overlap between feature rows and method-comparison rows.")

    return features.loc[common], method_df.loc[common]


# ============================================================
# TARGET CONSTRUCTION
# ============================================================


def build_target(method_df: pd.DataFrame, cp_variant: str):
    """
    Construct binary target:
        1 -> KB-KSSD better
        0 -> CP-SSD better

    Tie/both-wrong rows can be removed depending on config.
    """
    cp_err_col = f"{cp_variant}_abs_err"
    cp_steady_col = f"{cp_variant}_steady"

    required = [cp_err_col, "kb_abs_err", cp_steady_col, "kb_steady"]
    missing = [c for c in required if c not in method_df.columns]
    if missing:
        raise ValueError(f"Missing columns for {cp_variant}: {missing}")

    rows = []

    for key, r in method_df.iterrows():
        cp_err = r[cp_err_col]
        kb_err = r["kb_abs_err"]

        cp_bad = pd.isna(cp_err)
        kb_bad = pd.isna(kb_err)

        if cp_bad and kb_bad:
            outcome = "both_wrong"
            y = np.nan
        elif cp_bad:
            outcome = "kb_better"
            y = 1
        elif kb_bad:
            outcome = "cp_better"
            y = 0
        elif kb_err < cp_err:
            outcome = "kb_better"
            y = 1
        elif cp_err < kb_err:
            outcome = "cp_better"
            y = 0
        else:
            outcome = "tie"
            y = np.nan

        rows.append({
            "key": key,
            "cp_variant": cp_variant,
            "cp_abs_err": cp_err,
            "kb_abs_err": kb_err,
            "cp_steady": bool(r[cp_steady_col]),
            "kb_steady": bool(r["kb_steady"]),
            "outcome": outcome,
            "target_kb_better": y,
            "kb_minus_cp_abs_err": kb_err - cp_err if not (cp_bad or kb_bad) else np.nan,
            "cp_minus_kb_abs_err": cp_err - kb_err if not (cp_bad or kb_bad) else np.nan,
        })

    target_df = pd.DataFrame(rows).set_index("key", drop=False)

    mask = target_df["target_kb_better"].notna()

    if not DROP_TIES:
        # If desired, ties could be kept separately, but this script is binary.
        pass

    if DROP_BOTH_WRONG:
        mask &= target_df["outcome"] != "both_wrong"

    target_df = target_df[mask].copy()
    target_df["target_kb_better"] = target_df["target_kb_better"].astype(int)

    return target_df


# ============================================================
# MODELING
# ============================================================


def make_models():
    models = {
        "logistic_l1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "logistic_l2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return models



def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)



def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = safe_predict_proba(model, X_test)

    out = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_kb": precision_score(y_test, y_pred, zero_division=0),
        "recall_kb": recall_score(y_test, y_pred, zero_division=0),
        "f1_kb": f1_score(y_test, y_pred, zero_division=0),
    }

    if len(np.unique(y_test)) == 2:
        out["roc_auc"] = roc_auc_score(y_test, y_score)
        out["average_precision"] = average_precision_score(y_test, y_score)
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    return out, y_pred, y_score



def crossval_model(model, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    row = {}
    for k, vals in scores.items():
        if not k.startswith("test_"):
            continue
        metric = k.replace("test_", "")
        row[f"cv_{metric}_mean"] = float(np.mean(vals))
        row[f"cv_{metric}_std"] = float(np.std(vals))

    return row


# ============================================================
# PLOTS
# ============================================================


def save_confusion_matrix_plot(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()



def save_roc_pr_plots(y_test, y_score, prefix):
    if len(np.unique(y_test)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve: KB-better prediction")
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve: KB-better prediction")
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_precision_recall_curve.png", dpi=300)
    plt.close()



def save_feature_importance_plot(df_imp, title, path, top_n=TOP_N_FEATURES):
    sub = df_imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(sub))))
    ax.barh(sub["feature_short"], sub["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--", axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()



def short_feature_name(name: str, max_len=70) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def logistic_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "importance": np.abs(coefs),
        "direction": np.where(coefs > 0, "KB-better", "CP-better"),
    })
    df["family"] = df["feature"].map(feature_family)
    df["feature_short"] = df["feature"].map(short_feature_name)
    return df.sort_values("importance", ascending=False)



def random_forest_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    })
    df["family"] = df["feature"].map(feature_family)
    df["feature_short"] = df["feature"].map(short_feature_name)
    return df.sort_values("importance", ascending=False)



def permutation_feature_importance(model, X_test, y_test, feature_names, prefix):
    print("Computing permutation importance...")
    r = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": r.importances_mean,
        "importance_std": r.importances_std,
    })
    df["family"] = df["feature"].map(feature_family)
    df["feature_short"] = df["feature"].map(short_feature_name)
    df = df.sort_values("importance", ascending=False)
    df.to_csv(OUTDIR / f"{prefix}_permutation_importance.csv", index=False)

    save_feature_importance_plot(
        df,
        title=f"Permutation importance ({prefix})",
        path=OUTDIR / f"{prefix}_permutation_importance.png",
    )

    return df



def save_family_importance(df_imp, prefix, importance_col="importance"):
    fam = (
        df_imp.groupby("family", as_index=False)[importance_col]
        .sum()
        .sort_values(importance_col, ascending=False)
    )
    fam.to_csv(OUTDIR / f"{prefix}_family_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = fam.iloc[::-1]
    ax.barh(sub["family"], sub[importance_col])
    ax.set_xlabel("Summed importance")
    ax.set_title(f"Feature-family importance ({prefix})")
    ax.grid(alpha=0.25, linestyle="--", axis="x")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{prefix}_family_importance.png", dpi=300)
    plt.close()

    return fam


# ============================================================
# MAIN ANALYSIS
# ============================================================


def run_for_cp_variant(features_all, method_df_all, cp_variant: str):
    print("\n" + "=" * 80)
    print(f"PREDICTING KB vs {cp_variant.upper()} FROM TSFRESH FEATURES")
    print("=" * 80)

    target_df = build_target(method_df_all, cp_variant=cp_variant)

    common = features_all.index.intersection(target_df.index)
    X = features_all.loc[common].copy()
    y = target_df.loc[common, "target_kb_better"].astype(int)

    print(f"Usable rows: {len(y)}")
    print("Class balance:")
    print(y.value_counts().rename(index={0: "CP better", 1: "KB better"}))

    if len(y.unique()) < 2:
        print("Only one class present. Skipping classifier.")
        return

    class_balance = y.value_counts().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = make_models()

    metrics_rows = []
    cv_rows = []
    prediction_tables = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")

        cv = crossval_model(model, X, y)
        cv["model"] = name
        cv["cp_variant"] = cp_variant
        cv_rows.append(cv)

        test_metrics, y_pred, y_score = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        row = {
            "cp_variant": cp_variant,
            "model": name,
            "n_total": len(y),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_cp_better": int((y == 0).sum()),
            "n_kb_better": int((y == 1).sum()),
            **test_metrics,
        }
        metrics_rows.append(row)

        print(pd.Series(row))
        print(classification_report(
            y_test,
            y_pred,
            target_names=["CP better", "KB better"],
            zero_division=0,
        ))

        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix_plot(
            cm,
            labels=["CP better", "KB better"],
            title=f"Confusion matrix: {name}, {cp_variant}",
            path=OUTDIR / f"{cp_variant}_{name}_confusion_matrix.png",
        )

        save_roc_pr_plots(
            y_test,
            y_score,
            prefix=f"{cp_variant}_{name}",
        )

        pred_df = pd.DataFrame({
            "key": X_test.index,
            "cp_variant": cp_variant,
            "model": name,
            "true_kb_better": y_test.values,
            "pred_kb_better": y_pred,
            "prob_kb_better": y_score,
        }).set_index("key", drop=False)

        joined = pred_df.join(target_df[[
            "outcome", "cp_abs_err", "kb_abs_err", "cp_minus_kb_abs_err"
        ]], how="left")
        prediction_tables.append(joined)

        trained_models[name] = model

    df_metrics = pd.DataFrame(metrics_rows).sort_values(
        ["balanced_accuracy", "roc_auc"], ascending=False
    )
    df_cv = pd.DataFrame(cv_rows).sort_values(
        ["cv_balanced_accuracy_mean", "cv_roc_auc_mean"], ascending=False
    )
    df_pred = pd.concat(prediction_tables, ignore_index=True)

    df_metrics.to_csv(OUTDIR / f"{cp_variant}_test_metrics.csv", index=False)
    df_cv.to_csv(OUTDIR / f"{cp_variant}_crossval_metrics.csv", index=False)
    df_pred.to_csv(OUTDIR / f"{cp_variant}_test_predictions.csv", index=False)

    print("\nTest metrics:")
    print(df_metrics)

    print("\nCV metrics:")
    print(df_cv)

    # Pick best model by CV balanced accuracy.
    best_name = df_cv.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model by CV balanced accuracy: {best_name}")

    # Feature importance.
    feature_names = list(X.columns)

    if best_name.startswith("logistic"):
        df_imp = logistic_feature_importance(best_model, feature_names)
        imp_prefix = f"{cp_variant}_{best_name}"
        df_imp.to_csv(OUTDIR / f"{imp_prefix}_feature_importance.csv", index=False)
        save_feature_importance_plot(
            df_imp,
            title=f"Feature importance: {best_name}, {cp_variant}",
            path=OUTDIR / f"{imp_prefix}_feature_importance.png",
        )
        save_family_importance(df_imp, imp_prefix)

        # Split by direction for interpretability.
        df_imp[df_imp["direction"] == "KB-better"].to_csv(
            OUTDIR / f"{imp_prefix}_features_pointing_to_KB.csv", index=False
        )
        df_imp[df_imp["direction"] == "CP-better"].to_csv(
            OUTDIR / f"{imp_prefix}_features_pointing_to_CP.csv", index=False
        )

    elif best_name == "random_forest":
        df_imp = random_forest_feature_importance(best_model, feature_names)
        imp_prefix = f"{cp_variant}_{best_name}"
        df_imp.to_csv(OUTDIR / f"{imp_prefix}_feature_importance.csv", index=False)
        save_feature_importance_plot(
            df_imp,
            title=f"Feature importance: {best_name}, {cp_variant}",
            path=OUTDIR / f"{imp_prefix}_feature_importance.png",
        )
        save_family_importance(df_imp, imp_prefix)
        permutation_feature_importance(best_model, X_test, y_test, feature_names, imp_prefix)

    else:
        # HGB has no direct feature_importances_, so use permutation importance.
        imp_prefix = f"{cp_variant}_{best_name}"
        df_imp = permutation_feature_importance(best_model, X_test, y_test, feature_names, imp_prefix)
        save_family_importance(df_imp, imp_prefix)

    # Also save permutation importance for logistic_l1 because it is usually the most interpretable.
    if "logistic_l1" in trained_models and best_name != "logistic_l1":
        logistic_model = trained_models["logistic_l1"]
        df_log = logistic_feature_importance(logistic_model, feature_names)
        imp_prefix = f"{cp_variant}_logistic_l1_interpretable"
        df_log.to_csv(OUTDIR / f"{imp_prefix}_feature_importance.csv", index=False)
        save_feature_importance_plot(
            df_log,
            title=f"Interpretable L1-logistic features: {cp_variant}",
            path=OUTDIR / f"{imp_prefix}_feature_importance.png",
        )
        save_family_importance(df_log, imp_prefix)

    # Short text summary.
    summary_path = OUTDIR / f"{cp_variant}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"KB vs {cp_variant} TSFresh classifier summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Usable rows: {len(y)}\n")
        f.write(f"Class balance: {class_balance}\n\n")
        f.write("Best model by CV balanced accuracy:\n")
        f.write(str(df_cv.iloc[0].to_dict()) + "\n\n")
        f.write("Test metrics ranked by balanced accuracy:\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n")

    print(f"Saved summary: {summary_path}")



def main():
    features_all, method_df_all = load_inputs()

    for cp_variant in CP_VARIANTS:
        run_for_cp_variant(features_all, method_df_all, cp_variant)

    print("\nDone.")
    print(f"Outputs saved in: {OUTDIR}/")


if __name__ == "__main__":
    main()
