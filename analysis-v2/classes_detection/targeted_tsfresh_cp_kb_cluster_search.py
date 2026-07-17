#!/usr/bin/env python3
"""
Targeted TSFresh clustering + CP-SSD / KB-KSSD cluster interpretation.

Goal
----
Find clusterings of drop-first-1, normalized, Efficient TSFresh features where:

1. clusters are reasonably meaningful internally;
2. steady JAGT series are represented in clusters;
3. CP-SSD and KB-KSSD differ inside some clusters;
4. those clusters can be explained by raw TSFresh features and TSFresh subfamilies.

Two variants are evaluated:
    A) all_timeseries
       Clustering uses all time series.
       CP/KB evaluation uses only steady JAGT series inside each cluster.

    B) steady_jagt_only
       Clustering uses only steady JAGT series.
       CP/KB evaluation uses all clustered points.

This script assumes Efficient TSFresh features are already cached at:
    tsfresh_testing_2/feature_cache/drop_first1_normalized_efficient_features.pkl

Outputs are written to:
    tsfresh_targeted_clustering_results/
"""

import os
import json
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import hdbscan

try:
    import umap
except ImportError as exc:
    raise ImportError("This script needs umap-learn installed: pip install umap-learn") from exc

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


# ============================================================
# CONFIG
# ============================================================

OUTDIR = "tsfresh_targeted_clustering_results"
os.makedirs(OUTDIR, exist_ok=True)

PLOTS_DIR = os.path.join(OUTDIR, "plots")
TABLES_DIR = os.path.join(OUTDIR, "tables")
BANDS_DIR = os.path.join(OUTDIR, "cluster_bands")
SIG_DIR = os.path.join(OUTDIR, "significant_clusters")

for d in [PLOTS_DIR, TABLES_DIR, BANDS_DIR, SIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Per-configuration cache.
# Each evaluated clustering configuration is stored immediately, so the sweep is resumable.
CONFIG_CACHE_DIR = os.path.join(OUTDIR, "configuration_cache")
os.makedirs(CONFIG_CACHE_DIR, exist_ok=True)

INCREMENTAL_RESULTS_PATH = os.path.join(TABLES_DIR, "all_clustering_configurations_incremental.csv")
FAILED_CONFIGS_PATH = os.path.join(TABLES_DIR, "failed_configurations.csv")

FEATURE_CACHE_PATH = "tsfresh_testing_2/feature_cache/drop_first1_normalized_efficient_features.pkl"
SERIES_INDEX_PATH = "series_kbkssd.json"
JAGT_METHOD_TABLE_PATH = "../man_steady_comparison/ssd_best_cpssd_comparison_results/ssd_best_config_comparison_table.csv"
DATA_GLOB = "../../data/timeseries/all/*.json"

RANDOM_STATE = 42

# Keep modest first. Expand after seeing first results.
PCA_DIMS = [10, 20, 30, 50, 75, 100]
MIN_CLUSTER_SIZES = [10, 20, 30, 50]

UMAP_PCA_DIMS = [20, 50, 100]
UMAP_DIMS = [5, 10, 20]
UMAP_NEIGHBORS = [10, 15, 30]
UMAP_MIN_DISTS = [0.0, 0.05, 0.1]
UMAP_MIN_CLUSTER_SIZES = [10, 20, 30, 50]

MIN_JAGT_PER_CLUSTER = 5
MIN_NON_TIE_WINS = 5
SIGNIFICANCE_ALPHA = 0.05

TOP_FEATURES_PER_CLUSTER = 15
TOP_SUBFAMILIES_PER_CLUSTER = 12
TOP_BAND_CLUSTERS = 3

# Keep meeting outputs compact: no heatmaps, only a few bar/curve plots.
SAVE_METHOD_BARPLOTS = True
SAVE_CLUSTER_BANDS = True
SAVE_SUBFAMILY_BARPLOTS = True
MAX_SUBFAMILY_BAR_CLUSTERS = 3


# ============================================================
# BASIC UTILS
# ============================================================

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def preprocess_normalized(data):
    out = []
    for ts in data:
        ts = np.asarray(ts, dtype=float)
        std = np.std(ts)
        if std == 0:
            out.append(ts - np.mean(ts))
        else:
            out.append((ts - np.mean(ts)) / std)
    return out


def drop_first_n_with_keys(data, keys, n=1):
    data_out, keys_out, old_indices = [], [], []
    for i, (ts, key) in enumerate(zip(data, keys)):
        if len(ts) > n:
            data_out.append(np.asarray(ts, dtype=float)[n:])
            keys_out.append(key)
            old_indices.append(i)
    return data_out, keys_out, old_indices


def load_series_keys(path, expected_len=None):
    with open(path) as f:
        d = json.load(f)
    keys = list(d.keys())
    if expected_len is not None and len(keys) != expected_len:
        raise ValueError(f"Series-key count mismatch: {len(keys)} keys vs {expected_len} series")
    return keys


def load_jagt_method_table(path):
    df = pd.read_csv(path)
    if "key" not in df.columns:
        raise ValueError(f"Expected column 'key' in {path}")
    return df.set_index("key", drop=False)


def load_cached_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached Efficient TSFresh features not found:\n{path}\nRun the Efficient TSFresh feature extraction first."
        )
    return pd.read_pickle(path)


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid = ~np.isnan(pvals)
    pv = pvals[valid]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[valid] = tmp
    return out


# ============================================================
# TSFRESH FEATURE NAME PARSING
# ============================================================

def tsfresh_subfamily(feature_name):
    """
    Convert a full TSFresh feature name into a subfamily name.

    value__time_reversal_asymmetry_statistic__lag_1 -> value__time_reversal_asymmetry_statistic
    value__cwt_coefficients__coeff_5__w_10...       -> value__cwt_coefficients
    value__change_quantiles__f_agg_...              -> value__change_quantiles
    """
    parts = str(feature_name).split("__")
    if len(parts) >= 2:
        return "__".join(parts[:2])
    return str(feature_name)


def broad_family(feature_name):
    f = str(feature_name)
    if "cwt_coefficients" in f or "fft_" in f or "spkt_welch_density" in f:
        return "frequency_wavelet"
    if "autocorrelation" in f or "c3__" in f or "time_reversal" in f:
        return "temporal_dependence"
    if "entropy" in f or "lempel_ziv" in f or "cid_ce" in f:
        return "complexity_entropy"
    if "change_quantiles" in f or "mean_change" in f or "mean_abs_change" in f:
        return "local_variability"
    if "linear_trend" in f or "agg_linear_trend" in f or "augmented_dickey" in f:
        return "trend_nonstationarity"
    if "ratio_beyond" in f or "kurtosis" in f or "skewness" in f:
        return "bursts_heavy_tails"
    if "number_peaks" in f or "longest_strike" in f:
        return "peaks_strikes"
    if "quantile" in f or "median" in f or "mean" in f or "maximum" in f or "minimum" in f:
        return "level_distribution"
    return "other"


# ============================================================
# CLUSTERING
# ============================================================

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


def internal_metrics(labels, embedding):
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))
    outlier_frac = n_outliers / len(labels)

    if n_clusters >= 2 and np.sum(mask) > 5:
        try:
            sil = silhouette_score(embedding[mask], labels[mask])
        except Exception:
            sil = np.nan
        try:
            dbi = davies_bouldin_score(embedding[mask], labels[mask])
        except Exception:
            dbi = np.nan
    else:
        sil = np.nan
        dbi = np.nan

    return {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "outlier_frac": outlier_frac,
        "coverage": 1.0 - outlier_frac,
        "silhouette": sil,
        "davies_bouldin": dbi,
    }


def run_pca_hdbscan(features, pca_dim, min_cluster_size):
    Xs = StandardScaler().fit_transform(features.values)
    pca = PCA(n_components=min(pca_dim, Xs.shape[1]), random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(Xp)
    return labels, Xp


def run_pca_umap_hdbscan(features, pca_dim, umap_dim, n_neighbors, min_dist, min_cluster_size):
    Xs = StandardScaler().fit_transform(features.values)
    pca = PCA(n_components=min(pca_dim, Xs.shape[1]), random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    reducer = umap.UMAP(
        n_components=umap_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=RANDOM_STATE,
        init="random",  # avoids UMAP spectral-layout underflow crashes
        low_memory=True,
    )
    Xu = reducer.fit_transform(Xp)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(Xu)
    return labels, Xu


# ============================================================
# METHOD EVALUATION PER CLUSTER
# ============================================================

def compare_row(row, cp_prefix):
    cp_err = row[f"{cp_prefix}_abs_err"]
    kb_err = row["kb_abs_err"]
    cp_bad = pd.isna(cp_err)
    kb_bad = pd.isna(kb_err)
    if cp_bad and kb_bad:
        return "both_wrong"
    if cp_bad:
        return "kb"
    if kb_bad:
        return "cp"
    if kb_err < cp_err:
        return "kb"
    if cp_err < kb_err:
        return "cp"
    return "tie"


def safe_mean(series):
    x = pd.Series(series).dropna()
    return float(x.mean()) if len(x) else np.nan


def evaluate_methods_per_cluster(labels, series_keys, jagt_methods):
    labels = np.asarray(labels)
    rows = []
    total_jagt_available = sum(k in jagt_methods.index for k in series_keys)

    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        cluster_keys = [series_keys[i] for i in idx]
        jagt_keys = [k for k in cluster_keys if k in jagt_methods.index]
        n_cluster = len(idx)
        n_jagt = len(jagt_keys)

        row = {
            "cluster": cid,
            "cluster_size": n_cluster,
            "n_jagt": n_jagt,
            "jagt_pct_of_cluster": n_jagt / n_cluster if n_cluster else 0.0,
            "jagt_pct_of_all_jagt": n_jagt / total_jagt_available if total_jagt_available else 0.0,
        }

        if n_jagt == 0:
            for method in ["cp_best", "cp_orig", "kb"]:
                row[f"{method}_mae"] = np.nan
                row[f"{method}_median_abs_err"] = np.nan
                row[f"{method}_steady_accuracy"] = np.nan
                row[f"{method}_false_unsteady"] = 0
            for cp_prefix in ["cp_best", "cp_orig"]:
                row[f"kb_wins_vs_{cp_prefix}"] = 0
                row[f"{cp_prefix}_wins_vs_kb"] = 0
                row[f"ties_vs_{cp_prefix}"] = 0
                row[f"both_wrong_vs_{cp_prefix}"] = 0
                row[f"kb_minus_{cp_prefix}_win_margin"] = 0
                row[f"kb_minus_{cp_prefix}_win_rate_margin"] = np.nan
                row[f"kb_minus_{cp_prefix}_mae_advantage"] = np.nan
                row[f"sign_p_vs_{cp_prefix}"] = np.nan
            rows.append(row)
            continue

        sub = jagt_methods.loc[jagt_keys].copy()
        for method in ["cp_best", "cp_orig", "kb"]:
            err = sub[f"{method}_abs_err"]
            row[f"{method}_mae"] = safe_mean(err)
            row[f"{method}_median_abs_err"] = float(err.dropna().median()) if len(err.dropna()) else np.nan
            row[f"{method}_steady_accuracy"] = float(sub[f"{method}_steady"].mean())
            row[f"{method}_false_unsteady"] = int((~sub[f"{method}_steady"]).sum())

        for cp_prefix in ["cp_best", "cp_orig"]:
            res = sub.apply(lambda r: compare_row(r, cp_prefix), axis=1)
            counts = res.value_counts()
            kb_wins = int(counts.get("kb", 0))
            cp_wins = int(counts.get("cp", 0))
            ties = int(counts.get("tie", 0))
            both_wrong = int(counts.get("both_wrong", 0))
            non_tie = kb_wins + cp_wins
            row[f"kb_wins_vs_{cp_prefix}"] = kb_wins
            row[f"{cp_prefix}_wins_vs_kb"] = cp_wins
            row[f"ties_vs_{cp_prefix}"] = ties
            row[f"both_wrong_vs_{cp_prefix}"] = both_wrong
            row[f"kb_minus_{cp_prefix}_win_margin"] = kb_wins - cp_wins
            row[f"kb_minus_{cp_prefix}_win_rate_margin"] = (kb_wins - cp_wins) / non_tie if non_tie else np.nan
            row[f"kb_minus_{cp_prefix}_mae_advantage"] = row[f"{cp_prefix}_mae"] - row["kb_mae"]
            if binomtest is not None and non_tie >= MIN_NON_TIE_WINS:
                row[f"sign_p_vs_{cp_prefix}"] = float(binomtest(kb_wins, non_tie, 0.5).pvalue)
            else:
                row[f"sign_p_vs_{cp_prefix}"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    for cp_prefix in ["cp_best", "cp_orig"]:
        pcol = f"sign_p_vs_{cp_prefix}"
        df[f"sign_q_vs_{cp_prefix}"] = bh_adjust(df[pcol].values) if pcol in df else np.nan
    return df.sort_values(["n_jagt", "cluster_size"], ascending=False)


# ============================================================
# FEATURE SIGNATURES
# ============================================================

def cluster_feature_zscores(features, labels):
    X = features.values.astype(float)
    feature_names = list(features.columns)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    rows = []
    for cid in sorted(c for c in set(labels) if c != -1):
        idx = np.where(labels == cid)[0]
        z = (X[idx].mean(axis=0) - mu) / sd
        row = {"cluster": cid, "cluster_size": len(idx)}
        row.update({feature_names[i]: z[i] for i in range(len(feature_names))})
        rows.append(row)
    return pd.DataFrame(rows)


def top_raw_features_for_cluster(zscores, cid, top_k=TOP_FEATURES_PER_CLUSTER):
    row = zscores[zscores["cluster"] == cid]
    if row.empty:
        return pd.DataFrame()
    ignore = {"cluster", "cluster_size"}
    feature_cols = [c for c in zscores.columns if c not in ignore]
    z = row.iloc[0][feature_cols].astype(float)
    out = pd.DataFrame({
        "feature": feature_cols,
        "zscore": z.values,
        "abs_zscore": np.abs(z.values),
        "subfamily": [tsfresh_subfamily(f) for f in feature_cols],
        "broad_family": [broad_family(f) for f in feature_cols],
    })
    return out.sort_values("abs_zscore", ascending=False).head(top_k)


def subfamily_signature_for_cluster(zscores, cid):
    row = zscores[zscores["cluster"] == cid]
    if row.empty:
        return pd.DataFrame()
    ignore = {"cluster", "cluster_size"}
    feature_cols = [c for c in zscores.columns if c not in ignore]
    z = row.iloc[0][feature_cols].astype(float)
    tmp = pd.DataFrame({
        "feature": feature_cols,
        "zscore": z.values,
        "abs_zscore": np.abs(z.values),
        "subfamily": [tsfresh_subfamily(f) for f in feature_cols],
        "broad_family": [broad_family(f) for f in feature_cols],
    })
    return (
        tmp.groupby(["subfamily", "broad_family"], as_index=False)
        .agg(mean_z=("zscore", "mean"), mean_abs_z=("abs_zscore", "mean"), max_abs_z=("abs_zscore", "max"), n_features=("feature", "count"))
        .sort_values("mean_abs_z", ascending=False)
    )


# ============================================================
# SCORING
# ============================================================

def score_clustering(internal, cluster_eval):
    usable = cluster_eval[(cluster_eval["cluster"] != -1) & (cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER)].copy()
    if usable.empty:
        method_signal = 0.0
        n_sig_best = n_sig_orig = jagt_in_sig_best = jagt_in_sig_orig = 0
        max_abs_margin_best = 0.0
    else:
        sig_best = usable[(usable["sign_q_vs_cp_best"] < SIGNIFICANCE_ALPHA) & (usable["kb_wins_vs_cp_best"] + usable["cp_best_wins_vs_kb"] >= MIN_NON_TIE_WINS)]
        sig_orig = usable[(usable["sign_q_vs_cp_orig"] < SIGNIFICANCE_ALPHA) & (usable["kb_wins_vs_cp_orig"] + usable["cp_orig_wins_vs_kb"] >= MIN_NON_TIE_WINS)]
        n_sig_best = len(sig_best)
        n_sig_orig = len(sig_orig)
        jagt_in_sig_best = int(sig_best["n_jagt"].sum())
        jagt_in_sig_orig = int(sig_orig["n_jagt"].sum())
        total_jagt = max(usable["n_jagt"].sum(), 1)
        sig_jagt_frac_best = jagt_in_sig_best / total_jagt
        sig_jagt_frac_orig = jagt_in_sig_orig / total_jagt
        margins = usable["kb_minus_cp_best_win_rate_margin"].dropna()
        max_abs_margin_best = float(np.max(np.abs(margins))) if len(margins) else 0.0
        method_signal = 0.45 * sig_jagt_frac_best + 0.25 * sig_jagt_frac_orig + 0.15 * min(n_sig_best, 5) / 5 + 0.10 * min(n_sig_orig, 5) / 5 + 0.05 * max_abs_margin_best

    sil = internal["silhouette"]
    dbi = internal["davies_bouldin"]
    coverage = internal["coverage"]
    n_clusters = internal["n_clusters"]
    sil_term = 0 if pd.isna(sil) else max(sil, 0)
    dbi_term = 0 if pd.isna(dbi) else 1.0 / (1.0 + dbi)
    cluster_term = min(n_clusters, 20) / 20.0
    internal_score = 0.45 * sil_term + 0.25 * coverage + 0.20 * dbi_term + 0.10 * cluster_term
    hypothesis_score = 0.45 * internal_score + 0.55 * method_signal

    return {
        "internal_score": internal_score,
        "method_signal": method_signal,
        "hypothesis_score": hypothesis_score,
        "n_sig_clusters_cp_best": n_sig_best,
        "n_sig_clusters_cp_orig": n_sig_orig,
        "jagt_in_sig_clusters_cp_best": jagt_in_sig_best,
        "jagt_in_sig_clusters_cp_orig": jagt_in_sig_orig,
        "max_abs_win_rate_margin_cp_best": max_abs_margin_best,
    }


# ============================================================
# PLOTS
# ============================================================

def plot_method_bars(cluster_eval, prefix, top_n=15):
    sub = cluster_eval[cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER].head(top_n)
    if sub.empty:
        return
    x = np.arange(len(sub))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - width, sub["kb_wins_vs_cp_best"], width, label="KB wins vs CP-best")
    ax.bar(x, sub["cp_best_wins_vs_kb"], width, label="CP-best wins")
    ax.bar(x + width, sub["ties_vs_cp_best"], width, label="ties")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in sub["cluster"]])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("JAGT steady series")
    ax.set_title(f"CP-best vs KB-KSSD per cluster: {prefix}")
    ax.legend(frameon=False)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_method_wins_cpbest.png"), dpi=300)
    plt.close()


def plot_method_mae(cluster_eval, prefix, top_n=15):
    sub = cluster_eval[cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER].head(top_n)
    if sub.empty:
        return
    x = np.arange(len(sub))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - width, sub["kb_mae"], width, label="KB-KSSD")
    ax.bar(x, sub["cp_best_mae"], width, label="CP-SSD best")
    ax.bar(x + width, sub["cp_orig_mae"], width, label="CP-SSD original")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in sub["cluster"]])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("MAE vs JAGT index")
    ax.set_title(f"SSD index error per cluster: {prefix}")
    ax.legend(frameon=False)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_method_mae.png"), dpi=300)
    plt.close()


def plot_significant_cluster_bands(data_normalized, labels, cluster_eval, prefix, top_n=TOP_BAND_CLUSTERS):
    sig = cluster_eval[(cluster_eval["cluster"] != -1) & (cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER)].copy()
    sig["abs_margin"] = np.abs(sig["kb_minus_cp_best_win_margin"])
    sig = sig.sort_values(["abs_margin", "n_jagt"], ascending=False).head(top_n)
    labels = np.asarray(labels)
    for _, row in sig.iterrows():
        cid = row["cluster"]
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        series = np.vstack([data_normalized[i] for i in idx])
        mean = np.mean(series, axis=0)
        lo = np.min(series, axis=0)
        hi = np.max(series, axis=0)
        q25 = np.quantile(series, 0.25, axis=0)
        q75 = np.quantile(series, 0.75, axis=0)
        x = np.arange(len(mean))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.fill_between(x, lo, hi, alpha=0.12, label="min-max range")
        ax.fill_between(x, q25, q75, alpha=0.25, label="IQR band")
        ax.plot(x, mean, linewidth=2, label="mean")
        kb = int(row["kb_wins_vs_cp_best"])
        cp = int(row["cp_best_wins_vs_kb"])
        nj = int(row["n_jagt"])
        q = row["sign_q_vs_cp_best"]
        title = f"{prefix} | cluster {cid} | n={len(idx)}, JAGT={nj}\nKB wins={kb}, CP-best wins={cp}"
        if not pd.isna(q):
            title += f", q={q:.3g}"
        ax.set_title(title)
        ax.set_xlabel("Time index after dropping first point")
        ax.set_ylabel("Normalized value")
        ax.legend(frameon=False)
        prettify_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(BANDS_DIR, f"{prefix}_cluster_{cid}_band.png"), dpi=300)
        plt.close()



def plot_relevant_cluster_subfamily_bars(cluster_eval, zscores, prefix, top_n_clusters=MAX_SUBFAMILY_BAR_CLUSTERS, top_n_subfamilies=TOP_SUBFAMILIES_PER_CLUSTER):
    """
    Meeting-friendly replacement for heatmaps.

    For the most method-relevant clusters, save one horizontal barplot of the
    strongest TSFresh subfamilies. This is much easier to show than a dense
    heatmap.
    """
    usable = cluster_eval[(cluster_eval["cluster"] != -1) & (cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER)].copy()
    if usable.empty:
        return

    usable["abs_margin"] = np.abs(usable["kb_minus_cp_best_win_margin"])
    usable = usable.sort_values(["abs_margin", "n_jagt"], ascending=False).head(top_n_clusters)

    for _, row in usable.iterrows():
        cid = row["cluster"]
        sig = subfamily_signature_for_cluster(zscores, cid).head(top_n_subfamilies)
        if sig.empty:
            continue

        sig = sig.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(sig["subfamily"], sig["mean_z"])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axvline(1, linestyle=":", linewidth=1)
        ax.axvline(-1, linestyle=":", linewidth=1)
        ax.axvline(2, linestyle="-.", linewidth=1)
        ax.axvline(-2, linestyle="-.", linewidth=1)

        kb = int(row["kb_wins_vs_cp_best"])
        cp = int(row["cp_best_wins_vs_kb"])
        nj = int(row["n_jagt"])
        ax.set_title(f"Top TSFresh subfamilies: {prefix} | cluster {cid}\nJAGT={nj}, KB wins={kb}, CP-best wins={cp}")
        ax.set_xlabel("Mean subfamily Z-score")
        ax.set_ylabel("TSFresh subfamily")
        prettify_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_cluster_{cid}_top_subfamilies_bar.png"), dpi=300)
        plt.close()


def plot_subfamily_heatmap(cluster_eval, zscores, prefix, top_n_clusters=15, top_n_subfamilies=20):
    """Disabled intentionally: meeting version does not generate heatmaps."""
    return


def plot_sweep_summary(df):
    top = df.sort_values("hypothesis_score", ascending=False).head(20).copy()
    top["short_name"] = [f"{v}:{i}" for i, v in enumerate(top["variant"])]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top))
    ax.bar(x, top["hypothesis_score"], label="hypothesis score")
    ax.plot(x, top["internal_score"], marker="o", label="internal score")
    ax.plot(x, top["method_signal"], marker="o", label="method signal")
    ax.set_xticks(x)
    ax.set_xticklabels(top["short_name"], rotation=75, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Top clustering configurations by hypothesis score")
    ax.legend(frameon=False)
    prettify_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_configurations_scores.png"), dpi=300)
    plt.close()


# ============================================================
# REPORTING FOR SIGNIFICANT CLUSTERS
# ============================================================

def save_cluster_interpretation_tables(result, cluster_eval, zscores):
    prefix = safe_name(f"{result.variant}_{result.config_name}")
    cluster_eval.to_csv(os.path.join(TABLES_DIR, f"{prefix}_cluster_method_eval.csv"), index=False)
    zscores.to_csv(os.path.join(TABLES_DIR, f"{prefix}_cluster_feature_zscores.csv"), index=False)
    usable = cluster_eval[(cluster_eval["cluster"] != -1) & (cluster_eval["n_jagt"] >= MIN_JAGT_PER_CLUSTER)].copy()
    if usable.empty:
        return
    usable["abs_margin"] = np.abs(usable["kb_minus_cp_best_win_margin"])
    usable = usable.sort_values(["abs_margin", "n_jagt"], ascending=False)
    summary_rows = []
    for _, r in usable.iterrows():
        cid = r["cluster"]
        raw = top_raw_features_for_cluster(zscores, cid, TOP_FEATURES_PER_CLUSTER)
        subfam = subfamily_signature_for_cluster(zscores, cid).head(TOP_SUBFAMILIES_PER_CLUSTER)
        raw.to_csv(os.path.join(SIG_DIR, f"{prefix}_cluster_{cid}_top_raw_features.csv"), index=False)
        subfam.to_csv(os.path.join(SIG_DIR, f"{prefix}_cluster_{cid}_top_subfamilies.csv"), index=False)
        q_best = r["sign_q_vs_cp_best"]
        winner = "KB" if r["kb_minus_cp_best_win_margin"] > 0 else "CP-best" if r["kb_minus_cp_best_win_margin"] < 0 else "tie"
        summary_rows.append({
            "variant": result.variant,
            "config_name": result.config_name,
            "cluster": cid,
            "cluster_size": r["cluster_size"],
            "n_jagt": r["n_jagt"],
            "winner_vs_cp_best": winner,
            "kb_wins_vs_cp_best": r["kb_wins_vs_cp_best"],
            "cp_best_wins_vs_kb": r["cp_best_wins_vs_kb"],
            "ties_vs_cp_best": r["ties_vs_cp_best"],
            "sign_p_vs_cp_best": r["sign_p_vs_cp_best"],
            "sign_q_vs_cp_best": q_best,
            "kb_mae": r["kb_mae"],
            "cp_best_mae": r["cp_best_mae"],
            "cp_orig_mae": r["cp_orig_mae"],
            "kb_minus_cp_best_mae_advantage": r["kb_minus_cp_best_mae_advantage"],
            "top_raw_features": "; ".join(raw["feature"].head(5).tolist()),
            "top_subfamilies": "; ".join(subfam["subfamily"].head(5).tolist()),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(SIG_DIR, f"{prefix}_interpretable_cluster_summary.csv"), index=False)



def save_immediate_configuration_outputs(result, cluster_eval, force=False):
    """
    Save all useful diagnostics for one configuration immediately after it is
    computed or loaded.

    This makes the script robust to crashes: after each successful configuration
    you already have method-evaluation tables, feature-zscore tables,
    per-cluster raw-feature/subfamily summaries, method plots, cluster-band plots,
    and compact bar/curve plots.

    A marker file avoids regenerating diagnostics for cached configurations unless
    force=True.
    """
    prefix = safe_name(f"{result.variant}_{result.config_name}")
    marker_path = os.path.join(TABLES_DIR, f"{prefix}_DIAGNOSTICS_DONE.txt")

    if os.path.exists(marker_path) and not force:
        return

    print(f"Saving immediate diagnostics for: {prefix}")

    try:
        zscores = cluster_feature_zscores(result.features, result.labels)

        # CSV outputs:
        # - tables/<prefix>_cluster_method_eval.csv
        # - tables/<prefix>_cluster_feature_zscores.csv
        # - significant_clusters/<prefix>_cluster_<id>_top_raw_features.csv
        # - significant_clusters/<prefix>_cluster_<id>_top_subfamilies.csv
        # - significant_clusters/<prefix>_interpretable_cluster_summary.csv
        save_cluster_interpretation_tables(result, cluster_eval, zscores)

        # Compact meeting-friendly plot outputs only:
        # - plots/<prefix>_method_wins_cpbest.png      [barplot]
        # - plots/<prefix>_method_mae.png              [barplot]
        # - cluster_bands/<prefix>_cluster_<id>_band.png [curve + band]
        # - plots/<prefix>_cluster_<id>_top_subfamilies_bar.png [barplot]
        # No heatmaps are generated.
        if SAVE_METHOD_BARPLOTS:
            plot_method_bars(cluster_eval, prefix)
            plot_method_mae(cluster_eval, prefix)
        if SAVE_CLUSTER_BANDS:
            plot_significant_cluster_bands(
                result.data_normalized,
                result.labels,
                cluster_eval,
                prefix,
            )
        if SAVE_SUBFAMILY_BARPLOTS:
            plot_relevant_cluster_subfamily_bars(cluster_eval, zscores, prefix)

        with open(marker_path, "w") as f:
            f.write("done\n")

    except Exception as exc:
        # Do not fail the sweep just because diagnostics failed.
        # The expensive clustering result is already cached separately.
        diagnostic_fail_path = os.path.join(TABLES_DIR, "failed_diagnostics.csv")
        row = {
            "variant": result.variant,
            "config_name": result.config_name,
            "method": result.method,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if os.path.exists(diagnostic_fail_path):
            df = pd.read_csv(diagnostic_fail_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(diagnostic_fail_path, index=False)
        print(f"WARNING: diagnostics failed for {prefix}: {type(exc).__name__}: {exc}")



# ============================================================
# RESUMABLE CONFIGURATION CACHE
# ============================================================

def configuration_cache_path(variant, config_name):
    return os.path.join(
        CONFIG_CACHE_DIR,
        f"{safe_name(variant)}__{safe_name(config_name)}.pkl",
    )


def save_configuration_cache(result, cluster_eval):
    """
    Store one completed configuration immediately.

    This intentionally stores labels/embedding as well as metrics, so diagnostics
    for top configurations can be generated later without recomputing clustering.
    """
    path = configuration_cache_path(result.variant, result.config_name)
    pd.to_pickle(
        {
            "result": result,
            "cluster_eval": cluster_eval,
            "metrics": result.metrics,
        },
        path,
    )


def load_configuration_cache(variant, config_name):
    path = configuration_cache_path(variant, config_name)
    if not os.path.exists(path):
        return None

    try:
        payload = pd.read_pickle(path)
        return payload["result"], payload["cluster_eval"]
    except Exception as exc:
        print(f"WARNING: could not read cache {path}: {exc}")
        return None


def append_incremental_results(all_metric_rows):
    """
    Write all metrics seen in this run so far.
    If the process crashes, this file still contains all configurations processed
    before the crash.
    """
    if not all_metric_rows:
        return
    pd.DataFrame(all_metric_rows).to_csv(INCREMENTAL_RESULTS_PATH, index=False)


def log_failed_configuration(variant, config_name, method, error):
    row = {
        "variant": variant,
        "config_name": config_name,
        "method": method,
        "error_type": type(error).__name__,
        "error": str(error),
    }

    if os.path.exists(FAILED_CONFIGS_PATH):
        df = pd.read_csv(FAILED_CONFIGS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=["variant", "config_name", "method", "error_type", "error"])
    else:
        df = pd.DataFrame([row])

    df.to_csv(FAILED_CONFIGS_PATH, index=False)


def evaluate_or_load_configuration(
    variant,
    config_name,
    method,
    features,
    series_keys,
    data_normalized,
    jagt_methods,
    clustering_callable,
    all_metric_rows,
    all_results,
):
    """
    Load a completed configuration from cache, or evaluate it and cache it.

    After every successful configuration, the script immediately stores:
    - cache payload with labels/embedding/metrics;
    - incremental metrics table;
    - method evaluation table;
    - cluster feature z-score table;
    - raw TSFresh feature summaries;
    - TSFresh subfamily summaries;
    - method-win and MAE plots;
    - cluster-band plots;
    - compact subfamily barplots.

    Returns True if a result was available/evaluated, False if the configuration
    failed and was skipped.
    """

    cached = load_configuration_cache(variant, config_name)
    if cached is not None:
        result, cluster_eval = cached
        print(f"Loading cached {variant} / {config_name}")

        all_metric_rows.append(result.metrics)
        all_results.append((result, cluster_eval))
        append_incremental_results(all_metric_rows)

        # Also ensure diagnostics exist for old cached configurations.
        save_immediate_configuration_outputs(result, cluster_eval)

        return True

    try:
        result, cluster_eval = evaluate_one_configuration(
            variant,
            config_name,
            method,
            features,
            series_keys,
            data_normalized,
            jagt_methods,
            clustering_callable,
        )

        # 1) Save expensive computation immediately.
        save_configuration_cache(result, cluster_eval)

        # 2) Save metrics immediately.
        all_metric_rows.append(result.metrics)
        all_results.append((result, cluster_eval))
        append_incremental_results(all_metric_rows)

        # 3) Save interpretation outputs immediately.
        save_immediate_configuration_outputs(result, cluster_eval)

        return True

    except Exception as exc:
        print(f"\nFAILED {variant} / {config_name}: {type(exc).__name__}: {exc}")
        log_failed_configuration(variant, config_name, method, exc)
        return False


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def evaluate_one_configuration(variant, config_name, method, features, series_keys, data_normalized, jagt_methods, clustering_callable):
    labels, embedding = clustering_callable(features)
    internal = internal_metrics(labels, embedding)
    cluster_eval = evaluate_methods_per_cluster(labels, series_keys, jagt_methods)
    score = score_clustering(internal, cluster_eval)
    metrics = {
        "variant": variant,
        "config_name": config_name,
        "method": method,
        "n_series": len(labels),
        "n_features": features.shape[1],
        **internal,
        **score,
    }
    result = ClusteringResult(variant, config_name, method, np.asarray(labels), embedding, features, series_keys, data_normalized, metrics)
    return result, cluster_eval


# ============================================================
# MAIN SWEEP
# ============================================================

def build_variants(full_data, full_keys, features_all, jagt_methods):
    data_drop, keys_drop, _ = drop_first_n_with_keys(full_data, full_keys, n=1)
    data_norm = preprocess_normalized(data_drop)
    if len(features_all) != len(keys_drop):
        raise ValueError(f"Feature-cache length mismatch: features={len(features_all)}, keys after drop={len(keys_drop)}. Expected cached drop_first1 features.")
    all_variant = {"variant": "all_timeseries", "features": features_all.copy(), "series_keys": keys_drop, "data_normalized": data_norm}
    jagt_mask = np.array([k in jagt_methods.index for k in keys_drop])
    jagt_indices = np.where(jagt_mask)[0]
    jagt_variant = {
        "variant": "steady_jagt_only",
        "features": features_all.iloc[jagt_indices].reset_index(drop=True).copy(),
        "series_keys": [keys_drop[i] for i in jagt_indices],
        "data_normalized": [data_norm[i] for i in jagt_indices],
    }
    return [all_variant, jagt_variant]


def run_targeted_sweep():
    print("\nLoading data/features...")
    import cluster_detection as cd

    full_data = cd.load_all_json(DATA_GLOB)
    print(f"Loaded {len(full_data)} timeseries.")

    full_keys = load_series_keys(SERIES_INDEX_PATH, expected_len=len(full_data))
    features_all = load_cached_features(FEATURE_CACHE_PATH)
    jagt_methods = load_jagt_method_table(JAGT_METHOD_TABLE_PATH)
    variants = build_variants(full_data, full_keys, features_all, jagt_methods)

    all_metric_rows = []
    all_results = []

    for var in variants:
        variant = var["variant"]
        features = var["features"]
        series_keys = var["series_keys"]
        data_normalized = var["data_normalized"]

        print("\n" + "=" * 90)
        print(f"VARIANT: {variant} | n_series={len(features)}")
        print("=" * 90)

        # ------------------------------------------------------------
        # PCA-only configurations
        # ------------------------------------------------------------
        for pca_dim in PCA_DIMS:
            for mcs in MIN_CLUSTER_SIZES:
                config_name = f"pca{pca_dim}_mcs{mcs}"
                print(f"\nTesting {variant} / {config_name}")

                evaluate_or_load_configuration(
                    variant=variant,
                    config_name=config_name,
                    method="pca_hdbscan",
                    features=features,
                    series_keys=series_keys,
                    data_normalized=data_normalized,
                    jagt_methods=jagt_methods,
                    clustering_callable=(
                        lambda feats, pca_dim=pca_dim, mcs=mcs:
                        run_pca_hdbscan(feats, pca_dim, mcs)
                    ),
                    all_metric_rows=all_metric_rows,
                    all_results=all_results,
                )

        # ------------------------------------------------------------
        # PCA + UMAP configurations
        # ------------------------------------------------------------
        for pca_dim in UMAP_PCA_DIMS:
            for umap_dim in UMAP_DIMS:
                if umap_dim >= pca_dim:
                    continue

                for nn in UMAP_NEIGHBORS:
                    for md in UMAP_MIN_DISTS:
                        for mcs in UMAP_MIN_CLUSTER_SIZES:
                            config_name = f"pca{pca_dim}_umap{umap_dim}_nn{nn}_md{md}_mcs{mcs}"
                            print(f"\nTesting {variant} / {config_name}")

                            evaluate_or_load_configuration(
                                variant=variant,
                                config_name=config_name,
                                method="pca_umap_hdbscan",
                                features=features,
                                series_keys=series_keys,
                                data_normalized=data_normalized,
                                jagt_methods=jagt_methods,
                                clustering_callable=(
                                    lambda feats, pca_dim=pca_dim, umap_dim=umap_dim, nn=nn, md=md, mcs=mcs:
                                    run_pca_umap_hdbscan(feats, pca_dim, umap_dim, nn, md, mcs)
                                ),
                                all_metric_rows=all_metric_rows,
                                all_results=all_results,
                            )

    if not all_metric_rows:
        raise RuntimeError(
            "No clustering configurations were successfully evaluated or loaded from cache. "
            f"Check {FAILED_CONFIGS_PATH} for failures."
        )

    df_metrics = pd.DataFrame(all_metric_rows)
    df_metrics = df_metrics.drop_duplicates(subset=["variant", "config_name", "method"])
    df_metrics = df_metrics.sort_values("hypothesis_score", ascending=False)

    df_metrics.to_csv(os.path.join(TABLES_DIR, "all_clustering_configurations.csv"), index=False)
    df_metrics.to_csv(INCREMENTAL_RESULTS_PATH, index=False)

    plot_sweep_summary(df_metrics)

    print("\n" + "=" * 90)
    print("TOP CONFIGURATIONS")
    print("=" * 90)
    print(df_metrics.head(20))

    top_keys = set(tuple(x) for x in df_metrics.head(8)[["variant", "config_name"]].values.tolist())

    # Deduplicate all_results before diagnostics.
    unique_results = {}
    for result, cluster_eval in all_results:
        unique_results[(result.variant, result.config_name)] = (result, cluster_eval)

    summary_parts = []

    for key, (result, cluster_eval) in unique_results.items():
        if key not in top_keys:
            continue

        prefix = safe_name(f"{result.variant}_{result.config_name}")
        print(f"\nSaving diagnostics for top config: {prefix}")

        zscores = cluster_feature_zscores(result.features, result.labels)
        save_cluster_interpretation_tables(result, cluster_eval, zscores)
        if SAVE_METHOD_BARPLOTS:
            plot_method_bars(cluster_eval, prefix)
            plot_method_mae(cluster_eval, prefix)
        if SAVE_CLUSTER_BANDS:
            plot_significant_cluster_bands(result.data_normalized, result.labels, cluster_eval, prefix)
        if SAVE_SUBFAMILY_BARPLOTS:
            plot_relevant_cluster_subfamily_bars(cluster_eval, zscores, prefix)

        path = os.path.join(SIG_DIR, f"{prefix}_interpretable_cluster_summary.csv")
        if os.path.exists(path):
            summary_parts.append(pd.read_csv(path))

    if summary_parts:
        pd.concat(summary_parts, ignore_index=True).to_csv(
            os.path.join(OUTDIR, "top_configurations_interpretable_cluster_summary.csv"),
            index=False,
        )

    write_readme(df_metrics)

    print("\nDone.")
    print(f"Outputs saved in: {OUTDIR}/")
    print(f"Configuration cache: {CONFIG_CACHE_DIR}/")
    print(f"Incremental table: {INCREMENTAL_RESULTS_PATH}")
    if os.path.exists(FAILED_CONFIGS_PATH):
        print(f"Failed configurations table: {FAILED_CONFIGS_PATH}")


def write_readme(df_metrics):
    top = df_metrics.head(10)
    lines = [
        "Targeted TSFresh clustering summary",
        "=" * 40,
        "",
        "Purpose:",
        "- Search for visually/feature-coherent clusters where CP-SSD and KB-KSSD differ.",
        "- Use only drop-first-1 normalized Efficient TSFresh features.",
        "- Evaluate both all-timeseries clustering and steady-JAGT-only clustering.",
        "",
        "Primary score:",
        "- hypothesis_score = 0.45 * internal_score + 0.55 * method_signal",
        "- internal_score uses silhouette, coverage, Davies-Bouldin, and cluster count.",
        "- method_signal rewards significant CP-vs-KB asymmetry inside JAGT-bearing clusters.",
        "",
        "Top configurations:",
        top.to_string(index=False),
        "",
        "Important outputs:",
        "- tables/all_clustering_configurations.csv",
        "- top_configurations_interpretable_cluster_summary.csv",
        "- significant_clusters/*_top_raw_features.csv",
        "- significant_clusters/*_top_subfamilies.csv",
        "- cluster_bands/*_cluster_<id>_band.png",
        "- plots/*_top_subfamilies_bar.png",
        "",
    ]
    with open(os.path.join(OUTDIR, "README_SUMMARY.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_targeted_sweep()
