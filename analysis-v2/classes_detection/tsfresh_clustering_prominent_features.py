#!/usr/bin/env python3

import os
import json
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(under="ignore")

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)

import hdbscan
import umap


OUTDIR = "tsfresh_testing_2"
os.makedirs(OUTDIR, exist_ok=True)

FEATURE_CACHE_DIR = os.path.join(OUTDIR, "feature_cache")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

SERIES_INDEX_PATH = "series_kbkssd.json"

JAGT_METHOD_TABLE_PATH = (
    "../man_steady_comparison/"
    "ssd_best_cpssd_comparison_results/"
    "ssd_best_config_comparison_table.csv"
)

CLUSTER_METHOD_OUTDIR = os.path.join(OUTDIR, "cluster_method_comparison")
os.makedirs(CLUSTER_METHOD_OUTDIR, exist_ok=True)

# ======================================================
# METHOD-WIN FEATURE-SIGNATURE ANALYSIS CONFIG
# ======================================================

ANALYSIS_OUTDIR = os.path.join(OUTDIR, "method_feature_signature_analysis")
os.makedirs(ANALYSIS_OUTDIR, exist_ok=True)

# Only clusters with at least this many steady JAGT timeseries are interpreted.
MIN_JAGT_PER_CLUSTER = 3

# Winner must exceed the other method by at least this many cases.
MIN_WIN_MARGIN = 1

# Number of TSFresh features to show in plots/reports.
TOP_K_FEATURES = 15

# If True, group-level feature signatures are weighted by number of JAGT series.
WEIGHT_BY_JAGT = True

BEST_CLUSTER_CACHE_DIR = os.path.join(OUTDIR, "best_cluster_cache")
os.makedirs(BEST_CLUSTER_CACHE_DIR, exist_ok=True)

CLUSTER_BAND_OUTDIR = os.path.join(OUTDIR, "cluster_bands")
os.makedirs(CLUSTER_BAND_OUTDIR, exist_ok=True)

# Change this string whenever you change the cost function so stale best-clustering
# caches are not reused accidentally.
CLUSTER_COST_FUNCTION_NAME = "balanced_score_v1"


# ======================================================
# SWEEP CONFIGURATION
# ======================================================

EFFICIENT_PCA_DIMS = [10, 20, 30, 50, 75, 100]
EFFICIENT_MIN_CLUSTER_SIZES = [20]

EFFICIENT_UMAP_DIMS = [5, 10, 20]
EFFICIENT_UMAP_NEIGHBORS = [15]
EFFICIENT_UMAP_MIN_DIST = [0.0]

RANDOM_STATE = 42


# ======================================================
# PLOT STYLE
# ======================================================

def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_zscore_guides(ax):
    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.8, label="neutral")
    ax.axvline(1, linestyle=":", linewidth=1, alpha=0.8, label="moderate |z|=1")
    ax.axvline(-1, linestyle=":", linewidth=1, alpha=0.8)
    ax.axvline(2, linestyle="-.", linewidth=1, alpha=0.8, label="strong |z|=2")
    ax.axvline(-2, linestyle="-.", linewidth=1, alpha=0.8)


# ======================================================
# DATA PREPARATION
# ======================================================

def prepare_tsfresh_df(timeseries_list):
    dfs = []

    for i, ts in enumerate(timeseries_list):
        dfs.append(pd.DataFrame({
            "id": i,
            "time": np.arange(len(ts)),
            "value": np.asarray(ts, dtype=float),
        }))

    return pd.concat(dfs, ignore_index=True)


def drop_first_n(data, n=2):
    return [np.asarray(ts, dtype=float)[n:] for ts in data if len(ts) > n]


def drop_first_n_with_keys(data, keys, n=1):
    data_out = []
    keys_out = []

    for ts, key in zip(data, keys):
        if len(ts) > n:
            data_out.append(np.asarray(ts, dtype=float)[n:])
            keys_out.append(key)

    return data_out, keys_out


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


def load_series_keys(series_index_path, expected_len=None):
    d = json.load(open(series_index_path))
    keys = list(d.keys())

    if expected_len is not None and len(keys) != expected_len:
        raise ValueError(
            f"Series-key count mismatch: {len(keys)} keys vs {expected_len} series"
        )

    return keys


def load_jagt_method_table(path=JAGT_METHOD_TABLE_PATH):
    df = pd.read_csv(path)
    return df.set_index("key", drop=False)


# ======================================================
# TSFRESH FEATURE EXTRACTION
# ======================================================

def extract_tsfresh_features_minimal(timeseries_list):
    print("\nExtracting TSFresh features: MINIMAL mode...")

    df = prepare_tsfresh_df(timeseries_list)

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=8,
        disable_progressbar=True,
    )

    features = impute(features)

    print(f"Extracted {features.shape[1]} features")
    return features


def extract_tsfresh_features_efficient(timeseries_list):
    print("\nExtracting TSFresh features: EFFICIENT mode...")

    df = prepare_tsfresh_df(timeseries_list)

    settings = EfficientFCParameters()

    for bad_feature in [
        "friedrich_coefficients",
        "max_langevin_fixed_point",
    ]:
        settings.pop(bad_feature, None)

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        n_jobs=8,
        disable_progressbar=False,
    )

    features = impute(features)

    print(f"Extracted {features.shape[1]} features")
    return features


def get_cached_features(data_p, dataset_name, extractor):
    cache_path = os.path.join(
        FEATURE_CACHE_DIR,
        f"{dataset_name}_normalized_{extractor}_features.pkl"
    )

    if os.path.exists(cache_path):
        print(f"\nLoading cached features: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"\nNo cache found. Extracting features for {dataset_name} / {extractor}")

    if extractor == "minimal":
        features = extract_tsfresh_features_minimal(data_p)
    elif extractor in {"fast", "efficient"}:
        features = extract_tsfresh_features_efficient(data_p)
    else:
        raise ValueError(f"Unknown extractor: {extractor}")

    features.to_pickle(cache_path)
    print(f"Saved cached features: {cache_path}")

    return features


# ======================================================
# CLUSTERING
# ======================================================

def cluster_timeseries(features, pca_dim=10, min_cluster_size=20):
    print("\nRunning PCA + HDBSCAN...")

    X = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=min(pca_dim, X_scaled.shape[1]),
        random_state=RANDOM_STATE,
    )
    Xp = pca.fit_transform(X_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(Xp)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))

    print(f"Found {n_clusters} clusters")
    print(f"Outliers: {n_outliers} / {len(labels)}")

    return labels, Xp, scaler, pca


def cluster_timeseries_pca_umap(
    features,
    pca_dim=50,
    umap_dim=10,
    umap_neighbors=30,
    umap_min_dist=0.1,
    min_cluster_size=20,
):
    print(
        "\nRunning PCA + UMAP + HDBSCAN "
        f"(PCA={pca_dim}, UMAP={umap_dim}, nn={umap_neighbors}, "
        f"min_dist={umap_min_dist}, mcs={min_cluster_size})..."
    )

    X = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=min(pca_dim, X_scaled.shape[1]),
        random_state=RANDOM_STATE,
    )
    Xp = pca.fit_transform(X_scaled)

    reducer = umap.UMAP(
        n_components=umap_dim,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )

    Xu = reducer.fit_transform(Xp)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(Xu)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))

    print(f"Found {n_clusters} clusters")
    print(f"Outliers: {n_outliers} / {len(labels)}")

    return labels, Xu, scaler, pca, reducer


def get_top_clusters(labels, top_n=10):
    cluster_sizes = {}

    for l in labels:
        if l == -1:
            continue
        cluster_sizes[l] = cluster_sizes.get(l, 0) + 1

    top_clusters = sorted(
        cluster_sizes.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    print("\nTop clusters:")
    for cid, size in top_clusters:
        print(f"Cluster {cid}: {size} series")

    return [cid for cid, _ in top_clusters]


# ======================================================
# FEATURE SIGNATURE ANALYSIS
# ======================================================

def compute_cluster_feature_zscores(features, labels):
    X = features.values
    feature_names = list(features.columns)

    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0) + 1e-12

    clusters = sorted([c for c in set(labels) if c != -1])

    zscores = {}
    sizes = {}

    for cid in clusters:
        idx = np.where(labels == cid)[0]
        Xc = X[idx]

        cluster_mean = Xc.mean(axis=0)
        zscores[cid] = (cluster_mean - global_mean) / global_std
        sizes[cid] = len(idx)

    return zscores, sizes, feature_names


def print_cluster_feature_reports(features, labels, top_clusters, top_k=6):
    zscores, sizes, feature_names = compute_cluster_feature_zscores(features, labels)

    print("\n" + "=" * 80)
    print("CLUSTER FEATURE REPORTS")
    print("=" * 80)

    for cid in top_clusters:
        z = zscores[cid]
        idx = np.argsort(np.abs(z))[::-1][:top_k]

        print(f"\nCluster {cid} (n={sizes[cid]})")
        print("-" * 60)

        for i in idx:
            direction = "high" if z[i] > 0 else "low"
            print(f"{feature_names[i]:55s} {direction:4s} z={z[i]:+.2f}")


def plot_feature_importance_topn(features, labels, top_clusters, prefix, top_n_features=20):
    zscores, _, feature_names = compute_cluster_feature_zscores(features, labels)

    if len(zscores) == 0:
        return

    all_clusters = sorted(zscores.keys())

    Z_all = np.array([zscores[cid] for cid in all_clusters])
    Z_top = np.array([zscores[cid] for cid in top_clusters])

    imp_all = np.mean(np.abs(Z_all), axis=0)
    imp_top = np.mean(np.abs(Z_top), axis=0)

    order = np.argsort(imp_all)[::-1][:top_n_features]

    names = [feature_names[i] for i in order]
    vals_all = imp_all[order]
    vals_top = imp_top[order]

    x = np.arange(len(names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - width / 2, vals_all, width, label="All clusters")
    ax.bar(x + width / 2, vals_top, width, label="Top 10 clusters")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Mean |Z-score|")
    ax.set_title(f"Top {top_n_features} Feature Importances")
    ax.legend(title="Scope", frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_top{top_n_features}_feature_importance.png", dpi=300)
    plt.close()


def plot_compact_cluster_signature_heatmap(features, labels, top_clusters, prefix, top_n_features=20):
    zscores, sizes, feature_names = compute_cluster_feature_zscores(features, labels)

    if len(zscores) == 0:
        return

    all_clusters = sorted(zscores.keys())
    Z_all = np.array([zscores[cid] for cid in all_clusters])

    feature_importance = np.mean(np.abs(Z_all), axis=0)
    feature_order = np.argsort(feature_importance)[::-1][:top_n_features]

    cluster_ids = top_clusters
    Z = np.array([zscores[cid][feature_order] for cid in cluster_ids])

    order = np.argsort([sizes[cid] for cid in cluster_ids])[::-1]
    Z = Z[order]
    cluster_ids = [cluster_ids[i] for i in order]

    names = [feature_names[i] for i in feature_order]

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        Z,
        aspect="auto",
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Z-score: blue=lower, red=higher than global mean")

    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"C{cid} (n={sizes[cid]})" for cid in cluster_ids])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)

    ax.set_title(f"Compact Cluster Feature Signatures: Top {top_n_features} Features")
    ax.set_xlabel("Most discriminative TSFresh features")
    ax.set_ylabel("Largest clusters")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_compact_signature_heatmap.png", dpi=300)
    plt.close()


def plot_top_features_per_cluster(features, labels, top_clusters, top_k=8, prefix="tsfresh"):
    zscores, sizes, feature_names = compute_cluster_feature_zscores(features, labels)

    for cid in top_clusters:
        z = zscores[cid]
        idx = np.argsort(np.abs(z))[::-1][:top_k]

        vals = z[idx]
        names = [feature_names[i] for i in idx]

        y = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.barh(y, vals)
        add_zscore_guides(ax)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Cluster Z-score")
        ax.set_title(f"Cluster {cid} feature signature (n={sizes[cid]})")

        ax.text(
            0.01,
            0.02,
            "Interpretation: |z|≈1 moderate, |z|≥2 strong deviation from global feature mean",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.8,
        )

        ax.legend(frameon=False, fontsize=8, loc="lower right")
        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{prefix}_cluster_{cid}_zscore_barplot.png", dpi=300)
        plt.close()


def plot_most_important_clusters_zscores(features, labels, top_clusters, prefix, n_clusters=5, top_k=8):
    zscores, sizes, feature_names = compute_cluster_feature_zscores(features, labels)

    cluster_strength = {
        cid: np.mean(np.abs(zscores[cid]))
        for cid in top_clusters
    }

    chosen = sorted(
        top_clusters,
        key=lambda cid: cluster_strength[cid],
        reverse=True,
    )[:n_clusters]

    for cid in chosen:
        z = zscores[cid]
        idx = np.argsort(np.abs(z))[::-1][:top_k]

        vals = z[idx]
        names = [feature_names[i] for i in idx]

        y = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.barh(y, vals)
        add_zscore_guides(ax)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Cluster Z-score")
        ax.set_title(
            f"Most informative cluster C{cid}: top feature deviations "
            f"(n={sizes[cid]}, mean |z|={cluster_strength[cid]:.2f})"
        )

        ax.text(
            0.01,
            0.02,
            "Positive = feature higher than global mean; negative = lower. |z|≥2 is strong.",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.8,
        )

        ax.legend(frameon=False, fontsize=8, loc="lower right")
        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{prefix}_important_cluster_{cid}_zscore_barplot.png", dpi=300)
        plt.close()


def save_cluster_signature_tables(features, labels, top_clusters, prefix="tsfresh"):
    zscores, sizes, feature_names = compute_cluster_feature_zscores(features, labels)

    rows = []

    for cid, z in zscores.items():
        row = {"cluster": cid, "size": sizes[cid]}
        for i, fname in enumerate(feature_names):
            row[fname] = z[i]
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        f"{OUTDIR}/{prefix}_cluster_zscores.csv",
        index=False,
    )

    report_rows = []

    for cid in top_clusters:
        z = zscores[cid]
        idx = np.argsort(np.abs(z))[::-1]

        for rank, i in enumerate(idx, start=1):
            report_rows.append({
                "cluster": cid,
                "cluster_size": sizes[cid],
                "rank": rank,
                "feature": feature_names[i],
                "zscore": z[i],
                "abs_zscore": abs(z[i]),
                "direction": "high" if z[i] > 0 else "low",
                "strength": (
                    "strong" if abs(z[i]) >= 2 else
                    "moderate" if abs(z[i]) >= 1 else
                    "weak"
                ),
            })

    pd.DataFrame(report_rows).to_csv(
        f"{OUTDIR}/{prefix}_top_cluster_feature_report.csv",
        index=False,
    )


# ======================================================
# TIMESERIES PLOTS
# ======================================================

def _stack_cluster_series(data, labels, cid):
    idx = np.where(labels == cid)[0]
    return np.vstack([data[i] for i in idx])


def plot_cluster_means(data, labels, cluster_ids, prefix="tsfresh"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for cid in cluster_ids:
        cluster_series = _stack_cluster_series(data, labels, cid)

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(len(mean_curve))

        line, = ax.plot(x, mean_curve, linewidth=2, label=f"C{cid}")
        color = line.get_color()

        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.12,
            color=color,
        )

    ax.set_title("Cluster Mean Curves ± STD")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized value")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_cluster_means.png", dpi=300)
    plt.close()


def plot_derivatives(data, labels, cluster_ids, prefix="tsfresh"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for cid in cluster_ids:
        cluster_series = _stack_cluster_series(data, labels, cid)

        derivatives = np.diff(cluster_series, axis=1)
        mean_deriv = np.mean(derivatives, axis=0)

        x = np.arange(len(mean_deriv))
        ax.plot(x, mean_deriv, label=f"C{cid}")

    ax.set_title("Mean Derivatives per Cluster")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Mean Δ value")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_derivatives.png", dpi=300)
    plt.close()


def plot_cluster_distance_matrix(Xembed, labels, cluster_ids, prefix="tsfresh"):
    from scipy.spatial.distance import cdist

    centroids = []

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroids.append(np.mean(Xembed[idx], axis=0))

    centroids = np.array(centroids)
    dist_matrix = cdist(centroids, centroids)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_matrix)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Centroid distance in embedding space")

    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(cluster_ids)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)

    ax.set_title("Cluster Distance Matrix")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_distance_matrix.png", dpi=300)
    plt.close()


def plot_first_window(data, labels, cluster_ids, window=100, prefix="tsfresh"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for cid in cluster_ids:
        cluster_series = _stack_cluster_series(data, labels, cid)[:, :window]

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(window)

        line, = ax.plot(x, mean_curve, label=f"C{cid}")
        color = line.get_color()

        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.12,
            color=color,
        )

    ax.set_title(f"First {window} Samples: Mean ± STD")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized value")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{prefix}_first_window.png", dpi=300)
    plt.close()


# ======================================================
# COMPARISONS / QUALITY
# ======================================================

def summarize_clustering_quality(features, labels, Xembed, name):
    mask = labels != -1

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))
    outlier_frac = n_outliers / len(labels)

    if n_clusters >= 2 and np.sum(mask) > 5:
        sil = silhouette_score(Xembed[mask], labels[mask])
        dbi = davies_bouldin_score(Xembed[mask], labels[mask])
    else:
        sil = np.nan
        dbi = np.nan

    return {
        "name": name,
        "n_features": features.shape[1],
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "outlier_frac": outlier_frac,
        "silhouette": sil,
        "davies_bouldin": dbi,
    }


def feature_signature_strength(features, labels):
    zscores, _, _ = compute_cluster_feature_zscores(features, labels)

    if len(zscores) == 0:
        return np.nan, np.nan

    Z = np.array(list(zscores.values()))

    return float(np.mean(np.abs(Z))), float(np.max(np.abs(Z)))


def compare_two_clusterings(res_a, res_b, name_a, name_b):
    labels_a = res_a["labels"]
    labels_b = res_b["labels"]

    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)

    print("\n" + "=" * 70)
    print(f"CLUSTERING AGREEMENT: {name_a} vs {name_b}")
    print("=" * 70)
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")

    return ari, nmi


def compare_top_cluster_membership(res_a, res_b, name_a, name_b):
    labels_a = res_a["labels"]
    labels_b = res_b["labels"]

    top_a = set()
    for cid in res_a["top_clusters"]:
        top_a.update(np.where(labels_a == cid)[0])

    top_b = set()
    for cid in res_b["top_clusters"]:
        top_b.update(np.where(labels_b == cid)[0])

    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    jaccard = inter / union if union > 0 else 0

    print("\n" + "=" * 70)
    print(f"TOP-10 CLUSTER MEMBERSHIP OVERLAP: {name_a} vs {name_b}")
    print("=" * 70)
    print(f"{name_a} top-10 size: {len(top_a)}")
    print(f"{name_b} top-10 size: {len(top_b)}")
    print(f"intersection: {inter}")
    print(f"Jaccard overlap: {jaccard:.3f}")

    return jaccard


def compare_quality_table(results):
    rows = []

    for name, res in results.items():
        q = summarize_clustering_quality(
            res["features"],
            res["labels"],
            res["Xp"],
            name,
        )

        mean_abs_z, max_abs_z = feature_signature_strength(
            res["features"],
            res["labels"],
        )

        q["mean_abs_cluster_z"] = mean_abs_z
        q["max_abs_cluster_z"] = max_abs_z

        rows.append(q)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("CLUSTERING QUALITY COMPARISON")
    print("=" * 70)
    print(df)

    df.to_csv(f"{OUTDIR}/quality_comparison.csv", index=False)

    return df


def plot_quality_table_metrics(df):
    metrics = [
        "n_clusters",
        "outlier_frac",
        "silhouette",
        "davies_bouldin",
        "mean_abs_cluster_z",
        "max_abs_cluster_z",
    ]

    for m in metrics:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(df["name"], df[m])
        ax.set_title(m)
        ax.set_ylabel(m)
        ax.set_xticklabels(df["name"], rotation=45, ha="right")
        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/compare_{m}.png", dpi=300)
        plt.close()


def compare_cluster_counts(results):
    names = list(results.keys())
    counts = []

    for name in names:
        labels = results[name]["labels"]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        counts.append(n_clusters)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, counts)
    ax.set_title("Cluster Count Comparison")
    ax.set_ylabel("Number of clusters")
    ax.set_xticklabels(names, rotation=45, ha="right")
    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/cluster_count_comparison.png", dpi=300)
    plt.close()


# ======================================================
# CP-SSD VS KB-KSSD PER-CLUSTER EVALUATION
# ======================================================

def cp_vs_kb_result(row, cp_col="cp_best_abs_err"):
    cp_err = row[cp_col]
    kb_err = row["kb_abs_err"]

    cp_bad = pd.isna(cp_err)
    kb_bad = pd.isna(kb_err)

    if cp_bad and kb_bad:
        return "both_wrong"
    if cp_bad:
        return "kb_better"
    if kb_bad:
        return "cp_better"
    if cp_err < kb_err:
        return "cp_better"
    if kb_err < cp_err:
        return "kb_better"
    return "tie"


def evaluate_cp_kb_per_cluster(labels, series_keys, jagt_methods, prefix):
    """
    Evaluate CP-SSD best, original CP-SSD, and KB-KSSD within each cluster.

    Cluster size is based on all time series in the cluster, but method quality is
    computed only on steady JAGT series present in that cluster.
    """
    labels = np.asarray(labels)
    rows = []

    total_jagt = sum(k in jagt_methods.index for k in series_keys)

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
            "jagt_pct_of_all_jagt": n_jagt / total_jagt if total_jagt else 0.0,
        }

        if n_jagt == 0:
            for method in ["cp_best", "cp_orig", "kb"]:
                row[f"{method}_mae"] = np.nan
                row[f"{method}_median_abs_err"] = np.nan
                row[f"{method}_steady_accuracy"] = np.nan
                row[f"{method}_false_unsteady"] = 0

            row.update({
                "cp_best_better_than_kb": 0,
                "kb_better_than_cp_best": 0,
                "cp_best_kb_tie": 0,
                "cp_best_kb_both_wrong": 0,
                "cp_orig_better_than_kb": 0,
                "kb_better_than_cp_orig": 0,
                "cp_orig_kb_tie": 0,
                "cp_orig_kb_both_wrong": 0,
            })
            rows.append(row)
            continue

        sub = jagt_methods.loc[jagt_keys].copy()

        for method in ["cp_best", "cp_orig", "kb"]:
            vals = sub[f"{method}_abs_err"].dropna()
            row[f"{method}_mae"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{method}_median_abs_err"] = float(vals.median()) if len(vals) else np.nan
            row[f"{method}_steady_accuracy"] = float(sub[f"{method}_steady"].mean())
            row[f"{method}_false_unsteady"] = int((~sub[f"{method}_steady"]).sum())

        sub["cp_best_vs_kb"] = sub.apply(
            lambda r: cp_vs_kb_result(r, cp_col="cp_best_abs_err"),
            axis=1,
        )
        sub["cp_orig_vs_kb"] = sub.apply(
            lambda r: cp_vs_kb_result(r, cp_col="cp_orig_abs_err"),
            axis=1,
        )

        best_counts = sub["cp_best_vs_kb"].value_counts()
        orig_counts = sub["cp_orig_vs_kb"].value_counts()

        row.update({
            "cp_best_better_than_kb": int(best_counts.get("cp_better", 0)),
            "kb_better_than_cp_best": int(best_counts.get("kb_better", 0)),
            "cp_best_kb_tie": int(best_counts.get("tie", 0)),
            "cp_best_kb_both_wrong": int(best_counts.get("both_wrong", 0)),

            "cp_orig_better_than_kb": int(orig_counts.get("cp_better", 0)),
            "kb_better_than_cp_orig": int(orig_counts.get("kb_better", 0)),
            "cp_orig_kb_tie": int(orig_counts.get("tie", 0)),
            "cp_orig_kb_both_wrong": int(orig_counts.get("both_wrong", 0)),
        })

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        by=["n_jagt", "cluster_size"],
        ascending=False,
    )

    out_csv = os.path.join(
        CLUSTER_METHOD_OUTDIR,
        f"{prefix}_cp_kb_per_cluster.csv",
    )
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"CP-SSD vs KB-KSSD PER CLUSTER: {prefix}")
    print("=" * 80)
    print(df.head(20))

    plot_jagt_presence_per_cluster(df, prefix)
    plot_cp_kb_wins_per_cluster(df, prefix)
    plot_cp_kb_mae_per_cluster(df, prefix)
    plot_cp_kb_binary_accuracy_per_cluster(df, prefix)

    return df


def _top_clusters_with_jagt(df, top_n=15):
    return df[df["n_jagt"] > 0].sort_values("n_jagt", ascending=False).head(top_n)


def plot_jagt_presence_per_cluster(df, prefix, top_n=15):
    sub = _top_clusters_with_jagt(df, top_n)
    if sub.empty:
        return

    x = np.arange(len(sub))

    fig, ax1 = plt.subplots(figsize=(11, 4))

    ax1.bar(x, sub["n_jagt"], label="JAGT count")
    ax1.set_ylabel("Number of JAGT steady series")
    ax1.set_xlabel("Cluster")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in sub["cluster"]])

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        100 * sub["jagt_pct_of_cluster"],
        marker="o",
        label="JAGT % of cluster",
    )
    ax2.set_ylabel("JAGT as % of cluster")

    ax1.set_title(f"JAGT representation per cluster: {prefix}")

    prettify_axes(ax1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CLUSTER_METHOD_OUTDIR, f"{prefix}_jagt_presence.png"),
        dpi=300,
    )
    plt.close()


def plot_cp_kb_wins_per_cluster(df, prefix, top_n=15):
    sub = _top_clusters_with_jagt(df, top_n)
    if sub.empty:
        return

    x = np.arange(len(sub))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.bar(x - 1.5 * width, sub["cp_best_better_than_kb"], width, label="CP best > KB")
    ax.bar(x - 0.5 * width, sub["cp_orig_better_than_kb"], width, label="CP original > KB")
    ax.bar(x + 0.5 * width, sub["kb_better_than_cp_best"], width, label="KB > CP best")
    ax.bar(x + 1.5 * width, sub["kb_better_than_cp_orig"], width, label="KB > CP original")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in sub["cluster"]])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of JAGT steady series")
    ax.set_title(f"CP-SSD vs KB-KSSD wins per cluster: {prefix}")
    ax.legend(frameon=False, ncol=2)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CLUSTER_METHOD_OUTDIR, f"{prefix}_cp_kb_wins.png"),
        dpi=300,
    )
    plt.close()


def plot_cp_kb_mae_per_cluster(df, prefix, top_n=15):
    sub = _top_clusters_with_jagt(df, top_n)
    if sub.empty:
        return

    x = np.arange(len(sub))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(x - width, sub["cp_best_mae"], width, label="CP-SSD best")
    ax.bar(x, sub["cp_orig_mae"], width, label="CP-SSD original")
    ax.bar(x + width, sub["kb_mae"], width, label="KB-KSSD")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in sub["cluster"]])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("MAE vs JAGT index")
    ax.set_title(f"SSD index error per cluster: {prefix}")
    ax.legend(frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CLUSTER_METHOD_OUTDIR, f"{prefix}_cp_kb_mae.png"),
        dpi=300,
    )
    plt.close()


def plot_cp_kb_binary_accuracy_per_cluster(df, prefix, top_n=15):
    sub = _top_clusters_with_jagt(df, top_n)
    if sub.empty:
        return

    x = np.arange(len(sub))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(x - width, sub["cp_best_steady_accuracy"], width, label="CP-SSD best")
    ax.bar(x, sub["cp_orig_steady_accuracy"], width, label="CP-SSD original")
    ax.bar(x + width, sub["kb_steady_accuracy"], width, label="KB-KSSD")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in sub["cluster"]])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Binary steady accuracy")
    ax.set_title(f"False-unsteady behavior per cluster: {prefix}")
    ax.legend(frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CLUSTER_METHOD_OUTDIR, f"{prefix}_cp_kb_binary_accuracy.png"),
        dpi=300,
    )
    plt.close()


def plot_relevant_cluster_bands(data, labels, cluster_eval_df, prefix, top_n=10):
    """
    Plot individual cluster bands for clusters with the most JAGT series.
    The band is min--max range and the central line is the cluster mean curve.
    """
    sub = cluster_eval_df[cluster_eval_df["n_jagt"] > 0].sort_values(
        "n_jagt",
        ascending=False,
    ).head(top_n)

    if sub.empty:
        return

    labels = np.asarray(labels)

    for cid in sub["cluster"]:
        if cid == -1:
            continue

        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        cluster_series = np.vstack([data[i] for i in idx])

        mean_curve = np.mean(cluster_series, axis=0)
        lower = np.min(cluster_series, axis=0)
        upper = np.max(cluster_series, axis=0)

        x = np.arange(len(mean_curve))

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(x, mean_curve, linewidth=2, label="Mean curve")
        ax.fill_between(x, lower, upper, alpha=0.25, label="Min–max range")

        n_cluster = len(idx)
        n_jagt = int(sub[sub["cluster"] == cid]["n_jagt"].iloc[0])

        ax.set_title(
            f"Cluster {cid}: mean curve + range ({prefix})\n"
            f"cluster n={n_cluster}, JAGT n={n_jagt}"
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Normalized value")
        ax.legend(frameon=False)

        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(CLUSTER_BAND_OUTDIR, f"{prefix}_cluster_{cid}_mean_range.png"),
            dpi=300,
        )
        plt.close()


def plot_relevant_cluster_bands_overlay(data, labels, cluster_eval_df, prefix, top_n=10):
    """
    Overlay mean curves and min--max bands for the most JAGT-relevant clusters.
    """
    sub = cluster_eval_df[cluster_eval_df["n_jagt"] > 0].sort_values(
        "n_jagt",
        ascending=False,
    ).head(top_n)

    if sub.empty:
        return

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(11, 5))

    for cid in sub["cluster"]:
        if cid == -1:
            continue

        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        cluster_series = np.vstack([data[i] for i in idx])

        mean_curve = np.mean(cluster_series, axis=0)
        lower = np.min(cluster_series, axis=0)
        upper = np.max(cluster_series, axis=0)

        x = np.arange(len(mean_curve))

        line, = ax.plot(x, mean_curve, linewidth=2, label=f"C{cid}")
        color = line.get_color()

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.10,
        )

    ax.set_title(f"Relevant clusters: mean curve + min/max range ({prefix})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized value")
    ax.legend(ncol=2, fontsize=8, frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(CLUSTER_BAND_OUTDIR, f"{prefix}_relevant_clusters_overlay_mean_range.png"),
        dpi=300,
    )
    plt.close()


# ======================================================
# EFFICIENT SWEEP: PCA ONLY AND PCA+UMAP
# ======================================================

# ======================================================


def get_best_cluster_cache_path(dataset_name):
    return os.path.join(
        BEST_CLUSTER_CACHE_DIR,
        f"{dataset_name}_efficient_best_{CLUSTER_COST_FUNCTION_NAME}.pkl",
    )


def load_cached_best_clustering(dataset_name):
    cache_path = get_best_cluster_cache_path(dataset_name)

    if os.path.exists(cache_path):
        print(f"\nLoading cached best clustering: {cache_path}")
        return pd.read_pickle(cache_path)

    return None


def save_cached_best_clustering(dataset_name, payload):
    cache_path = get_best_cluster_cache_path(dataset_name)
    pd.to_pickle(payload, cache_path)
    print(f"Saved best clustering cache: {cache_path}")

def balanced_score(row):
    if pd.isna(row["silhouette"]):
        return -np.inf

    coverage = 1.0 - row["outlier_frac"]
    cluster_bonus = min(row["n_clusters"], 10) / 10.0

    return (
        row["silhouette"] * coverage
        + 0.10 * cluster_bonus
        - 0.05 * row["davies_bouldin"]
    )


def evaluate_clustering_pca_only(features, pca_dim, min_cluster_size):
    labels, Xp, scaler, pca = cluster_timeseries(
        features,
        pca_dim=pca_dim,
        min_cluster_size=min_cluster_size,
    )

    name = f"pca_pca{pca_dim}_mcs{min_cluster_size}"

    q = summarize_clustering_quality(
        features,
        labels,
        Xp,
        name=name,
    )

    mean_abs_z, max_abs_z = feature_signature_strength(features, labels)

    q.update({
        "method": "pca",
        "pca_dim": pca_dim,
        "umap_dim": np.nan,
        "umap_neighbors": np.nan,
        "umap_min_dist": np.nan,
        "min_cluster_size": min_cluster_size,
        "mean_abs_cluster_z": mean_abs_z,
        "max_abs_cluster_z": max_abs_z,
    })

    q["coverage"] = 1.0 - q["outlier_frac"]
    q["balanced_score"] = balanced_score(q)

    artifact = {
        "labels": labels,
        "Xp": Xp,
        "scaler": scaler,
        "pca": pca,
        "umap": None,
        "features": features,
        "method": "pca",
    }

    return q, artifact


def evaluate_clustering_pca_umap(
    features,
    pca_dim,
    umap_dim,
    umap_neighbors,
    umap_min_dist,
    min_cluster_size,
):
    labels, Xu, scaler, pca, reducer = cluster_timeseries_pca_umap(
        features,
        pca_dim=pca_dim,
        umap_dim=umap_dim,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        min_cluster_size=min_cluster_size,
    )

    name = (
        f"pca_umap_pca{pca_dim}_u{umap_dim}_nn{umap_neighbors}_"
        f"md{umap_min_dist}_mcs{min_cluster_size}"
    )

    q = summarize_clustering_quality(
        features,
        labels,
        Xu,
        name=name,
    )

    mean_abs_z, max_abs_z = feature_signature_strength(features, labels)

    q.update({
        "method": "pca_umap",
        "pca_dim": pca_dim,
        "umap_dim": umap_dim,
        "umap_neighbors": umap_neighbors,
        "umap_min_dist": umap_min_dist,
        "min_cluster_size": min_cluster_size,
        "mean_abs_cluster_z": mean_abs_z,
        "max_abs_cluster_z": max_abs_z,
    })

    q["coverage"] = 1.0 - q["outlier_frac"]
    q["balanced_score"] = balanced_score(q)

    artifact = {
        "labels": labels,
        "Xp": Xu,
        "scaler": scaler,
        "pca": pca,
        "umap": reducer,
        "features": features,
        "method": "pca_umap",
    }

    return q, artifact


def sweep_efficient_clustering(data, dataset_name):
    print("\n" + "=" * 80)
    print(f"EFFICIENT CLUSTERING SWEEP: {dataset_name}")
    print("=" * 80)

    data_p = preprocess_normalized(data)

    features = get_cached_features(
        data_p,
        dataset_name=dataset_name,
        extractor="efficient",
    )

    cached = load_cached_best_clustering(dataset_name)
    if cached is not None:
        return cached["df_sweep"], cached["best_artifact"]

    rows = []
    artifacts = {}

    for pca_dim in EFFICIENT_PCA_DIMS:
        for mcs in EFFICIENT_MIN_CLUSTER_SIZES:
            print(f"\nTesting PCA-only Efficient: PCA={pca_dim}, mcs={mcs}")

            q, artifact = evaluate_clustering_pca_only(
                features,
                pca_dim=pca_dim,
                min_cluster_size=mcs,
            )

            key = f"pca_pca{pca_dim}_mcs{mcs}"
            rows.append(q)
            artifacts[key] = artifact

    for pca_dim in EFFICIENT_PCA_DIMS:
        for umap_dim in EFFICIENT_UMAP_DIMS:
            if umap_dim >= pca_dim:
                continue

            for nn in EFFICIENT_UMAP_NEIGHBORS:
                for min_dist in EFFICIENT_UMAP_MIN_DIST:
                    for mcs in EFFICIENT_MIN_CLUSTER_SIZES:
                        print(
                            f"\nTesting PCA+UMAP Efficient: PCA={pca_dim}, "
                            f"UMAP={umap_dim}, nn={nn}, min_dist={min_dist}, mcs={mcs}"
                        )

                        q, artifact = evaluate_clustering_pca_umap(
                            features,
                            pca_dim=pca_dim,
                            umap_dim=umap_dim,
                            umap_neighbors=nn,
                            umap_min_dist=min_dist,
                            min_cluster_size=mcs,
                        )

                        key = (
                            f"pca_umap_pca{pca_dim}_u{umap_dim}_"
                            f"nn{nn}_md{min_dist}_mcs{mcs}"
                        )

                        rows.append(q)
                        artifacts[key] = artifact

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="balanced_score",
        ascending=False,
    )

    out_csv = f"{OUTDIR}/{dataset_name}_efficient_pca_umap_sweep.csv"
    df.to_csv(out_csv, index=False)

    print("\nTop Efficient configurations:")
    print(df.head(20))

    plot_efficient_sweep_results(df, dataset_name)
    plot_method_comparison_summary(df, dataset_name)

    best = df.iloc[0]
    best_key = best["name"]

    best_art = artifacts[best_key]
    top_clusters = get_top_clusters(best_art["labels"], top_n=10)

    prefix = f"{dataset_name}_normalized_efficient_BEST_{best_key}"

    plot_cluster_means(data_p, best_art["labels"], top_clusters, prefix=prefix)
    plot_derivatives(data_p, best_art["labels"], top_clusters, prefix=prefix)
    plot_cluster_distance_matrix(best_art["Xp"], best_art["labels"], top_clusters, prefix=prefix)
    plot_first_window(data_p, best_art["labels"], top_clusters, window=100, prefix=prefix)

    plot_feature_importance_topn(features, best_art["labels"], top_clusters, prefix=prefix, top_n_features=20)
    plot_compact_cluster_signature_heatmap(features, best_art["labels"], top_clusters, prefix=prefix, top_n_features=20)
    plot_top_features_per_cluster(features, best_art["labels"], top_clusters, top_k=8, prefix=prefix)
    plot_most_important_clusters_zscores(features, best_art["labels"], top_clusters, prefix=prefix, n_clusters=5, top_k=8)

    print_cluster_feature_reports(features, best_art["labels"], top_clusters, top_k=6)
    save_cluster_signature_tables(features, best_art["labels"], top_clusters, prefix=prefix)

    save_cached_best_clustering(
        dataset_name,
        {
            "df_sweep": df,
            "best_artifact": best_art,
            "cost_function": CLUSTER_COST_FUNCTION_NAME,
            "best_name": best_key,
            "best_score": float(best["balanced_score"]),
        },
    )

    return df, best_art


def plot_efficient_sweep_results(df, dataset_name):
    metrics = [
        "n_clusters",
        "outlier_frac",
        "silhouette",
        "davies_bouldin",
        "mean_abs_cluster_z",
        "coverage",
        "balanced_score",
    ]

    for method in ["pca", "pca_umap"]:
        df_m = df[df["method"] == method].copy()

        if df_m.empty:
            continue

        for metric in metrics:
            if method == "pca":
                table = df_m.pivot_table(
                    index="pca_dim",
                    columns="min_cluster_size",
                    values=metric,
                    aggfunc="mean",
                )
            else:
                agg = df_m.groupby(["pca_dim", "min_cluster_size"])[metric].max().reset_index()
                table = agg.pivot_table(
                    index="pca_dim",
                    columns="min_cluster_size",
                    values=metric,
                    aggfunc="mean",
                )

            fig, ax = plt.subplots(figsize=(7, 5))

            im = ax.imshow(table.values, aspect="auto")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric)

            ax.set_xticks(range(len(table.columns)))
            ax.set_xticklabels(table.columns)

            ax.set_yticks(range(len(table.index)))
            ax.set_yticklabels(table.index)

            ax.set_xlabel("HDBSCAN min_cluster_size")
            ax.set_ylabel("PCA dimensions")
            ax.set_title(f"{dataset_name}: {method} sweep — {metric}")

            plt.tight_layout()
            plt.savefig(
                f"{OUTDIR}/{dataset_name}_{method}_sweep_{metric}.png",
                dpi=300,
            )
            plt.close()


def plot_method_comparison_summary(df, dataset_name):
    metrics = [
        "balanced_score",
        "silhouette",
        "outlier_frac",
        "coverage",
        "davies_bouldin",
        "n_clusters",
        "mean_abs_cluster_z",
    ]

    rows = []

    for method in ["pca", "pca_umap"]:
        df_m = df[df["method"] == method]

        if df_m.empty:
            continue

        for metric in metrics:
            rows.append({
                "method": method,
                "metric": metric,
                "best": df_m[metric].max() if metric not in ["outlier_frac", "davies_bouldin"] else df_m[metric].min(),
                "median": df_m[metric].median(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUTDIR}/{dataset_name}_method_comparison_summary.csv", index=False)

    for metric in metrics:
        sub = summary[summary["metric"] == metric]

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width / 2, sub["best"], width, label="best")
        ax.bar(x + width / 2, sub["median"], width, label="median")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"])
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_name}: PCA vs PCA+UMAP — {metric}")
        ax.legend(frameon=False)

        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(
            f"{OUTDIR}/{dataset_name}_method_compare_{metric}.png",
            dpi=300,
        )
        plt.close()


# ======================================================
# ORIGINAL PIPELINE
# ======================================================

def run_pipeline(data, dataset_name, extractor="minimal", pca_dim=10):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name} | MODE: normalized | EXTRACTOR: {extractor}")
    print("=" * 70)

    data_p = preprocess_normalized(data)

    features = get_cached_features(
        data_p,
        dataset_name=dataset_name,
        extractor=extractor,
    )

    labels, Xp, scaler, pca = cluster_timeseries(
        features,
        pca_dim=pca_dim,
        min_cluster_size=20,
    )

    top_clusters = get_top_clusters(labels, top_n=10)

    prefix = f"{dataset_name}_normalized_{extractor}_pca{pca_dim}"

    plot_cluster_means(data_p, labels, top_clusters, prefix=prefix)
    plot_derivatives(data_p, labels, top_clusters, prefix=prefix)
    plot_cluster_distance_matrix(Xp, labels, top_clusters, prefix=prefix)
    plot_first_window(data_p, labels, top_clusters, window=100, prefix=prefix)

    plot_feature_importance_topn(features, labels, top_clusters, prefix=prefix, top_n_features=20)
    plot_compact_cluster_signature_heatmap(features, labels, top_clusters, prefix=prefix, top_n_features=20)
    plot_top_features_per_cluster(features, labels, top_clusters, top_k=8, prefix=prefix)
    plot_most_important_clusters_zscores(features, labels, top_clusters, prefix=prefix, n_clusters=5, top_k=8)

    print_cluster_feature_reports(features, labels, top_clusters, top_k=6)
    save_cluster_signature_tables(features, labels, top_clusters, prefix=prefix)

    return {
        "dataset": dataset_name,
        "mode": "normalized",
        "extractor": extractor,
        "labels": labels,
        "Xp": Xp,
        "top_clusters": top_clusters,
        "features": features,
        "scaler": scaler,
        "pca": pca,
    }


# ======================================================
# METHOD-WIN FEATURE-SIGNATURE ANALYSIS
# ======================================================

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


# ======================================================
# FILE DISCOVERY
# ======================================================


def discover_prefixes():
    """
    Find all cluster-method evaluation files and infer prefixes.
    Expected file pattern:
        <prefix>_cp_kb_per_cluster.csv
    """

    pattern = os.path.join(CLUSTER_METHOD_OUTDIR, "*_cp_kb_per_cluster.csv")
    files = sorted(glob.glob(pattern))

    prefixes = []
    for f in files:
        base = os.path.basename(f)
        prefix = base.replace("_cp_kb_per_cluster.csv", "")
        prefixes.append(prefix)

    return prefixes



def find_zscore_file(prefix):
    """
    Find the cluster-zscore table corresponding to a clustering prefix.

    Existing clustering code saves files like:
        <prefix>_cluster_zscores.csv

    For best efficient clustering, prefix may be shortened in method-comparison
    output, so we also search loosely.
    """

    exact = os.path.join(OUTDIR, f"{prefix}_cluster_zscores.csv")
    if os.path.exists(exact):
        return exact

    candidates = sorted(glob.glob(os.path.join(OUTDIR, f"{prefix}*_cluster_zscores.csv")))
    if candidates:
        return candidates[0]

    # Special case: method prefix may be drop_first1_efficient_best, while
    # zscore prefix contains drop_first1_normalized_efficient_BEST_<config>.
    if prefix.endswith("_efficient_best"):
        dataset = prefix.replace("_efficient_best", "")
        candidates = sorted(glob.glob(os.path.join(
            OUTDIR,
            f"{dataset}_normalized_efficient_BEST*_cluster_zscores.csv"
        )))
        if candidates:
            return candidates[0]

    return None


# ======================================================
# CORE ANALYSIS
# ======================================================


def classify_cluster_rows(df):
    """
    Add boolean/group labels describing which method dominates per cluster.
    """

    df = df.copy()

    df = df[df["n_jagt"] >= MIN_JAGT_PER_CLUSTER].copy()

    if df.empty:
        return df

    df["cp_best_margin_vs_kb"] = df["cp_best_better_than_kb"] - df["kb_better_than_cp_best"]
    df["kb_margin_vs_cp_best"] = -df["cp_best_margin_vs_kb"]

    df["cp_orig_margin_vs_kb"] = df["cp_orig_better_than_kb"] - df["kb_better_than_cp_orig"]
    df["kb_margin_vs_cp_orig"] = -df["cp_orig_margin_vs_kb"]

    df["group_cp_best_dominant"] = df["cp_best_margin_vs_kb"] >= MIN_WIN_MARGIN
    df["group_kb_over_cp_best"] = df["kb_margin_vs_cp_best"] >= MIN_WIN_MARGIN

    df["group_cp_orig_dominant"] = df["cp_orig_margin_vs_kb"] >= MIN_WIN_MARGIN
    df["group_kb_over_cp_orig"] = df["kb_margin_vs_cp_orig"] >= MIN_WIN_MARGIN

    # MAE-based variants are useful when win counts are sparse.
    df["group_cp_best_lower_mae"] = df["cp_best_mae"] < df["kb_mae"]
    df["group_kb_lower_mae_than_cp_best"] = df["kb_mae"] < df["cp_best_mae"]
    df["group_cp_orig_lower_mae"] = df["cp_orig_mae"] < df["kb_mae"]
    df["group_kb_lower_mae_than_cp_orig"] = df["kb_mae"] < df["cp_orig_mae"]

    return df



def load_feature_zscores(path):
    df = pd.read_csv(path)

    if "cluster" not in df.columns:
        raise ValueError(f"Missing 'cluster' column in {path}")

    feature_cols = [c for c in df.columns if c not in ["cluster", "size"]]
    return df, feature_cols



def aggregate_group_signature(cluster_eval, zscores, feature_cols, group_col):
    """
    Compute average feature signature for clusters in a given group.
    Uses cluster-level z-scores.
    """

    sub_eval = cluster_eval[cluster_eval[group_col]].copy()
    if sub_eval.empty:
        return None

    sub = zscores[zscores["cluster"].isin(sub_eval["cluster"])].copy()
    if sub.empty:
        return None

    sub = sub.merge(
        sub_eval[["cluster", "n_jagt", "cluster_size"]],
        on="cluster",
        how="left",
    )

    X = sub[feature_cols].values.astype(float)

    if WEIGHT_BY_JAGT:
        weights = sub["n_jagt"].values.astype(float)
    else:
        weights = np.ones(len(sub))

    weights = np.maximum(weights, 1.0)

    mean_z = np.average(X, axis=0, weights=weights)
    mean_abs_z = np.average(np.abs(X), axis=0, weights=weights)

    out = pd.DataFrame({
        "feature": feature_cols,
        "mean_z": mean_z,
        "mean_abs_z": mean_abs_z,
        "n_clusters": len(sub),
        "n_jagt_total": int(sub["n_jagt"].sum()),
        "group": group_col,
    })

    out = out.sort_values("mean_abs_z", ascending=False)
    return out



def plot_top_group_features(sig, prefix, group_col, top_k=TOP_K_FEATURES):
    if sig is None or sig.empty:
        return

    sub = sig.sort_values("mean_abs_z", ascending=False).head(top_k).copy()
    sub = sub.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sub["feature"], sub["mean_z"])

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(1, linestyle=":", linewidth=1)
    ax.axvline(-1, linestyle=":", linewidth=1)
    ax.axvline(2, linestyle="-.", linewidth=1)
    ax.axvline(-2, linestyle="-.", linewidth=1)

    readable = group_col.replace("group_", "").replace("_", " ")
    ax.set_title(
        f"Prominent TSFresh features: {readable}\n"
        f"{prefix} | n clusters={int(sig['n_clusters'].iloc[0])}, "
        f"JAGT n={int(sig['n_jagt_total'].iloc[0])}"
    )
    ax.set_xlabel("Mean cluster feature Z-score")
    ax.set_ylabel("TSFresh feature")

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ANALYSIS_OUTDIR, f"{safe_name(prefix)}_top_features_{safe_name(group_col)}.png"),
        dpi=300,
    )
    plt.close()



def plot_difference_between_groups(sig_a, sig_b, prefix, label_a, label_b, fname_suffix, top_k=TOP_K_FEATURES):
    """
    Plot feature-signature difference:
        group A mean_z - group B mean_z
    """

    if sig_a is None or sig_b is None or sig_a.empty or sig_b.empty:
        return

    a = sig_a[["feature", "mean_z"]].rename(columns={"mean_z": "mean_z_a"})
    b = sig_b[["feature", "mean_z"]].rename(columns={"mean_z": "mean_z_b"})

    d = a.merge(b, on="feature", how="inner")
    d["diff"] = d["mean_z_a"] - d["mean_z_b"]
    d["abs_diff"] = np.abs(d["diff"])

    d = d.sort_values("abs_diff", ascending=False).head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(d["feature"], d["diff"])

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(f"Feature-signature difference: {label_a} minus {label_b}\n{prefix}")
    ax.set_xlabel("Difference in mean feature Z-score")
    ax.set_ylabel("TSFresh feature")

    ax.text(
        0.01,
        0.02,
        f"Positive = more characteristic of {label_a}; negative = more characteristic of {label_b}",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ANALYSIS_OUTDIR, f"{safe_name(prefix)}_{fname_suffix}.png"),
        dpi=300,
    )
    plt.close()



def analyze_prefix(prefix):
    method_path = os.path.join(CLUSTER_METHOD_OUTDIR, f"{prefix}_cp_kb_per_cluster.csv")
    zscore_path = find_zscore_file(prefix)

    if not os.path.exists(method_path):
        print(f"Skipping {prefix}: missing method table {method_path}")
        return []

    if zscore_path is None:
        print(f"Skipping {prefix}: no matching cluster_zscores file found")
        return []

    print("\n" + "=" * 80)
    print(f"ANALYZING PREFIX: {prefix}")
    print("=" * 80)
    print(f"Method table: {method_path}")
    print(f"Z-score table: {zscore_path}")

    cluster_eval = pd.read_csv(method_path)
    cluster_eval = classify_cluster_rows(cluster_eval)

    if cluster_eval.empty:
        print(f"No clusters with n_jagt >= {MIN_JAGT_PER_CLUSTER} for {prefix}")
        return []

    zscores, feature_cols = load_feature_zscores(zscore_path)

    group_cols = [
        "group_kb_over_cp_best",
        "group_cp_best_dominant",
        "group_kb_over_cp_orig",
        "group_cp_orig_dominant",
        "group_kb_lower_mae_than_cp_best",
        "group_cp_best_lower_mae",
        "group_kb_lower_mae_than_cp_orig",
        "group_cp_orig_lower_mae",
    ]

    all_sigs = []
    sig_map = {}

    for g in group_cols:
        sig = aggregate_group_signature(cluster_eval, zscores, feature_cols, g)
        if sig is None:
            continue

        sig.insert(0, "prefix", prefix)
        all_sigs.append(sig)
        sig_map[g] = sig

        plot_top_group_features(sig, prefix, g)

        print(f"\nTop features for {g}:")
        print(sig[["feature", "mean_z", "mean_abs_z", "n_clusters", "n_jagt_total"]].head(8))

    if all_sigs:
        df_sig = pd.concat(all_sigs, ignore_index=True)
        df_sig.to_csv(
            os.path.join(ANALYSIS_OUTDIR, f"{safe_name(prefix)}_group_feature_signature.csv"),
            index=False,
        )
    else:
        df_sig = pd.DataFrame()

    # Difference plots: KB-dominant minus CP-dominant signatures.
    plot_difference_between_groups(
        sig_map.get("group_kb_over_cp_best"),
        sig_map.get("group_cp_best_dominant"),
        prefix,
        label_a="KB > CP best clusters",
        label_b="CP best > KB clusters",
        fname_suffix="kb_vs_cpbest_feature_difference",
    )

    plot_difference_between_groups(
        sig_map.get("group_kb_over_cp_orig"),
        sig_map.get("group_cp_orig_dominant"),
        prefix,
        label_a="KB > CP original clusters",
        label_b="CP original > KB clusters",
        fname_suffix="kb_vs_cporig_feature_difference",
    )

    return all_sigs



def make_global_summary(all_sigs):
    if not all_sigs:
        print("No signatures to summarize globally.")
        return

    df = pd.concat(all_sigs, ignore_index=True)
    df.to_csv(os.path.join(ANALYSIS_OUTDIR, "global_group_feature_signature.csv"), index=False)

    # Aggregate across prefixes by group and feature.
    grouped = (
        df.groupby(["group", "feature"], as_index=False)
        .agg(
            mean_z=("mean_z", "mean"),
            mean_abs_z=("mean_abs_z", "mean"),
            n_clusters=("n_clusters", "sum"),
            n_jagt_total=("n_jagt_total", "sum"),
        )
    )

    grouped.to_csv(
        os.path.join(ANALYSIS_OUTDIR, "global_group_feature_signature_aggregated.csv"),
        index=False,
    )

    for g in sorted(grouped["group"].unique()):
        sub = grouped[grouped["group"] == g].sort_values("mean_abs_z", ascending=False).head(TOP_K_FEATURES)
        sub = sub.iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(sub["feature"], sub["mean_z"])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axvline(1, linestyle=":", linewidth=1)
        ax.axvline(-1, linestyle=":", linewidth=1)
        ax.axvline(2, linestyle="-.", linewidth=1)
        ax.axvline(-2, linestyle="-.", linewidth=1)

        readable = g.replace("group_", "").replace("_", " ")
        ax.set_title(f"Global prominent TSFresh features: {readable}")
        ax.set_xlabel("Mean feature Z-score across clustering outputs")
        ax.set_ylabel("TSFresh feature")
        prettify_axes(ax)

        plt.tight_layout()
        plt.savefig(
            os.path.join(ANALYSIS_OUTDIR, f"global_top_features_{safe_name(g)}.png"),
            dpi=300,
        )
        plt.close()


# ======================================================
# MAIN
# ======================================================


def run_method_feature_signature_analysis():
    prefixes = discover_prefixes()

    if not prefixes:
        raise RuntimeError(
            f"No '*_cp_kb_per_cluster.csv' files found in {CLUSTER_METHOD_OUTDIR}. "
            "Run the clustering script first."
        )

    print("Found prefixes:")
    for p in prefixes:
        print(f"  - {p}")

    all_sigs = []

    for prefix in prefixes:
        all_sigs.extend(analyze_prefix(prefix))

    make_global_summary(all_sigs)

    print("\nDone.")
    print(f"Results saved in: {ANALYSIS_OUTDIR}/")





def ensure_best_cluster_signature_tables(dataset_name, df_sweep, best_artifact):
    """
    Ensure the best cached Efficient clustering has cluster-zscore tables on disk.
    This is needed when the best clustering is loaded from cache, because the
    sweep plotting/signature-table code is skipped in that branch.
    """
    if df_sweep is None or len(df_sweep) == 0:
        return

    best = df_sweep.sort_values(by="balanced_score", ascending=False).iloc[0]
    best_name = best["name"]
    prefix = f"{dataset_name}_normalized_efficient_BEST_{best_name}"

    zscore_path = os.path.join(OUTDIR, f"{prefix}_cluster_zscores.csv")
    report_path = os.path.join(OUTDIR, f"{prefix}_top_cluster_feature_report.csv")

    if os.path.exists(zscore_path) and os.path.exists(report_path):
        return

    print(f"Creating missing best-clustering signature tables for: {prefix}")
    labels = best_artifact["labels"]
    features = best_artifact["features"]
    top_clusters = get_top_clusters(labels, top_n=10)
    save_cluster_signature_tables(features, labels, top_clusters, prefix=prefix)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    import cluster_detection as cd

    full_data = cd.load_all_json("../../data/timeseries/all/*.json")

    full_keys = load_series_keys(
        SERIES_INDEX_PATH,
        expected_len=len(full_data),
    )

    jagt_methods = load_jagt_method_table()

    results = {}
    pca_dim = 10

    datasets = {
        "drop_first1": drop_first_n_with_keys(full_data, full_keys, n=1),
    }

    efficient_sweep_results = {}

    for dataset_name, dataset_payload in datasets.items():
        data, series_keys = dataset_payload

        df_sweep, best_artifact = sweep_efficient_clustering(
            data,
            dataset_name=dataset_name,
        )

        ensure_best_cluster_signature_tables(
            dataset_name=dataset_name,
            df_sweep=df_sweep,
            best_artifact=best_artifact,
        )

        efficient_sweep_results[dataset_name] = {
            "sweep": df_sweep,
            "best": best_artifact,
        }

        best_prefix = f"{dataset_name}_efficient_best"
        cluster_eval = evaluate_cp_kb_per_cluster(
            labels=best_artifact["labels"],
            series_keys=series_keys,
            jagt_methods=jagt_methods,
            prefix=best_prefix,
        )

        data_normalized = preprocess_normalized(data)
        plot_relevant_cluster_bands(
            data=data_normalized,
            labels=best_artifact["labels"],
            cluster_eval_df=cluster_eval,
            prefix=best_prefix,
            top_n=10,
        )
        plot_relevant_cluster_bands_overlay(
            data=data_normalized,
            labels=best_artifact["labels"],
            cluster_eval_df=cluster_eval,
            prefix=best_prefix,
            top_n=10,
        )

    for dataset_name, dataset_payload in datasets.items():
        data, series_keys = dataset_payload

        for extractor in ["minimal", "efficient"]:

            key = f"{dataset_name}_normalized_{extractor}"

            results[key] = run_pipeline(
                data,
                dataset_name=dataset_name,
                extractor=extractor,
                pca_dim=pca_dim,
            )

            cluster_eval = evaluate_cp_kb_per_cluster(
                labels=results[key]["labels"],
                series_keys=series_keys,
                jagt_methods=jagt_methods,
                prefix=key,
            )

            data_normalized = preprocess_normalized(data)
            plot_relevant_cluster_bands(
                data=data_normalized,
                labels=results[key]["labels"],
                cluster_eval_df=cluster_eval,
                prefix=key,
                top_n=10,
            )
            plot_relevant_cluster_bands_overlay(
                data=data_normalized,
                labels=results[key]["labels"],
                cluster_eval_df=cluster_eval,
                prefix=key,
                top_n=10,
            )

    for dataset_name in datasets:

        compare_two_clusterings(
            results[f"{dataset_name}_normalized_minimal"],
            results[f"{dataset_name}_normalized_efficient"],
            name_a=f"{dataset_name}_normalized_minimal",
            name_b=f"{dataset_name}_normalized_efficient",
        )

        compare_top_cluster_membership(
            results[f"{dataset_name}_normalized_minimal"],
            results[f"{dataset_name}_normalized_efficient"],
            name_a=f"{dataset_name}_normalized_minimal",
            name_b=f"{dataset_name}_normalized_efficient",
        )

    df_quality = compare_quality_table(results)
    plot_quality_table_metrics(df_quality)
    compare_cluster_counts(results)

    summary_rows = []

    for dataset_name, dataset_payload in datasets.items():
        data, _ = dataset_payload

        for extractor in ["minimal", "efficient"]:

            key = f"{dataset_name}_normalized_{extractor}"
            res = results[key]

            labels = res["labels"]
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = int(np.sum(labels == -1))
            outlier_frac = n_outliers / len(labels)

            mean_abs_z, max_abs_z = feature_signature_strength(
                res["features"],
                labels,
            )

            row = {
                "scenario": dataset_name,
                "extractor": extractor,
                "n_series": len(labels),
                "n_features": res["features"].shape[1],
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "outlier_frac": outlier_frac,
                "mean_abs_cluster_z": mean_abs_z,
                "max_abs_cluster_z": max_abs_z,
            }

            summary_rows.append(row)

    df_ablation = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print("CLUSTERING SUMMARY")
    print("=" * 80)
    print(df_ablation)

    df_ablation.to_csv(
        f"{OUTDIR}/ablation_summary.csv",
        index=False,
    )

    scenario_order = ["drop_first1"]

    for extractor in ["minimal", "efficient"]:

        df_e = df_ablation[df_ablation["extractor"] == extractor].copy()
        df_e["scenario"] = pd.Categorical(
            df_e["scenario"],
            categories=scenario_order,
            ordered=True,
        )
        df_e = df_e.sort_values("scenario")

        for metric in [
            "n_clusters",
            "outlier_frac",
            "mean_abs_cluster_z",
            "max_abs_cluster_z",
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(df_e["scenario"].astype(str), df_e[metric])
            ax.set_title(f"{metric} ({extractor})")
            ax.set_ylabel(metric)
            ax.set_xlabel("Scenario")
            ax.set_xticklabels(df_e["scenario"].astype(str), rotation=30, ha="right")

            prettify_axes(ax)

            plt.tight_layout()
            plt.savefig(
                f"{OUTDIR}/ablation_{extractor}_{metric}.png",
                dpi=300,
            )
            plt.close()

    # ------------------------------------------------------
    # Method-win feature-signature analysis
    # ------------------------------------------------------
    # This post-processes the already saved per-cluster CP/KB tables
    # and the corresponding cluster_zscores tables.
    run_method_feature_signature_analysis()

    print("\nDone.")
    print(f"Results saved in: {OUTDIR}/")


# ======================================================
# MEETING-READY METHOD SUMMARY EXTENSION
# ======================================================
#!/usr/bin/env python3
"""
Meeting-ready summary of CP-SSD vs KB-KSSD behavior across TSFresh clusters.

This script is intentionally post-processing only. It does NOT rerun TSFresh,
PCA, UMAP, or HDBSCAN. It reads the outputs already produced by your clustering
script:

  tsfresh_testing_2/cluster_method_comparison/*_cp_kb_per_cluster.csv
  tsfresh_testing_2/*_cluster_zscores.csv

It creates concise CSV summaries and plots in:

  tsfresh_testing_2/meeting_method_summary/

Main outputs:
  - method_support_summary.csv
  - feature_family_summary_<prefix>.csv
  - top_features_<prefix>_<group>.csv
  - representative_clusters_<prefix>.csv
  - *_method_support.png
  - *_feature_family_summary.png
  - *_complexity_vs_kb_advantage.png
  - *_representative_cluster_zscores.png

Interpretation idea:
  KB-favored clusters are summarized by complexity / entropy / variability / burstiness.
  CP-favored clusters are summarized by wavelet / autocorrelation / peak / ordered structure.
"""

import os
import re
import glob
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# CONFIGURATION
# ======================================================

OUTDIR = "tsfresh_testing_2"
CLUSTER_METHOD_DIR = os.path.join(OUTDIR, "cluster_method_comparison")
SUMMARY_OUTDIR = os.path.join(OUTDIR, "meeting_method_summary")
os.makedirs(SUMMARY_OUTDIR, exist_ok=True)

TOP_N_FEATURES = 12
MIN_JAGT_FOR_CLUSTER_SUMMARY = 1

# Optional: set to True if you want to ignore HDBSCAN outlier cluster -1 in summaries.
EXCLUDE_OUTLIER_CLUSTER = False


# ======================================================
# PLOT HELPERS
# ======================================================

def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


# ======================================================
# INPUT DISCOVERY
# ======================================================

def find_method_tables() -> Dict[str, str]:
    pattern = os.path.join(CLUSTER_METHOD_DIR, "*_cp_kb_per_cluster.csv")
    paths = sorted(glob.glob(pattern))

    prefix_to_path = {}
    for p in paths:
        name = os.path.basename(p).replace("_cp_kb_per_cluster.csv", "")
        prefix_to_path[name] = p

    if not prefix_to_path:
        raise FileNotFoundError(
            f"No CP/KB per-cluster tables found in {CLUSTER_METHOD_DIR}. "
            "Run the clustering script first."
        )

    return prefix_to_path


def find_zscore_table(prefix: str) -> str | None:
    """
    Locate the cluster_zscores table belonging to a given method-comparison prefix.

    Expected cases:
      drop_first1_normalized_minimal -> drop_first1_normalized_minimal_pca10_cluster_zscores.csv
      drop_first1_normalized_efficient -> drop_first1_normalized_efficient_pca10_cluster_zscores.csv
      drop_first1_efficient_best -> drop_first1_normalized_efficient_BEST_..._cluster_zscores.csv
    """

    candidates = []

    # Exact-ish baseline match
    candidates.extend(glob.glob(os.path.join(OUTDIR, f"{prefix}*_cluster_zscores.csv")))

    # Best efficient match
    if prefix.endswith("efficient_best"):
        dataset = prefix.replace("_efficient_best", "")
        candidates.extend(glob.glob(
            os.path.join(OUTDIR, f"{dataset}_normalized_efficient_BEST*_cluster_zscores.csv")
        ))

    # Fallbacks
    if "normalized_efficient" in prefix:
        candidates.extend(glob.glob(os.path.join(OUTDIR, "*normalized_efficient*_cluster_zscores.csv")))
    if "normalized_minimal" in prefix:
        candidates.extend(glob.glob(os.path.join(OUTDIR, "*normalized_minimal*_cluster_zscores.csv")))

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    if not uniq:
        return None

    # Prefer BEST for best prefix, otherwise prefer pca10 baseline.
    if prefix.endswith("efficient_best"):
        best = [c for c in uniq if "BEST" in os.path.basename(c)]
        if best:
            return sorted(best)[-1]

    pca10 = [c for c in uniq if "pca10" in os.path.basename(c)]
    if pca10:
        return sorted(pca10)[0]

    return sorted(uniq)[0]


# ======================================================
# FEATURE FAMILY CLASSIFICATION
# ======================================================

def feature_family(feature: str) -> str:
    f = feature.lower()

    if any(s in f for s in [
        "lempel_ziv", "entropy", "cid_ce", "complexity", "permutation_entropy",
        "sample_entropy", "approximate_entropy"
    ]):
        return "Complexity / entropy"

    if any(s in f for s in [
        "change_quantiles", "mean_abs_change", "mean_change", "variance",
        "standard_deviation", "abs_energy", "root_mean_square", "variation_coefficient"
    ]):
        return "Local variability"

    if any(s in f for s in [
        "ratio_beyond_r_sigma", "kurtosis", "skewness", "large_standard_deviation",
        "count_above", "count_below", "range_count"
    ]):
        return "Bursts / heavy tails"

    if any(s in f for s in [
        "cwt_coefficients", "spkt_welch_density", "fft_coefficient", "fft_aggregated",
        "fourier", "welch"
    ]):
        return "Frequency / wavelet structure"

    if any(s in f for s in [
        "autocorrelation", "partial_autocorrelation", "c3", "time_reversal_asymmetry"
    ]):
        return "Autocorrelation / temporal dependence"

    if any(s in f for s in [
        "linear_trend", "agg_linear_trend", "augmented_dickey", "number_crossing_m"
    ]):
        return "Trend / nonstationarity"

    if any(s in f for s in [
        "number_peaks", "longest_strike", "symmetry_looking", "has_duplicate", "first_location", "last_location"
    ]):
        return "Peaks / shape structure"

    if any(s in f for s in [
        "quantile", "minimum", "maximum", "median", "mean", "sum_values", "absolute_maximum"
    ]):
        return "Level / distribution position"

    return "Other"


KB_FAMILIES = {
    "Complexity / entropy",
    "Local variability",
    "Bursts / heavy tails",
    "Trend / nonstationarity",
}

CP_FAMILIES = {
    "Frequency / wavelet structure",
    "Autocorrelation / temporal dependence",
    "Peaks / shape structure",
}


# ======================================================
# GROUP DEFINITIONS
# ======================================================

def add_method_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "cp_best_better_than_kb", "kb_better_than_cp_best",
        "cp_orig_better_than_kb", "kb_better_than_cp_orig",
        "cp_best_mae", "cp_orig_mae", "kb_mae", "n_jagt"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column in method table: {c}")

    df["cp_best_win_margin"] = df["cp_best_better_than_kb"] - df["kb_better_than_cp_best"]
    df["kb_vs_cp_best_win_margin"] = df["kb_better_than_cp_best"] - df["cp_best_better_than_kb"]

    df["cp_orig_win_margin"] = df["cp_orig_better_than_kb"] - df["kb_better_than_cp_orig"]
    df["kb_vs_cp_orig_win_margin"] = df["kb_better_than_cp_orig"] - df["cp_orig_better_than_kb"]

    df["kb_advantage_vs_cp_best_mae"] = df["cp_best_mae"] - df["kb_mae"]
    df["kb_advantage_vs_cp_orig_mae"] = df["cp_orig_mae"] - df["kb_mae"]

    df["group_kb_over_cp_best"] = df["kb_vs_cp_best_win_margin"] > 0
    df["group_cp_best_over_kb"] = df["cp_best_win_margin"] > 0
    df["group_kb_over_cp_orig"] = df["kb_vs_cp_orig_win_margin"] > 0
    df["group_cp_orig_over_kb"] = df["cp_orig_win_margin"] > 0

    df["group_kb_lower_mae_than_cp_best"] = df["kb_advantage_vs_cp_best_mae"] > 0
    df["group_cp_best_lower_mae_than_kb"] = df["kb_advantage_vs_cp_best_mae"] < 0
    df["group_kb_lower_mae_than_cp_orig"] = df["kb_advantage_vs_cp_orig_mae"] > 0
    df["group_cp_orig_lower_mae_than_kb"] = df["kb_advantage_vs_cp_orig_mae"] < 0

    return df


GROUPS = {
    "KB wins more often than CP-best": "group_kb_over_cp_best",
    "CP-best wins more often than KB": "group_cp_best_over_kb",
    "KB wins more often than CP-original": "group_kb_over_cp_orig",
    "CP-original wins more often than KB": "group_cp_orig_over_kb",
    "KB lower MAE than CP-best": "group_kb_lower_mae_than_cp_best",
    "CP-best lower MAE than KB": "group_cp_best_lower_mae_than_kb",
    "KB lower MAE than CP-original": "group_kb_lower_mae_than_cp_orig",
    "CP-original lower MAE than KB": "group_cp_orig_lower_mae_than_kb",
}


# ======================================================
# SUMMARY COMPUTATION
# ======================================================

def prepare_joined_tables(method_path: str, zscore_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method_df = pd.read_csv(method_path)
    z_df = pd.read_csv(zscore_path)

    method_df = add_method_group_columns(method_df)

    if EXCLUDE_OUTLIER_CLUSTER:
        method_df = method_df[method_df["cluster"] != -1].copy()
        z_df = z_df[z_df["cluster"] != -1].copy()

    method_df = method_df[method_df["n_jagt"] >= MIN_JAGT_FOR_CLUSTER_SUMMARY].copy()

    # Ensure matching cluster type.
    method_df["cluster"] = method_df["cluster"].astype(int)
    z_df["cluster"] = z_df["cluster"].astype(int)

    return method_df, z_df


def weighted_feature_summary(
    method_df: pd.DataFrame,
    z_df: pd.DataFrame,
    group_col: str,
    top_n: int = TOP_N_FEATURES,
) -> pd.DataFrame:
    clusters = method_df.loc[method_df[group_col], ["cluster", "n_jagt"]]

    if clusters.empty:
        return pd.DataFrame(columns=["feature", "mean_z", "mean_abs_z", "n_clusters", "n_jagt_total", "family"])

    joined = clusters.merge(z_df, on="cluster", how="inner")

    if joined.empty:
        return pd.DataFrame(columns=["feature", "mean_z", "mean_abs_z", "n_clusters", "n_jagt_total", "family"])

    feature_cols = [
        c for c in joined.columns
        if c not in {"cluster", "size", "n_jagt"}
        and pd.api.types.is_numeric_dtype(joined[c])
    ]

    weights = joined["n_jagt"].astype(float).values
    if np.sum(weights) <= 0:
        weights = np.ones(len(joined), dtype=float)

    rows = []
    for f in feature_cols:
        vals = joined[f].astype(float).values
        mean_z = np.average(vals, weights=weights)
        mean_abs_z = np.average(np.abs(vals), weights=weights)
        rows.append({
            "feature": f,
            "mean_z": mean_z,
            "mean_abs_z": mean_abs_z,
            "n_clusters": joined["cluster"].nunique(),
            "n_jagt_total": int(joined["n_jagt"].sum()),
            "family": feature_family(f),
        })

    out = pd.DataFrame(rows).sort_values("mean_abs_z", ascending=False)
    return out.head(top_n)


def feature_family_summary(
    method_df: pd.DataFrame,
    z_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    clusters = method_df.loc[method_df[group_col], ["cluster", "n_jagt"]]

    if clusters.empty:
        return pd.DataFrame(columns=["family", "mean_abs_z", "mean_z", "n_clusters", "n_jagt_total"])

    joined = clusters.merge(z_df, on="cluster", how="inner")

    if joined.empty:
        return pd.DataFrame(columns=["family", "mean_abs_z", "mean_z", "n_clusters", "n_jagt_total"])

    feature_cols = [
        c for c in joined.columns
        if c not in {"cluster", "size", "n_jagt"}
        and pd.api.types.is_numeric_dtype(joined[c])
    ]

    weights = joined["n_jagt"].astype(float).values
    if np.sum(weights) <= 0:
        weights = np.ones(len(joined), dtype=float)

    rows = []
    for f in feature_cols:
        vals = joined[f].astype(float).values
        rows.append({
            "feature": f,
            "family": feature_family(f),
            "mean_z_feature": np.average(vals, weights=weights),
            "mean_abs_z_feature": np.average(np.abs(vals), weights=weights),
            "n_clusters": joined["cluster"].nunique(),
            "n_jagt_total": int(joined["n_jagt"].sum()),
        })

    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return pd.DataFrame(columns=["family", "mean_abs_z", "mean_z", "n_clusters", "n_jagt_total"])

    out = tmp.groupby("family", as_index=False).agg(
        mean_abs_z=("mean_abs_z_feature", "mean"),
        mean_z=("mean_z_feature", "mean"),
        n_clusters=("n_clusters", "max"),
        n_jagt_total=("n_jagt_total", "max"),
    )

    out = out.sort_values("mean_abs_z", ascending=False)
    return out


def compute_complexity_score(z_row: pd.Series) -> float:
    vals = []
    for col, val in z_row.items():
        if col in {"cluster", "size"}:
            continue
        if not isinstance(val, (int, float, np.number)):
            continue
        fam = feature_family(col)
        if fam in KB_FAMILIES:
            vals.append(abs(float(val)))
    return float(np.mean(vals)) if vals else np.nan


def compute_cp_structure_score(z_row: pd.Series) -> float:
    vals = []
    for col, val in z_row.items():
        if col in {"cluster", "size"}:
            continue
        if not isinstance(val, (int, float, np.number)):
            continue
        fam = feature_family(col)
        if fam in CP_FAMILIES:
            vals.append(abs(float(val)))
    return float(np.mean(vals)) if vals else np.nan


def representative_clusters(method_df: pd.DataFrame, z_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    candidates = [
        ("Strongest KB over CP-best", "kb_vs_cp_best_win_margin", False),
        ("Strongest CP-best over KB", "cp_best_win_margin", False),
        ("Strongest KB MAE advantage over CP-best", "kb_advantage_vs_cp_best_mae", False),
        ("Strongest CP-best MAE advantage over KB", "kb_advantage_vs_cp_best_mae", True),
        ("Strongest KB over CP-original", "kb_vs_cp_orig_win_margin", False),
        ("Strongest CP-original over KB", "cp_orig_win_margin", False),
    ]

    for label, metric, ascending in candidates:
        sub = method_df[method_df["n_jagt"] > 0].copy()
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            continue
        row = sub.sort_values(metric, ascending=ascending).iloc[0]
        cid = int(row["cluster"])

        zrow_match = z_df[z_df["cluster"].astype(int) == cid]
        complexity = np.nan
        cp_structure = np.nan
        if not zrow_match.empty:
            zrow = zrow_match.iloc[0]
            complexity = compute_complexity_score(zrow)
            cp_structure = compute_cp_structure_score(zrow)

        rows.append({
            "representative_type": label,
            "cluster": cid,
            "cluster_size": int(row["cluster_size"]),
            "n_jagt": int(row["n_jagt"]),
            "cp_best_better_than_kb": int(row["cp_best_better_than_kb"]),
            "kb_better_than_cp_best": int(row["kb_better_than_cp_best"]),
            "cp_orig_better_than_kb": int(row["cp_orig_better_than_kb"]),
            "kb_better_than_cp_orig": int(row["kb_better_than_cp_orig"]),
            "cp_best_mae": row["cp_best_mae"],
            "cp_orig_mae": row["cp_orig_mae"],
            "kb_mae": row["kb_mae"],
            "complexity_score": complexity,
            "cp_structure_score": cp_structure,
        })

    return pd.DataFrame(rows)


# ======================================================
# PLOTS
# ======================================================

def plot_method_support(method_df: pd.DataFrame, prefix: str):
    support = pd.DataFrame([
        {
            "group": "KB > CP-best",
            "jagt_support": int(method_df.loc[method_df["group_kb_over_cp_best"], "n_jagt"].sum()),
        },
        {
            "group": "CP-best > KB",
            "jagt_support": int(method_df.loc[method_df["group_cp_best_over_kb"], "n_jagt"].sum()),
        },
        {
            "group": "KB > CP-original",
            "jagt_support": int(method_df.loc[method_df["group_kb_over_cp_orig"], "n_jagt"].sum()),
        },
        {
            "group": "CP-original > KB",
            "jagt_support": int(method_df.loc[method_df["group_cp_orig_over_kb"], "n_jagt"].sum()),
        },
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(support["group"], support["jagt_support"])
    ax.set_ylabel("JAGT steady series in favored clusters")
    ax.set_title(f"Method support by cluster group: {prefix}")
    ax.set_xticklabels(support["group"], rotation=25, ha="right")
    prettify_axes(ax)
    savefig(os.path.join(SUMMARY_OUTDIR, f"{prefix}_method_support.png"))

    return support


def plot_family_comparison(family_rows: List[pd.DataFrame], prefix: str):
    if not family_rows:
        return

    df = pd.concat(family_rows, ignore_index=True)
    if df.empty:
        return

    # Keep the most interpretable families.
    keep_families = [
        "Complexity / entropy",
        "Local variability",
        "Bursts / heavy tails",
        "Trend / nonstationarity",
        "Frequency / wavelet structure",
        "Autocorrelation / temporal dependence",
        "Peaks / shape structure",
        "Level / distribution position",
    ]
    df = df[df["family"].isin(keep_families)].copy()

    # One concise plot: KB vs CP-best groups only.
    df = df[df["comparison_group"].isin([
        "KB wins more often than CP-best",
        "CP-best wins more often than KB",
    ])]

    if df.empty:
        return

    pivot = df.pivot_table(
        index="family",
        columns="comparison_group",
        values="mean_abs_z",
        aggfunc="mean",
        fill_value=0.0,
    )

    # Sort by total signal.
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True).drop(columns="_total")

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(pivot.index))
    width = 0.38

    cols = list(pivot.columns)
    if len(cols) == 1:
        ax.barh(y, pivot[cols[0]], height=0.5, label=cols[0])
    else:
        ax.barh(y - width / 2, pivot[cols[0]], height=width, label=cols[0])
        ax.barh(y + width / 2, pivot[cols[1]], height=width, label=cols[1])

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Mean |cluster feature Z-score|")
    ax.set_title(f"Feature-family profile of KB- vs CP-favored clusters: {prefix}")
    ax.legend(frameon=False, fontsize=8)
    prettify_axes(ax)
    savefig(os.path.join(SUMMARY_OUTDIR, f"{prefix}_feature_family_summary.png"))


def plot_top_feature_bars(summary: pd.DataFrame, prefix: str, group_name: str):
    if summary.empty:
        return

    group_slug = sanitize_filename(group_name)
    sub = summary.head(10).copy()
    sub = sub.sort_values("mean_abs_z", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r.feature}\n[{r.family}]" for r in sub.itertuples()]
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_abs_z"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Weighted mean |Z-score|")
    ax.set_title(f"Top TSFresh features: {group_name}\n{prefix}")
    prettify_axes(ax)
    savefig(os.path.join(SUMMARY_OUTDIR, f"{prefix}_top_features_{group_slug}.png"))


def plot_complexity_vs_advantage(method_df: pd.DataFrame, z_df: pd.DataFrame, prefix: str):
    merged = method_df.merge(z_df, on="cluster", how="inner", suffixes=("", "_z"))
    if merged.empty:
        return

    rows = []
    for _, r in merged.iterrows():
        if r["n_jagt"] <= 0:
            continue
        complexity = compute_complexity_score(r)
        cp_structure = compute_cp_structure_score(r)
        rows.append({
            "cluster": int(r["cluster"]),
            "n_jagt": int(r["n_jagt"]),
            "complexity_score": complexity,
            "cp_structure_score": cp_structure,
            "kb_advantage_vs_cp_best_mae": r["kb_advantage_vs_cp_best_mae"],
            "kb_win_margin_vs_cp_best": r["kb_vs_cp_best_win_margin"],
        })

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return

    df.to_csv(os.path.join(SUMMARY_OUTDIR, f"{prefix}_cluster_complexity_advantage.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.clip(df["n_jagt"].values, 5, 120)
    ax.scatter(df["complexity_score"], df["kb_advantage_vs_cp_best_mae"], s=sizes, alpha=0.65)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Complexity / variability / burstiness score")
    ax.set_ylabel("KB advantage over CP-best in MAE\npositive = KB lower MAE")
    ax.set_title(f"Complexity vs KB advantage: {prefix}")
    prettify_axes(ax)
    savefig(os.path.join(SUMMARY_OUTDIR, f"{prefix}_complexity_vs_kb_advantage.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["cp_structure_score"], df["kb_advantage_vs_cp_best_mae"], s=sizes, alpha=0.65)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Wavelet / autocorrelation / peak-structure score")
    ax.set_ylabel("KB advantage over CP-best in MAE\npositive = KB lower MAE")
    ax.set_title(f"Structured-shape score vs KB advantage: {prefix}")
    prettify_axes(ax)
    savefig(os.path.join(SUMMARY_OUTDIR, f"{prefix}_structure_vs_kb_advantage.png"))


def plot_representative_cluster_zscores(rep_df: pd.DataFrame, z_df: pd.DataFrame, prefix: str, top_n_features: int = 8):
    if rep_df.empty:
        return

    for _, rep in rep_df.iterrows():
        cid = int(rep["cluster"])
        zrow = z_df[z_df["cluster"].astype(int) == cid]
        if zrow.empty:
            continue
        zrow = zrow.iloc[0]

        feats = []
        for col, val in zrow.items():
            if col in {"cluster", "size"}:
                continue
            if isinstance(val, (int, float, np.number)):
                feats.append((col, float(val), abs(float(val)), feature_family(col)))

        if not feats:
            continue

        fdf = pd.DataFrame(feats, columns=["feature", "z", "abs_z", "family"])
        fdf = fdf.sort_values("abs_z", ascending=False).head(top_n_features)
        fdf = fdf.sort_values("abs_z", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{r.feature}\n[{r.family}]" for r in fdf.itertuples()]
        y = np.arange(len(fdf))
        ax.barh(y, fdf["z"])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Cluster feature Z-score")
        ax.set_title(
            f"Representative cluster {cid}: {rep['representative_type']}\n"
            f"n={int(rep['cluster_size'])}, JAGT n={int(rep['n_jagt'])}"
        )
        prettify_axes(ax)
        savefig(os.path.join(
            SUMMARY_OUTDIR,
            f"{prefix}_representative_cluster_{cid}_{sanitize_filename(rep['representative_type'])}.png"
        ))




# ======================================================
# PURE TSFRESH FEATURE SUMMARY (NO FAMILY AGGREGATION)
# ======================================================

def _feature_columns_from_zdf(z_df: pd.DataFrame) -> List[str]:
    return [
        c for c in z_df.columns
        if c not in {"cluster", "size"}
        and pd.api.types.is_numeric_dtype(z_df[c])
    ]


def _weighted_mean_safe(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def pure_tsfresh_feature_contrast(
    method_df: pd.DataFrame,
    z_df: pd.DataFrame,
    kb_group_col: str,
    cp_group_col: str,
    comparison_name: str,
    prefix: str,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Compare raw TSFresh feature z-scores between KB-favored and CP-favored clusters.

    This deliberately avoids feature-family aggregation. Each row is one original
    TSFresh feature with weighted mean z-scores in KB-favored and CP-favored clusters.
    Weights are n_jagt, so clusters with more JAGT evidence matter more.
    """

    merged = method_df.merge(z_df, on="cluster", how="inner", suffixes=("", "_z"))
    if merged.empty:
        return pd.DataFrame()

    feature_cols = _feature_columns_from_zdf(z_df)
    if not feature_cols:
        return pd.DataFrame()

    kb_sub = merged[merged[kb_group_col]].copy()
    cp_sub = merged[merged[cp_group_col]].copy()

    if kb_sub.empty and cp_sub.empty:
        return pd.DataFrame()

    rows = []
    for feat in feature_cols:
        kb_weights = kb_sub["n_jagt"].values if not kb_sub.empty else np.array([])
        cp_weights = cp_sub["n_jagt"].values if not cp_sub.empty else np.array([])

        kb_vals = kb_sub[feat].values if not kb_sub.empty else np.array([])
        cp_vals = cp_sub[feat].values if not cp_sub.empty else np.array([])

        kb_mean_z = _weighted_mean_safe(kb_vals, kb_weights)
        cp_mean_z = _weighted_mean_safe(cp_vals, cp_weights)
        kb_mean_abs_z = _weighted_mean_safe(np.abs(kb_vals), kb_weights)
        cp_mean_abs_z = _weighted_mean_safe(np.abs(cp_vals), cp_weights)

        rows.append({
            "comparison": comparison_name,
            "feature": feat,
            "family": feature_family(feat),
            "kb_mean_z": kb_mean_z,
            "cp_mean_z": cp_mean_z,
            "kb_mean_abs_z": kb_mean_abs_z,
            "cp_mean_abs_z": cp_mean_abs_z,
            "delta_abs_z_kb_minus_cp": kb_mean_abs_z - cp_mean_abs_z,
            "delta_z_kb_minus_cp": kb_mean_z - cp_mean_z,
            "abs_delta_abs_z": abs(kb_mean_abs_z - cp_mean_abs_z) if np.isfinite(kb_mean_abs_z) and np.isfinite(cp_mean_abs_z) else np.nan,
            "n_kb_clusters": int(len(kb_sub)),
            "n_cp_clusters": int(len(cp_sub)),
            "n_jagt_kb": int(kb_sub["n_jagt"].sum()) if not kb_sub.empty else 0,
            "n_jagt_cp": int(cp_sub["n_jagt"].sum()) if not cp_sub.empty else 0,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("abs_delta_abs_z", ascending=False)

    fname = f"{prefix}_pure_tsfresh_feature_contrast_{sanitize_filename(comparison_name)}.csv"
    out.to_csv(os.path.join(SUMMARY_OUTDIR, fname), index=False)

    plot_pure_tsfresh_feature_contrast(out, prefix, comparison_name, top_n=top_n)
    plot_pure_tsfresh_feature_profiles(out, prefix, comparison_name, top_n=top_n)

    return out


def plot_pure_tsfresh_feature_contrast(
    contrast_df: pd.DataFrame,
    prefix: str,
    comparison_name: str,
    top_n: int = 20,
):
    """
    Plot raw TSFresh features with the largest KB-vs-CP contrast in |z|.
    Positive values mean the feature is more prominent in KB-favored clusters;
    negative values mean it is more prominent in CP-favored clusters.
    """
    if contrast_df.empty:
        return

    sub = contrast_df.dropna(subset=["delta_abs_z_kb_minus_cp"]).head(top_n).copy()
    if sub.empty:
        return

    sub = sub.sort_values("delta_abs_z_kb_minus_cp", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [str(f) for f in sub["feature"]]
    y = np.arange(len(sub))

    ax.barh(y, sub["delta_abs_z_kb_minus_cp"])
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ mean |Z-score|  (KB-favored − CP-favored)")
    ax.set_title(f"Raw TSFresh feature contrast: {comparison_name}\n{prefix}")
    ax.text(
        0.01,
        0.01,
        "Positive = more prominent in KB-favored clusters; negative = more prominent in CP-favored clusters.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    prettify_axes(ax)

    savefig(os.path.join(
        SUMMARY_OUTDIR,
        f"{prefix}_pure_tsfresh_feature_contrast_{sanitize_filename(comparison_name)}.png"
    ))


def plot_pure_tsfresh_feature_profiles(
    contrast_df: pd.DataFrame,
    prefix: str,
    comparison_name: str,
    top_n: int = 15,
):
    """
    Plot KB and CP weighted mean |z| side by side for top raw TSFresh features.
    This is easier to read in meetings than the signed contrast alone.
    """
    if contrast_df.empty:
        return

    sub = contrast_df.dropna(subset=["abs_delta_abs_z"]).head(top_n).copy()
    if sub.empty:
        return

    sub = sub.sort_values("abs_delta_abs_z", ascending=True)

    y = np.arange(len(sub))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y - height / 2, sub["kb_mean_abs_z"], height, label="KB-favored clusters")
    ax.barh(y + height / 2, sub["cp_mean_abs_z"], height, label="CP-favored clusters")

    ax.set_yticks(y)
    ax.set_yticklabels([str(f) for f in sub["feature"]], fontsize=7)
    ax.set_xlabel("Weighted mean |Z-score|")
    ax.set_title(f"Raw TSFresh feature profiles: {comparison_name}\n{prefix}")
    ax.legend(frameon=False)
    prettify_axes(ax)

    savefig(os.path.join(
        SUMMARY_OUTDIR,
        f"{prefix}_pure_tsfresh_feature_profiles_{sanitize_filename(comparison_name)}.png"
    ))


def run_pure_tsfresh_feature_summaries(method_df: pd.DataFrame, z_df: pd.DataFrame, prefix: str):
    """
    Create concise raw-feature summaries for the main KB-vs-CP comparisons.
    These complement the feature-family summaries but keep the original TSFresh
    feature names intact.
    """

    comparisons = [
        (
            "win_count_cp_best",
            "group_kb_over_cp_best",
            "group_cp_best_over_kb",
        ),
        (
            "mae_cp_best",
            "group_kb_lower_mae_than_cp_best",
            "group_cp_best_lower_mae_than_kb",
        ),
        (
            "win_count_cp_original",
            "group_kb_over_cp_orig",
            "group_cp_orig_over_kb",
        ),
        (
            "mae_cp_original",
            "group_kb_lower_mae_than_cp_orig",
            "group_cp_orig_lower_mae_than_kb",
        ),
    ]

    all_rows = []
    for comparison_name, kb_col, cp_col in comparisons:
        if kb_col not in method_df.columns or cp_col not in method_df.columns:
            continue
        contrast = pure_tsfresh_feature_contrast(
            method_df=method_df,
            z_df=z_df,
            kb_group_col=kb_col,
            cp_group_col=cp_col,
            comparison_name=comparison_name,
            prefix=prefix,
            top_n=25,
        )
        if not contrast.empty:
            all_rows.append(contrast)

    if all_rows:
        all_contrasts = pd.concat(all_rows, ignore_index=True)
        all_contrasts.to_csv(
            os.path.join(SUMMARY_OUTDIR, f"{prefix}_pure_tsfresh_all_feature_contrasts.csv"),
            index=False,
        )

        # A very compact top-table for quick inspection: top 10 raw features per comparison.
        top_rows = []
        for comparison_name in all_contrasts["comparison"].unique():
            sub = all_contrasts[all_contrasts["comparison"] == comparison_name].head(10)
            top_rows.append(sub)
        pd.concat(top_rows, ignore_index=True).to_csv(
            os.path.join(SUMMARY_OUTDIR, f"{prefix}_pure_tsfresh_top10_by_comparison.csv"),
            index=False,
        )

        print(f"Saved pure TSFresh feature summaries for {prefix}")


# ======================================================
# MAIN ANALYSIS
# ======================================================

def analyze_prefix(prefix: str, method_path: str, zscore_path: str):
    print("\n" + "=" * 80)
    print(f"MEETING SUMMARY FOR PREFIX: {prefix}")
    print("=" * 80)
    print(f"Method table: {method_path}")
    print(f"Z-score table: {zscore_path}")

    method_df, z_df = prepare_joined_tables(method_path, zscore_path)

    if method_df.empty:
        print("No clusters with JAGT data. Skipping.")
        return

    # 1) Who wins where?
    support = plot_method_support(method_df, prefix)
    support["prefix"] = prefix
    support.to_csv(os.path.join(SUMMARY_OUTDIR, f"{prefix}_method_support_summary.csv"), index=False)

    # 2) Top features and feature-family summaries for each group.
    family_rows = []
    for group_name, group_col in GROUPS.items():
        top_features = weighted_feature_summary(method_df, z_df, group_col, top_n=TOP_N_FEATURES)
        top_features.to_csv(
            os.path.join(SUMMARY_OUTDIR, f"{prefix}_top_features_{sanitize_filename(group_name)}.csv"),
            index=False,
        )
        plot_top_feature_bars(top_features, prefix, group_name)

        fam = feature_family_summary(method_df, z_df, group_col)
        if not fam.empty:
            fam["comparison_group"] = group_name
            family_rows.append(fam)

    if family_rows:
        fam_all = pd.concat(family_rows, ignore_index=True)
        fam_all.to_csv(os.path.join(SUMMARY_OUTDIR, f"{prefix}_feature_family_summary.csv"), index=False)
        plot_family_comparison(family_rows, prefix)

    # 2b) Pure raw TSFresh-feature summaries, without family aggregation.
    run_pure_tsfresh_feature_summaries(method_df, z_df, prefix)

    # 3) Complexity / structured-shape relationship with method advantage.
    plot_complexity_vs_advantage(method_df, z_df, prefix)

    # 4) Representative clusters for slide-level discussion.
    reps = representative_clusters(method_df, z_df)
    reps.to_csv(os.path.join(SUMMARY_OUTDIR, f"{prefix}_representative_clusters.csv"), index=False)
    plot_representative_cluster_zscores(reps, z_df, prefix)

    # 5) Concise human-readable text summary.
    write_text_summary(prefix, method_df, family_rows, reps)


def write_text_summary(prefix: str, method_df: pd.DataFrame, family_rows: List[pd.DataFrame], reps: pd.DataFrame):
    support = {
        "KB > CP-best": int(method_df.loc[method_df["group_kb_over_cp_best"], "n_jagt"].sum()),
        "CP-best > KB": int(method_df.loc[method_df["group_cp_best_over_kb"], "n_jagt"].sum()),
        "KB > CP-original": int(method_df.loc[method_df["group_kb_over_cp_orig"], "n_jagt"].sum()),
        "CP-original > KB": int(method_df.loc[method_df["group_cp_orig_over_kb"], "n_jagt"].sum()),
    }

    lines = []
    lines.append(f"Meeting summary for {prefix}")
    lines.append("=" * (len(lines[-1])))
    lines.append("")
    lines.append("Method support by JAGT-bearing clusters:")
    for k, v in support.items():
        lines.append(f"- {k}: {v} JAGT steady series in favored clusters")

    if family_rows:
        fam_all = pd.concat(family_rows, ignore_index=True)
        lines.append("")
        lines.append("Most recurring feature-family signals:")
        for group in ["KB wins more often than CP-best", "CP-best wins more often than KB"]:
            sub = fam_all[fam_all["comparison_group"] == group].sort_values("mean_abs_z", ascending=False).head(5)
            if sub.empty:
                continue
            lines.append(f"\n{group}:")
            for _, r in sub.iterrows():
                lines.append(f"- {r['family']}: mean |z|={r['mean_abs_z']:.3f}, JAGT n={int(r['n_jagt_total'])}")

    if not reps.empty:
        lines.append("")
        lines.append("Representative clusters:")
        for _, r in reps.iterrows():
            lines.append(
                f"- {r['representative_type']}: cluster {int(r['cluster'])}, "
                f"cluster n={int(r['cluster_size'])}, JAGT n={int(r['n_jagt'])}, "
                f"KB MAE={r['kb_mae']:.2f}, CP-best MAE={r['cp_best_mae']:.2f}"
            )

    lines.append("")
    lines.append("Interpretation template:")
    lines.append(
        "KB-KSSD-favored clusters should be described in terms of complexity, entropy, "
        "local variability, burstiness, and trend uncertainty when those families dominate."
    )
    lines.append(
        "CP-SSD-favored clusters should be described in terms of wavelet/frequency structure, "
        "autocorrelation, peak structure, and ordered dynamics when those families dominate."
    )
    lines.append(
        "Use the generated scatter plots to check whether complexity/variability is actually "
        "associated with positive KB MAE advantage in this prefix."
    )

    path = os.path.join(SUMMARY_OUTDIR, f"{prefix}_meeting_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines[:20]))
    print(f"\nSaved text summary: {path}")


def run_meeting_ready_method_summary():
    warnings.filterwarnings("ignore", category=UserWarning)

    prefix_to_method = find_method_tables()

    print("Found method-comparison prefixes:")
    for p in prefix_to_method:
        print(f"  - {p}")

    missing = []
    for prefix, method_path in prefix_to_method.items():
        zscore_path = find_zscore_table(prefix)
        if zscore_path is None:
            missing.append(prefix)
            print(f"WARNING: no z-score table found for {prefix}; skipping.")
            continue
        analyze_prefix(prefix, method_path, zscore_path)

    if missing:
        with open(os.path.join(SUMMARY_OUTDIR, "missing_zscore_tables.txt"), "w") as f:
            f.write("\n".join(missing))

    print("\nDone.")
    print(f"Meeting-ready outputs saved in: {SUMMARY_OUTDIR}/")


if __name__ == "__main__":
    run_meeting_ready_method_summary()
