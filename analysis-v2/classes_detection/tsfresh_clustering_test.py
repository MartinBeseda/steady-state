#!/usr/bin/env python3

"""
TSFresh-based clustering pipeline for timeseries analysis.

Replaces handcrafted features with TSFresh descriptors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import hdbscan


# ======================================================
# 1. PREPARE DATA FOR TSFRESH
# ======================================================

def prepare_tsfresh_df(timeseries_list):
    """
    Convert list of arrays into TSFresh long-format DataFrame.

    Output columns:
    - id: timeseries id
    - time: time index
    - value: signal value
    """
    dfs = []

    for i, ts in enumerate(timeseries_list):
        df = pd.DataFrame({
            "id": i,
            "time": np.arange(len(ts)),
            "value": ts
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def preprocess_raw(data):
    return data


def preprocess_shifted(data):
    """
    Remove mean only (keeps scale + shape)
    """
    return [ts - np.mean(ts) for ts in data]


def preprocess_normalized(data):
    """
    Z-normalization (shape-only comparison)
    """
    out = []
    for ts in data:
        std = np.std(ts)
        if std == 0:
            out.append(ts - np.mean(ts))
        else:
            out.append((ts - np.mean(ts)) / std)
    return out


# ======================================================
# 2. FEATURE EXTRACTION (TSFRESH)
# ======================================================

def extract_tsfresh_features(timeseries_list):
    print("\nExtracting TSFresh features...")

    df = prepare_tsfresh_df(timeseries_list)

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        disable_progressbar=False
    )

    # Handle NaNs/infs
    features = impute(features)

    print(f"Extracted {features.shape[1]} features")

    return features.values


def extract_tsfresh_features_fast(timeseries_list):
    df = prepare_tsfresh_df(timeseries_list)

    print("\nExtracting TSFresh (FAST mode)...")

    settings = EfficientFCParameters()  # ⭐ KEY SPEEDUP

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        disable_progressbar=False,
        n_jobs=8  # increase if you have CPU
    )

    features = impute(features)

    print(f"Features shape: {features.shape}")

    return features.values


def extract_tsfresh_features_minimal(timeseries_list):
    df = prepare_tsfresh_df(timeseries_list)

    print("\nExtracting TSFresh (MINIMAL mode)...")

    settings = MinimalFCParameters()

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        n_jobs=8,
        disable_progressbar=True
    )

    features = impute(features)

    print(f"Features shape: {features.shape}")

    return features.values


# ======================================================
# 3. CLUSTERING
# ======================================================

def cluster_timeseries(features, pca_dim=10):
    print("\nRunning PCA + HDBSCAN...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=min(pca_dim, X_scaled.shape[1]))
    Xp = pca.fit_transform(X_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        metric='euclidean'
    )

    labels = clusterer.fit_predict(Xp)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")

    return labels, Xp


# ======================================================
# 4. SELECT TOP CLUSTERS
# ======================================================

def get_top_clusters(labels, top_n=10):
    cluster_sizes = {}

    for l in labels:
        if l == -1:
            continue
        cluster_sizes[l] = cluster_sizes.get(l, 0) + 1

    top_clusters = sorted(cluster_sizes.items(),
                          key=lambda x: x[1],
                          reverse=True)[:top_n]

    print("\nTop clusters:")
    for cid, size in top_clusters:
        print(f"Cluster {cid}: {size} series")

    return [cid for cid, _ in top_clusters]


def run_pipeline(data, mode_name="raw", extractor="minimal", pca_dim=10):
    print("\n" + "=" * 60)
    print(f"RUNNING MODE: {mode_name}")
    print("=" * 60)

    # -------------------------
    # preprocessing
    # -------------------------
    if mode_name == "raw":
        data_p = preprocess_raw(data)
    elif mode_name == "shifted":
        data_p = preprocess_shifted(data)
    elif mode_name == "normalized":
        data_p = preprocess_normalized(data)
    else:
        raise ValueError("Unknown mode")

    # -------------------------
    # feature extraction
    # -------------------------
    if extractor == "full":
        features = extract_tsfresh_features(data_p)
    elif extractor == "fast":
        features = extract_tsfresh_features_fast(data_p)
    elif extractor == "minimal":
        features = extract_tsfresh_features_minimal(data_p)
    else:
        raise ValueError("Unknown extractor")

    # -------------------------
    # clustering
    # -------------------------
    labels, Xp = cluster_timeseries(features, pca_dim=pca_dim)

    # -------------------------
    # top clusters
    # -------------------------
    top_clusters = get_top_clusters(labels, top_n=10)

    # -------------------------
    # plots
    # -------------------------
    prefix = f"{mode_name}_{extractor}"

    plot_cluster_means(data_p, labels, top_clusters, prefix=prefix)
    plot_derivatives(data_p, labels, top_clusters, prefix=prefix)
    plot_cluster_distance_matrix(Xp, labels, top_clusters, prefix=prefix)
    plot_first_window(data_p, labels, top_clusters, window=100, prefix=prefix)

    return {
        "labels": labels,
        "Xp": Xp,
        "top_clusters": top_clusters,
        "features": features
    }


# ======================================================
# 5. VISUALIZATION: MEAN + STD
# ======================================================

def plot_cluster_means(data, labels, cluster_ids, prefix="tsfresh"):
    plt.figure(figsize=(10, 6))

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        cluster_series = np.array(data)[idx]

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(len(mean_curve))

        line, = plt.plot(x, mean_curve, linewidth=2, label=f"C{cid}")
        color = line.get_color()

        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color
        )

    plt.title("Cluster Mean Curves (TSFresh)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"tsfresh_testing/{prefix}_cluster_means.png")
    plt.show()


# ======================================================
# 6. DERIVATIVE OVERLAY (VERY USEFUL)
# ======================================================

def plot_derivatives(data, labels, cluster_ids, prefix="tsfresh"):
    plt.figure(figsize=(10, 6))

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        cluster_series = np.array(data)[idx]

        derivatives = np.diff(cluster_series, axis=1)

        mean_deriv = np.mean(derivatives, axis=0)

        x = np.arange(len(mean_deriv))
        plt.plot(x, mean_deriv, label=f"C{cid}")

    plt.title("Mean Derivatives per Cluster (TSFresh)")
    plt.xlabel("Time")
    plt.ylabel("Δ Value")
    plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"tsfresh_testing/{prefix}_derivatives.png")
    plt.show()


# ======================================================
# 7. DISTANCE BETWEEN CLUSTERS (CENTROIDS)
# ======================================================

def plot_cluster_distance_matrix(Xp, labels, cluster_ids, prefix="tsfresh"):
    from scipy.spatial.distance import cdist

    centroids = []

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroids.append(np.mean(Xp[idx], axis=0))

    centroids = np.array(centroids)

    dist_matrix = cdist(centroids, centroids)

    plt.figure(figsize=(6, 5))
    plt.imshow(dist_matrix)
    plt.colorbar(label="Distance")

    plt.xticks(range(len(cluster_ids)), cluster_ids)
    plt.yticks(range(len(cluster_ids)), cluster_ids)

    plt.title("Cluster Distance Matrix (TSFresh)")
    plt.tight_layout()
    plt.savefig(f"tsfresh_testing/{prefix}_distance_matrix.png")
    plt.show()


def compare_cluster_counts(results):
    modes = list(results.keys())

    counts = []
    for m in modes:
        labels = results[m]["labels"]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        counts.append(n_clusters)

    plt.figure()
    plt.bar(modes, counts)
    plt.title("Cluster count comparison (TSFresh modes)")
    plt.ylabel("Number of clusters")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================
# 8. OPTIONAL: FIRST WINDOW VIEW
# ======================================================

def plot_first_window(data, labels, cluster_ids, window=100, prefix="tsfresh"):
    plt.figure(figsize=(10, 6))

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        cluster_series = np.array(data)[idx][:, :window]

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(window)

        line, = plt.plot(x, mean_curve, label=f"C{cid}")
        color = line.get_color()

        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color
        )

    plt.title(f"First {window} Samples (TSFresh)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"tsfresh_testing/{prefix}_first_window.png")
    plt.show()


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    import cluster_detection as cd
    data = cd.load_all_json('../../data/timeseries/all/*.json')

    results = {}

    # ======================================================
    # RUN ALL MODES
    # ======================================================

    results["raw"] = run_pipeline(
        data,
        mode_name="raw",
        extractor="minimal",
        pca_dim=10
    )

    results["shifted"] = run_pipeline(
        data,
        mode_name="shifted",
        extractor="minimal",
        pca_dim=10
    )

    results["normalized"] = run_pipeline(
        data,
        mode_name="normalized",
        extractor="minimal",
        pca_dim=10
    )

    compare_cluster_counts(results)