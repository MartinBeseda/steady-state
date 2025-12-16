#!/usr/bin/env python3
"""
Human-like clustering of timeseries based on structural shape features.
Enhanced version with:
- Safe scalar fix for best_overall comparison
- Cluster-size scaled UMAP visualization
- Full cluster summary printout
- Visualization of all timeseries in the clusters
- ADDITIONAL PLOTS: cluster size distribution, heatmaps, centroids,
  pairwise cluster distances, feature variability, scatter-matrix.
"""

import os, glob, json, warnings
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, silhouette_samples
import hdbscan
import matplotlib.pyplot as plt
import umap
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(0)

DATA_GLOB = "../../data/timeseries/all/*.json"
RESAMPLE_LEN = 500

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def load_all_json(glob_pattern=DATA_GLOB):
    series = []
    for fp in glob.glob(glob_pattern):
        try:
            data = json.load(open(fp))
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    for s in data:
                        series.append(np.array(s, dtype=float))
                else:
                    series.append(np.array(data, dtype=float))
        except Exception:
            continue
    print(f"Loaded {len(series)} total timeseries from {len(glob.glob(glob_pattern))} files.")
    return series


def z_norm(ts):
    ts = np.asarray(ts)
    ts = ts - np.nanmean(ts)
    std = np.nanstd(ts)
    return ts / (std + 1e-8)


def resample(ts, target_len):
    x_old = np.linspace(0, 1, len(ts))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, ts)


# -------------------------------------------------------------------
# SHAPE SIGNATURE EXTRACTION
# -------------------------------------------------------------------

def extract_shape_signature(ts, smooth_sigma=3):
    ts = gaussian_filter1d(z_norm(ts), sigma=smooth_sigma)
    n = len(ts)

    peaks, _ = find_peaks(ts)
    troughs, _ = find_peaks(-ts)
    n_peaks = len(peaks)
    n_troughs = len(troughs)

    amp = np.max(ts) - np.min(ts)
    mean_val = np.mean(ts)
    std_val = np.std(ts)
    p10, p90 = np.percentile(ts, [10, 90])
    spread = p90 - p10

    diff = np.diff(ts)
    pos_slope = np.mean(diff[diff > 0]) if np.any(diff > 0) else 0
    neg_slope = np.mean(diff[diff < 0]) if np.any(diff < 0) else 0
    pos_ratio = np.sum(diff > 0) / len(diff)

    if n_peaks > 1:
        avg_peak_dist = np.mean(np.diff(peaks))
    else:
        avg_peak_dist = n

    symmetry = abs(n_peaks - n_troughs) / (n_peaks + n_troughs + 1e-6)
    curvature = np.mean(np.abs(np.diff(diff)))
    complexity = (n_peaks + n_troughs) / n
    autocorr = np.corrcoef(ts[:-1], ts[1:])[0, 1] if n > 2 else 0

    return np.array([
        n_peaks, n_troughs, amp, mean_val, std_val,
        pos_slope, neg_slope, pos_ratio, avg_peak_dist,
        symmetry, curvature, complexity, autocorr, spread
    ], dtype=float)


# -------------------------------------------------------------------
# CLUSTERING
# -------------------------------------------------------------------

def evaluate_hdbscan(features, metric='euclidean',
                     mcs_values=[3, 5, 7, 10, 12, 15],
                     eps_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    best = None
    for mcs in mcs_values:
        for eps in eps_values:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, cluster_selection_epsilon=eps, metric=metric)
                labels = clusterer.fit_predict(features)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = np.sum(labels == -1)
                if n_clusters < 2:
                    continue

                mask = labels >= 0
                if np.sum(mask) > 5:
                    silhouette = silhouette_score(features[mask], labels[mask])
                    dbi = davies_bouldin_score(features[mask], labels[mask])
                else:
                    silhouette, dbi = -1, np.inf

                score = silhouette - 0.1 * dbi - (n_outliers / len(features))

                print(f"PCA=? | metric={metric:8s} | clusters={n_clusters:4d} | "
                      f"outliers={n_outliers:5d} ({n_outliers/len(features)*100:5.2f}%) | "
                      f"mcs={mcs:3d} | eps={eps:4.2f} | silhouette={silhouette:5.3f} | DBI={dbi:5.3f}")

                if best is None or float(score) > float(np.mean(best[-1])):
                    best = (n_clusters, n_outliers, mcs, eps, labels, score)
            except Exception:
                continue
    return best


def summarize_clusters(features, labels):
    clusters = np.unique(labels[labels >= 0])
    summaries = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        cluster_feats = features[idx]
        avg = np.mean(cluster_feats, axis=0)
        size = len(idx)
        desc = (
            f"Cluster {c:3d} | size={size:4d} | peaks={avg[0]:4.1f} | "
            f"troughs={avg[1]:4.1f} | amp={avg[2]:6.3f} | std={avg[4]:5.3f} | "
            f"pos_ratio={avg[7]:4.2f} | symmetry={avg[9]:5.3f} | autocorr={avg[12]:5.3f}"
        )
        summaries.append((size, desc))
    summaries.sort(reverse=True, key=lambda x: x[0])
    return summaries


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    series = load_all_json(DATA_GLOB)
    if not series:
        return
    data = np.array([resample(z_norm(ts), RESAMPLE_LEN) for ts in series])

    print("\nExtracting structural shape signatures...")
    shape_features = np.array([extract_shape_signature(ts) for ts in data])
    shape_features = np.nan_to_num(shape_features)

    metrics = ['euclidean']
    pca_dims = [5, 8, 10, 15]

    print("\nFinding best clustering configuration...\n")
    best_overall = None

    for n_pca in pca_dims:
        Xp = PCA(n_components=min(n_pca, shape_features.shape[1])).fit_transform(shape_features)
        for metric in metrics:
            cfg = evaluate_hdbscan(Xp, metric=metric)
            if cfg is None:
                continue
            clusters, outliers, mcs, eps, labels, score = cfg

            score_val = float(np.mean(score)) if np.ndim(score) > 0 else float(score)
            if best_overall is None:
                best_val = -np.inf
            else:
                try:
                    best_val = float(np.mean(best_overall[-1])) if np.ndim(best_overall[-1]) > 0 else float(best_overall[-1])
                except Exception:
                    best_val = float(np.mean(np.ravel(best_overall[-1])))

            if score_val > best_val:
                best_overall = (n_pca, metric, clusters, outliers, mcs, eps, labels, score_val, Xp)

    if not best_overall:
        print("No valid configuration found.")
        return

    n_pca, metric, clusters, outliers, mcs, eps, labels, score, Xp = best_overall
    print(f"\n--- BEST CONFIGURATION ---")
    print(f"PCA={n_pca}, metric={metric}, clusters={clusters}, outliers={outliers}, mcs={mcs}, eps={eps:.2f}")

    summaries = summarize_clusters(shape_features, labels)
    print("\n--- CLUSTER SUMMARIES (all clusters) ---")
    for _, s in summaries:
        print(s)

    # -------------------------------------------------------------------
    # UMAP VISUALIZATION
    # -------------------------------------------------------------------
    print("\nComputing UMAP projection (2D)...")
    # UMAP embedding
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(Xp)

    plt.figure(figsize=(8, 6), dpi=300)
    unique_labels = np.unique(labels)
    cluster_sizes = {c: np.sum(labels == c) for c in unique_labels if c >= 0}

    for c in unique_labels:
        idx = labels == c
        if c == -1:
            plt.scatter(emb[idx, 0], emb[idx, 1], s=8, c="gray", alpha=0.2, label="Outliers")
        else:
            plt.scatter(emb[idx, 0], emb[idx, 1],
                        s=np.clip(cluster_sizes[c] / 3, 10, 150),
                        alpha=0.6, label=f"C{c}")

    # Axes labels with larger font
    # plt.xlabel("UMAP Component 1", fontsize=16)
    # plt.ylabel("UMAP Component 2", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Optional: subtle grey grid
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)

    # Optional: legend
    # plt.legend(loc='upper right', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig("testing/plots/umap.png", dpi=300)
    plt.savefig("testing/plots/umap.eps", format='eps')
    plt.close()

    os.makedirs("testing/plots/clusters", exist_ok=True)

    # -------------------------------------------------------------------
    # VISUALIZE ALL CLUSTERS' TIMESERIES
    # -------------------------------------------------------------------
    print("\n--- VISUALIZING ALL CLUSTERS ---\n")
    for i, (size, summary_text) in enumerate(summaries):
        c = int(summary_text.split()[1])
        idx = np.where(labels == c)[0]
        plt.figure(figsize=(10, 4))
        for j in idx:
            plt.plot(data[j], alpha=0.4)
        plt.title(f"Cluster {c} (size={len(idx)})")
        plt.tight_layout()
        plt.savefig(f"testing/plots/clusters/cluster_{c}.png")
        plt.close()

    # ===============================================================
    # NEW SUMMARY PLOTS (ADDED)
    # ===============================================================

    # 1. Cluster Size Histogram
    sizes = np.array([s for s, _ in summaries])
    plt.figure(figsize=(8,4))
    plt.hist(sizes, bins=40, alpha=0.7)
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("testing/plots/cluster_size_hist.png")
    plt.close()

    # 1b. Log Histogram
    plt.figure(figsize=(8,4))
    plt.hist(np.log1p(sizes), bins=40, alpha=0.7)
    plt.title("Cluster Size Distribution (log-scale)")
    plt.xlabel("log(1 + size)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("testing/plots/cluster_size_loghist.png")
    plt.close()

    # 2. Feature Heatmap (cluster means)
    cluster_ids = [int(s.split()[1]) for _, s in summaries]
    cluster_means = []
    for size, stext in summaries:
        c = int(stext.split()[1])
        idx = np.where(labels == c)[0]
        cluster_means.append(np.mean(shape_features[idx], axis=0))
    cluster_means = np.array(cluster_means)

    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_means, aspect="auto", cmap="viridis")
    plt.colorbar(label="Feature value")
    plt.yticks(range(len(cluster_means)), cluster_ids)
    plt.xticks(range(shape_features.shape[1]), [
        "peaks","troughs","amp","mean","std","pos_slope","neg_slope",
        "pos_ratio","peak_dist","symmetry","curvature","complexity",
        "autocorr","spread"
    ], rotation=90)
    plt.title("Cluster Feature Means (Heatmap)")
    plt.tight_layout()
    plt.savefig("testing/plots/feature_heatmap.png")
    plt.close()

    # 3. Cluster Centroids Timeseries
    os.makedirs("testing/plots/centroids", exist_ok=True)
    for size, summary_text in summaries:
        c = int(summary_text.split()[1])
        idx = np.where(labels == c)[0]
        centroid = np.mean(data[idx], axis=0)
        plt.figure(figsize=(10, 3))
        plt.plot(centroid, linewidth=2)
        plt.title(f"Cluster {c} – Centroid Timeseries")
        plt.tight_layout()
        plt.savefig(f"testing/plots/centroids/cluster_{c}_centroid.png")
        plt.close()

    # 4. Pairwise Cluster Distances (in PCA space)
    centroids_pca = []
    for size, summary_text in summaries:
        c = int(summary_text.split()[1])
        idx = np.where(labels == c)[0]
        centroids_pca.append(np.mean(Xp[idx], axis=0))
    centroids_pca = np.array(centroids_pca)

    from sklearn.metrics import pairwise_distances
    dist_mat = pairwise_distances(centroids_pca)

    plt.figure(figsize=(10, 8))
    plt.imshow(dist_mat, cmap="magma", aspect="auto")
    plt.colorbar(label="Distance")
    plt.title("Pairwise Cluster Distances (PCA space)")
    plt.tight_layout()
    plt.savefig("testing/plots/pairwise_cluster_distances.png")
    plt.close()

    # 5. Feature Variance Heatmap
    variances = []
    for size, summary_text in summaries:
        c = int(summary_text.split()[1])
        idx = np.where(labels == c)[0]
        variances.append(np.var(shape_features[idx], axis=0))
    variances = np.array(variances)

    plt.figure(figsize=(10, 8))
    plt.imshow(variances, aspect="auto", cmap="inferno")
    plt.colorbar(label="Feature variance")
    plt.title("Cluster Feature Variability (Variance Heatmap)")
    plt.xticks(range(shape_features.shape[1]), [
        "peaks","troughs","amp","mean","std","pos_slope","neg_slope",
        "pos_ratio","peak_dist","symmetry","curvature","complexity",
        "autocorr","spread"
    ], rotation=90)
    plt.yticks(range(len(variances)), cluster_ids)
    plt.tight_layout()
    plt.savefig("testing/plots/feature_variance_heatmap.png")
    plt.close()

    # 6. Scatter-matrix style: amplitude vs curvature, peaks vs troughs
    plt.figure(figsize=(6,5))
    plt.scatter(shape_features[:,2], shape_features[:,10],
                c=labels, s=5, alpha=0.5, cmap="tab20")
    plt.xlabel("Amplitude")
    plt.ylabel("Curvature")
    plt.title("Amplitude vs Curvature colored by cluster")
    plt.tight_layout()
    plt.savefig("testing/plots/scatter_amp_curvature.png")
    plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(shape_features[:,0], shape_features[:,1],
                c=labels, s=5, alpha=0.5, cmap="tab20")
    plt.xlabel("Peaks")
    plt.ylabel("Troughs")
    plt.title("Peaks vs Troughs colored by cluster")
    plt.tight_layout()
    plt.savefig("testing/plots/scatter_peaks_troughs.png")
    plt.close()

    import matplotlib.ticker as mtick

    # --- Mask for valid (non-outlier) points ---
    mask = labels >= 0
    valid_labels = labels[mask]
    unique_clusters = np.unique(valid_labels)

    # --- 1. Per-cluster silhouette score (in PCA space) ---
    sample_silhouette_values = silhouette_samples(Xp[mask], valid_labels)
    mean_silhouette_per_cluster = [np.mean(sample_silhouette_values[valid_labels == c]) for c in unique_clusters]

    # --- 2. Fraction of outliers per cluster ---
    outlier_mask = labels == -1
    if np.sum(outlier_mask) > 0:
        Xp_outliers = Xp[outlier_mask]  # PCA-space representation of outliers
        # nearest cluster centroid in PCA space
        nearest_cluster = np.argmin(pairwise_distances(Xp_outliers, centroids_pca), axis=1)
        outlier_fraction_per_cluster = [
            np.sum(nearest_cluster == i) / np.sum(valid_labels == c)
            for i, c in enumerate(unique_clusters)
        ]
    else:
        outlier_fraction_per_cluster = [0] * len(unique_clusters)

    # --- 3. Mean intra-cluster distance (in PCA space) ---
    mean_intra_dist = []
    for c in unique_clusters:
        idx = np.where(valid_labels == c)[0]
        if len(idx) > 1:
            dist_mat = pairwise_distances(Xp[mask][idx])
            mean_intra_dist.append(np.mean(dist_mat[np.triu_indices(len(idx), k=1)]))
        else:
            mean_intra_dist.append(0)

    # --- Plotting all three metrics together ---
    x = np.arange(len(unique_clusters))
    width = 0.25

    plt.figure(figsize=(10, 4))
    plt.bar(x - width, mean_silhouette_per_cluster, width, label="Silhouette", color='C1', alpha=0.7)
    plt.bar(x, outlier_fraction_per_cluster, width, label="Outlier Fraction", color='C2', alpha=0.7)
    plt.bar(x + width, mean_intra_dist, width, label="Mean Intra-dist", color='C3', alpha=0.7)

    plt.xticks(x, unique_clusters)
    plt.xlabel("Cluster ID")
    plt.title("Cluster Quality Metrics (PCA space)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("testing/plots/cluster_quality_metrics.png")
    plt.close()

    # --------------------------------------------------
    # PLOT: QUALITY DISTRIBUTIONS
    # --------------------------------------------------

    plt.figure(figsize=(3.4, 1.9), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    titles = [
        "Silh. Sc.",
        "IC dist",
        "Out. fr."
    ]

    datasets = [
        mean_silhouette_per_cluster,
        mean_intra_dist,
        outlier_fraction_per_cluster
    ]

    for i, (title, data) in enumerate(zip(titles, datasets), start=1):
        ax = plt.subplot(1, 3, i)

        ax.boxplot(
            data,
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor="lightgray", edgecolor="black"),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )

        # ---- Tighten x-axis around the box ----
        ax.set_xlim(0.7, 1.3)
        ax.margins(x=0)

        # ---- Styling (paper-ready) ----
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xticks([])

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks([])

    # plt.suptitle(
    #     "Clustering Quality Distributions (PCA space)",
    #     fontsize=12,
    #     y=1.05
    # )

    plt.tight_layout()
    plt.savefig("testing/plots/cluster_quality_distributions_split.png", bbox_inches="tight")
    plt.savefig("testing/plots/cluster_quality_distributions_split.pdf", bbox_inches="tight")
    plt.savefig("testing/plots/cluster_quality_distributions_split.eps", format='eps', bbox_inches="tight")
    plt.close()

    # Silhouette
    sil_vals = mean_silhouette_per_cluster
    sil_mean = np.mean(sil_vals)
    sil_std = np.std(sil_vals)
    sil_min = np.min(sil_vals)
    sil_max = np.max(sil_vals)

    # Intra-cluster distance
    intra_vals = mean_intra_dist
    intra_mean = np.mean(intra_vals)
    intra_std = np.std(intra_vals)
    intra_min = np.min(intra_vals)
    intra_max = np.max(intra_vals)

    # Outlier fraction
    outlier_vals = outlier_fraction_per_cluster
    outlier_mean = np.mean(outlier_vals)
    outlier_std = np.std(outlier_vals)
    outlier_min = np.min(outlier_vals)
    outlier_max = np.max(outlier_vals)
    total_outliers_frac = np.sum(labels == -1) / len(labels)

    # Print nicely
    print("=== Cluster Quality Metrics Summary ===")
    print(f"Silhouette: mean={sil_mean:.3f}, std={sil_std:.3f}, min={sil_min:.3f}, max={sil_max:.3f}")
    print(
        f"Intra-cluster distance: mean={intra_mean:.3f}, std={intra_std:.3f}, min={intra_min:.3f}, max={intra_max:.3f}")
    print(
        f"Outlier fraction per cluster: mean={outlier_mean:.3f}, std={outlier_std:.3f}, min={outlier_min:.3f}, max={outlier_max:.3f}")
    print(f"Total fraction of outliers: {total_outliers_frac:.3f}")


if __name__ == "__main__":
    main()
