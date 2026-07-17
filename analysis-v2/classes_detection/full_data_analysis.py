#!/usr/bin/env python3

"""This file contains a clustering analysis performed on the full dataset of 5860 timeseries."""
import os

from scipy.stats import gaussian_kde, chi2_contingency, entropy, f_oneway
from sklearn.decomposition import PCA

import cluster_detection as cd
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns


def analyse_time_scale(data: list[np.ndarray]) -> dict:
    """
    Analyze time scales of full dataset.
    """

    def detect_time_scale(t):
        """
        Classify a time value (in seconds) into 's', 'ms', 'us' and 'ns'.
        """
        t = abs(np.max(t))

        if t >= 1:
            return "s"
        elif t >= 1e-3:
            return "ms"
        elif t >= 1e-6:
            return "us"
        else:
            return "ns"

    # Print bar plot of number of magnitudes (seconds, milliseconds, ...)
    time_scales = {'s': [], 'ms': [], 'us': [], 'ns': []}

    timeseries_name = list(timeseries_ssd_idx.keys())
    for i, e in enumerate(data):
        time_scales[detect_time_scale(e)].append(timeseries_name[i])

    plt.figure()
    plt.title(f'Time scales of full dataset ({len(data)} timeseries)')
    plt.bar(list(time_scales.keys()), [len(e) for e in time_scales.values()])
    plt.savefig('full_data_results/time_scales.png')
    plt.show()

    return time_scales


def analyse_clusters(data: list[np.ndarray]):
    data = np.array([cd.resample(cd.z_norm(ts), 500) for ts in data])

    print("\nExtracting structural shape signatures...")
    shape_features = np.array([cd.extract_shape_signature(ts) for ts in data])
    shape_features = np.nan_to_num(shape_features)

    metrics = ['euclidean']
    pca_dims = [5, 8, 10, 15]

    print("\nFinding best clustering configuration...\n")
    best_overall = None

    for n_pca in pca_dims:
        pca = PCA(n_components=min(n_pca, shape_features.shape[1]))
        Xp = pca.fit_transform(shape_features)

        for metric in metrics:
            cfg = cd.evaluate_hdbscan(Xp, metric=metric)
            if cfg is None:
                continue

            clusters, outliers, mcs, eps, labels, score = cfg

            score_val = float(np.mean(score)) if np.ndim(score) > 0 else float(score)

            if best_overall is None:
                best_val = -np.inf
            else:
                best_val = float(best_overall[7])

            if score_val > best_val:
                best_overall = (
                    n_pca, metric, clusters, outliers, mcs,
                    eps, labels, score_val, Xp, pca, shape_features
                )

    if not best_overall:
        print("No valid configuration found.")
        return

    return best_overall


def aggregate_cluster_data(ssd_idxs, time_scales):
    #------------------------
    # Aggregate cluster data
    #------------------------
    cluster_detail = {}
    for k, v in ssd_idxs.items():
        if v['cluster_idx'] not in cluster_detail:
            cluster_detail[v['cluster_idx']] = {'n_series': 0, 'n_unsteady': 0, 'projects': {}, 'benchmarks': {},
                                                'forks': [],
                                                'time_scales': {'s': 0, 'ms': 0, 'us': 0, 'ns': 0}}

        cluster_detail[v['cluster_idx']]['n_series'] += 1
        cluster_detail[v['cluster_idx']]['forks'].append(k)

        if k in time_scales['s']:
            cluster_detail[v['cluster_idx']]['time_scales']['s'] += 1
        elif k in time_scales['ms']:
            cluster_detail[v['cluster_idx']]['time_scales']['ms'] += 1
        elif k in time_scales['us']:
            cluster_detail[v['cluster_idx']]['time_scales']['us'] += 1
        else:
            cluster_detail[v['cluster_idx']]['time_scales']['ns'] += 1

        project_name = k.split('#')[0]
        benchmark_name = k.rsplit('#', 1)[0]

        if project_name not in cluster_detail[v['cluster_idx']]['projects']:
            cluster_detail[v['cluster_idx']]['projects'][project_name] = 1
        else:
            cluster_detail[v['cluster_idx']]['projects'][project_name] += 1

        if benchmark_name not in cluster_detail[v['cluster_idx']]['benchmarks']:
            cluster_detail[v['cluster_idx']]['benchmarks'][benchmark_name] = 1
        else:
            cluster_detail[v['cluster_idx']]['benchmarks'][benchmark_name] += 1

        if v['steadiness_idx_kbkssd'] == -1:
            cluster_detail[v['cluster_idx']]['n_unsteady'] += 1

    # ---- select top 10 clusters by size ----
    top_clusters = sorted({k: v for (k, v) in cluster_detail.items() if k != -1}.items(),
                          key=lambda x: x[1]['n_series'], reverse=True)[:10]

    cluster_ids = [cid for cid, _ in top_clusters]
    sizes = [c['n_series'] for _, c in top_clusters]
    unsteady = [c['n_unsteady'] for _, c in top_clusters]

    return top_clusters, cluster_ids, sizes, unsteady, cluster_detail


def plot_cluster_info(clusters):

    mean_time_scales = np.array((0,0,0,0), dtype=np.float64)

    for cid, c in clusters:
        projects = c['projects']
        benchmarks = c['benchmarks']
        time_scales = c['time_scales']

        plt.figure()
        plt.xticks(rotation=90, ha='right')
        plt.bar(range(len(projects)), projects.values(), label=projects.keys(), tick_label=projects.keys())
        plt.title(f"Cluster {cid} Projects (n={c['n_series']})")
        plt.tight_layout()
        plt.savefig(f'full_data_results/cluster_{cid}_projects.png')
        plt.show()

        plt.figure()
        plt.xticks(rotation=90, ha='right')
        plt.bar(range(1, 5), [time_scales['s'], time_scales['ms'], time_scales['us'], time_scales['ns']],
                tick_label=time_scales.keys())
        plt.title(f"Cluster {cid} Time Scales")
        plt.tight_layout()
        plt.savefig(f'full_data_results/cluster_{cid}_time_scales.png')
        plt.show()

        # Add time scales, so that we can plot their mean later
        mean_time_scales += np.array((time_scales['s'], time_scales['ms'], time_scales['us'], time_scales['ns']))

    mean_time_scales /= len(clusters)

    # Plot mean time scales
    plt.figure()
    plt.bar(range(1, 5), mean_time_scales, tick_label=('s', 'ms', 'us', 'ns'))
    plt.title('Mean time scales for largest clusters')
    plt.tight_layout()
    plt.savefig('full_data_results/mean_time_scales.png')
    plt.show()


def plot_cluster_importance(cluster_detail, suffix=''):
    """
    Plot cluster importance using:
    - cumulative coverage curve
    - rank-size distribution
    """

    # --- remove outliers cluster (-1) ---
    cluster_sizes = [
        v['n_series']
        for k, v in cluster_detail.items()
        if k != -1
    ]

    sizes = np.array(cluster_sizes)
    sizes_sorted = np.sort(sizes)[::-1]

    # ======================================================
    # 1. CUMULATIVE COVERAGE CURVE (MAIN RESULT)
    # ======================================================
    cum_sizes = np.cumsum(sizes_sorted)
    cum_fraction = cum_sizes / cum_sizes[-1]
    cluster_fraction = np.arange(1, len(sizes_sorted) + 1) / len(sizes_sorted)

    plt.figure(figsize=(6,5))
    plt.plot(cluster_fraction, cum_fraction, marker='o')

    plt.xlabel("Fraction of clusters")
    plt.ylabel("Fraction of timeseries")
    plt.title("Cluster Importance (Cumulative Coverage)")

    plt.axhline(0.8, linestyle='--', alpha=0.6)
    plt.axhline(0.9, linestyle='--', alpha=0.6)

    plt.fill_between(cluster_fraction, cum_fraction, alpha=0.2)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"full_data_results/cumulative_cluster_coverage_{suffix}.png")
    plt.show()

    # ======================================================
    # 2. RANK-SIZE DISTRIBUTION
    # ======================================================
    plt.figure(figsize=(6,5))
    plt.plot(sizes_sorted, marker='o')

    plt.yscale('log')
    plt.xlabel("Cluster rank")
    plt.ylabel("Cluster size (log)")
    plt.title("Cluster Size Rank Distribution")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"full_data_results/rank_size_clusters_{suffix}.png")
    plt.show()

    # ======================================================
    # 3. PRINT INTERPRETABLE NUMBERS (VERY IMPORTANT)
    # ======================================================
    total_clusters = len(sizes_sorted)

    for threshold in [0.5, 0.8, 0.9]:
        idx = np.argmax(cum_fraction >= threshold) + 1
        print(f"{idx} clusters ({idx/total_clusters:.1%}) explain {threshold:.0%} of data")

    # optional: top-k dominance
    top5 = np.sum(sizes_sorted[:5]) / np.sum(sizes_sorted)
    top10 = np.sum(sizes_sorted[:10]) / np.sum(sizes_sorted)

    print(f"Top 5 clusters explain {top5:.1%} of data")
    print(f"Top 10 clusters explain {top10:.1%} of data")


def plot_cluster_size_distribution(cluster_detail, suffix=''):
    sizes = np.array([
        v['n_series']
        for k, v in cluster_detail.items()
        if k != -1
    ])

    # KDE
    kde = gaussian_kde(sizes)
    x = np.linspace(1, np.max(sizes), 500)
    y = kde(x)

    plt.figure(figsize=(6,5))
    plt.plot(x, y, linewidth=2)
    plt.fill_between(x, y, alpha=0.2)

    plt.xlabel("Cluster size")
    plt.ylabel("Density")
    plt.title("Smoothed Cluster Size Distribution")
    plt.savefig(f'full_data_results/cluster_size_distribution_{suffix}.png')


def plot_log_distribution(cluster_detail, suffix=''):
    sizes = np.array([
        v['n_series']
        for k, v in cluster_detail.items()
        if k != -1
    ])

    log_sizes = np.log1p(sizes)

    plt.figure(figsize=(6,5))
    plt.hist(log_sizes, bins=30, alpha=0.7)

    plt.xlabel("log(1 + cluster size)")
    plt.ylabel("Count")
    plt.title("Log-Scaled Cluster Size Distribution")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"full_data_results/cluster_size_loghist_{suffix}.png")
    plt.show()


def print_cluster_size_stats(cluster_detail):
    sizes = np.array([
        v['n_series']
        for k, v in cluster_detail.items()
        if k != -1
    ])

    mean = np.mean(sizes)
    median = np.median(sizes)
    std = np.std(sizes)

    skew_indicator = mean / median if median > 0 else np.nan

    print("\n=== Cluster Size Statistics ===")
    print(f"Mean size:   {mean:.2f}")
    print(f"Median size: {median:.2f}")
    print(f"Std dev:     {std:.2f}")
    print(f"Mean/Median ratio: {skew_indicator:.2f}")

    if skew_indicator > 2:
        print("→ Strong right-skew (few large clusters dominate)")
    elif skew_indicator > 1.2:
        print("→ Moderate skew")
    else:
        print("→ Fairly balanced clusters")


def plot_violin(cluster_detail, suffix=''):
    sizes = np.array([
        v['n_series']
        for k, v in cluster_detail.items()
        if k != -1
    ])

    plt.figure(figsize=(4,5))
    plt.violinplot(sizes, showmeans=True)

    plt.ylabel("Cluster size")
    plt.title("Cluster Size Distribution (Violin)")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"full_data_results/cluster_size_violin_{suffix}.png")
    plt.show()


def compare_methods_full(cpssd_idx, kbkssd_idx, jagt_idx):
    """
    Compare CP-SSD and KB-KSSD against JAGT for a single timeseries.

    Returns:
        "cp_better", "kbk_better", or "tie"
    """

    # --- Convert to steady/unsteady ---
    cp_steady = cpssd_idx >= 0
    kbk_steady = kbkssd_idx >= 0
    jagt_steady = jagt_idx >= 0

    # --- Step 1: classification correctness ---
    cp_correct = (cp_steady == jagt_steady)
    kbk_correct = (kbk_steady == jagt_steady)

    # Case A: only one is correct
    if cp_correct and not kbk_correct:
        return "cp_better"
    if kbk_correct and not cp_correct:
        return "kbk_better"

    # Case B: both incorrect → tie
    if not cp_correct and not kbk_correct:
        return "tie"

    # --- Step 2: both correct ---
    # If unsteady → both are correct and identical
    if not jagt_steady:
        return "tie"

    # If steady → compare distances
    cp_dist = abs(cpssd_idx - jagt_idx)
    kbk_dist = abs(kbkssd_idx - jagt_idx)

    if cp_dist < kbk_dist:
        return "cp_better"
    elif kbk_dist < cp_dist:
        return "kbk_better"
    else:
        return "tie"


def compare_methods_binary(cpssd_idx, kbkssd_idx, jagt_idx):
    cp_steady = cpssd_idx >= 0
    kbk_steady = kbkssd_idx >= 0
    jagt_steady = jagt_idx >= 0

    cp_correct = (cp_steady == jagt_steady)
    kbk_correct = (kbk_steady == jagt_steady)

    if cp_correct and not kbk_correct:
        return "cp_better"
    elif kbk_correct and not cp_correct:
        return "kbk_better"
    else:
        return "tie"


def compute_cluster_comparison(cluster_source, timeseries_ssd_idx,
                               unsteady_keynames, steady_jagt,
                               use_binary=False):
    """
    cluster_source: dict OR list of (cluster_id, cluster_detail)
    """

    if isinstance(cluster_source, dict):
        cluster_items = [(k, v) for k, v in cluster_source.items() if k != -1]
    else:
        cluster_items = cluster_source

    results = {}

    for cluster_idx, cluster_detail in cluster_items:
        results[cluster_idx] = {
            'n_jagt_forks': 0,
            'cp_better': 0,
            'kbk_better': 0,
            'tie': 0
        }

        for fork in cluster_detail['forks']:
            if fork not in unsteady_keynames:
                continue

            # --- JAGT reference ---
            if unsteady_keynames[fork] == -1:
                jagt_sidx = -1
            else:
                jagt_sidx = steady_jagt[fork]['steady_idx']

            kbkssd_pred = timeseries_ssd_idx[fork]['steadiness_idx_kbkssd']

            fork_name, fork_idx = fork.rsplit('_', 1)
            cpssd_pred = json.load(
                open(f'../man_steady_comparison/orig_classification/{fork_name}')
            )['steady_state_starts'][int(fork_idx)]

            # --- Choose comparison mode ---
            if use_binary:
                res = compare_methods_binary(cpssd_pred, kbkssd_pred, jagt_sidx)
            else:
                res = compare_methods_full(cpssd_pred, kbkssd_pred, jagt_sidx)

            results[cluster_idx][res] += 1
            results[cluster_idx]['n_jagt_forks'] += 1

    return results


def plot_cp_vs_kbk(cluster_stats, suffix=''):
    clusters = sorted(cluster_stats.keys())

    cp_vals = [cluster_stats[c]['cp_better'] for c in clusters]
    kbk_vals = [cluster_stats[c]['kbk_better'] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(x - width/2, cp_vals, width, label="CP-SSD")
    plt.bar(x + width/2, kbk_vals, width, label="KB-KSSD")

    plt.xticks(x, clusters)
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.title("CP-SSD vs KB-KSSD (per cluster)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'full_data_results/cpssd_vs_kbkssd_{suffix}.png')
    plt.show()


def plot_cp_vs_kbk_ratio(cluster_stats, suffix=''):
    clusters = sorted(cluster_stats.keys())

    cp_ratios = []
    kbk_ratios = []

    for c in clusters:
        total = cluster_stats[c]['n_jagt_forks']
        cp = cluster_stats[c]['cp_better']
        kbk = cluster_stats[c]['kbk_better']

        cp_ratios.append(cp / total if total > 0 else 0)
        kbk_ratios.append(kbk / total if total > 0 else 0)

    x = np.arange(len(clusters))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(x - width/2, cp_ratios, width, label="CP-SSD")
    plt.bar(x + width/2, kbk_ratios, width, label="KB-KSSD")

    plt.xticks(x, clusters)
    plt.xlabel("Cluster ID")
    plt.ylabel("Fraction of wins")
    plt.title("CP-SSD vs KB-KSSD (normalized per cluster)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'full_data_results/cpssd_vs_kbkssd_normalized_{suffix}.png')
    plt.show()


def plot_difference(cluster_stats, suffix=''):
    clusters = sorted(cluster_stats.keys())

    diff = []
    for c in clusters:
        total = cluster_stats[c]['n_jagt_forks']
        cp = cluster_stats[c]['cp_better']
        kbk = cluster_stats[c]['kbk_better']

        val = (kbk - cp) / total if total > 0 else 0
        diff.append(val)

    x = np.arange(len(clusters))

    plt.figure(figsize=(10, 5))
    plt.bar(x, diff)

    plt.axhline(0, linestyle='--')

    plt.xticks(x, clusters)
    plt.xlabel("Cluster ID")
    plt.ylabel("(KBK - CP) / total")
    plt.title("Which method wins per cluster")

    plt.tight_layout()
    plt.savefig(f'full_data_results/method_success_per_cluster_{suffix}.png')
    plt.show()


def compare_by_timescale(timeseries_ssd_idx, time_scales,
                         unsteady_keynames, steady_jagt,
                         use_binary=False):

    results = {k: {'cp_better': 0, 'kbk_better': 0, 'tie': 0, 'total': 0}
               for k in ['s', 'ms', 'us', 'ns']}

    for scale, forks in time_scales.items():
        for fork in forks:

            if fork not in unsteady_keynames:
                continue

            # --- JAGT reference ---
            if unsteady_keynames[fork] == -1:
                jagt_sidx = -1
            else:
                jagt_sidx = steady_jagt[fork]['steady_idx']

            kbkssd_pred = timeseries_ssd_idx[fork]['steadiness_idx_kbkssd']

            fork_name, fork_idx = fork.rsplit('_', 1)
            cpssd_pred = json.load(
                open(f'../man_steady_comparison/orig_classification/{fork_name}')
            )['steady_state_starts'][int(fork_idx)]

            # --- comparison ---
            if use_binary:
                res = compare_methods_binary(cpssd_pred, kbkssd_pred, jagt_sidx)
            else:
                res = compare_methods_full(cpssd_pred, kbkssd_pred, jagt_sidx)

            results[scale][res] += 1
            results[scale]['total'] += 1

    return results


def plot_timescale_comparison(results, title, filename):
    scales = ['s', 'ms', 'us', 'ns']

    cp = [results[s]['cp_better'] / results[s]['total'] for s in scales]
    kbk = [results[s]['kbk_better'] / results[s]['total'] for s in scales]

    x = np.arange(len(scales))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, cp, width, label="CP-SSD")
    plt.bar(x + width/2, kbk, width, label="KB-KSSD")

    plt.xticks(x, scales)
    plt.ylabel("Fraction of wins")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



def test_timescale_dependency(results):
    """
    Chi-square test:
    Does method performance depend on timescale?
    """

    scales = ['s', 'ms', 'us', 'ns']

    table = []
    for s in scales:
        table.append([
            results[s]['cp_better'],
            results[s]['kbk_better'],
            results[s]['tie']
        ])

    table = np.array(table)

    chi2, p, dof, expected = chi2_contingency(table)

    print("\n=== Timescale Dependency Test ===")
    print("Contingency table (rows=scales, cols=[CP, KBK, tie]):")
    print(table)

    print(f"\nChi2 = {chi2:.3f}, p-value = {p:.5f}")

    if p < 0.05:
        print("→ Significant: performance depends on timescale")
    else:
        print("→ Not significant: no strong dependency")

    return chi2, p


def cramers_v(chi2, table):
    n = np.sum(table)
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r-1, k-1))))


def plot_timescale_stacked(results, suffix=""):
    scales = ['s', 'ms', 'us', 'ns']

    cp = np.array([results[s]['cp_better'] for s in scales])
    kbk = np.array([results[s]['kbk_better'] for s in scales])
    tie = np.array([results[s]['tie'] for s in scales])

    total = cp + kbk + tie

    cp = cp / total
    kbk = kbk / total
    tie = tie / total

    x = np.arange(len(scales))

    plt.figure(figsize=(6,4))

    plt.bar(x, cp, label="CP", alpha=0.8)
    plt.bar(x, kbk, bottom=cp, label="KBK", alpha=0.8)
    plt.bar(x, tie, bottom=cp+kbk, label="Tie", alpha=0.8)

    plt.xticks(x, scales)
    plt.ylabel("Fraction")
    plt.title("Method Performance by Timescale")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"full_data_results/timescale_stacked{suffix}.png")
    plt.show()


def timescale_cluster_purity(cluster_detail):
    entropies = []

    for cid, c in cluster_detail.items():
        if cid == -1:
            continue

        counts = np.array(list(c['time_scales'].values()))
        probs = counts / np.sum(counts)

        ent = entropy(probs)
        entropies.append(ent)

    mean_ent = np.mean(entropies)

    print("\n=== Timescale–Cluster Relationship ===")
    print(f"Mean entropy: {mean_ent:.3f}")

    if mean_ent < 0.5:
        print("→ Clusters are timescale-specific")
    elif mean_ent < 1.0:
        print("→ Moderate mixing")
    else:
        print("→ Timescale not strongly related to clusters")


def compute_cluster_difficulty(cluster_detail, shape_features, labels):
    difficulty = {}

    for cid, c in cluster_detail.items():
        if cid == -1:
            continue

        idx = np.where(labels == cid)[0]
        feats = shape_features[idx]

        # simple proxy: variability = difficulty
        var = np.mean(np.var(feats, axis=0))

        difficulty[cid] = var

    return difficulty


def plot_expressibility(results, difficulty, title, fname):
    xs = []
    cp = []
    kbk = []

    for cid in results:
        if cid not in difficulty:
            continue

        total = results[cid]['n_jagt_forks']
        if total == 0:
            continue

        xs.append(difficulty[cid])
        cp.append(results[cid]['cp_better'] / total)
        kbk.append(results[cid]['kbk_better'] / total)

    plt.figure(figsize=(6,5))
    plt.scatter(xs, cp, label="CP-SSD", alpha=0.6)
    plt.scatter(xs, kbk, label="KB-KSSD", alpha=0.6)

    plt.xlabel("Cluster difficulty (feature variance)")
    plt.ylabel("Win rate")
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

def error_distribution(results):
        vals = []
        for cid, r in results.items():
            total = r['n_jagt_forks']
            if total == 0:
                continue

            error_rate = 1 - (r['cp_better'] + r['kbk_better']) / total
            vals.append(error_rate)

        return np.array(vals)


def compare_error_distributions(results_full):
    cp_errors = []
    kbk_errors = []

    for cid, r in results_full.items():
        total = r['n_jagt_forks']
        if total == 0:
            continue

        cp_err = 1 - r['cp_better'] / total
        kbk_err = 1 - r['kbk_better'] / total

        cp_errors.append(cp_err)
        kbk_errors.append(kbk_err)

    # histogram → probability
    bins = np.linspace(0, 1, 20)
    cp_hist, _ = np.histogram(cp_errors, bins=bins, density=True)
    kbk_hist, _ = np.histogram(kbk_errors, bins=bins, density=True)

    kl = entropy(cp_hist + 1e-8, kbk_hist + 1e-8)

    print(f"KL divergence (CP || KBK): {kl:.4f}")


def prediction_entropy(timeseries_ssd_idx):
    cp_preds = []
    kbk_preds = []

    for k, v in timeseries_ssd_idx.items():
        kbk = v['steadiness_idx_kbkssd'] >= 0
        kbk_preds.append(int(kbk))

        # load CP
        fork_name, fork_idx = k.rsplit('_', 1)
        cp = json.load(
            open(f'../man_steady_comparison/orig_classification/{fork_name}')
        )['steady_state_starts'][int(fork_idx)] >= 0

        cp_preds.append(int(cp))

    def ent(x):
        probs = np.bincount(x) / len(x)
        return entropy(probs)

    print("\n=== Prediction Entropy ===")
    print(f"CP entropy:  {ent(cp_preds):.3f}")
    print(f"KBK entropy: {ent(kbk_preds):.3f}")


def failure_concentration(results):
    errors = []

    for cid, r in results.items():
        total = r['n_jagt_forks']
        if total == 0:
            continue

        err = 1 - max(r['cp_better'], r['kbk_better']) / total
        errors.append(err)

    errors = np.array(errors)

    print("\n=== Failure Concentration ===")
    print(f"Mean error: {np.mean(errors):.3f}")
    print(f"Std error:  {np.std(errors):.3f}")

    if np.std(errors) > 0.2:
        print("→ Failures concentrated (method struggles in specific regions)")
    else:
        print("→ Failures uniform (global limitation)")


def plot_cluster_envelopes(data, labels, top_clusters, mode="std"):
    """
    Visual comparison of clusters using mean/median + envelope.

    mode:
        "range" → min/max band
        "std"   → mean ± std
        "median_std" → median ± std
    """

    plt.figure(figsize=(10, 6))

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = data[idx]  # shape: (n_series, T)

        mean_curve = np.mean(cluster_series, axis=0)
        median_curve = np.median(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        if mode == "range":
            lower = np.min(cluster_series, axis=0)
            upper = np.max(cluster_series, axis=0)
            center = mean_curve

        elif mode == "std":
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
            center = mean_curve

        elif mode == "median_std":
            lower = median_curve - std_curve
            upper = median_curve + std_curve
            center = median_curve

        else:
            raise ValueError("Unknown mode")

        x = np.arange(len(center))

        plt.plot(x, center, label=f"C{cid}", linewidth=2)
        plt.fill_between(x, lower, upper, alpha=0.15)

    plt.title(f"Cluster Shape Comparison ({mode})")
    plt.xlabel("Time")
    plt.ylabel("Normalized value")
    plt.legend(ncol=2, fontsize=9)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_visual_comparison/cluster_comparison_{mode}.png")
    plt.show()


def plot_cluster_single(data, labels, top_clusters, mode="std"):
    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = data[idx]

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)
        median_curve = np.median(cluster_series, axis=0)

        if mode == "range":
            lower = np.min(cluster_series, axis=0)
            upper = np.max(cluster_series, axis=0)
            center = mean_curve
        elif mode == "std":
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
            center = mean_curve
        elif mode == "median_std":
            lower = median_curve - std_curve
            upper = median_curve + std_curve
            center = median_curve

        x = np.arange(len(center))

        plt.figure(figsize=(8, 4))
        plt.plot(x, center, linewidth=2, label="Center")
        plt.fill_between(x, lower, upper, alpha=0.2, label="Spread")

        plt.title(f"Cluster {cid} shape")
        plt.xlabel("Time")
        plt.ylabel("Normalized value")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"cluster_visual_comparison/cluster_{cid}_envelope_{mode}.png")
        plt.show()


def plot_cluster_distance_heatmap(data, labels, top_clusters):
    centroids = []
    cluster_ids = []

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = data[idx]
        centroid = np.mean(cluster_series, axis=0)

        centroids.append(centroid)
        cluster_ids.append(cid)

    centroids = np.array(centroids)

    from sklearn.metrics import pairwise_distances
    dist_mat = pairwise_distances(centroids)

    plt.figure(figsize=(8, 6))
    plt.imshow(dist_mat, cmap="viridis", aspect="auto")
    plt.colorbar(label="Distance")

    plt.xticks(range(len(cluster_ids)), cluster_ids)
    plt.yticks(range(len(cluster_ids)), cluster_ids)

    plt.title("Cluster Similarity (Centroid Distance)")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")

    plt.tight_layout()
    plt.savefig("cluster_visual_comparison/cluster_distance_heatmap.png")
    plt.show()


def plot_cluster_embedding(data, labels, top_clusters):
    from sklearn.decomposition import PCA

    centroids = []
    cluster_ids = []

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = data[idx]
        centroid = np.mean(cluster_series, axis=0)

        centroids.append(centroid)
        cluster_ids.append(cid)

    centroids = np.array(centroids)

    # Reduce to 2D
    emb = PCA(n_components=2).fit_transform(centroids)

    plt.figure(figsize=(6, 5))

    for i, cid in enumerate(cluster_ids):
        plt.scatter(emb[i, 0], emb[i, 1], s=100)
        plt.text(emb[i, 0], emb[i, 1], f"C{cid}", fontsize=10)

    plt.title("Cluster Similarity Map (Centroids)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cluster_visual_comparison/cluster_embedding.png")
    plt.show()


def print_most_similar_clusters(data, labels, top_clusters, top_k=5):
    from sklearn.metrics import pairwise_distances

    centroids = []
    cluster_ids = []

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        centroid = np.mean(data[idx], axis=0)

        centroids.append(centroid)
        cluster_ids.append(cid)

    centroids = np.array(centroids)
    dist_mat = pairwise_distances(centroids)

    pairs = []
    for i in range(len(cluster_ids)):
        for j in range(i+1, len(cluster_ids)):
            pairs.append((cluster_ids[i], cluster_ids[j], dist_mat[i, j]))

    pairs = sorted(pairs, key=lambda x: x[2])

    print("\n=== Most similar clusters ===")
    for c1, c2, d in pairs[:top_k]:
        print(f"C{c1} ↔ C{c2}  | distance = {d:.3f}")


def plot_cluster_raw(data, labels, top_clusters, subtract_mean=False):
    plt.figure(figsize=(10, 6))

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = np.array(data)[idx]

        # optionally remove vertical shift
        if subtract_mean:
            cluster_series = cluster_series - np.mean(cluster_series, axis=1, keepdims=True)

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(len(mean_curve))

        # --- plot mean and capture color ---
        line, = plt.plot(x, mean_curve, label=f"C{cid}", linewidth=2)
        color = line.get_color()

        # --- use SAME color for std band ---
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2
        )

    title = "Cluster Mean Curves (raw)"
    if subtract_mean:
        title = "Cluster Mean Curves (mean-shift removed)"

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=9)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_visual_comparison/cluster_raw_{'shifted' if subtract_mean else 'raw'}.png")
    plt.show()


def plot_cluster_zoom(data, labels, top_clusters, n_points=100, subtract_mean=False):
    plt.figure(figsize=(10, 6))

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_series = np.array(data)[idx][:, :n_points]

        if subtract_mean:
            cluster_series = cluster_series - np.mean(cluster_series, axis=1, keepdims=True)

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(n_points)

        line, = plt.plot(x, mean_curve, label=f"C{cid}", linewidth=2)
        color = line.get_color()

        plt.fill_between(x,
                         mean_curve - std_curve,
                         mean_curve + std_curve,
                         color=color,
                         alpha=0.2)

    plt.title(f"Cluster Comparison (first {n_points} samples)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=9)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_visual_comparison/cluster_zoom_{n_points}.png")
    plt.show()


def plot_cluster_zoom_raw(data, labels, top_clusters, n_points=100):
    """
    Zoomed-in plot (first N samples) using RAW data:
    - no normalization
    - no mean shifting
    """

    plt.figure(figsize=(10, 6))

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]

        # take first N points only
        cluster_series = np.array(data)[idx][:, :n_points]

        mean_curve = np.mean(cluster_series, axis=0)
        std_curve = np.std(cluster_series, axis=0)

        x = np.arange(n_points)

        # --- plot mean and capture color ---
        line, = plt.plot(x, mean_curve, label=f"C{cid}", linewidth=2)
        color = line.get_color()

        # --- std band ---
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2
        )

    plt.title(f"Cluster Comparison (RAW, first {n_points} samples)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=9)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_visual_comparison/cluster_zoom_raw_{n_points}.png")
    plt.show()


def plot_cluster_derivatives(data, labels, top_clusters, n_points=100):
    """
    Plot derivatives (first differences) for clusters:
    mean derivative + std band
    """

    plt.figure(figsize=(10, 6))

    for cid, _ in top_clusters:
        idx = np.where(labels == cid)[0]

        # take first N points
        cluster_series = np.array(data)[idx][:, :n_points]

        # --- compute derivative ---
        deriv = np.diff(cluster_series, axis=1)

        mean_curve = np.mean(deriv, axis=0)
        std_curve = np.std(deriv, axis=0)

        x = np.arange(len(mean_curve))

        line, = plt.plot(x, mean_curve, label=f"C{cid}", linewidth=2)
        color = line.get_color()

        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2
        )

    plt.title(f"Cluster Derivative Comparison (first {n_points} samples)")
    plt.xlabel("Time")
    plt.ylabel("Δ value")
    plt.legend(ncol=2, fontsize=9)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_visual_comparison/cluster_derivatives_{n_points}.png")
    plt.show()


def analyze_feature_importance(shape_features, labels, feature_names=None, top_k=10):
    """
    For each cluster:
    - compute mean feature vector
    - compare against global mean
    - show most distinguishing features
    """

    shape_features = np.array(shape_features)

    global_mean = np.mean(shape_features, axis=0)

    clusters = sorted(set(labels))
    clusters = [c for c in clusters if c != -1]

    for cid in clusters:
        idx = np.where(labels == cid)[0]
        feats = shape_features[idx]

        cluster_mean = np.mean(feats, axis=0)

        # difference from global
        diff = cluster_mean - global_mean

        # rank features
        important_idx = np.argsort(np.abs(diff))[::-1][:top_k]

        print(f"\n=== Cluster {cid} ===")
        for i in important_idx:
            fname = feature_names[i] if feature_names is not None else f"f{i}"
            print(f"{fname:20s}  value={cluster_mean[i]: .3f}  Δ={diff[i]: .3f}")


def analyze_pca_feature_contributions(Xp, shape_features, labels, pca, feature_names=None, top_k=10):
    """
    Interpret clusters in ORIGINAL feature space using PCA back-projection.
    """

    shape_features = np.array(shape_features)
    global_mean = np.mean(shape_features, axis=0)

    clusters = sorted(set(labels))
    clusters = [c for c in clusters if c != -1]

    for cid in clusters:
        idx = np.where(labels == cid)[0]

        if len(idx) == 0:
            continue

        # --- centroid in PCA space ---
        centroid_pca = np.mean(Xp[idx], axis=0)

        # --- back-project to original space ---
        centroid_orig = pca.inverse_transform(centroid_pca)

        # --- compare to global mean ---
        diff = centroid_orig - global_mean

        important_idx = np.argsort(np.abs(diff))[::-1][:top_k]

        print(f"\n=== Cluster {cid} (PCA → original space) ===")
        for i in important_idx:
            fname = feature_names[i] if feature_names else f"f{i}"
            print(f"{fname:20s}  value={centroid_orig[i]: .3f}  Δ={diff[i]: .3f}")



def build_feature_dataframe(shape_features, labels, feature_names):
    df = pd.DataFrame(shape_features, columns=feature_names)
    df['cluster'] = labels
    df = df[df['cluster'] != -1]
    return df


# ======================================================
# 1. FEATURE DISTRIBUTION PER CLUSTER
# ======================================================
def plot_feature_boxplots(df, top_clusters, feature_names, suffix=""):
    cluster_ids = [cid for cid, _ in top_clusters]

    df = df[df['cluster'].isin(cluster_ids)]

    for f in feature_names:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x='cluster', y=f)
        plt.title(f"{f} distribution across clusters")
        plt.tight_layout()
        plt.savefig(f"feature_analysis/{f}_boxplot{suffix}.png")
        plt.show()


# ======================================================
# 2. FEATURE HEATMAP (cluster means)
# ======================================================
def plot_feature_heatmap(df, top_clusters, feature_names, suffix=""):
    cluster_ids = [cid for cid, _ in top_clusters]

    means = []
    for cid in cluster_ids:
        means.append(df[df['cluster'] == cid][feature_names].mean())

    mat = np.array(means)

    # normalize (z-score across clusters)
    mat = (mat - np.mean(mat, axis=0)) / (np.std(mat, axis=0) + 1e-8)

    plt.figure(figsize=(10, 6))
    sns.heatmap(mat, xticklabels=feature_names, yticklabels=cluster_ids, cmap="coolwarm")
    plt.title("Feature importance per cluster (z-scored)")
    plt.tight_layout()
    plt.savefig(f"feature_analysis/feature_heatmap{suffix}.png")
    plt.show()


# ======================================================
# 3. ANOVA + EFFECT SIZE
# ======================================================
def feature_anova(df, feature_names):
    print("\n=== FEATURE SIGNIFICANCE (ANOVA) ===")

    results = []

    for f in feature_names:
        groups = [g[f].values for _, g in df.groupby('cluster')]

        if len(groups) < 2:
            continue

        F, p = f_oneway(*groups)

        # effect size η²
        grand_mean = df[f].mean()
        ss_between = sum(len(g) * (g[f].mean() - grand_mean) ** 2 for _, g in df.groupby('cluster'))
        ss_total = sum((df[f] - grand_mean) ** 2)

        eta2 = ss_between / ss_total if ss_total > 0 else 0

        results.append((f, F, p, eta2))

    results = sorted(results, key=lambda x: x[3], reverse=True)

    for f, F, p, eta2 in results:
        print(f"{f:20s}  F={F:8.2f}  p={p:.2e}  η²={eta2:.3f}")

    return results


# ======================================================
# 4. CLUSTER FEATURE PROFILES (Z-SCORES)
# ======================================================
def plot_cluster_feature_profiles(df, top_clusters, feature_names, suffix=""):
    cluster_ids = [cid for cid, _ in top_clusters]

    global_mean = df[feature_names].mean()
    global_std = df[feature_names].std() + 1e-8

    plt.figure(figsize=(10, 6))

    for cid in cluster_ids:
        mean = df[df['cluster'] == cid][feature_names].mean()
        z = (mean - global_mean) / global_std

        plt.plot(z.values, label=f"C{cid}")

    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.title("Cluster Feature Profiles (z-score)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"feature_analysis/feature_profiles{suffix}.png")
    plt.show()


# ======================================================
# 5. FEATURE CORRELATION
# ======================================================
def plot_feature_correlation(df, feature_names, suffix=""):
    corr = df[feature_names].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"feature_analysis/feature_correlation{suffix}.png")
    plt.show()


# ======================================================
# 6. PAIRWISE SCATTER (top features only)
# ======================================================
def plot_pairwise_scatter(df, top_features, suffix=""):
    sns.pairplot(df, vars=top_features, hue="cluster", corner=True)
    plt.savefig(f"feature_analysis/pairplot{suffix}.png")
    plt.show()


def plot_feature_boxplots_weighted(df, top_clusters, feature_names, suffix=""):
    """
    Boxplots of NORMALIZED features across top clusters.
    This approximates feature "importance" visually.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    cluster_ids = [cid for cid, _ in top_clusters]
    df = df[df['cluster'].isin(cluster_ids)].copy()

    # -----------------------------------------
    # 1. Normalize features (critical!)
    # -----------------------------------------
    for f in feature_names:
        mu = df[f].mean()
        sigma = df[f].std() + 1e-8
        df[f] = (df[f] - mu) / sigma

    # -----------------------------------------
    # 2. Melt dataframe (for seaborn)
    # -----------------------------------------
    df_melt = df.melt(
        id_vars="cluster",
        value_vars=feature_names,
        var_name="feature",
        value_name="value"
    )

    # -----------------------------------------
    # 3. Plot
    # -----------------------------------------
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df_melt,
        x="feature",
        y="value",
        hue="cluster",
        showfliers=False
    )

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Normalized value (z-score)")
    plt.title("Feature distributions across top clusters (normalized)")

    plt.legend(title="Cluster", ncol=2, fontsize=9)
    plt.tight_layout()

    plt.savefig(f"feature_analysis/feature_boxplots_weighted{suffix}.png")
    plt.show()


def compute_cluster_zscores(df, top_clusters, feature_names):
    """
    Compute per-cluster Z-scores for each feature.

    Returns:
        dict: {cluster_id: np.array of z-scores}
    """

    cluster_ids = [cid for cid, _ in top_clusters]

    # global stats
    global_mean = df[feature_names].mean()
    global_std = df[feature_names].std() + 1e-8

    zscores = {}

    for cid in cluster_ids:
        cluster_df = df[df['cluster'] == cid]

        if len(cluster_df) == 0:
            continue

        cluster_mean = cluster_df[feature_names].mean()

        z = (cluster_mean - global_mean) / global_std
        zscores[cid] = z.values

    return zscores, global_mean, global_std


def print_cluster_zscores(zscores, feature_names, top_k=10):
    """
    Print most important features per cluster (by |z|)
    """

    for cid, z in zscores.items():
        idx = np.argsort(np.abs(z))[::-1][:top_k]

        print(f"\n=== Cluster {cid} ===")
        for i in idx:
            print(f"{feature_names[i]:20s}  z = {z[i]: .3f}")


def save_cluster_zscores(zscores, feature_names, filename):
    import pandas as pd

    data = []
    for cid, z in zscores.items():
        row = {"cluster": cid}
        for i, f in enumerate(feature_names):
            row[f] = z[i]
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def plot_cluster_zscores(zscores, feature_names, suffix=""):
    import matplotlib.pyplot as plt

    for cid, z in zscores.items():
        plt.figure(figsize=(8, 4))

        x = np.arange(len(feature_names))

        plt.bar(x, z)
        plt.axhline(0, linestyle='--')

        plt.xticks(x, feature_names, rotation=45, ha='right')
        plt.ylabel("Z-score")
        plt.title(f"Cluster {cid} feature profile")

        plt.tight_layout()
        plt.savefig(f"feature_analysis/cluster_{cid}_zscores{suffix}.png")
        plt.show()


def plot_mean_feature_profiles_zscore(df, top_clusters, feature_names, suffix=""):
    import matplotlib.pyplot as plt
    import numpy as np

    df = df.copy()
    df = df[df['cluster'] != -1]

    # ----------------------------------------
    # global normalization
    # ----------------------------------------
    global_mean = df[feature_names].mean()
    global_std = df[feature_names].std() + 1e-8

    df_z = df.copy()
    df_z[feature_names] = (df_z[feature_names] - global_mean) / global_std

    # ----------------------------------------
    # compute means
    # ----------------------------------------
    mean_all = df_z[feature_names].mean()

    top_ids = [cid for cid, _ in top_clusters]
    mean_top = df_z[df_z['cluster'].isin(top_ids)][feature_names].mean()

    # ----------------------------------------
    # plot
    # ----------------------------------------
    x = np.arange(len(feature_names))

    plt.figure(figsize=(10, 5))

    plt.plot(x, mean_all.values, marker='o', label="All clusters (z-score)")
    plt.plot(x, mean_top.values, marker='o', label="Top 10 clusters (z-score)")

    plt.axhline(0, linestyle='--', alpha=0.5)

    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.ylabel("Z-scored mean feature value")
    plt.title("Feature Profiles (Normalized): All vs Top Clusters")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"feature_analysis/mean_feature_profiles_z{suffix}.png")
    plt.show()


def plot_mean_feature_profiles(df, top_clusters, feature_names, suffix=""):
    """
    Plot mean feature values:
    (1) across ALL clusters
    (2) across TOP clusters only
    """

    import matplotlib.pyplot as plt
    import numpy as np

    def compute_means(sub_df):
        return sub_df[feature_names].mean()

    # ----------------------------------------
    # ALL clusters (excluding noise)
    # ----------------------------------------
    df_all = df[df['cluster'] != -1]
    mean_all = compute_means(df_all)

    # ----------------------------------------
    # TOP clusters only
    # ----------------------------------------
    top_ids = [cid for cid, _ in top_clusters]
    df_top = df[df['cluster'].isin(top_ids)]
    mean_top = compute_means(df_top)

    # ----------------------------------------
    # PLOT
    # ----------------------------------------
    x = np.arange(len(feature_names))

    plt.figure(figsize=(10, 5))

    plt.plot(x, mean_all.values, marker='o', label="All clusters")
    plt.plot(x, mean_top.values, marker='o', label="Top 10 clusters")

    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.ylabel("Mean feature value")
    plt.title("Mean Feature Profiles: All vs Top Clusters")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"feature_analysis/mean_feature_profiles{suffix}.png")
    plt.show()


import numpy as np

def build_cluster_reports(shape_features, labels, feature_names, top_k=5):
    """
    Creates interpretable cluster reports:
    - Z-score signatures
    - top distinguishing features
    - simple automatic label
    """

    shape_features = np.array(shape_features)

    global_mean = np.mean(shape_features, axis=0)
    global_std = np.std(shape_features, axis=0) + 1e-12

    clusters = sorted(set(labels))
    clusters = [c for c in clusters if c != -1]

    reports = {}

    for cid in clusters:
        idx = np.where(labels == cid)[0]
        feats = shape_features[idx]

        cluster_mean = np.mean(feats, axis=0)

        z = (cluster_mean - global_mean) / global_std

        # -------------------------
        # top distinguishing features
        # -------------------------
        top_idx = np.argsort(np.abs(z))[::-1][:top_k]

        signature = [
            {
                "feature": feature_names[i] if feature_names else f"f{i}",
                "z": float(z[i])
            }
            for i in top_idx
        ]

        # -------------------------
        # simple automatic label
        # -------------------------
        pos = [feature_names[i] for i in np.where(z > 1.5)[0]]
        neg = [feature_names[i] for i in np.where(z < -1.5)[0]]

        if len(pos) == 0 and len(neg) == 0:
            label = "neutral / unstructured"
        else:
            label = ""
            if pos:
                label += "high " + ", ".join(pos[:2])
            if neg:
                if label:
                    label += " | "
                label += "low " + ", ".join(neg[:2])

        # -------------------------
        # store report
        # -------------------------
        reports[cid] = {
            "label": label,
            "signature": signature,
            "z_vector": z
        }

    return reports


def print_cluster_reports(reports):
    for cid, r in reports.items():
        print("\n" + "="*60)
        print(f"CLUSTER {cid}")
        print(f"SUMMARY: {r['label']}")
        print("-"*60)

        for f in r["signature"]:
            print(f"{f['feature']:20s}  Z = {f['z']:+.2f}")


def plot_feature_importance_global_vs_top10(shape_features, labels, feature_names, top_clusters):
    """
    Compare feature importance:
    - globally across all clusters
    - restricted to top-10 clusters
    """

    shape_features = np.array(shape_features)

    global_mean = np.mean(shape_features, axis=0)
    global_std = np.std(shape_features, axis=0) + 1e-12

    clusters = sorted(set(labels))
    clusters = [c for c in clusters if c != -1]

    top_cluster_ids = [cid for cid, _ in top_clusters]

    # -----------------------------
    # compute Z-scores per cluster
    # -----------------------------
    Z_all = []
    Z_top = []

    for cid in clusters:
        idx = np.where(labels == cid)[0]
        feats = shape_features[idx]

        cluster_mean = np.mean(feats, axis=0)
        z = (cluster_mean - global_mean) / global_std

        Z_all.append(z)

        if cid in top_cluster_ids:
            Z_top.append(z)

    Z_all = np.array(Z_all)
    Z_top = np.array(Z_top)

    # -----------------------------
    # feature importance = mean absolute Z
    # -----------------------------
    global_importance = np.mean(np.abs(Z_all), axis=0)
    top_importance = np.mean(np.abs(Z_top), axis=0)

    # -----------------------------
    # sort by global importance
    # -----------------------------
    order = np.argsort(global_importance)[::-1]

    features_sorted = [feature_names[i] for i in order]
    global_sorted = global_importance[order]
    top_sorted = top_importance[order]

    # -----------------------------
    # plot
    # -----------------------------
    x = np.arange(len(features_sorted))
    width = 0.4

    plt.figure(figsize=(12, 5))

    plt.bar(x - width/2, global_sorted, width, label="All clusters")
    plt.bar(x + width/2, top_sorted, width, label="Top 10 clusters")

    plt.xticks(x, features_sorted, rotation=90)
    plt.ylabel("Mean |Z-score| (importance)")
    plt.title("Feature Importance: Global vs Top 10 Clusters")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the full dataset
    full_data = cd.load_all_json('../../data/timeseries/all/*.json')

    # Load the DB od unsteady forks in JAGT
    unsteady_jagt = json.load(open('../man_steady_comparison/benchmark_database_binary.json'))['main']
    unsteady_keynames = {e['keyname']: e['value'] for e in unsteady_jagt}

    # Load the DB with steady-series JAGT
    steady_jagt = json.load(open('../man_steady_comparison/full_classification.json'))

    # Perform cluster analysis
    n_pca, metric, clusters, outliers, mcs, eps, labels, score, Xp, pca, shape_features = analyse_clusters(full_data)

    # Load the JAGT - timeseries steady-state idx (via KB-KSSD)
    timeseries_ssd_idx = json.load(open('series_kbkssd.json'))

    # Detect time scales of full dataset
    time_scales = analyse_time_scale(full_data)

    # Aggregate data about all the timeseries w.r.t. KB-KSSD taking 10 largest clusters into account
    top_clusters, cluster_idxs, cluster_sizes, cluster_unsteady, full_cluster_detail = aggregate_cluster_data(timeseries_ssd_idx,
                                                                                         time_scales)

    # Plot info about the largest clusters
    plot_cluster_info(top_clusters)

    # Compare CP-SSD and KB-KSSD on the largest clusters
    results_per_cluster = {}
    for cluster_idx, cluster_detail in top_clusters:
        results_per_cluster[cluster_idx] = {'n_jagt_forks': 0, 'cp_better': 0, 'kbk_better': 0, 'tie': 0}

        n_jagt_forks = 0
        for fork in cluster_detail['forks']:
            if fork in unsteady_keynames.keys():
                n_jagt_forks += 1

                # Reference SSD idx
                jagt_sidx = None
                if unsteady_keynames[fork] == -1:
                    jagt_sidx = -1
                else:
                    jagt_sidx = steady_jagt[fork]['steady_idx']

                kbkssd_pred = timeseries_ssd_idx[fork]['steadiness_idx_kbkssd']

                fork_name, fork_idx = fork.rsplit('_', 1)
                cpssd_pred = \
                json.load(open(f'../man_steady_comparison/orig_classification/{fork_name}'))['steady_state_starts'][
                    int(fork_idx)]

                results_per_cluster[cluster_idx][compare_methods_full(cpssd_pred, kbkssd_pred, jagt_sidx)] += 1

        results_per_cluster[cluster_idx]['n_jagt_forks'] = n_jagt_forks

    # ---- ALL clusters (full metric) ----
    all_cluster_results_full = compute_cluster_comparison(
        full_cluster_detail,
        timeseries_ssd_idx,
        unsteady_keynames,
        steady_jagt,
        use_binary=False
    )

    # ---- ALL clusters (binary only) ----
    all_cluster_results_binary = compute_cluster_comparison(
        full_cluster_detail,
        timeseries_ssd_idx,
        unsteady_keynames,
        steady_jagt,
        use_binary=True
    )

    # ---- TOP 10 (full metric) ----
    results_top10_full = compute_cluster_comparison(
        top_clusters,
        timeseries_ssd_idx,
        unsteady_keynames,
        steady_jagt,
        use_binary=False
    )

    # ---- TOP 10 (binary only) ----
    results_top10_binary = compute_cluster_comparison(
        top_clusters,
        timeseries_ssd_idx,
        unsteady_keynames,
        steady_jagt,
        use_binary=True
    )

    plot_cp_vs_kbk(results_per_cluster, "_default")
    plot_cp_vs_kbk_ratio(results_per_cluster, "_default")
    plot_difference(results_per_cluster, "_default")

    plot_cp_vs_kbk(results_top10_full, "_top10_full")
    plot_cp_vs_kbk_ratio(results_top10_full, "_top10_full")
    plot_difference(results_top10_full, "_top10_full")

    plot_cp_vs_kbk(results_top10_binary, "_top10_binary")
    plot_cp_vs_kbk_ratio(results_top10_binary, "_top10_binary")
    plot_difference(results_top10_binary, "_top10_binary")

    plot_cp_vs_kbk_ratio(all_cluster_results_full, "_all_full")
    plot_cp_vs_kbk_ratio(all_cluster_results_binary, "_all_binary")

    plot_cluster_importance(full_cluster_detail, "_all")
    plot_cluster_size_distribution(full_cluster_detail, "_all")
    plot_log_distribution(full_cluster_detail, "_all")
    plot_violin(full_cluster_detail, "_all")

    # ---- TIMESCALE comparisons ----
    ts_full = compare_by_timescale(
        timeseries_ssd_idx, time_scales,
        unsteady_keynames, steady_jagt,
        use_binary=False
    )

    ts_binary = compare_by_timescale(
        timeseries_ssd_idx, time_scales,
        unsteady_keynames, steady_jagt,
        use_binary=True
    )

    plot_timescale_comparison(
        ts_full,
        "CP vs KBK (Full metric) by timescale",
        "full_data_results/timescale_full.png"
    )

    plot_timescale_comparison(
        ts_binary,
        "CP vs KBK (Binary) by timescale",
        "full_data_results/timescale_binary.png"
    )

    chi2, p = test_timescale_dependency(ts_full)
    # v = cramers_v(chi2, table)
    # print(f"Cramér's V (full) = {v:.3f}")
    print(f'Timescale dependency (full): chi2 {chi2}, p {p}')

    chi2, p = test_timescale_dependency(ts_binary)
    # v = cramers_v(chi2, table)
    # print(f"Cramér's V (binary) = {v:.3f}")
    print(f'Timescale dependency (binary): chi2 {chi2}, p {p}')

    plot_timescale_stacked(ts_full, "_full")
    plot_timescale_stacked(ts_binary, "_binary")

    timescale_cluster_purity(full_cluster_detail)

    # ======================================================
    # EXPRESSIBILITY ANALYSIS
    # ======================================================

    # print("\n==============================")
    # print(" EXPRESSIBILITY ANALYSIS")
    # print("==============================\n")
    #
    # # ------------------------------------------------------
    # # 1. Cluster difficulty (based on feature variability)
    # # ------------------------------------------------------
    # data_resampled = np.array([
    #     cd.resample(cd.z_norm(ts), 500)
    #     for ts in full_data
    # ])
    #
    # plot_cluster_envelopes(data_resampled, labels, top_clusters, mode="range")
    # plot_cluster_single(data_resampled, labels, top_clusters, mode="std")
    # plot_cluster_envelopes(data_resampled, labels, top_clusters, "std")
    # plot_cluster_distance_heatmap(data_resampled, labels, top_clusters)
    # plot_cluster_embedding(data_resampled, labels, top_clusters)
    # print_most_similar_clusters(data_resampled, labels, top_clusters)
    # plot_cluster_zoom(data_resampled, labels, top_clusters, n_points=100)
    # plot_cluster_zoom(data_resampled, labels, top_clusters, n_points=50)
    #
    # plot_cluster_zoom_raw(data_resampled, labels, top_clusters, n_points=100)
    # plot_cluster_zoom_raw(data_resampled, labels, top_clusters, n_points=50)
    #
    # plot_cluster_derivatives(data_resampled, labels, top_clusters, n_points=100)

    # # ORIGINAL DATA (not z-normalized!)
    # plot_cluster_raw(full_data, labels, top_clusters, subtract_mean=False)
    #
    # # REMOVE OFFSET ONLY
    # plot_cluster_raw(full_data, labels, top_clusters, subtract_mean=True)
    #
    # shape_features = np.array([
    #     cd.extract_shape_signature(ts)
    #     for ts in data_resampled
    # ])
    #
    # shape_features = np.nan_to_num(shape_features)
    # cluster_difficulty = compute_cluster_difficulty(
    #     full_cluster_detail,
    #     shape_features,
    #     labels
    # )
    #
    # # ------------------------------------------------------
    # # 2. Performance vs difficulty (KEY PLOTS)
    # # ------------------------------------------------------
    #
    # # --- ALL clusters (full metric) ---
    # plot_expressibility(
    #     all_cluster_results_full,
    #     cluster_difficulty,
    #     title="Expressibility (Full metric, All clusters)",
    #     fname="full_data_results/expressibility_all_full.png"
    # )
    #
    # # --- ALL clusters (binary) ---
    # plot_expressibility(
    #     all_cluster_results_binary,
    #     cluster_difficulty,
    #     title="Expressibility (Binary, All clusters)",
    #     fname="full_data_results/expressibility_all_binary.png"
    # )
    #
    # # --- TOP 10 clusters (optional but useful) ---
    # plot_expressibility(
    #     results_top10_full,
    #     cluster_difficulty,
    #     title="Expressibility (Full metric, Top 10)",
    #     fname="full_data_results/expressibility_top10_full.png"
    # )
    #
    # plot_expressibility(
    #     results_top10_binary,
    #     cluster_difficulty,
    #     title="Expressibility (Binary, Top 10)",
    #     fname="full_data_results/expressibility_top10_binary.png"
    # )
    #
    # # ------------------------------------------------------
    # # 3. KL divergence of error distributions
    # # ------------------------------------------------------
    # print("\n--- KL Divergence (Error Distributions) ---")
    # compare_error_distributions(all_cluster_results_full)
    #
    # # ------------------------------------------------------
    # # 4. Prediction entropy (global flexibility)
    # # ------------------------------------------------------
    # print("\n--- Prediction Entropy ---")
    # prediction_entropy(timeseries_ssd_idx)
    #
    # # ------------------------------------------------------
    # # 5. Failure concentration (where models break)
    # # ------------------------------------------------------
    # print("\n--- Failure Concentration ---")
    # failure_concentration(all_cluster_results_full)


    # ------------------------------------------------------
    # 6. Original features share
    # ------------------------------------------------------
    feature_names = ['n_peaks', 'n_troughs', 'amp', 'mean_val', 'std_val', 'pos_slope', 'neg_slope', 'pos_ratio',
                     'avg_peak_dist', 'symmetry', 'curvature', 'complexity', 'autocorr', 'spread']

    analyze_feature_importance(shape_features, labels, feature_names)

    print("\n==============================")
    print(" PCA FEATURE INTERPRETATION")
    print("==============================\n")

    analyze_pca_feature_contributions(
        Xp,
        shape_features,
        labels,
        pca,
        feature_names
    )

    print("\n==============================")
    print(" GLOBAL PCA COMPONENTS")
    print("==============================")

    for i, comp in enumerate(pca.components_[:5]):
        print(f"\nComponent {i}:")
        idx = np.argsort(np.abs(comp))[::-1][:5]
        for j in idx:
            print(f"  {feature_names[j]}: {comp[j]:.3f}")

    print("\n==============================")
    print(" FEATURE-LEVEL ANALYSIS")
    print("==============================\n")

    os.makedirs("feature_analysis", exist_ok=True)

    df = build_feature_dataframe(shape_features, labels, feature_names)

    # --- restrict to top 10 clusters ---
    df_top = df[df['cluster'].isin([cid for cid, _ in top_clusters])]

    # 1. Boxplots
    plot_feature_boxplots(df_top, top_clusters, feature_names, "_top10")

    # 2. Heatmap (cluster means)
    plot_feature_heatmap(df_top, top_clusters, feature_names, "_top10")

    # 3. ANOVA (VERY IMPORTANT)
    anova_results = feature_anova(df_top, feature_names)

    # 4. Feature profiles (z-score)
    plot_cluster_feature_profiles(df_top, top_clusters, feature_names, "_top10")

    # 5. Correlation
    plot_feature_correlation(df_top, feature_names, "_top10")

    # 6. Pairplot (top discriminative features only)
    top_features = [f for f, _, _, _ in anova_results[:5]]
    plot_pairwise_scatter(df_top, top_features, "_top10")

    plot_feature_boxplots_weighted(
        df_top,
        top_clusters,
        feature_names,
        "_top10"
    )

    print("\n==============================")
    print(" CLUSTER Z-SCORE PROFILES")
    print("==============================\n")

    zscores, gmean, gstd = compute_cluster_zscores(
        df_top,
        top_clusters,
        feature_names
    )

    print_cluster_zscores(zscores, feature_names, top_k=8)

    save_cluster_zscores(
        zscores,
        feature_names,
        "feature_analysis/cluster_zscores_top10.csv"
    )

    plot_cluster_zscores(
        zscores,
        feature_names,
        "_top10"
    )

    print("\n==============================")
    print(" FEATURE PROFILE COMPARISON")
    print("==============================\n")

    plot_mean_feature_profiles(
        df,
        top_clusters,
        feature_names,
        "_raw"
    )

    plot_mean_feature_profiles_zscore(
        df,
        top_clusters,
        feature_names,
        "_zscore"
    )

    reports = build_cluster_reports(
        shape_features=shape_features,
        labels=labels,
        feature_names=feature_names,
        top_k=5
    )

    print_cluster_reports(reports)

    plot_feature_importance_global_vs_top10(
        shape_features,
        labels,
        feature_names,
        top_clusters
    )