#!/usr/bin/env python3

"""This file contains a clustering analysis performed on the full dataset of 5860 timeseries."""
from sklearn.decomposition import PCA

import cluster_detection as cd
import matplotlib.pyplot as plt
import numpy as np
import json


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
        Xp = PCA(n_components=min(n_pca, shape_features.shape[1])).fit_transform(shape_features)
        for metric in metrics:
            cfg = cd.evaluate_hdbscan(Xp, metric=metric)
            if cfg is None:
                continue
            clusters, outliers, mcs, eps, labels, score = cfg

            score_val = float(np.mean(score)) if np.ndim(score) > 0 else float(score)
            if best_overall is None:
                best_val = -np.inf
            else:
                try:
                    best_val = float(np.mean(best_overall[-1])) if np.ndim(best_overall[-1]) > 0 else float(
                        best_overall[-1])
                except Exception:
                    best_val = float(np.mean(np.ravel(best_overall[-1])))

            if score_val > best_val:
                best_overall = (n_pca, metric, clusters, outliers, mcs, eps, labels, score_val, Xp)

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

    return top_clusters, cluster_ids, sizes, unsteady


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


def compare_methods(cpssd_idx, kbkssd_idx, jagt_idx):
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


def plot_cp_vs_kbk(cluster_stats):
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
    plt.savefig('full_data_results/cpssd_vs_kbkssd.png')
    plt.show()


def plot_cp_vs_kbk_ratio(cluster_stats):
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
    plt.savefig(f'full_data_results/cpssd_vs_kbkssd_normalized.png')
    plt.show()


def plot_difference(cluster_stats):
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
    plt.savefig('full_data_results/method_success_per_cluster.png')
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
    n_pca, metric, clusters, outliers, mcs, eps, labels, score, Xp = analyse_clusters(full_data)

    # Load the JAGT - timeseries steady-state idx (via KB-KSSD)
    timeseries_ssd_idx = json.load(open('series_kbkssd.json'))

    # Detect time scales of full dataset
    time_scales = analyse_time_scale(full_data)

    # Aggregate data about all the timeseries w.r.t. KB-KSSD taking 10 largest clusters into account
    top_clusters, cluster_idxs, cluster_sizes, cluster_unsteady = aggregate_cluster_data(timeseries_ssd_idx,
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

                results_per_cluster[cluster_idx][compare_methods(cpssd_pred, kbkssd_pred, jagt_sidx)] += 1

        results_per_cluster[cluster_idx]['n_jagt_forks'] = n_jagt_forks

    plot_cp_vs_kbk(results_per_cluster)
    plot_cp_vs_kbk_ratio(results_per_cluster)
    plot_difference(results_per_cluster)
