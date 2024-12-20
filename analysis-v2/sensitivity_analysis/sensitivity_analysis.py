#!/usr/bin/env python3

"""Sensitivity analysis of the new SSD approach w.r.t. the selected parameters."""

import json
import numpy as np
import simpleJDB
import sys
from matplotlib import pyplot as plt
import plotly

sys.path.append('..')
import steady_state_detection as ssd

import scipy.signal as ssi
import sklearn.cluster
from scipy.stats import norm

# Selected parameters to test
default_params_tup = (100, 500, 4.5, 80, 0.85)

outliers_window_size_tup = (80, 90, 100, 110, 120)
prob_win_size_tup = (300, 400, 500, 600, 700)
t_crit_tup = (3.5, 4.0, 4.5, 5.0, 5.5)
step_win_size_tup = (60, 70, 80, 90, 100)
prob_threshold_tup = (0.75, 0.8, 0.85, 0.9, 0.95)

all_param_vals = (outliers_window_size_tup, prob_win_size_tup, t_crit_tup, step_win_size_tup, prob_threshold_tup)
for j, param in enumerate(all_param_vals):
    for k, val in enumerate(param):
        params = list(default_params_tup)
        params[j] = all_param_vals[j][k]

        params_str = '_'.join(str(el) for el in params)

        # Load the data together with manual labeling and the original auto-classification
        data_vittorio = simpleJDB.database('benchmark_database_steady_indices_vittorio')
        data_michele = simpleJDB.database('benchmark_database_steady_indices_michele')
        data_daniele = simpleJDB.database('benchmark_database_steady_indices_daniele')
        data_luca = simpleJDB.database('benchmark_database_steady_indices_luca')
        data_martin = simpleJDB.database('benchmark_database_steady_indices_martin')

        # Differences of original and new methods w.r.t. the steady-state reference index
        orig_diffs_clustered = []
        orig_diffs_scattered = []
        new_diffs_clustered = []
        new_diffs_scattered = []

        # Structure containing all the data including manual labeling and both automatic detections
        data_sum = {}

        # Numbers of series with clustered and scattered manual labels
        no_clusters = 0
        no_scattered = 0

        for i, e in enumerate(data_vittorio.data['main']):
            # Remove the bad filename suffix and load the filenames with fork indices
            fname, fork_idx = e['keyname'].rsplit('_', 1)[0].rsplit('_', 1)
            fork_idx = int(fork_idx)
            series_key = f'{fname}_{fork_idx}'

            # The original automatic classification
            orig_clas_idx = json.load(open(f'orig_classification/{fname}'))['steady_state_starts'][fork_idx]

            # Load the corresponding timeseries
            timeseries = json.load(open(f'../sensitivity_analysis/data_st/{fname}_{fork_idx}'))
            timeseries1 = timeseries.copy()
            timeseries, _ = ssd.substitute_outliers_percentile(timeseries, percentile_threshold_upper=85,
                                                               percentile_threshold_lower=2,
                                                               window_size=params[0])
            timeseries2 = timeseries.copy()
            # timeseries = ssi.medfilt(timeseries, kernel_size=3)

            # Apply the new approach
            P, warmup_end = ssd.detect_steady_state(timeseries, prob_win_size=params[1], t_crit=params[2],
                                                    step_win_size=params[3], medfilt_kernel_size=1)
            res = ssd.get_compact_result(P, warmup_end)
            new_clas_idx = ssd.get_ssd_idx(res, prob_threshold=params[4], min_steady_length=0)

            # print(e['value'], data_michele.getkey(e['keyname']), data_daniele.getkey(e['keyname']),
            #       data_luca.getkey(e['keyname']), data_martin.getkey(e['keyname']), orig_clas_idx, new_clas_idx)

            data_sum[series_key] = {'idxs': [e['value'],
                                             data_michele.getkey(e['keyname']),
                                             data_daniele.getkey(e['keyname']),
                                             data_luca.getkey(e['keyname']),
                                             data_martin.getkey(e['keyname']),
                                             orig_clas_idx,
                                             new_clas_idx],
                                    'series': timeseries1}

            # Check the correctness of the loaded timeseries
            ref_series = json.load(open(f'../../data/timeseries/all/{fname}'))[fork_idx]
            ref_idx = json.load(open(f'../../data/classification/{fname}'))['steady_state_starts'][fork_idx]

            assert ref_series == data_sum[series_key]['series']
            assert ref_idx == data_sum[series_key]['idxs'][-2]

            # Recognize the SSD reference point
            # If there's cluster (min 3 points), then take the middle point or the 3rd point (if there are 4 in the cluster)
            # If there's no cluster, take the last point, as the most conservative one
            dbscan_inst = sklearn.cluster.DBSCAN(eps=50, min_samples=3)
            man_labels = np.array(data_sum[series_key]['idxs'][:5])
            cluster_idxs = dbscan_inst.fit(np.array(man_labels).reshape(-1, 1)).labels_

            scattered = False
            if not (cluster_idxs > -1).any():
                scattered = True
                # continue
                data_sum[series_key]['steady_idx'] = max(man_labels)
                data_sum[series_key]['clustered'] = False
            elif sum(cluster_idxs > -1) == 4:
                data_sum[series_key]['steady_idx'] = man_labels[sorted(np.where(cluster_idxs > -1)[0])[-2]]
                data_sum[series_key]['clustered'] = True

            else:
                # continue
                data_sum[series_key]['steady_idx'] = int(np.median(man_labels[np.where(cluster_idxs > -1)[0]]))
                data_sum[series_key]['clustered'] = True

            # Compute differences of results
            if -1 not in data_sum[series_key]['idxs'][-2:]:
                if scattered:
                    orig_diffs_scattered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-2])
                    new_diffs_scattered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-1])
                else:
                    orig_diffs_clustered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-2])
                    new_diffs_clustered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-1])

        # No. unsteady series detected
        no_unsteady_orig = 0
        no_unsteady_new = 0

        for k, v in data_sum.items():
            if v['idxs'][-2] == -1:
                no_unsteady_orig += 1
            if v['idxs'][-1] == -1:
                no_unsteady_new += 1

        # Plot bar plots comparing numbers of incorrectly detected unsteady series
        # print(no_unsteady_orig, no_unsteady_new)
        plt.figure()
        plt.title('No. unsteady series detected')
        plt.bar([1, 2], [no_unsteady_orig, no_unsteady_new], tick_label=['Orig', 'New'])
        plt.savefig(f'barplots/no_unsteady_{params_str}.png')
        plt.close()

        # Plot histogram of SSD differences for clustered points
        mean_orig, std_orig = norm.fit(orig_diffs_clustered)
        mean_new, std_new = norm.fit(new_diffs_clustered)

        # plt.figure(figsize=(10, 6))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title('SSD Reference idx - detected idx (clustered labels)')
        # plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
        # plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='red')
        ax1.hist([orig_diffs_clustered, new_diffs_clustered], bins=30, alpha=0.5, label=['original', 'new'],
                 color=['blue', 'red'])

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
        ax1.axvline(mean_orig - 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI lower')
        ax1.axvline(mean_orig + 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI upper')

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
        ax1.axvline(mean_new - 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI lower')
        ax1.axvline(mean_new + 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI upper')
        ax2 = ax1.twinx()
        # ---- Fit and plot Gaussian for new data ----
        p_new = norm.pdf(x, mean_new, std_new)
        ax2.plot(x, p_new, 'r-', linewidth=2, label=f'New Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
        p_orig = norm.pdf(x, mean_orig, std_orig)
        ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'Original Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')

        # Place the legend outside the plot
        plt.legend(loc='upper left')  # Adjust position of the legend

        plt.savefig(f'histograms/differences_cluster_30_bins_{params_str}.png')
        plt.close()
        print(f'No. clustered diffs:{len(orig_diffs_clustered)}')
        print(
            f'Clustered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}')
        with open(f'summaries/sum_{params_str}.txt', 'w') as f:
            f.write(f'No. clustered diffs:{len(orig_diffs_clustered)}\n')
            f.write(f'Clustered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}\n')

        # Plot histogram of SSD differences for scattered points
        mean_orig, std_orig = norm.fit(orig_diffs_scattered)
        mean_new, std_new = norm.fit(new_diffs_scattered)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # plt.figure(figsize=(10, 6))
        plt.title('SSD Reference idx - detected idx (scattered labels)')
        # plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
        # plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='red')
        ax1.hist([orig_diffs_scattered, new_diffs_scattered], bins=30, alpha=0.5, label=['original', 'new'],
                 color=['blue', 'red'])

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
        ax1.axvline(mean_orig - 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI lower')
        ax1.axvline(mean_orig + 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI upper')

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
        ax1.axvline(mean_new - 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI lower')
        ax1.axvline(mean_new + 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI upper')
        ax2 = ax1.twinx()
        # ---- Fit and plot Gaussian for new data ----
        p_new = norm.pdf(x, mean_new, std_new)
        ax2.plot(x, p_new, 'r-', linewidth=2, label=f'New Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
        p_orig = norm.pdf(x, mean_orig, std_orig)
        ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'Original Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')

        # Place the legend outside the plot
        plt.legend(loc='upper left')  # Adjust position of the legend

        plt.savefig(f'histograms/differences_scattered_30_bins_{params_str}.png')
        plt.close()
        print(f'No. scattered diffs:{len(orig_diffs_scattered)}')
        print(
            f'Scattered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}')
        with open(f'summaries/sum_{params_str}.txt', 'a') as f:
            f.write(f'No. scattered diffs:{len(orig_diffs_scattered)}\n')
            f.write(f'Scattered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}\n')

        # Plot histogram of SSD differences for all points
        mean_orig, std_orig = norm.fit(orig_diffs_scattered + orig_diffs_clustered)
        mean_new, std_new = norm.fit(new_diffs_scattered + new_diffs_clustered)

        # plt.figure(figsize=(10, 6))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title('SSD Reference idx - detected idx (all labels)')
        # plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
        # plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='red')
        ax1.hist([orig_diffs_scattered + orig_diffs_clustered, new_diffs_scattered + new_diffs_clustered],
                 bins=30, alpha=0.5, label=['original', 'new'], color=['blue', 'red'])

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
        ax1.axvline(mean_orig - 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI lower')
        ax1.axvline(mean_orig + 2 * std_orig, color='blue', linestyle='--', linewidth=1.5,
                    label='Original 95% CI upper')

        # Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
        ax1.axvline(mean_new - 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI lower')
        ax1.axvline(mean_new + 2 * std_new, color='red', linestyle='--', linewidth=1.5, label='New 95% CI upper')
        ax2 = ax1.twinx()
        # ---- Fit and plot Gaussian for new data ----
        p_new = norm.pdf(x, mean_new, std_new)
        ax2.plot(x, p_new, 'r-', linewidth=2, label=f'New Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
        p_orig = norm.pdf(x, mean_orig, std_orig)
        ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'Original Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')

        # Place the legend outside the plot
        plt.legend(loc='upper left')  # Adjust position of the legend

        plt.savefig(f'histograms/differences_all_30_bins_{params_str}.png')
        plt.close()
        print(f'No. diffs:{len(orig_diffs_scattered + orig_diffs_clustered)}')
        print(f'All diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}')
        print(f'No. all series: {len(data_sum.keys())}')
        with open(f'summaries/sum_{params_str}.txt', 'a') as f:
            f.write(f'No. diffs:{len(orig_diffs_scattered + orig_diffs_clustered)}\n')
            f.write(f'All diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}\n')
            f.write(f'No. all series: {len(data_sum.keys())}')

        # Create boxplots of differences
        plt.figure()
        plt.title('Detection differences (clustered labels)')
        plt.boxplot([orig_diffs_clustered, new_diffs_clustered], tick_labels=['orig', 'new'])
        plt.savefig(f'boxplots/differences_clustered_{params_str}.png')
        plt.close()

        plt.figure()
        plt.title('Detection differences (scattered labels)')
        plt.boxplot([orig_diffs_scattered, new_diffs_scattered], tick_labels=['orig', 'new'])
        plt.savefig(f'boxplots/differences_scattered_{params_str}.png')
        plt.close()

        plt.figure()
        plt.title('Detection differences (all labels)')
        plt.boxplot([orig_diffs_clustered + orig_diffs_scattered, new_diffs_clustered + new_diffs_scattered],
                    tick_labels=['orig', 'new'])
        plt.savefig(f'boxplots/differences_all_{params_str}.png')
        plt.close()
