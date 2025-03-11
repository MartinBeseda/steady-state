#!/usr/bin/env python3

"""Compare both approaches on manually selected steady states."""
import os
import json
import shutil

import numpy as np
import scipy
import simpleJDB
import sys

from matplotlib import pyplot as plt
import plotly

sys.path.append('..')
import steady_state_detection as ssd

import scipy.signal as ssi
import sklearn.cluster
from scipy.stats import norm
from scipy.optimize import differential_evolution

# # Are the data labeled OK? They are - the following code checks it.
# for f in os.listdir('../man_steady_indices/data_st'):
#     loc_series = json.load(open(f'../man_steady_indices/data_st/{f}'))
#
#     fname, fork_idx = f.rsplit('_', 1)
#     fork_idx = int(fork_idx)
#     prev_series = json.load(open(f'../../data/timeseries/all/{fname}'))[fork_idx]
#
#     if not loc_series == prev_series:
#         print(loc_series)
#         print(prev_series)

# # Copy the original classification
# for f in os.listdir('data_st'):
#     fname, fork_idx = f.rsplit('_', 1)
#     fork_idx = int(fork_idx)
#     shutil.copyfile(f'../../data/classification/{fname}', f'orig_classification/{fname}')

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

# Differences between absolute values of models' predictions
pred_abs_diffs = []

# Parameters
outliers_window_size = 100
prob_win_size = 500
t_crit = 4
step_win_size = 70
prob_threshold = 0.95
median_kernel_size = 1

print(f'Number of steady timeseries: {len(list(data_vittorio.data['main']))}')

for i, e in enumerate(data_vittorio.data['main']):
    # Remove the bad filename suffix and load the filenames with fork indices
    fname, fork_idx = e['keyname'].rsplit('_', 1)[0].rsplit('_', 1)
    fork_idx = int(fork_idx)
    series_key = f'{fname}_{fork_idx}'

    # The original automatic classification
    orig_clas_idx = json.load(open(f'orig_classification/{fname}'))['steady_state_starts'][fork_idx]

    # Load the corresponding timeseries
    timeseries = json.load(open(f'../compare_configurations/data_st/{fname}_{fork_idx}'))
    timeseries1 = timeseries.copy()
    timeseries, _ = ssd.substitute_outliers_percentile(timeseries,
                                                       percentile_threshold_upper=98,
                                                       percentile_threshold_lower=2,
                                                       window_size=outliers_window_size)
    timeseries2 = timeseries.copy()
    #timeseries = ssi.medfilt(timeseries, kernel_size=3)

    # Apply the new approach
    P, warmup_end = ssd.detect_steady_state(timeseries, prob_win_size=prob_win_size, t_crit=t_crit,
                                            step_win_size=step_win_size, medfilt_kernel_size=median_kernel_size)
    res = ssd.get_compact_result(P, warmup_end)
    new_clas_idx = ssd.get_ssd_idx(res, prob_threshold=prob_threshold, min_steady_length=1)

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
        data_sum[series_key]['steady_idx'] = sorted(man_labels)[-3]#max(man_labels)#
        data_sum[series_key]['clustered'] = False
    elif sum(cluster_idxs > -1) == 4:
        data_sum[series_key]['steady_idx'] = man_labels[sorted(np.where(cluster_idxs > -1)[0])[-2]]
        data_sum[series_key]['clustered'] = True

    else:
        #continue
        data_sum[series_key]['steady_idx'] = int(np.median(man_labels[np.where(cluster_idxs > -1)[0]]))
        data_sum[series_key]['clustered'] = True

    # print(f'Steady idx:{data_sum[e['keyname']]['steady_idx']}')
    # # Plot all the data with detected indices
    # plt.figure()
    # plt.title(e['keyname'])
    # plt.plot(data_sum[e['keyname']]['series'])
    # plt.vlines([data_sum[e['keyname']]['idxs'][-2:]],
    #            min(data_sum[e['keyname']]['series']),
    #            max(data_sum[e['keyname']]['series']),
    #            colors=['green','#D85C5C'],
    #            label=['orig', 'new'])
    # plt.vlines([data_sum[e['keyname']]['steady_idx']],
    #            min(data_sum[e['keyname']]['series']),
    #            max(data_sum[e['keyname']]['series']),
    #            colors=['black'],
    #            label='steady idx')
    # plt.savefig(f'plots/plot_{i}.png')
    # plt.close()
    # Compute differences of results
    if -1 not in data_sum[series_key]['idxs'][-2:]:
        if scattered:
            orig_diffs_scattered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-2])
            new_diffs_scattered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-1])
        else:
            orig_diffs_clustered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-2])
            new_diffs_clustered.append(data_sum[series_key]['steady_idx'] - data_sum[series_key]['idxs'][-1])

    pred_abs_diffs.append(abs(data_sum[series_key]['idxs'][-2]) - abs(data_sum[series_key]['idxs'][-1]))

# Compute the Manhattan metric of errors for both clustered and scattered GT indices
manhattan_new_clust = sum([abs(e) for e in new_diffs_clustered])
manhattan_orig_clust = sum([abs(e) for e in orig_diffs_clustered])
manhattan_new_scat = sum([abs(e) for e in new_diffs_scattered])
manhattan_orig_scat = sum([abs(e) for e in orig_diffs_scattered])

print(f'Manhattan metric:\nNew clustered: {manhattan_new_clust}, '
      f'Orig clustered: {manhattan_orig_clust}, ')
print(f'New scattered: {manhattan_new_scat} '
      f'Orig scattered: {manhattan_orig_scat}')
# print(scipy.stats.wilcoxon(new_diffs_clustered,orig_diffs_clustered))
# print(scipy.stats.wilcoxon(orig_diffs_scattered,new_diffs_scattered))
# print(scipy.stats.ttest_rel(new_diffs_clustered,orig_diffs_clustered))
# print(scipy.stats.ttest_rel(orig_diffs_scattered,new_diffs_scattered))
# print(scipy.stats.shapiro(new_diffs_clustered))
# print(scipy.stats.shapiro(orig_diffs_clustered))
# print(scipy.stats.shapiro(new_diffs_scattered))
# print(scipy.stats.shapiro(orig_diffs_scattered))
# exit(-1)

# No. unsteady series detected
no_unsteady_orig_scattered = 0
no_unsteady_orig_clustered = 0
no_unsteady_new_scattered = 0
no_unsteady_new_clustered = 0

for k, v in data_sum.items():
    if v['idxs'][-2] == -1:
        if v['clustered']:
            no_unsteady_orig_clustered += 1
        else:
            no_unsteady_orig_scattered += 1
    if v['idxs'][-1] == -1:
        if v['clustered']:
            no_unsteady_new_clustered += 1
        else:
            no_unsteady_new_scattered += 1

# Plot bar plots comparing numbers of incorrectly detected unsteady series
# print(no_unsteady_orig, no_unsteady_new)
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.title('No. unsteady series detected')
plt.tick_params(axis='both', labelsize=14)
bars_orig = plt.bar([1, 3],
                    [no_unsteady_orig_scattered, no_unsteady_orig_clustered],
                    #    tick_label=['Scattered', 'Clustered'],
                    color='#4c72b0',
                    edgecolor='#2A4D69')
bars_new = plt.bar([2, 4],
                   [no_unsteady_new_scattered, no_unsteady_new_clustered],
                   #  tick_label=['Scattered', 'Clustered'],
                   color='#D85C5C',
                   edgecolor='#9E3D3D')
plt.xticks(ticks=(1.5,3.5), labels=('Scattered', 'Clustered'))
plt.legend([bars_orig, bars_new], ['CP-SSD', 'KB-KSSD'], fontsize=14)
plt.savefig('barplots/no_unsteady.png')
plt.savefig('barplots/no_unsteady.eps', format='eps')
plt.close()

print(f'Number of false negatives: {no_unsteady_orig_scattered, no_unsteady_orig_clustered, no_unsteady_new_scattered,
no_unsteady_new_clustered}')


# Plot the number of steadiness detections on larger dataset
larger_data = json.load(open('benchmark_database_binary.json'))
print(f'Total number (including unsteady) of timeseries: {len(larger_data['main'])}')

no_agreements_orig = 0
false_positives_orig = 0
false_negatives_orig = 0
no_agreements_new = 0
false_positives_new = 0
false_negatives_new = 0
n_total_agreements = 0
n_method_agreements = 0
n_std = 0
n_std_new = 0
n_std_orig = 0
for e in larger_data['main']:
    fork_name, fork_idx = e['keyname'].rsplit('_', 1)
    fork_idx = int(fork_idx)

    is_steady = True if e['value'] > -1 else False
    if is_steady:
        n_std += 1
    else:
        a=1
    # Obtain the original classification
    orig_classification = True \
        if json.load(open(f'orig_classification/{fork_name}'))['steady_state_starts'][fork_idx] > -1 else False

    dbg1 = False
    dbg2 = False
    if orig_classification:
        n_std_orig += 1
        dbg1=True
    if is_steady == orig_classification:
        no_agreements_orig += 1
    elif not is_steady and orig_classification:
        false_positives_orig += 1
        dbg2=True
    else:
        false_negatives_orig += 1

    if not dbg1 and dbg2:
        a=1

    # Detect steadiness via the new approach
    # Load the corresponding timeseries
    timeseries = json.load(open(f'data_st/{fork_name}'))[fork_idx]
    timeseries1 = timeseries.copy()
    timeseries, _ = ssd.substitute_outliers_percentile(timeseries, percentile_threshold_upper=98,
                                                       percentile_threshold_lower=2,
                                                       window_size=outliers_window_size)
    timeseries2 = timeseries.copy()
    # timeseries = ssi.medfilt(timeseries, kernel_size=3)

    # Apply the new approach
    #'100_500_4.0_70_0.95_1', 128076)
    #(outliers_window_size_tup, prob_win_size_tup, t_crit_tup, step_win_size_tup, prob_threshold_tup,
    #            median_kernel_size)
    P, warmup_end = ssd.detect_steady_state(timeseries, prob_win_size=prob_win_size, t_crit=t_crit,
                                            step_win_size=step_win_size, medfilt_kernel_size=median_kernel_size)
    res = ssd.get_compact_result(P, warmup_end)

    new_clas_idx = True if ssd.get_ssd_idx(res, prob_threshold=prob_threshold, min_steady_length=0) > -1 else False
    if new_clas_idx:
        n_std_new += 1
    if is_steady == new_clas_idx:
        no_agreements_new += 1
    elif not is_steady and new_clas_idx:
        false_positives_new += 1
    else:
        false_negatives_new += 1

    # Both methods and the ground truth agree with each other
    if is_steady == new_clas_idx == orig_classification:
        n_total_agreements += 1

    # Both methods agree with each other, but not with the ground truth
    if (is_steady != new_clas_idx) and (new_clas_idx == orig_classification):
        n_method_agreements += 1

print(n_std, n_std_orig, n_std_new)
print(no_agreements_orig, no_agreements_new)
print(false_positives_orig, false_negatives_orig, false_positives_new, false_negatives_new)
print(n_total_agreements, n_method_agreements)


fig, ax = plt.subplots()
bar_width = 0.35
bar_positions = np.arange(2)
# Data for original and new
no_agreements = [no_agreements_orig, no_agreements_new]
false_positives = [false_positives_orig, false_positives_new]
false_negatives = [false_negatives_orig, false_negatives_new]

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# First subplot for original data
ax1.bar(bar_positions[0], no_agreements[0], bar_width, label='Agreements', color='lightgrey')
ax1.bar(bar_positions[0] -bar_width/4, false_positives[0], bar_width/2, label='False Positives', color='#D85C5C',
        edgecolor='#9E3D3D')
ax1.bar(bar_positions[0] + bar_width/4, false_negatives[0], bar_width/2, label='False Negatives', color='#4c72b0',
        edgecolor='#2A4D69')

# Add horizontal lines to the first subplot (Original)
ax1.axhline(y=n_total_agreements, color='green', linewidth=3,linestyle='--', label='Total Agmts.')
ax1.axhline(y=n_method_agreements, color='black', linewidth=3, linestyle='-.', label='Method Agmts.')

# Second subplot for new data
ax2.bar(bar_positions[1], no_agreements[1], bar_width, label='Agreements with JAGT', color='lightgrey')
ax2.bar(bar_positions[1] - bar_width/4, false_positives[1], bar_width/2, label='False Positives', color='#D85C5C',
        edgecolor='#9E3D3D')
ax2.bar(bar_positions[1] + bar_width/4, false_negatives[1], bar_width/2, label='False Negatives', color='#4c72b0',
        edgecolor='#2A4D69')

# Add horizontal lines to the second subplot (New)
ax2.axhline(y=n_total_agreements, color='green', linewidth=3,linestyle='--', label='Total Agmts.')
ax2.axhline(y=n_method_agreements, color='black', linewidth=3,linestyle='-.', label='Method Agmts.')

# Set labels and title for original subplot
# ax1.set_ylabel('Counts')
ax1.set_xlabel('CP-SSD', fontsize=20)

ax1.set_xticks([])

# Set labels and title for new subplot
ax2.set_xlabel('KB-KSSD', fontsize=20)

# ax2.set_ylabel('Counts')
ax2.set_xticks([])

ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

ax1.set(ylim=(0, 420))
ax2.set(ylim=(0, 420))

# ax2.set_yticks([])

# Add legends
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2,fontsize=20)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=20)
# ax2.legend()
# Get the legend handles and labels from ax1 (first subplot)
handles, labels = ax1.get_legend_handles_labels()

# Add a common legend using the handles and labels from ax1
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3,fontsize=20)

# Adjust layout to ensure the legend fits above the plots
plt.tight_layout(rect=[0, 0, 1, 0.85])

plt.savefig('barplots/agreements.png')
plt.savefig('barplots/agreements.eps', format='eps')
plt.close()

print(f'Number of agreements with GT: total number of samples: {no_agreements_orig + false_negatives_orig
                                                                + false_positives_orig} '
      f'& {no_agreements_new + false_negatives_new + false_positives_new}')

# Plot the number of agreements with the larger dataset containing even unsteady timeseries
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.title('Number of agreements w.r.t. steadiness (larger dataset)')
plt.bar([1, 2], [no_agreements_orig, no_agreements_new], tick_label=['KB-KSSD', 'CP-SSD'], color='#4c72b0',
        edgecolor='#2A4D69')
plt.savefig('barplots/larger_set_agreements.png')
plt.savefig('barplots/larger_set_agreements.eps', format='eps')
plt.close()

# Plot histogram of SSD differences for clustered points
mean_orig, std_orig = norm.fit(orig_diffs_clustered)
mean_new, std_new = norm.fit(new_diffs_clustered)

# plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots(figsize=(10,6))
plt.title('SSD Reference idx - detected idx (clustered labels)')
#plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
#plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='#D85C5C')
ax1.hist([orig_diffs_clustered, new_diffs_clustered], bins=30, alpha=0.5, label=['original', 'new'],
         color=['#4c72b0', '#D85C5C'])

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
ax1.axvline(mean_orig - 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI lower')
ax1.axvline(mean_orig + 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI upper')

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
ax1.axvline(mean_new - 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI lower')
ax1.axvline(mean_new + 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI upper')
ax2 = ax1.twinx()
# ---- Fit and plot Gaussian for new data ----
p_new = norm.pdf(x, mean_new, std_new)
ax2.plot(x, p_new, 'r-', linewidth=2, label=f'KB-KSSD Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
p_orig = norm.pdf(x, mean_orig, std_orig)
ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'CP-SSD Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')


# Place the legend outside the plot
plt.legend(loc='upper left')  # Adjust position of the legend

plt.savefig('histograms/differences_cluster_30_bins.png')
plt.savefig('histograms/differences_cluster_30_bins.eps', format='eps')
plt.close()
print(f'No. clustered diffs:{len(orig_diffs_clustered)}')
print(f'Clustered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}')


# Plot histogram of SSD differences for scattered points
mean_orig, std_orig = norm.fit(orig_diffs_scattered)
mean_new, std_new = norm.fit(new_diffs_scattered)
fig, ax1 = plt.subplots(figsize=(10,6))
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.figure(figsize=(10, 6))
plt.title('SSD Reference idx - detected idx (scattered labels)')
#plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
#plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='#D85C5C')
ax1.hist([orig_diffs_scattered, new_diffs_scattered], bins=30, alpha=0.5, label=['original', 'new'], color=['#4c72b0', '#D85C5C'])

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
ax1.axvline(mean_orig - 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI lower')
ax1.axvline(mean_orig + 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI upper')

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
ax1.axvline(mean_new - 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI lower')
ax1.axvline(mean_new + 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI upper')
ax2 = ax1.twinx()
# ---- Fit and plot Gaussian for new data ----
p_new = norm.pdf(x, mean_new, std_new)
ax2.plot(x, p_new, 'r-', linewidth=2, label=f'KB-KSSD Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
p_orig = norm.pdf(x, mean_orig, std_orig)
ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'CP-SSD Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')

# Place the legend outside the plot
plt.legend(loc='upper left')  # Adjust position of the legend
# plt.show()
# exit(-1)
plt.savefig('histograms/differences_scattered_30_bins.png')
plt.savefig('histograms/differences_scattered_30_bins.eps', format='eps')
plt.close()
print(f'No. scattered diffs:{len(orig_diffs_scattered)}')
print(f'Scattered diffs: New std: {std_new}, New mean: {mean_new}, Orig std: {std_orig}, Orig mean: {mean_orig}')

# Plot histogram of SSD errors for all points
mean_orig, std_orig = norm.fit(np.array(orig_diffs_scattered + orig_diffs_clustered))
mean_new, std_new = norm.fit(np.array(new_diffs_scattered + new_diffs_clustered))

# plt.figure(figsize=(10, 6))
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

# Remove top and right spines for ax1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Remove top and right spines for ax2
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# plt.title('SSD Reference idx - detected idx (all labels)')
#plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
#plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='#D85C5C')
ax1.hist([np.array(orig_diffs_scattered + orig_diffs_clustered), np.array(new_diffs_scattered + new_diffs_clustered)],
         bins=30, alpha=1, label=['CP-SSD', 'KB-KSSD'], color=['#4c72b0', '#D85C5C'],  edgecolor='#4D4D4D')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# ---- Fit and plot Gaussian for new data ----
p_new = norm.pdf(x, mean_new, std_new)
ax2.plot(x, p_new, '-', linewidth=2, label=f'KB-KSSD Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}', color='#4c72b0')
p_orig = norm.pdf(x, mean_orig, std_orig)
ax2.plot(x, p_orig, '-', linewidth=2, label=f'CP-SSD Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}', color='#D85C5C')

#ax1.set_xlim(xmin=0)

ax1.tick_params(axis='both', which='both', labelsize=20)
ax2.tick_params(axis='both', which='both', labelsize=20)
ax2.set_yticks([])

# Place the legend outside the plot
# plt.legend(loc='upper left', fontsize=20)  # Adjust position of the legend


# Add a common legend using the handles and labels from ax1
fig.legend( loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=20)

# Adjust layout to ensure the legend fits above the plots
fig.tight_layout(rect=[0, 0, 1, 0.8])

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
ax1.axvline(mean_orig - 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI lower')
ax1.axvline(mean_orig + 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI upper')

# Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
ax1.axvline(mean_new - 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI lower')
ax1.axvline(mean_new + 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI upper')

plt.savefig('histograms/differences_all_30_bins.png')
plt.savefig('histograms/differences_all_30_bins.eps', format='eps')
plt.close()

print(f'No. abs errs:{len(orig_diffs_scattered + orig_diffs_clustered)}')
print(f'All errs: New std: {std_new}, New mean: {mean_new} New median: {np.median(new_diffs_scattered +
                                                                                  new_diffs_clustered)}, '
      f'Orig std: {std_orig}, Orig mean: {mean_orig}, Orig median: {np.median(orig_diffs_scattered +
                                                                              orig_diffs_clustered)}')

# Plot histogram of SSD absolute errors for all points
mean_orig, std_orig = norm.fit(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)))
mean_new, std_new = norm.fit(abs(np.array(new_diffs_scattered + new_diffs_clustered)))

# plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Remove top and right spines for ax1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Remove top and right spines for ax2
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# plt.title('SSD Reference idx - detected idx (all labels)')
#plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
#plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='#D85C5C')
ax1.hist([abs(np.array(orig_diffs_scattered + orig_diffs_clustered)), abs(np.array(new_diffs_scattered + new_diffs_clustered))],
         bins=30, alpha=1, label=['CP-SSD', 'KB-KSSD'], color=['#4c72b0', '#D85C5C'], edgecolor='#4D4D4D')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# ---- Fit and plot Gaussian for new data ----
p_new = norm.pdf(x, mean_new, std_new)
ax2.plot(x, p_new, '-', linewidth=2, label=f'KB-KSSD Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}', color='#4c72b0')
p_orig = norm.pdf(x, mean_orig, std_orig)
ax2.plot(x, p_orig, '-', linewidth=2, label=f'CP-SSD Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}', color='#D85C5C')

ax1.set_xlim(xmin=0)

ax1.tick_params(axis='both', which='both', labelsize=20)
ax2.tick_params(axis='both', which='both', labelsize=20)
ax2.set_yticks([])

# Place the legend outside the plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.8])
plt.savefig('histograms/abs_errs_all_30_bins.png')
plt.savefig('histograms/abs_errs_all_30_bins.eps', format='eps')
plt.close()
print(f'No. abs errs:{len(orig_diffs_scattered + orig_diffs_clustered)}')
print(f'All abs errs: New std: {std_new}, New mean: {mean_new}, '
      f'New median: {np.median(abs(np.array(new_diffs_scattered + new_diffs_scattered)))}, Orig std: {std_orig}, '
      f'Orig mean: {mean_orig}, Orig median: {np.median(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)))}')

# Plot histogram of SSD differences for all points
mean_orig, std_orig = norm.fit(orig_diffs_scattered + orig_diffs_clustered)
mean_new, std_new = norm.fit(new_diffs_scattered + new_diffs_clustered)

# # plt.figure(figsize=(10, 6))
# fig, ax1 = plt.subplots(figsize=(10,6))
# # plt.title('SSD Reference idx - detected idx (all labels)')
# #plt.hist([orig_diffs, new_diffs], label=['orig', 'new'])
# #plt.hist(new_diffs, bins=20, density=True, alpha=0.3, label='new', color='#D85C5C')
# ax1.hist([orig_diffs_scattered + orig_diffs_clustered, new_diffs_scattered + new_diffs_clustered],
#          bins=30, alpha=0.5, label=['CP-SSD', 'KB-KSSD'], color=['#4c72b0', '#D85C5C'])
#
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
#
# # Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
# ax1.axvline(mean_orig - 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI lower')
# ax1.axvline(mean_orig + 2 * std_orig, color='#4c72b0', linestyle='--', linewidth=1.5, label='Original 95% CI upper')
#
# # Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
# ax1.axvline(mean_new - 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI lower')
# ax1.axvline(mean_new + 2 * std_new, color='#D85C5C', linestyle='--', linewidth=1.5, label='New 95% CI upper')
# ax2 = ax1.twinx()
# # ---- Fit and plot Gaussian for new data ----
# p_new = norm.pdf(x, mean_new, std_new)
# ax2.plot(x, p_new, 'r-', linewidth=2, label=f'New Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')
# p_orig = norm.pdf(x, mean_orig, std_orig)
# ax2.plot(x, p_orig, 'b-', linewidth=2, label=f'Original Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')
#
#
# # Place the legend outside the plot
# plt.legend(loc='upper left')  # Adjust position of the legend
#
# plt.savefig('histograms/differences_all_30_bins.png')
# plt.close()

# Create boxplots of differences
plt.figure()
plt.title('Detection differences (clustered labels)')
plt.boxplot([orig_diffs_clustered, new_diffs_clustered], tick_labels=['CP-SSD', 'KB-KSSD'])
plt.savefig('boxplots/differences_clustered.png')
plt.savefig('boxplots/differences_clustered.eps', format='eps')
plt.close()

plt.figure()
plt.title('Detection differences (scattered labels)')
plt.boxplot([orig_diffs_scattered, new_diffs_scattered], tick_labels=['CP-SSD', 'KB-KSSD'])
plt.savefig('boxplots/differences_scattered.png')
plt.savefig('boxplots/differences_scattered.eps', format='eps')
plt.close()

plt.figure()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.title('Detection differences (all labels)')
plt.boxplot([orig_diffs_clustered + orig_diffs_scattered, new_diffs_clustered + new_diffs_scattered],
            tick_labels=['CP-SSD', 'KB-KSSD'])
plt.ylim(-3000, 3000)
plt.savefig('boxplots/differences_all.png')
plt.savefig('boxplots/differences_all.eps', format='eps')
plt.close()

# Plot the Manhattan distances of differences from GT

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# First subplot for original data
ax1.bar(bar_positions[0], manhattan_orig_scat + manhattan_orig_clust, bar_width, label='Total distance',
        color='lightgrey')
ax1.bar(bar_positions[0] -bar_width/4, manhattan_orig_clust, bar_width/2, label='Distance w.r.t. clustered idxs',
        color='#D85C5C', edgecolor='#9E3D3D')
ax1.bar(bar_positions[0] + bar_width/4, manhattan_orig_scat, bar_width/2, label='Distance w.r.t. scattered idxs',
        color='#4c72b0', edgecolor='#2A4D69')


# Second subplot for new data
ax2.bar(bar_positions[1], manhattan_new_scat + manhattan_new_clust, bar_width, label='Total distance',
        color='lightgrey')
ax2.bar(bar_positions[1] - bar_width/4, manhattan_new_clust, bar_width/2, label='Distance w.r.t. clustered idxs',
        color='#D85C5C', edgecolor='#9E3D3D')
ax2.bar(bar_positions[1] + bar_width/4, manhattan_new_scat, bar_width/2, label='Distance w.r.t. scattered idxs',
        color='#4c72b0', edgecolor='#2A4D69')

# Set labels and title for original subplot
ax1.set_ylabel('Manhattan distance w.r.t. GT', fontsize=20)
ax1.set_xlabel('CP-SSD', fontsize=20)

ax1.set_xticks([])
ax1.tick_params(axis='both', labelsize=20)

# Set labels and title for new subplot
ax2.set_xlabel('KB-KSSD', fontsize=20)

# ax2.set_ylabel('Counts')
ax2.set_xticks([])
ax2.tick_params(axis='both', labelsize=20)

# Add legends

handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(loc='upper right')
fig.legend(handles, labels, loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.0), fontsize=19)

ax1.set_ylim(0, 165000)
ax2.set_ylim(0, 165000)
plt.ylim(0, 165000)

# Display the plot
fig.tight_layout(rect=[0.05, 0, 0.95, 0.85])

plt.savefig(f'barplots/manhattan.png')
plt.savefig(f'barplots/manhattan.eps', format='eps')
plt.close()

# Check, if differences among models' predictions are normally distributed and plot them
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.title('Differences of absolute values of models\' prediction distances from GT (orig-new)')
plt.hist(pred_abs_diffs, bins=30, color='#4c72b0', edgecolor='#2A4D69')
plt.savefig(f'histograms/pred_abs_diffs.png')
plt.savefig(f'histograms/pred_abs_diffs.eps', format='eps')
plt.close()

plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.title('Diffs of all orig - new differences w.r.t. GT')
plt.hist(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered),
         color='#4c72b0',
         edgecolor='#2A4D69')
plt.savefig('histograms/diffs_of_diffs_all.png')
plt.savefig('histograms/diffs_of_diffs_all.eps', format='eps')
plt.close()

print(f'Mean and median of differences of ALL orig - new diffs from GT: '
      f'{np.mean(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))}, '
      f'{np.median(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))}')

print(f'Mean and median of differences of ALL abs(orig) - abs(new) diffs from GT: '
      f'{np.mean(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)) - abs(np.array(new_diffs_scattered + new_diffs_clustered)))}, '
      f'{np.median(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)) - abs(np.array(new_diffs_scattered + new_diffs_clustered)))}')

print(f'Skewness of differences of ALL orig - new diffs from GT: {scipy.stats.skew(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))}')
print(f'Skewness of differences of ALL abs(orig) - abs(new) diffs from GT: {scipy.stats.skew(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)) - abs(np.array(new_diffs_scattered + new_diffs_clustered)))}')

from scipy.stats import binomtest

# Differences between paired observations
differences = np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered)

# Count positive and negative differences
pos_differences = np.sum(differences > 0)
neg_differences = np.sum(differences < 0)

# Perform a binomial test
n = pos_differences + neg_differences
res = binomtest(pos_differences, n=n, p=0.5, alternative='two-sided')
statistic = res.statistic
pval = res.pvalue
print(f"Sign test for differences of ALL orig - new diffs from GT: {statistic, pval}")

# Differences between paired observations
differences = abs(np.array(orig_diffs_scattered + orig_diffs_clustered)) - abs(np.array(new_diffs_scattered + new_diffs_clustered))

# Count positive and negative differences
pos_differences = np.sum(differences > 0)
neg_differences = np.sum(differences < 0)

# Perform a binomial test
n = pos_differences + neg_differences
res = binomtest(pos_differences, n=n, p=0.5, alternative='two-sided')
statistic = res.statistic
pval = res.pvalue
print(f"Sign test for differences of ALL abs(orig) - abs(new) diffs from GT: {statistic, pval}")


statistic, pval = scipy.stats.shapiro(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))
print(f'Shapiro-Wilk over differences of ALL orig - new diffs from GT: statistic: {statistic}, p-value: {pval}')

statistic, pval = scipy.stats.shapiro(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)) - abs(np.array(new_diffs_scattered + new_diffs_clustered)))
print(f'Shapiro-Wilk over differences of ALL abs(orig) - abs(new) diffs from GT: statistic: {statistic}, p-value: {pval}')

# plt.figure()
# plt.title('Diffs of clustered orig - new differences w.r.t. GT')
# plt.hist(np.array(orig_diffs_clustered) - np.array(new_diffs_clustered))
# plt.savefig('histograms/diffs_of_diffs_clust.png')
# plt.close()
# statistic, pval = scipy.stats.shapiro(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))
# print(f'Shapiro-Wilk over differences of CLUST orig - new diffs from GT: statistic: {statistic}, p-value: {pval}')
#
# plt.figure()
# plt.title('Diffs of scattered orig - new differences w.r.t. GT')
# plt.hist(np.array(orig_diffs_clustered) - np.array(new_diffs_clustered))
# plt.savefig('histograms/diffs_of_diffs_scat.png')
# plt.close()
# statistic, pval = scipy.stats.shapiro(np.array(orig_diffs_scattered + orig_diffs_clustered) - np.array(new_diffs_scattered + new_diffs_clustered))
# print(f'Shapiro-Wilk over differences of SCAT orig - new diffs from GT: statistic: {statistic}, p-value: {pval}')

# statistic, pval = scipy.stats.wilcoxon(orig_diffs_scattered + orig_diffs_clustered,
#                                        new_diffs_scattered + new_diffs_clustered)
# print(f'Wilcoxon two-tail signed-rank test comparing ALL orig/new diffs from GT: statistic: {statistic}, pval: {pval}')
#
# statistic, pval = scipy.stats.wilcoxon(orig_diffs_scattered,
#                                        new_diffs_scattered)
# print(f'Wilcoxon two-tail signed-rank test comparing SCAT orig/new diffs from GT: statistic: {statistic}, pval: {pval}')
#
# statistic, pval = scipy.stats.wilcoxon(orig_diffs_clustered,
#                                        new_diffs_clustered)
# print(f'Wilcoxon two-tail signed-rank test comparing CLUST orig/new diffs from GT: statistic: {statistic}, '
#       f'pval: {pval}')
#
# statistic, pval = scipy.stats.wilcoxon(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)),
#                                        abs(np.array(new_diffs_scattered + new_diffs_clustered)))
# print(f'Wilcoxon two-tail signed-rank test comparing ALL orig/new ABS ERRS from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.ks_2samp(orig_diffs_scattered + orig_diffs_clustered,
                                       new_diffs_scattered + new_diffs_clustered)
print(f'KS test comparing ALL orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.ks_2samp(orig_diffs_scattered, new_diffs_scattered)
print(f'KS test comparing SCAT orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.ks_2samp(orig_diffs_clustered, new_diffs_clustered)
print(f'KS test comparing CLUST orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.ks_2samp(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)),
                                       abs(np.array(new_diffs_scattered + new_diffs_clustered)))
print(f'KS test comparing ALL orig/new ABS ERRS from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.levene(orig_diffs_scattered + orig_diffs_clustered,
                                     new_diffs_scattered + new_diffs_clustered)
print(f'Levene test comparing ALL orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.levene(orig_diffs_scattered, new_diffs_scattered)
print(f'Levene test comparing SCAT orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.levene(orig_diffs_clustered, new_diffs_clustered)
print(f'Levene test comparing CLUST orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

statistic, pval = scipy.stats.levene(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)),
                                     abs(np.array(new_diffs_scattered + new_diffs_clustered)))
print(f'Levene test comparing ALL orig/new ABS ERRS from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.cramervonmises_2samp(orig_diffs_scattered + orig_diffs_clustered,
                                       new_diffs_scattered + new_diffs_clustered)
statistic = res.statistic
pval = res.pvalue
print(f'Cramer-von Mises test comparing ALL orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.cramervonmises_2samp(orig_diffs_scattered, new_diffs_scattered)
statistic = res.statistic
pval = res.pvalue
print(f'Cramer-von Mises test comparing SCAT orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.cramervonmises_2samp(orig_diffs_clustered, new_diffs_clustered)
statistic = res.statistic
pval = res.pvalue
print(f'Cramer-von Mises test comparing CLUST orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res= scipy.stats.cramervonmises_2samp(abs(np.array(orig_diffs_scattered + orig_diffs_clustered)),
                                      abs(np.array(new_diffs_scattered + new_diffs_clustered)))
statistic = res.statistic
pval = res.pvalue
print(f'Cramer-von Mises two-tail signed-rank test comparing ALL orig/new ABS ERRS from GT: statistic: {statistic}, '
      f'pval: {pval}')

res = scipy.stats.anderson_ksamp([orig_diffs_scattered + orig_diffs_clustered,
                                  new_diffs_scattered + new_diffs_clustered])
statistic = res.statistic
pval = res.pvalue
print(f'Anderson-Darling test comparing ALL orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.anderson_ksamp([orig_diffs_scattered, new_diffs_scattered])
statistic = res.statistic
pval = res.pvalue
print(f'Anderson-Darling test comparing SCAT orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.anderson_ksamp([orig_diffs_clustered, new_diffs_clustered])
statistic = res.statistic
pval = res.pvalue
print(f'Anderson-Darling test comparing CLUST orig/new diffs from GT: statistic: {statistic}, pval: {pval}')

res = scipy.stats.anderson_ksamp([abs(np.array(orig_diffs_scattered + orig_diffs_clustered)),
                                  abs(np.array(new_diffs_scattered + new_diffs_clustered))])
statistic = res.statistic
pval = res.pvalue
print(f'Anderson-Darling test comparing ALL orig/new ABS ERRS from GT: statistic: {statistic}, pval: {pval}')

pvals_adjusted_all = scipy.stats.false_discovery_control([0.003498590902895466, 0.01851105994215968, 0.012905978306124622, 0.03670550960639096, 0.011283427463200917])
print(f'Adjusted p-values for ALL indices: {pvals_adjusted_all}')

pvals_adjusted_all = scipy.stats.false_discovery_control([6.704061638247396e-06, 0.21479443462137032, 0.12505597757035683, 0.15724747079744394, 0.1494288234321643])
print(f'Adjusted p-values for ALL indices of ABS ERRS: {pvals_adjusted_all}')

# res = scipy.special.kl_div(orig_diffs_scattered + orig_diffs_clustered,
#                                        new_diffs_scattered + new_diffs_clustered)
# print(f'KL div comparing ALL orig/new diffs from GT: {res}')
#
# res = scipy.special.kl_div(orig_diffs_scattered,
#                                        new_diffs_scattered)
# print(f'KL div comparing SCAT orig/new diffs from GT: {res}')
#
# res = scipy.special.kl_div(orig_diffs_clustered,
#                                        new_diffs_clustered)
# print(f'KL div comparing CLUST orig/new diffs from GT: {res}')


# Save the data structure
json.dump(data_sum, open('full_classification.json', 'w'), cls=plotly.utils.PlotlyJSONEncoder)
