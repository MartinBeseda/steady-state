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

from ossdem import SlopeDetectionMethod, TTest, ExactBayes, FTest, RTest
from itertools import batched
import scipy.signal as ssi
import sklearn.cluster
from scipy.stats import norm
from scipy.optimize import differential_evolution

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix

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

# Plot the number of steadiness detections on larger dataset
larger_data = json.load(open('benchmark_database_binary.json'))
print(f'Total number (including unsteady) of timeseries: {len(larger_data['main'])}')

no_agreements_orig = 0
false_positives_orig = 0
false_negatives_orig = 0

no_agreements_new = 0
false_positives_new = 0
false_negatives_new = 0

no_agreements_sdm = 0
false_positives_sdm = 0
false_negatives_sdm = 0

no_agreements_ttest = 0
false_positives_ttest = 0
false_negatives_ttest = 0

no_agreements_ftest = 0
false_positives_ftest = 0
false_negatives_ftest = 0

no_agreements_rtest = 0
false_positives_rtest = 0
false_negatives_rtest = 0

no_agreements_eb = 0
false_positives_eb = 0
false_negatives_eb = 0

n_total_agreements = 0
n_method_agreements = 0
n_std = 0
n_std_new = 0
n_std_orig = 0
n_std_sdm = 0
n_std_ttest = 0
n_std_ftest = 0
n_std_rtest = 0
n_std_eb = 0

# Binary classification
jagt_bin = []
methods_bin = [list() for _ in range(5)]

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

    # Detection via SDM - using raw timeseries, without any preprocessing
    batch_size = 15
    timeseries_batches = list(batched(timeseries1, batch_size))

    sdm = SlopeDetectionMethod(slope_crit=1e-8)
    sdm_steady_idx = None
    for j, batch in enumerate(timeseries_batches):
        sdm.insert(batch)
        sdm_steady_idx = sdm.steady_state_start_point()

    # Detection via t-test
    t_test = TTest(T_crit=0.009)
    ttest_steady_idx = None
    for j, batch in enumerate(timeseries_batches):
        t_test.insert(batch)
        ttest_steady_idx = t_test.steady_state_start_point()

    # # Detection via Exact Bayes
    # eb = ExactBayes(m=20, s_0=1e-8)
    # eb_steady_idx = None
    # for j, batch in enumerate(timeseries_batches):
    #     eb.insert(batch)
    #     eb_steady_idx = eb.steady_state_start_point()

    # # Detection via F-test
    # f_test = FTest(F_crit=1.2)
    # ftest_steady_idx = None
    # for j, batch in enumerate(timeseries_batches):
    #     f_test.insert(batch)
    #     ftest_steady_idx = f_test.steady_state_start_point()

    # Detection via R-test
    r_test = RTest(R_crit=1.2, lambda1=0.03, lambda2=0.05, lambda3=0.05)
    rtest_steady_idx = None
    for j, batch in enumerate(timeseries_batches):
        r_test.insert(batch)
        rtest_steady_idx = r_test.steady_state_start_point()

    # Number of agreements, etc. for SDM method
    sdm_steady = True if sdm_steady_idx > -1 else False
    if sdm_steady > -1:
        n_std_sdm += 1
    if is_steady == sdm_steady:
        no_agreements_sdm += 1
    elif not is_steady and sdm_steady_idx:
        false_positives_sdm += 1
    else:
        false_negatives_sdm += 1

    ttest_steady = True if ttest_steady_idx > -1 else False
    if ttest_steady > -1:
        n_std_ttest += 1
    if is_steady == ttest_steady:
        no_agreements_ttest += 1
    elif not is_steady and ttest_steady_idx:
        false_positives_ttest += 1
    else:
        false_negatives_ttest += 1

    rtest_steady = True if rtest_steady_idx > -1 else False
    if rtest_steady > -1:
        n_std_rtest += 1
    if is_steady == rtest_steady:
        no_agreements_rtest += 1
    elif not is_steady and rtest_steady_idx:
        false_positives_rtest += 1
    else:
        false_negatives_rtest += 1

    jagt_bin.append(is_steady)
    methods_bin[0].append(orig_classification)
    methods_bin[1].append(new_clas_idx)
    methods_bin[2].append(sdm_steady)
    methods_bin[3].append(ttest_steady)
    methods_bin[4].append(rtest_steady)

    # ftest_steady = True if ftest_steady_idx > -1 else False
    # if ftest_steady > -1:
    #     n_std_ftest += 1
    # if is_steady == ftest_steady:
    #     no_agreements_ftest += 1
    # elif not is_steady and ftest_steady_idx:
    #     false_positives_ftest += 1
    # else:
    #     false_negatives_ftest += 1

    # eb_steady = True if eb_steady_idx > -1 else False
    # if eb_steady > -1:
    #     n_std_eb += 1
    # if is_steady == eb_steady:
    #     no_agreements_eb += 1
    # elif not is_steady and eb_steady_idx:
    #     false_positives_eb += 1
    # else:
    #     false_negatives_eb += 1

    # Both methods and the ground truth agree with each other
    if is_steady == new_clas_idx == orig_classification:
        n_total_agreements += 1

    # Both methods agree with each other, but not with the ground truth
    if (is_steady != new_clas_idx) and (new_clas_idx == orig_classification):
        n_method_agreements += 1

# print(n_std, n_std_orig, n_std_new, n_std_sdm)
# print(no_agreements_orig, no_agreements_new, no_agreements_sdm, no_agreements_ttest)
# print(false_positives_orig, false_negatives_orig, false_positives_new, false_negatives_new)
# print(n_total_agreements, n_method_agreements)

jagt_bin = np.array(jagt_bin).astype(int)
methods_bin = np.array(methods_bin).astype(int)

print(jagt_bin)
print(methods_bin)

method_names = ['CP-SSD', 'KB-KSSD', 'SDM', 't-test', 'R-test']

# Step 1: Compute confusion matrices and accuracies
print("Confusion Matrices and Accuracies:")
accuracies = []
for i, preds in enumerate(methods_bin):
    cm = confusion_matrix(jagt_bin, preds, labels=[1, 0])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / np.sum(cm)
    accuracies.append(acc)
    print(f"{method_names[i]}:\n{cm}")
    print(f"Accuracy: {acc:.3f}\n")

accuracies = np.array(accuracies)
best_idx = np.argmax(accuracies)
print(f"Best performing method: {method_names[best_idx]} with accuracy {accuracies[best_idx]:.3f}\n")

# Step 2: Prepare data for Cochran's Q Test
correct_matrix = (methods_bin == jagt_bin).astype(int).T  # shape: (samples, methods)

q_stat, p_value, df = cochrans_q(correct_matrix, return_object=False)
print(f"Cochran's Q test statistic: {q_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    # Selecting KB-KSSD


    print("Significant difference detected among methods, proceeding with post-hoc McNemar's tests...\n")

    # Step 3: Post-hoc pairwise McNemar tests vs best method
    p_values = []
    comparisons = []
    best_preds = methods_bin[best_idx]

    for i, preds in enumerate(methods_bin):
        if i == best_idx:
            continue

        best_correct = (best_preds == jagt_bin)
        other_correct = (preds == jagt_bin)

        table = np.zeros((2, 2), dtype=int)
        for bc, oc in zip(best_correct, other_correct):
            table[bc, oc] += 1

        result = mcnemar(table, exact=False)
        p_values.append(result.pvalue)
        comparisons.append((method_names[best_idx], method_names[i]))
        print(f"McNemar test between {method_names[best_idx]} and {method_names[i]}: p-value={result.pvalue:.4f}")

    # Step 4: Correct for multiple comparisons using Benjamini-Hochberg (FDR)
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    print("\nAfter Benjamini-Hochberg correction:")
    for (m1, m2), pval, rej in zip(comparisons, pvals_corrected, reject):
        print(f"Comparison {m1} vs {m2}: corrected p-value={pval:.4f}, reject H0={rej}")

else:
    print("No significant difference detected; no need for post-hoc testing.")

# Plot number of agreements with the larger data set for all the compared methods
plt.figure()
values = [0.75, 0.85, 0.65, 0.80,]# 0.70, 0.90, 0.78]

# Custom colors and edgecolors
fill_colors = [
    '#D85C5C',  # red
    '#4c72b0',  # blue
    '#BFBFBF',  # gray
    '#55A868',  # green
    # '#C44E52',  # dark red
    # '#8172B3',  # purple
    # '#CCB974',  # goldenrod
]

edge_colors = [
    '#9E3D3D',  # red edge
    '#2A4D69',  # blue edge
    '#7F7F7F',  # gray edge
    '#2C6B4D',  # green edge
    # '#872529',  # dark red edge
    # '#4B3D72',  # purple edge
    # '#7A653E',  # goldenrod edge
]
hatches = []# ['/', '\\', 'x', '.', '-', '*', 'o']

# Create bar plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(['CP-SSD', 'KB-KSSD', 'SDM', 't-test'],#, 'F-test', 'R-test', 'ExactBayes'],
              [no_agreements_orig, no_agreements_new, no_agreements_sdm, no_agreements_ttest],
              #, no_agreements_ftest, no_agreements_rtest, no_agreements_eb],
              color=fill_colors,
              edgecolor=edge_colors,
              linewidth=1.5)

# Apply hatches
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Labels and title
ax.set_ylabel('Agreements')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate values
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('barplots/agreements_all.eps', format='eps')
plt.savefig('barplots/agreements_all.png')
plt.show()
plt.close()
exit(-1)

# Detailed number of agreements and false positives / negatives for CP-SSD and KB-KSSD
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
