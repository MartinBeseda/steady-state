#!/usr/bin/env python3

"""The comparison of the original approach and the new one based on the manually labeled 'ground truth' benchmark."""

import json
import os

import simpleJDB
from scipy.stats import norm

import steady_state_detection as ssd

import numpy as np
from matplotlib import pyplot as plt

# # Join the databases
# with open('/tmp/benchmark_database__1_daniele_2_michele.json') as fd:
#     with open('/tmp/benchmark_database_1_luca_2_vittorio.json') as fm:
#         with open('/tmp/benchmark_database__1_michele_2_daniele.json') as fl:
#             with open('/tmp/benchmark_database_1_vittorio_2_luca.json') as fv:
#                 fd_db = json.load(fd)
#                 fm_db = json.load(fm)
#                 fl_db = json.load(fl)
#                 fv_db = json.load(fv)
#                 print(len(fd_db['main']))
#                 print(len(fm_db['main']))
#                 print(len(fl_db['main']))
#                 print(len(fv_db['main']))
#
#                 new_db = {'main': fd_db['main'] + fm_db['main'] + fl_db['main'] + fv_db['main']}
#                 with open('ground_truth_benchmark_2.json', 'w') as fgt:
#                     json.dump(new_db, fgt)
#
# exit(-1)

# Merge the ground truth
ground_truth = {}

gt_db_1 = simpleJDB.database('ground_truth_benchmark')
gt_db_2 = simpleJDB.database('ground_truth_benchmark_2')
gt_db_dis = simpleJDB.database('relabelled_disagreements')

# for e in gt_db_1.data['main']:
#     print(e)

# print(len(gt_db_1.data['main']))
# print(len(gt_db_2.data['main']))
# print(len(gt_db_dis.data['main']))

for el in gt_db_1.data['main']:
    k = el['keyname']
    e = el['value']

    if e == -1 and gt_db_2.getkey(k) == -1:
        continue

    try:
        if gt_db_dis.getkey(k):
            t = (e, gt_db_2.getkey(k), gt_db_dis.getkey(k))
            if sum(np.array(t) == -1) == 2:
                continue
            else:
                ground_truth[k] = sorted(t)[1:]
    except TypeError:
        ground_truth[k] = (e, gt_db_2.getkey(k))

# print(len(ground_truth))
print(ground_truth)

# Compute F-metrics for every steady series
new_f_scores = []
new_harm_diffs = []
harm_diffs_forks = {}
for series, vals in ground_truth.items():
    fork_name, fork_idx = series.rsplit('_', 1)
    fork_idx = int(fork_idx)
    fork = json.load(open(f'../data/timeseries/all/{fork_name}'))[fork_idx]
    # P, warmup_end = ssd.detect_steady_state(fork, prob_win_size=100, t_crit=1.9, step_win_size=50, medfilt_kernel_size=15)
    P, warmup_end = ssd.detect_steady_state(fork, prob_win_size=150, t_crit=3.75, step_win_size=100, medfilt_kernel_size=1)
    res = ssd.get_compact_result(P, warmup_end)
    new_steadiness_idx = ssd.get_ssd_idx(res, 0.8)
    new_f_scores.append(ssd.f_measure({i: [e] for i, e in enumerate(vals)}, [new_steadiness_idx], margin=2000))
    new_harm_diffs.append(ssd.harmonic_mean_of_diffs(new_steadiness_idx, vals))
    harm_diffs_forks[series] = ssd.harmonic_mean_of_diffs(new_steadiness_idx, vals)

orig_f_scores = []
orig_harm_diffs = []
for series, vals in ground_truth.items():
    fork_name, fork_idx = series.rsplit('_', 1)
    fork_idx = int(fork_idx)
    orig_steadiness_idx = json.load(open(f'../data/classification/{fork_name}'))['steady_state_starts'][fork_idx]
    orig_f_scores.append(ssd.f_measure({i: [e] for i, e in enumerate(vals)}, [orig_steadiness_idx], margin=2000))
    orig_harm_diffs.append(ssd.harmonic_mean_of_diffs(orig_steadiness_idx, vals))

plt.figure()
plt.title('F1 metric')
plt.hist([new_f_scores, orig_f_scores])
plt.legend(['New', 'Orig'])
plt.show()

plt.figure()
plt.title('Harmonic mean of differences')
plt.hist([new_harm_diffs, orig_harm_diffs])
plt.legend(['New', 'Orig'])
plt.show()

# Compare the ground truth with both SSD approaches
full_steadiness_idxs = {}
steadiness_idxs = {}
for i, fname in enumerate(('ground_truth_benchmark.json', 'ground_truth_benchmark_2.json')):
    f = json.load(open(fname))['main']
    #print(f)
    for e in f:
        timeseries_name, timeseries_idx = e['keyname'].rsplit('_', 1)
        timeseries_idx = int(timeseries_idx)
        steadiness_idx = e['value']

        # Load the data for the original approach
        with open(f'../data/classification/{timeseries_name}') as f_orig:
            orig_steadiness_idx = json.load(f_orig)['steady_state_starts'][timeseries_idx]

        # Load the data for the new approach
        # TODO write a function for the "final verdict"
        prob_threshold = 0.8
        with open(f'../data/timeseries/all/{timeseries_name}') as f_new:
            fork = json.load(f_new)[timeseries_idx]
            P, warmup_end = ssd.detect_steady_state(fork,
                                                    prob_win_size=150, t_crit=3.75, step_win_size=100, medfilt_kernel_size=1)


            res = ssd.get_compact_result(P, warmup_end)
            new_steadiness_idx = ssd.get_ssd_idx(res, prob_threshold)

        # plt.figure()
        # plt.plot(fork[10:])
        # plt.show()


        steadiness_idxs[e['keyname']] = (steadiness_idx, orig_steadiness_idx, new_steadiness_idx)
        if i == 0:
            full_steadiness_idxs[e['keyname']] = [steadiness_idx, -2, orig_steadiness_idx, new_steadiness_idx]
        else:
            full_steadiness_idxs[e['keyname']][1] = steadiness_idx

        if i == 0:
            try:
                ground_truth[e['keyname']] = [*ground_truth[e['keyname']], orig_steadiness_idx, new_steadiness_idx]
            except KeyError:
                pass

    # Compare number of detected un-/steady series
    no_unsteady_manual = sum([e[0] == -1 for e in list(steadiness_idxs.values())])
    no_unsteady_orig = sum([e[1] == -1 for e in list(steadiness_idxs.values())])
    no_unsteady_new = sum([e[2] == -1 for e in list(steadiness_idxs.values())])

    plt.figure()
    plt.title(f'Number of unsteady timeseries detected (GT {i+1})')
    plt.bar([1, 2, 3], [no_unsteady_manual, no_unsteady_orig, no_unsteady_new])
    plt.xticks([1, 2, 3], ['Manual', 'Original', 'New'])
    plt.show()

    # Compare the number of agreements in unsteadiness without the specific index
    manual_orig_aggr = sum([e[0] == -1 and e[1] == -1 for e in list(steadiness_idxs.values())])
    manual_new_aggr = sum([e[0] == -1 and e[2] == -1 for e in list(steadiness_idxs.values())])
    orig_new_aggr = sum([e[1] == -1 and e[2] == -1 for e in list(steadiness_idxs.values())])

    plt.figure()
    plt.title(f'Number of unsteadiness agreements (GT {i+1})')
    plt.bar([1, 2, 3], [manual_orig_aggr, manual_new_aggr, orig_new_aggr])
    plt.xticks([1, 2, 3], ['Manual X Orig', 'Manual X New', 'Orig X New'])
    plt.show()


    # Compute deviations
    vals = np.array([e for e in list(steadiness_idxs.values()) if -1 not in (e[0], e[1])])
    orig_devs = vals[:, 1] - vals[:, 0]

    vals = np.array([e for e in list(steadiness_idxs.values()) if -1 not in (e[0], e[2])])
    new_devs = vals[:, 2] - vals[:, 0]

    plt.figure()
    plt.title(f'Difference = Detection - ground truth {i+1}')
    plt.hist([orig_devs, new_devs], label=['original', 'new'], bins=20)
    plt.legend()
    plt.show()

    print(f'Absolute difference orig: mean: {np.mean(np.abs(orig_devs))} median: {np.median(np.abs(orig_devs))}')
    print(f'Absolute difference new:  mean: {np.mean(np.abs(new_devs))}  median: {np.median(np.abs(new_devs))} ')


    # Assuming orig_devs and new_devs are already defined as your datasets

    # Plot the histograms
    plt.figure(figsize=(8, 6))  # Increase figure size to make room for the legend
    plt.title(f'Difference = Detection - ground truth {i + 1}')

    # Create histograms for orig_devs and new_devs
    plt.hist(orig_devs, bins=10, density=True, alpha=0.3, label='original', color='blue')
    plt.hist(new_devs, bins=10, density=True, alpha=0.3, label='new', color='green')

    # Get plot limits
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # ---- Fit and plot Gaussian for original data ----
    mean_orig, std_orig = norm.fit(orig_devs)
    p_orig = norm.pdf(x, mean_orig, std_orig)
    plt.plot(x, p_orig, 'b-', linewidth=2, label=f'Original Gaussian: μ={mean_orig:.2f}, σ={std_orig:.2f}')

    # Plot vertical lines for 95% confidence interval (mean ± 2σ) for original data
    plt.axvline(mean_orig - 2 * std_orig, color='blue', linestyle='--', linewidth=1.5, label='Original 95% CI lower')
    plt.axvline(mean_orig + 2 * std_orig, color='blue', linestyle='--', linewidth=1.5, label='Original 95% CI upper')

    # ---- Fit and plot Gaussian for new data ----
    mean_new, std_new = norm.fit(new_devs)
    p_new = norm.pdf(x, mean_new, std_new)
    plt.plot(x, p_new, 'g-', linewidth=2, label=f'New Gaussian: μ={mean_new:.2f}, σ={std_new:.2f}')

    # Plot vertical lines for 95% confidence interval (mean ± 2σ) for new data
    plt.axvline(mean_new - 2 * std_new, color='green', linestyle='--', linewidth=1.5, label='New 95% CI lower')
    plt.axvline(mean_new + 2 * std_new, color='green', linestyle='--', linewidth=1.5, label='New 95% CI upper')

    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust position of the legend

    # Show the plot
    plt.tight_layout()  # Adjust plot to make room for the legend
    plt.show()


# print(len(ground_truth))
# print(ground_truth)
problematic_forks = sorted(harm_diffs_forks.items(), key=lambda e: e[1], reverse=True)
print(problematic_forks)
for e in problematic_forks:
    print(e[0], ground_truth[e[0]])



fname = 'prestodb__presto#com.facebook.presto.operator.BenchmarkBlockFlattener.benchmarkWithoutFlatten#blockSize=1000000&nestedLevel=2&numberOfIterations=1000.json_0'
# fname = 'eclipse-vertx__vert.x#io.vertx.benchmarks.HeadersContainsBenchmark.nettySmallMiss#.json_0'
# fname = 'apache__hive#org.apache.hive.benchmark.vectorization.operators.VectorGroupByOperatorBench.testAggCount#aggregation=min&dataType=timestamp&evalMode=PARTIAL1&hasNulls=false&isRepeating=true&processMode=HASH.json_0'
name, idx = fname.rsplit('_', 1)
idx = int(idx)
fork = json.load(open(f'../data/timeseries/all/{name}'))[idx]

plt.figure()
plt.plot(fork)
plt.show()

exit(-1)


with open('full_relabeled_gt.json', 'w') as f:
    json.dump(ground_truth, f)
exit(-1)

print(steadiness_idxs)
print(full_steadiness_idxs)
with open('labeling_predictions.json', 'w') as f:
    json.dump(full_steadiness_idxs, f)
exit(-1)

# Plot the distribution of the ground truth differences



# Select the timeseries which were disagreed and copy them to separate file to enable manual "relabeling"

# Relabel it manually

# Rewrite the loaded manual labeling according to the new "decisive" one

# Compute F-metrics for the original approach w.r.t. the ground truth (always 2 indices)

# Compute F-metrics for the new approach w.r.t. the ground truth (always 2 indices)

# Create histograms (range 0-1) from both F-metrics
