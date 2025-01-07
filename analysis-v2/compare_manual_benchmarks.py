#!/usr/bin/env python3

"""Comparison of the manual 'ground-truth' benchmarks."""
import json

import numpy as np
from matplotlib import pyplot as plt

steadiness_idxs = {}
with open('ground_truth_benchmark.json') as f1:
    with open('ground_truth_benchmark_2.json') as f2:
        data_1 = json.load(f1)['main']
        data_2 = json.load(f2)['main']

        for e in data_1:
            steadiness_idxs[e['keyname']] = [e['value'], -2]

        for e in data_2:
            steadiness_idxs[e['keyname']][1] = e['value']

# Numbers of unsteady timeseries
no_unst_1 = sum([e[0] == -1 for e in steadiness_idxs.values()])
no_unst_2 = sum([e[1] == -1 for e in steadiness_idxs.values()])
no_unst_aggreements = sum([e[0] == -1 and e[1] == -1 for e in steadiness_idxs.values()])

plt.figure()
plt.title('No. unsteady timeseries detected')
plt.bar([1, 2, 3], [no_unst_1, no_unst_2, no_unst_aggreements])
plt.xticks([1, 2, 3], ['Labeling 1', 'Labeling 2', 'Agreement'])
plt.show()

print(list(steadiness_idxs.values()))

# Distribution of labeling differences
label_diffs = [e[0]-e[1] for e in steadiness_idxs.values()]

plt.figure()
plt.title('Labeling 1 - labeling 2 (all series)')
plt.hist(label_diffs, bins=20)
plt.show()

print(f'All timeseries differences: mean = {np.mean(label_diffs)}, median = {np.median(label_diffs)}')

# Distribution of labeling differences considering only timeseries detected as steady by both approaches
label_diffs = [e[0]-e[1] for e in steadiness_idxs.values() if -1 not in (e[0], e[1])]

plt.figure()
plt.title('Labeling 1 - labeling 2 (steady by both)')
plt.hist(label_diffs, bins=20)
plt.show()

print(f'Steady timeseries differences: mean = {np.mean(label_diffs)}, median = {np.median(label_diffs)}')