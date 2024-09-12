"""
The utility script for performing of cumulative statistics on classification results.
"""

import os
import json

import numpy as np
from matplotlib import pyplot as plt

data_dir = '../data'
path = f'{data_dir}/classification'

new_data_raw = {}
classifications = {}
for filename in os.listdir(path):
    f_old = os.path.join(path, filename)
    f_new = os.path.join(data_dir, f'new_classification/{filename}')

    if os.path.isfile(f_old):
        # Read all the forks for the configuration
        forks_old = json.load(open(f_old))['steady_state_starts']
        forks_new_data = json.load(open(f_new))

        new_data_raw[filename] = forks_new_data

        # Parse new forks and store steady-state starting indices together with info about warm-up detection
        forks_new = []
        for fork in forks_new_data:
            # Check, if steady-state was detected
            steady_start = -1
            for interval in fork['data'][::-1]:
                if interval[1] >= 0.8:
                    steady_start = interval[0][0]
                else:
                    break

            if steady_start > fork['data'][-1][0][1] - 500:
                steady_start = -1

            forks_new.append(steady_start)

        classifications[filename] = (forks_old, forks_new)

# Compare how many times we agree on steadiness or not

#  Number of times both approaches agree, that the timeseries is steady or it's not
no_agree = 0

# Number of times the original classification marks the series as unsteady and the new approach as steady
no_original_unsteady = 0

# Number of times the new classification marks the series as unsteady and the original approach as steady
no_new_unsteady = 0
for benchmark in classifications:
    for i, orig_start in enumerate(classifications[benchmark][0]):
        if orig_start == -1 and classifications[benchmark][1][i] != -1:
            no_original_unsteady += 1
        elif classifications[benchmark][1][i] == -1 and orig_start != -1:
            no_new_unsteady += 1
        else:
            no_agree += 1

# print(no_agree, no_original_unsteady, no_new_unsteady)
plt.figure()
plt.bar(range(3), (no_agree, no_original_unsteady, no_new_unsteady))
plt.xticks(range(3), ('Agreements', 'Unsteady st. (original)',
                      'Unsteady st. (new)'))
plt.show()

# Deviations of detected steady-state when both approaches detected steadiness
diffs = []
for benchmark in classifications:
    for i, orig_start in enumerate(classifications[benchmark][0]):
        if -1 not in (orig_start, classifications[benchmark][1][i]):
            diffs.append(orig_start - classifications[benchmark][1][i])

plt.figure()
plt.title('Deviations when steady-state detected via both methods')
plt.hist(diffs, 30)
plt.show()

# Compute variance of the deviations
print(np.var(diffs))

# Deviations of warm-up detection WITHOUT respect to any other considerations of steadiness
warmup_diffs = []
for benchmark in new_data_raw:
    for i, fork in enumerate(new_data_raw[benchmark]):
        if fork['warm_start_detected'] and len(fork['data']) > 1:
            warmup_diffs.append(classifications[benchmark][0][i] - fork['data'][1][0][0])


plt.figure()
plt.title('Deviations of warm-up phase detected (no Kelly!)')
plt.hist(warmup_diffs, 30)
plt.show()

# Compute variance of the deviations
print(np.var(warmup_diffs))
