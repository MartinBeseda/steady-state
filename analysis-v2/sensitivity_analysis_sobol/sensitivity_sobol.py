#!/usr/bin/env python3

"""This script performs Sobol's parameter sensitivity analysis of the new SSD approach."""

import json
import sys

import scipy.stats
from SALib import ProblemSpec
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
from SALib.test_functions import Ishigami
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
import steady_state_detection as ssd


# # Defining the model
# def dummy_model(params: tuple[int, float, float]) -> float:
#     # print(f'params: {params}')
#     return params[0] + 10*params[1]
#
#
# # Defining to-be-analyzed parameters and their properties
# problem = ProblemSpec({
#     'names': ['x1', 'x2', 'x3'],
#     'bounds': [[-3.5, 5.5],
#                [-3.14159265359, 3.14159265359],
#                [-3.14159265359, 3.14159265359]],
#     'outputs': 'y'
# })
#
# # Generate the samples for different parameter configurations
# param_values = problem.sample_sobol(2**12)
# param_values.samples = np.array([(round(e[0]), e[1], e[2]) for e in param_values.samples])
#
# a=1
#
# Y = np.zeros([param_values.samples.shape[0]])
#
# for i, X in enumerate(param_values.samples):
#     Y[i] = dummy_model(X)
#
# # Perform analysis
# analysis_res = analyze(problem, Y, print_to_console=True)
#
# print(analysis_res)
# plt.figure()
# analysis_res.plot()
# plt.show()

# outliers_window_size = 100
# prob_win_size = 500
# t_crit = 4
# step_win_size = 70
# prob_threshold = 0.95
# median_kernel_size = 1


# Load the data
ground_truth_data = json.load(open('full_classification.json'))

# Export the deviations w.r.t. the GT into CSV
new_diffs = []
orig_diffs = []
with open('absolute_differences.csv', 'w+') as f:
    for k, e in ground_truth_data.items():
        if e['idxs'][-1] > 0 and e['idxs'][-2] > 0:
            print(k, np.abs(e['steady_idx'] - e['idxs'][-1]), np.abs(e['steady_idx'] - e['idxs'][-2]))
            new_diffs.append(np.abs(e['steady_idx'] - e['idxs'][-1]))
            orig_diffs.append(np.abs(e['steady_idx'] - e['idxs'][-2]))
            # f.write(f'{k} {np.abs(e['steady_idx'] - e['idxs'][-1])} {np.abs(e['steady_idx'] - e['idxs'][-2])}\n')

# Defining to-be-analyzed parameters and their properties
#
# prob_win_size and step_win_size are integer variables, thus the intervals are
# shifted by 0.5 and they have to be rounded before passing them as inputs to
# the model
problem = ProblemSpec({
    'names': ['prob_win_size', 'step_win_size', 't_crit', 'prob_threshold', 'outlier_win_size'],
    'bounds': [[400.5, 600.5],
               [50.5, 90.5],
               [3.5, 5.5],
               [0.75, 0.95],
               [80.5, 120.5]],
    'outputs': 'steadiness_idx'
})

# Generating parameter configurations via Sobol sampling
param_values = problem.sample_sobol(2**9)
param_values.samples = np.array([(round(e[0]),
                                  round(e[1]),
                                  e[2],
                                  e[3],
                                  round(e[4]) if round(e[4]) % 2 else round(e[4]) - 1)
                                for e in param_values.samples])

# Running the model for the different configurations of parameters
outputs = np.zeros([param_values.samples.shape[0]])
for i, params in enumerate(param_values.samples):
    print(f'Processing parameters set ({params}): {i+1}/{len(param_values.samples)}...')
    for key, el in ground_truth_data.items():
        timeseries = el['series']

        # timeseries1 = timeseries.copy()
        timeseries, _ = ssd.substitute_outliers_percentile(timeseries, percentile_threshold_upper=85,
                                                           percentile_threshold_lower=2,
                                                           window_size=int(round(params[4])))

        P, warmup_end = ssd.detect_steady_state(timeseries,
                                                prob_win_size=int(round(params[0])),
                                                step_win_size=int(round(params[1])),
                                                t_crit=params[2],
                                                medfilt_kernel_size=1)
        res = ssd.get_compact_result(P, warmup_end)
        new_ssd_idx = ssd.get_ssd_idx(res, prob_threshold=params[3], min_steady_length=0)
        outputs[i] += (el['steady_idx'] - new_ssd_idx)**2

# Perform analysis
analysis_res = analyze(problem, outputs)
print(analysis_res)

# Print the plots
plt.figure()
fig, axes = plt.subplots(1, 3)
analysis_res.plot(axes)
plt.suptitle('Sobols\' indices')
plt.tight_layout()
plt.savefig('barplots/sobol_idxs.png')
plt.close()

with open('sobol_analysis.txt', 'w+') as f:
    f.write(analysis_res)
