#!/usr/bin/env python3

"""Sensitivity analysis of the CP-SSD w.r.t. the selected parameters."""

import json
import time
import os
import numpy as np
from itertools import product
import multiprocessing
from fit_cpssd import run_cpsssd_with_params

# sys.path.append('..')
# import steady_state_detection as ssd

import sklearn.cluster

# Selected parameters to test
s1_vals = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
curve_vals = ["convex"]
direction_vals = ["decreasing"]
es_vals = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
significance_vals = (0.03, 0.04, 0.05, 0.06, 0.07)
interpolation_methods = [('interp1d', 1), ('polynomial', 2), ('polynomial', 3), ('polynomial', 4), ('polynomial', 5),
                         ('polynomial', 6), ('polynomial', 7)]
all_param_vals = (s1_vals, curve_vals, direction_vals, es_vals, significance_vals, interpolation_methods)

# List of all combinations of parameters
config_lst = list(product(*all_param_vals))
n_configs = len(config_lst)
print(f'Total number of parameter configurations: {n_configs}')

# Shared list for the pool of processes
manager = multiprocessing.Manager()
param_errs = manager.list()

# Load the JAGT dataset and extract the manually-obtained labels to compute the "true" steady-state idx later
jagt = json.load(open('analysis-v2/man_steady_comparison/full_classification.json'))

def worker(params_idx: int) -> None:
    params = config_lst[params_idx]

    params_str = '_'.join(str(el) for el in params)

    # Manhattan distance from GT per one parameter combination
    param_manhattan_err = 0

    # Obtain CP-SSD SSD IDX detection
    run_cpsssd_with_params(S1=params[0], curve=params[1], direction=params[2], es=params[3], significance=params[4],
                           interp_method=params[5][0], poly_degree=params[5][1],
                           changepoints_dir=f'analysis-v2/cpssd-new-results/changepoints_{params_idx}',
                           classification_dir=f'analysis-v2/cpssd-new-results/classification_{params_idx}')

    for k, e in jagt.items():
        # Remove the bad filename suffix and load the filenames with fork indices
        series_key = k
        fname, fork_idx = series_key.rsplit('_', 1)
        fork_idx = int(fork_idx)

        # Check the correctness of the loaded timeseries
        # ref_series = json.load(open(f'../../data/timeseries/all/{fname}'))[fork_idx]
        # ref_idx = json.load(open(f'../../data/classification/{fname}'))['steady_state_starts'][fork_idx]
        #
        # assert ref_series == data_sum[series_key]['series']
        # assert ref_idx == data_sum[series_key]['idxs'][-2]

        # Recognize the SSD reference point
        # If there's cluster (min 3 points), then take the middle point or the 3rd point (if there are 4 in the cluster)
        # If there's no cluster, take the last point, as the most conservative one
        dbscan_inst = sklearn.cluster.DBSCAN(eps=50, min_samples=3)
        man_labels = np.array((e['idxs']['vittorio'], e['idxs']['michele'], e['idxs']['daniele'], e['idxs']['luca'],
                               e['idxs']['martin']))
        cluster_idxs = dbscan_inst.fit(np.array(man_labels).reshape(-1, 1)).labels_

        steady_idx = None
        if not (cluster_idxs > -1).any():
            steady_idx = sorted(man_labels)[-3]
            # data_sum[series_key]['steady_idx'] = sorted(man_labels)[-3]
            # data_sum[series_key]['clustered'] = False
        elif sum(cluster_idxs > -1) == 4:
            steady_idx = man_labels[sorted(np.where(cluster_idxs > -1)[0])[-2]]
            # data_sum[series_key]['steady_idx'] = man_labels[sorted(np.where(cluster_idxs > -1)[0])[-2]]
            # data_sum[series_key]['clustered'] = True
        else:
            steady_idx = int(np.median(man_labels[np.where(cluster_idxs > -1)[0]]))
            # data_sum[series_key]['steady_idx'] = int(np.median(man_labels[np.where(cluster_idxs > -1)[0]]))
            # data_sum[series_key]['clustered'] = True

        # Extract the SSD idx from CP-SSD
        cp_ssd_idx = json.load(open(f'analysis-v2/cpssd-new-results/classification_{params_idx}/{fname}'))['steady_state_starts'][fork_idx]

        # Compute differences of results
        if -1 not in (cp_ssd_idx, e['idxs']['kbkssd']):
            param_manhattan_err += abs(steady_idx - cp_ssd_idx)

    param_errs.append((params_str, int(param_manhattan_err)))


if __name__ == '__main__':
    print(f'Number of CPU cores available: {len(os.sched_getaffinity(0))} / {multiprocessing.cpu_count()} / '
          f'{os.cpu_count()}')
    start = time.time()
    pool_inst = multiprocessing.Pool()
    ans = pool_inst.map(worker, range(n_configs))
    # ans = pool_inst.map(worker, [0,1,2])
    end = time.time()
    print(f'Total runtime: {end - start}')

    # Detect the best configuration
    print(f'Sorted params:\n{sorted(param_errs, key=lambda x: x[1])}')
    print(f'Best params: {min(param_errs, key=lambda x: x[1])}')

    # Store the classification results
    cpssd_config_results = dict(sorted(param_errs, key=lambda x: x[1]))
    json.dump(cpssd_config_results, open('analysis-v2/cpssd-new-results/cpssd_config_results.json', 'w'))
