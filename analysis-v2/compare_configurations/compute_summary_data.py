#!/usr/bin/env python3

"""The script for computation of different Manhattan distances for configurations around the optimal one."""

import json
import sys
import numpy as np

sys.path.append('..')
import steady_state_detection as ssd

# The ground truth
gt_data = json.load(open('full_classification.json'))

# Parameters
optimal_config = (100, 500, 4, 70, 0.95)

outliers_window_size = (80, 120)
prob_win_size = (400, 600)
t_crit = (3, 5)
step_win_size = (60, 80)
prob_threshold = (0.75, 0.85)

config_lst = [outliers_window_size, prob_win_size, t_crit, step_win_size, prob_threshold]

def get_err(params: np.ndarray | list[float] | tuple[float]) -> float:
    err = 0
    for timeseries_key, data in gt_data.items():
        # Skip over the unsteady timeseries
        if -1 in (data['idxs'][-2], data['steady_idx']):
            continue

        timeseries = data['series'].copy()
        gt_steady_idx = data['steady_idx']


        # Obtain the prediction
        timeseries, _ = ssd.substitute_outliers_percentile(timeseries,
                                                           percentile_threshold_upper=98,
                                                           percentile_threshold_lower=2,
                                                           window_size=params[0])

        P, warmup_end = ssd.detect_steady_state(timeseries,
                                                prob_win_size=params[1],
                                                t_crit=params[2],
                                                step_win_size=params[3],
                                                medfilt_kernel_size=1)

        res = ssd.get_compact_result(P, warmup_end)

        new_steady_idx = ssd.get_ssd_idx(res,
                                         prob_threshold=params[4],
                                         min_steady_length=0)

        err += abs(new_steady_idx - gt_steady_idx)
    return err


if __name__ == "__main__":
    # Obtain the error of the optimal configuration
    ideal_err = get_err(optimal_config)
    print('_'.join([str(e) for e in optimal_config]), ideal_err)

    for i in range(5):
        for j in range(2):
            current_config = list(optimal_config)
            current_config[i] = config_lst[i][j]
            config_str = '_'.join([str(e) for e in current_config])
            err = get_err(current_config)
            print(config_str, err)
