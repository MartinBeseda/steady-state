"""
Module providing functionality for detection of runtime steady-state
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ssi

np.seterr(all='raise')


def plot_forks(timeseries: list[list], classification: dict, vline_min, vline_max):
    n_timeseries = len(timeseries)
    fig, axs = plt.subplots(n_timeseries, 1, figsize=(10, 20))
    for i in range(n_timeseries):
        x = range(len(timeseries[i]))
        axs[i].plot(x, timeseries[i])
        axs[i].vlines(classification['steady_state_starts'][i], vline_min, vline_max, colors='r')
        axs[0].grid(True)

    plt.tight_layout()
    plt.show()


def remove_outliers_iqr(arr: np.ndarray | list, window_size: int = 100) -> np.ndarray:
    arr = np.array(arr)

    subarrays = np.array_split(arr, len(arr) / float(window_size))
    new_subarrays = []

    for subarr in subarrays:
        q1 = np.percentile(subarr, 25, method='midpoint')
        q3 = np.percentile(subarr, 75, method='midpoint')
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        upper_array = np.where(subarr >= upper)[0]
        tmpsubarr = np.delete(subarr, upper_array)

        lower_array = np.where(tmpsubarr <= lower)[0]
        new_subarrays.append(np.delete(subarr, lower_array))

    return np.concatenate(new_subarrays)


def remove_outliers_percentile(arr: np.ndarray | list,
                               window_size: int = 100,
                               percentile_threshold_upper: int = 99,
                               percentile_threshold_lower: int = 1) -> tuple[np.ndarray, np.ndarray]:
    subarrays = np.array_split(arr, int(len(arr) / int(window_size)))
    new_subarrays = []
    outliers = []

    for subarr in subarrays:
        perc_up = np.percentile(subarr, percentile_threshold_upper)
        perc_low = np.percentile(subarr, percentile_threshold_lower)

        outliers_idxs = np.concatenate((np.where(subarr > perc_up)[0], np.where(subarr < perc_low)[0]))
        new_subarrays.append(np.delete(subarr, outliers_idxs))
        outliers.append(outliers_idxs)

    return np.concatenate(new_subarrays), np.concatenate(outliers)


def ssd_Kelly(x: np.ndarray, n: int, t_crit: float) -> np.ndarray:
    P = np.zeros(len(x))

    if n > len(x):
        n = len(x)

    k = 0
    should_break = False

    intervals = []
    while True:
        from_idx = k
        to_idx = from_idx + n
        if to_idx >= len(x):
            to_idx = len(x)
            n = to_idx - from_idx
            should_break = True

        if to_idx - from_idx < 3:
            intervals[-1][1] = to_idx
        else:
            intervals.append([from_idx, to_idx])

        if should_break:
            break

        k += n

    for interval in intervals:
        from_idx, to_idx = interval[0], interval[1]
        n = to_idx - from_idx

        x_active = x[from_idx:to_idx]

        # Calculate the slope (m) of the drift component
        m = np.mean(np.diff(x_active))

        # Calculate the mean (mu)
        mu = (np.sum(x_active) - np.sum(np.arange(1, n + 1) * m)) / n

        # Calculate the standard deviation (sd)
        sd = np.sqrt(np.sum((x_active - m * np.arange(1, n + 1) - mu) ** 2) / (n - 2))

        # Calculate the steady-state probability (y)
        y = np.mean(np.abs(x_active - mu) <= t_crit * sd)

        P[from_idx:to_idx] = y

        k += n

    return P


def print_all_forks(timeseries, P, warmup_ends, classification, vline_min, vline_max):
    n_timeseries = len(timeseries)
    for i in range(n_timeseries):
        # Plotting the noisy step response
        fig, ax1 = plt.subplots(figsize=(10, 2))

        color = 'tab:blue'
        ax1.set_xlabel('n-th sample')
        ax1.set_ylabel('Noisy Response', color=color)
        ax1.plot(timeseries[i], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim([0, len(timeseries[i])])
        ax1.set_ylim([min(timeseries[i]) - 0.01, max(timeseries[i]) + 0.01])
        ax1.vlines(warmup_ends[i], min(timeseries[i]) - 0.01, max(timeseries[i]) + 0.01, colors='g')
        ax1.vlines(classification['steady_state_starts'][i], vline_min, vline_max, colors='r')


        # Plotting the steady-state probability
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('SS Probability', color=color)
        ax2.plot(P[i], color=color, label='SS Probability')
        ax2.plot(np.ones(len(timeseries[i])), 'k', label='SS Threshold')
        ax2.hlines(0.8, 0, len(timeseries[i]), colors='r', linestyles='dashed')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0.0, 1])

        fig.tight_layout()
        plt.title('Noisy Step Response and Steady-State Probability')
        plt.grid()
        plt.show()

        # Print steadiness probabilities
        idx_start = 0
        len_series = len(P[i])

        if warmup_ends[i] > -1:
            P[i][:warmup_ends[i]] = 0

        prev_e = P[i][0]

        for j, e in enumerate(P[i]):
            if prev_e != e:
                print(f'({idx_start}, {j-1}): {prev_e}')
                idx_start = j
                prev_e = e
            elif j == len_series-1:
                print(f'({idx_start}, {j}): {prev_e}')


def print_fork(timeseries: np.ndarray, P: np.ndarray, warmup_end: int, classification: int,
               vline_min: int, vline_max: int):

    if warmup_end > -1:
        P[:warmup_end] = 0

    # Plotting the noisy step response
    fig, ax1 = plt.subplots(figsize=(10, 2))

    color = 'tab:blue'
    ax1.set_xlabel('n-th sample')
    ax1.set_ylabel('Noisy Response', color=color)
    ax1.plot(timeseries, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([0, len(timeseries)])
    ax1.set_ylim([min(timeseries), max(timeseries)])
    ax1.vlines(warmup_end, min(timeseries), max(timeseries), colors='g')
    ax1.vlines(classification, vline_min, vline_max, colors='r')

    # Plotting the steady-state probability
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SS Probability', color=color)
    ax2.plot(range(len(P)), P, color=color, label='SS Probability')
    # ax2.plot(np.ones(len(timeseries)), 'k')
    ax2.hlines(0.8, 0, len(timeseries), colors='r', linestyles='dashed', label='SS Threshold')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1])

    fig.tight_layout()
    plt.title('Noisy Step Response and Steady-State Probability')
    plt.grid()
    # plt.legend()
    plt.show()

    # Print steadiness probabilities
    idx_start = 0
    len_series = len(P)

    if warmup_end > -1:
        P[:warmup_end] = 0

    prev_e = P[0]
    for j, e in enumerate(P):
        if prev_e != e:
            print(f'({idx_start}, {j-1}): {prev_e}')
            idx_start = j
            prev_e = e
        elif j == len_series-1:
            print(f'({idx_start}, {j}): {prev_e}')


def get_compact_result(P: np.ndarray, warmup_end: int) -> dict:
    results = {'data': list(), 'warm_start_detected': True if warmup_end > -1 else False}

    # Print steadiness probabilities
    idx_start = 0
    len_series = len(P)

    if warmup_end > -1:
        P[:warmup_end] = 0

    prev_e = P[0]
    for j, e in enumerate(P):
        if prev_e != e:
            results['data'].append(((idx_start, j - 1), prev_e))
            idx_start = j
            prev_e = e
        elif j == len_series - 1:
            results['data'].append(((idx_start, j), prev_e))

    return results


def detect_step(data: np.ndarray, win_size: int = 50) -> tuple[int, np.ndarray]:

    data_len = len(data)

    # Detect step in the data with convolution kernel spanning the whole domain
    data -= np.average(data)
    step = np.hstack((np.ones(len(data)), -1 * np.ones(len(data))))
    timeseries_step = np.convolve(data, step, mode='valid')
    large_kernel_step_idx = np.argmin(timeseries_step) - 1
    # print(f'large step idx: {large_kernel_step_idx}')
    # print(f'large step: {data[large_kernel_step_idx] - data[large_kernel_step_idx + 1]}')

    # Detect step in the data with small convolution kernel (4 elements) to take data tails into account
    filter_len = 10
    step2 = np.array(filter_len*[1]+filter_len*[-1])
    timeseries_step2 = np.convolve(data, step2, mode='valid')
    small_kernel_step_idx = np.argmin(timeseries_step2) + filter_len - 1
    # print(f'small step idx {small_kernel_step_idx}')
    # print(f'small step: {data[small_kernel_step_idx] - data[small_kernel_step_idx + 1]}')

    # Check, whether the steps are significant enough and which one to accept, if any, based on median difference of
    # their surrounding windows
    large_left_win = data[max(large_kernel_step_idx - win_size, 0):large_kernel_step_idx + 1]
    large_right_win = data[large_kernel_step_idx + 1:min(large_kernel_step_idx + win_size + 1, data_len)]
    large_diff = np.median(large_left_win) - np.median(large_right_win)

    small_left_win = data[max(small_kernel_step_idx - win_size, 0):small_kernel_step_idx + 1]
    small_right_win = data[small_kernel_step_idx + 1:min(small_kernel_step_idx + win_size + 1, data_len)]
    small_diff = np.median(small_left_win) - np.median(small_right_win)

    # step_idx = large_kernel_step_idx
    if small_kernel_step_idx > large_kernel_step_idx:
        tmp = small_kernel_step_idx
        small_kernel_step_idx = large_kernel_step_idx

        large_kernel_step_idx = tmp

    elif small_kernel_step_idx == large_kernel_step_idx:
        return small_kernel_step_idx, timeseries_step
    if small_kernel_step_idx == 0:
        return large_kernel_step_idx, timeseries_step


    #print('medians', np.median(data[small_kernel_step_idx+1:large_kernel_step_idx] , np.median(data[large_kernel_step_idx+1:])))
    # if np.median(data[small_kernel_step_idx+1:large_kernel_step_idx]) < 1.5*np.median(data[large_kernel_step_idx+1:]):
    # #if small_diff > large_diff:
    #     step_idx = small_kernel_step_idx

    # print(small_kernel_step_idx)
    step_idx = small_kernel_step_idx
    right_med = np.median(data[large_kernel_step_idx+1:])

    if np.median(data[small_kernel_step_idx+1:large_kernel_step_idx]) >0.5*np.abs(large_diff)+ right_med:#1.5 * np.abs(right_med) + right_med:
        step_idx = large_kernel_step_idx



    #ADD
    # print(f'Tresh: {np.abs(np.median(data)/2)}')
    # print(f'Diff: {np.median(data[:step_idx])-np.median(data[step_idx:])}')
    if np.median(data[:step_idx])-np.median(data[step_idx:]) < np.abs(np.median(data)/2):
        if step_idx > small_kernel_step_idx:
            step_idx = small_kernel_step_idx
        else:
            print(f'Chosen step not valid!!!')
            return -1, timeseries_step

    # print(f'{step_idx}: {data[step_idx-1]}, {data[step_idx]}, {data[step_idx+1]}')
    return step_idx, timeseries_step


def plot_step(data, warmup_end, win, label):
    # print(f'Step Idx {warmup_end}')
    win_data = data[max(warmup_end-win,0):min(warmup_end+win, len(data))-1]
    plt.figure(figsize=(12, 6))

    plt.scatter( np.linspace(0,len(win_data),len(win_data)),win_data, label='Signal', color='gray')
    plt.vlines(min(win, warmup_end), min(win_data), max(win_data), colors='g')
    plt.title(label)

    plt.legend()
    plt.show()


def detect_steady_state(x: np.ndarray | list,
                        prob_win_size: int,
                        t_crit: float,
                        step_win_size: int = 150,
                        medfilt_kernel_size: int = 15) -> tuple[np.ndarray, int]:
    # Apply median filter to data for warm-up detection
    x_smooth = ssi.medfilt(x, kernel_size=medfilt_kernel_size)

    # Detect a significant step in data, if there is one
    warmup_end = detect_step(x_smooth, step_win_size)[0]

    # Compute window-based probabilities of steadiness in data, around the step, if detected
    probabilities = None
    if warmup_end > -1:
        left_len = len(x[:warmup_end])
        right_len = len(x[warmup_end + 1:])
        if left_len == 0:
            left_probabilities = np.array([])
        elif left_len == 1:
            left_probabilities = np.array([0])
        else:
            left_probabilities = ssd_Kelly(x[:warmup_end], prob_win_size, t_crit) \
                if len(x[:warmup_end]) > 0 else np.array([])

        if right_len == 0:
            right_probabilities = np.array([])
        elif right_len == 1:
            right_probabilities = np.array([0])
        else:
            right_probabilities = ssd_Kelly(x[warmup_end + 1:], prob_win_size, t_crit) \
                if len(x[warmup_end + 1:]) > 0 else np.array([])
        probabilities = np.concatenate((left_probabilities, np.array([0]), right_probabilities))
    else:
        probabilities = ssd_Kelly(x, prob_win_size, t_crit)

    return probabilities, warmup_end
