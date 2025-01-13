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


def substitute_outliers_percentile(arr: np.ndarray | list,
                               window_size: int = 100,
                               percentile_threshold_upper: int = 99,
                               percentile_threshold_lower: int = 1) -> tuple[np.ndarray, np.ndarray]:
    subarrays = np.array_split(arr[1:], int(len(arr[1:]) / int(window_size)))
    new_subarrays = [np.array(arr[0]).flatten()]
    outliers = []

    for subarr in subarrays:
        perc_up = np.percentile(subarr, percentile_threshold_upper)
        perc_low = np.percentile(subarr, percentile_threshold_lower)

        outliers_idxs = np.concatenate((np.where(subarr > perc_up)[0], np.where(subarr < perc_low)[0]))

        for idx in outliers_idxs:
            subarr[idx] = np.median(subarr)

        new_subarrays.append(subarr)

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
            if intervals:
                intervals[-1][1] = to_idx
            else:
                # print('There are too few points provided for Kelly to determine the steadiness probability! '
                #       'The probabilities for these points are gonna be set to -1.')

                intervals.append([from_idx, to_idx])
                P[from_idx:to_idx] = -1
                return P
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
    if warmup_end > -1:
        ax1.vlines(warmup_end, min(timeseries), max(timeseries), colors='g')
    ax1.vlines(classification, vline_min, vline_max, colors='r')

    # Plotting the steady-state probability
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SS Probability', color=color)
    ax2.plot(range(len(P)), P, color=color, label='SS Probability')
    ax2.hlines(0.8, 0, len(timeseries), colors='r', linestyles='dashed', label='SS Threshold')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1])

    fig.tight_layout()
    plt.title('Noisy Step Response and Steady-State Probability')
    plt.grid()
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
    large_up = np.argmax(timeseries_step) -1

    # Detect step in the data with small convolution kernel (4 elements) to take data tails into account
    filter_len = 15
    step2 = np.array(filter_len*[1]+filter_len*[-1])
    timeseries_step2 = np.convolve(data, step2, mode='valid')
    small_kernel_step_idx = np.argmin(timeseries_step2) + filter_len - 1
    small_up = np.argmax(timeseries_step2) + filter_len- 1

    #return np.min(np.array([large_up, large_kernel_step_idx, small_up, small_kernel_step_idx])), timeseries_step

    # print(small_kernel_step_idx, large_kernel_step_idx, small_up, large_up)
    # plt.figure()
    # plt.plot(timeseries_step)
    # plt.title(f'Convolution with larger kernel')
    # plt.savefig('convolution_larger')
    # plt.close()
    #
    # plt.figure()
    # plt.plot(timeseries_step2)
    # plt.title(f'Convolution with smaller kernel')
    # plt.savefig('convolution_smaller')
    # plt.close()
    # exit(-1)
    # If the step is detected in the very last sample of the series, we consider it non-existent
    if large_kernel_step_idx == len(data) - 1:
        large_kernel_step_idx = -1

    if small_kernel_step_idx == len(data) - 1:
        small_kernel_step_idx = -1

    # If both the kernels detected steps at the same index or differing just by 1, discard the one detected via large
    # kernel
    if np.abs(large_kernel_step_idx - small_kernel_step_idx) in (0, 1):
        large_kernel_step_idx = -1

    step_idx = -1


    # If large kernel detected a step, continue with its processing
    if large_kernel_step_idx > -1:

        if small_kernel_step_idx > large_kernel_step_idx:
            # If "short kernel detection" is at larger index than "large window" one, switch them for convenience
            tmp = small_kernel_step_idx
            small_kernel_step_idx = large_kernel_step_idx
            large_kernel_step_idx = tmp

        elif small_kernel_step_idx == large_kernel_step_idx:
            return small_kernel_step_idx, timeseries_step
        if small_kernel_step_idx == 0:
            return large_kernel_step_idx, timeseries_step

        # Check, whether the steps are significant enough and which one to accept, if any, based on median difference of
        # their surrounding windows
        large_left_win = data[max(large_kernel_step_idx - win_size, 0):large_kernel_step_idx + 1]
        large_right_win = data[large_kernel_step_idx + 1:min(large_kernel_step_idx + win_size + 1, data_len)]

        # Difference of medians of windows around the detected "large step"
        large_diff = np.median(large_left_win) - np.median(large_right_win)



        # Choose the step detected via "short kernel" as a default one
        step_idx = small_kernel_step_idx
        right_med = np.median(data[large_kernel_step_idx+1:])

        # Check, if the step detected via "short kernel" is significant enough, otherwise take the "large kernel" one
        if np.median(data[small_kernel_step_idx+1:large_kernel_step_idx]) > 2.0*np.abs(large_diff) + right_med:
            step_idx = large_kernel_step_idx

    elif small_kernel_step_idx > -1:
        step_idx = small_kernel_step_idx

    if np.median(data[:step_idx])-np.median(data[step_idx:]) < np.abs(np.median(data)/2):
        if step_idx > small_kernel_step_idx:
            step_idx = small_kernel_step_idx
        else:
            return -1, timeseries_step

    # up_step = max(large_up, small_up)
    # if up_step >= data_len - 1:
    #     return step_idx, timeseries_step
    # up_left_win = data[max(up_step - win_size, 0):up_step + 1]
    # up_right_win = data[up_step + 1:min(up_step + win_size + 1, data_len)]
    # up_diff = np.median(up_right_win) - np.median(up_left_win)
    #
    # chosen_left_win = data[max(step_idx - win_size, 0):step_idx + 1]
    # chosen_right_win = data[step_idx + 1:min(step_idx + win_size + 1, data_len)]
    # chosen_diff = np.median(chosen_left_win) - np.median(chosen_right_win)
    # if up_diff > 2*chosen_diff:
    #     step_idx = up_step
    #     print(f'Chosen UP')
    return step_idx, timeseries_step


def plot_step(data, warmup_end, win, label):
    win_data = data[max(warmup_end-win,0):min(warmup_end+win, len(data))-1]
    plt.figure(figsize=(12, 6))

    plt.scatter(np.linspace(0,len(win_data),len(win_data)),win_data, label='Signal', color='gray')
    plt.vlines(min(win, warmup_end), min(win_data), max(win_data), colors='g')
    plt.title(label)

    plt.legend()
    plt.show()


def detect_steady_state(x: np.ndarray | list,
                        prob_win_size: int,
                        t_crit: float,
                        step_win_size: int = 150,
                        medfilt_kernel_size: int = 15) -> tuple[np.ndarray, int]:
    # TODO probably substitute outliers with the median of their surroundings

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


def get_ssd_idx(compact_result: dict, prob_threshold: float, min_steady_length: int = 500) -> int:
    """Obtain an index of steady-state start or -1,
    if the timeseries is unsteady by parsing of the full result."""

    # if not compact_result['warm_start_detected']:
    #     return -1

    new_steadiness_idx = -1
    for interval in compact_result['data'][::-1]:
        if interval[1] >= prob_threshold:
            new_steadiness_idx = interval[0][0]
        else:
            break


    if new_steadiness_idx > compact_result['data'][-1][0][1] - min_steady_length:
        new_steadiness_idx = -1


    return new_steadiness_idx


def true_positives(T, X, margin=5):
    """Compute true positives without double counting

    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too

    Remember that all CP locations are 0-based!

    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def harmonic_mean_of_diffs(prediction: float, references: list[float]) -> float:
    try:
        return len(references) / sum(1/np.abs(prediction - e + 1e-10) for e in references)
    except FloatingPointError:
        print(prediction)
        print(references)
        exit(-1)