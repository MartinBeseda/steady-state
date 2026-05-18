import json
import os
from kalibera import test

STEADY_STATE = "steady state"
NO_STEADY_STATE = "no steady state"
INCONSISTENT = "inconsistent"


jagt = json.load(open('analysis-v2/man_steady_comparison/full_classification.json'))


def load(path):
    with open(path) as f:
        data = json.load(f)
    return data


def write(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def steady_state_starts(ts, cpts, es, significance):
    cpts = [0] + cpts
    i = -2
    last_segment = ts[cpts[i] + 1:]
    while cpts[i] > 0:
        cpt = cpts[i]
        i -= 1
        prev_cpt = cpts[i]

        start = prev_cpt + 1 if prev_cpt > 0 else prev_cpt
        stop = cpt + 1
        current_segment = ts[start:stop]

        if test(last_segment, current_segment, es, significance):
            return cpt

    return 0


def _classify_benchmark(forks):
    if STEADY_STATE not in forks:
        return NO_STEADY_STATE
    elif NO_STEADY_STATE in forks:
        return INCONSISTENT
    else:
        return STEADY_STATE


def classify_fork(ts, cpts, es, significance):
    steady_state_starts_at = steady_state_starts(ts, cpts, es, significance)
    if steady_state_starts_at <= 2499:
        classification = STEADY_STATE
    else:
        classification = NO_STEADY_STATE
        steady_state_starts_at = -1
    return classification, steady_state_starts_at


def classify_benchmark(ts_list, cpts_list, es, significance):
    forks = []
    steady_state_starts_at = []
    for ts, cpts in zip(ts_list, cpts_list):
        fork_class, steady_state_starts_ = classify_fork(ts, cpts, es, significance)
        forks.append(fork_class)
        steady_state_starts_at.append(steady_state_starts_)

    classification = _classify_benchmark(forks)
    return {"run": classification, "forks": forks, "steady_state_starts": steady_state_starts_at}


def classify_runs(es,
                  significance,
                  changepoints_dir,
                  classification_dir):

    if not os.path.exists(classification_dir):
        os.mkdir(classification_dir)

    # TODO rewrite in a way, that only JAGT timeseries are considered and not every single one
    for series_name_fork, e in jagt.items():
        filename, fork_idx = series_name_fork.rsplit('_', 1)
        changepoints_path = f'{changepoints_dir}/{filename}'
        classification_path = f'{classification_dir}/{filename}'

        if not os.path.exists(classification_path):
            print("Executing {}".format(filename))
            ts_list = load(f'./data/timeseries/all/{filename}')
            cpts_list = load(changepoints_path)

            classification = classify_benchmark(ts_list, cpts_list, es, significance)
            write(classification, classification_path)
        else:
            print('Skipped: {} already exists'.format(classification_path))
