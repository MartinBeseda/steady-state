import os
from changepoint_fit import changepoint
from prepare_dir import prepare_dir
import json

jagt = json.load(open('analysis-v2/man_steady_comparison/full_classification.json'))

def changepoint_analysis(S, curve, direction, dirpath):
    changepoints_path = prepare_dir(dirpath)

    # Iterate over JAGT
    for series_name_fork, e in jagt.items():
        filename, fork_idx = series_name_fork.rsplit('_', 1)
        jmh_path = f'./data/timeseries/all/{filename}'
        filtered_path = f'./data/timeseries/filtered/{filename}'

        if not os.path.exists(f'{changepoints_path}/{filename}'):
            print('Executing ', filename, '..')
            changepoint(jmh_path, filtered_path, f'{changepoints_path}/{filename}', S, curve, direction)
        else:
            print(filename, 'already exists')
