from glob import glob
import os
from changepoint_fit import changepoint
import sys

filtered_dirpath = './data/timeseries/filtered'
jmh_dirpath = './data/timeseries/all'
changepoints_dirpath = './data/changepoints_cpssd_fitted'


def changepoint_analysis(S, curve, direction):
    if not os.path.exists(changepoints_dirpath):
        os.mkdir(changepoints_dirpath)

    for filtered_path in glob('{}/*.json'.format(filtered_dirpath)):
    # for filtered_path in glob('{}/h2oai__h2o-3#water.ChunkBench.colsRowsRead#cols=10000&rows=100000.json'.format(filtered_dirpath)):

        print(filtered_path)
        filename = filtered_path.split('/')[-1]
        jmh_path = '{}/{}'.format(jmh_dirpath, filename)
        changepoints_path = '{}/{}'.format(changepoints_dirpath, filename)

        jmh_path, filtered_path, changepoints_path = [path for path in [jmh_path, filtered_path, changepoints_path]]
        if not os.path.exists(changepoints_path):
            print('Executing ', filename, '..')
            changepoint(jmh_path, filtered_path, changepoints_path, S, curve, direction)
        else:
            print(filename, 'already exists')
