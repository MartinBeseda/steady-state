#!/usr/bin/env python3

"""Obtain steady timeseries from the JSON database."""
import json

import simpleJDB
import os
import shutil

db = simpleJDB.database('benchmark_database')

# Create a new folder for manually selected steady timeseries
data_dir = 'data_st'
os.makedirs(data_dir)

for e in db.data['main']:
    fname = e['keyname'].rsplit('_', 1)[0]
    idx = int(e['keyname'].rsplit('_', 1)[1])
    if e['value'] > -1:
        timeseries = json.load(open(f'../data/timeseries/all/{fname}'))
        json.dump(timeseries[idx], open(f'{data_dir}/{fname}_{idx}', 'w'))
