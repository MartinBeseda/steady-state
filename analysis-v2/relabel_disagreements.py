#!/usr/bin/env python3

"""The utility script for relabelling of timeseries, where the previous manual atttempts disagree about stesdiness.
For such cases another JSON database will be created to be later used as a "patch" for the data, if necessary.
"""

import json

import numpy as np
import simpleJDB
from matplotlib import pyplot as plt

def on_pick(event):
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    val = int(np.clip(xmouse, 0, len(timeseries)-1))
    db.setkey(f'{filename}_{fork_idx}', val)
    db.commit()
    plt.close(event.canvas.figure)


def on_esc(event):
    if event.key == 'escape':
        db.setkey(f'{filename}_{fork_idx}', -1)
        db.commit()
        plt.close(event.canvas.figure)


if __name__ == '__main__':
    # Timeseries where manual labeling disagrees w.r.t. the steadiness
    steadiness_idxs_manual_comp = {}
    with open('ground_truth_benchmark.json') as f1:
        with open('ground_truth_benchmark_2.json') as f2:
            data_1 = json.load(f1)['main']
            data_2 = json.load(f2)['main']

            for e in data_1:
                steadiness_idxs_manual_comp[e['keyname']] = [e['value'], -2]

            for e in data_2:
                steadiness_idxs_manual_comp[e['keyname']][1] = e['value']

    disagreements = {k: e
                     for k, e
                     in steadiness_idxs_manual_comp.items()
                     if (e[0] == -1) != (e[1] == -1)}

    # Iterate through the identified timeseries and relabel them
    db = simpleJDB.database('relabelled_disagreements')
    for series in disagreements:
        filename, fork_idx = series.rsplit('_', 1)
        fork_idx = int(fork_idx)

        timeseries = json.load(open(f'../data/timeseries/all/{filename}'))[fork_idx]
        try:
            # Do NOT reload already saved data
            if db.getkey(f'{filename}_{fork_idx}'):
                continue
        except TypeError:
            # SimpleJDB raises TypeError when the key is not in the database - not sure about "correct" solution
            # if I'm asking about it's existence...
            pass

        print(f'{filename}: {fork_idx}')
        plt.close()
        fig, ax = plt.subplots()
        tolerance = 10 # points
        ax.plot(range(len(timeseries)), timeseries, 'ro-', picker=tolerance)
        fig.canvas.callbacks.connect('pick_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', on_esc)
        plt.show()
