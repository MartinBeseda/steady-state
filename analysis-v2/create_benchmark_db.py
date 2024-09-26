#!/usr/bin/env python3

"""Utility script for adding benchmark elements into JSON database"""

import json
import os

import numpy as np
import simpleJDB
import matplotlib.pyplot as plt

data_dir = '../data'
db = simpleJDB.database('benchmark_database')


def on_pick(event):
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    val = int(np.clip(xmouse, 0, len(fork)-1))
    db.setkey(f'{filename}_{fork_idx}', val)
    db.commit()
    plt.close(event.canvas.figure)


def on_esc(event):
    if event.key == 'escape':
        db.setkey(f'{filename}_{fork_idx}', -1)
        db.commit()
        plt.close(event.canvas.figure)


if __name__ == '__main__':
    for filename in os.listdir(f'{data_dir}/timeseries/all'):
        timeseries = json.load(open(os.path.join(data_dir, 'timeseries/all', filename)))
        for fork_idx, fork in enumerate(timeseries):
            print(f'{filename}: {fork_idx}')
            plt.close()
            # fork = timeseries[fork_idx]
            fig, ax = plt.subplots()
            tolerance = 10 # points
            ax.plot(range(len(fork)), fork, 'ro-', picker=tolerance)
            fig.canvas.callbacks.connect('pick_event', on_pick)
            fig.canvas.callbacks.connect('key_press_event', on_esc)
            plt.show()

