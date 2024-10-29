#!/usr/bin/env python3

"""Classification tool for manually-selected steady-states."""

data_dir = 'man_select_steady'

#!/usr/bin/env python3

"""Utility script for adding benchmark elements into JSON database"""

import json
import os

import numpy as np
import simpleJDB
import matplotlib.pyplot as plt
import sys

data_dir = 'data_st'
db = simpleJDB.database('benchmark_database_steady_indices')

plt.rcParams.update({'font.size': 22})

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
    ts = os.listdir(f'{data_dir}')
    tot = len(ts)
    for c, filename in enumerate(ts):
        timeseries = json.load(open(os.path.join(data_dir, filename)))
        name, fork_idx = filename.rsplit('_', 1)
        # for fork_idx, fork in enumerate(timeseries):
        try:
            # Do NOT reload already saved data
            if db.getkey(f'{name}_{fork_idx}'):
                continue
        except TypeError:
            # SimpleJDB raises TypeError when the key is not in the database - not sure about "correct" solution
            # if I'm asking about it's existence...
            pass
        print(f'{name}: {fork_idx} - {c}/{tot}')
        plt.close()
        fig, ax = plt.subplots(figsize=(16, 9))
        tolerance = 10 # points
        ax.plot(range(len(timeseries)), timeseries, 'ko-', picker=tolerance,
                markersize=0.5)
        ax.grid()
        plt.tight_layout()
        fig.canvas.callbacks.connect('pick_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', on_esc)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
