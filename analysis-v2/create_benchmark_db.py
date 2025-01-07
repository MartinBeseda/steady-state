#!/usr/bin/env python3

"""Utility script for adding benchmark elements into JSON database"""

import json
import os

import numpy as np
import simpleJDB
import matplotlib.pyplot as plt
import sys

from matplotlib import font_manager

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
    # my_name = sys.argv[1]
    #
    # if my_name not in ('michele', 'daniele', 'luca', 'vittorio'):
    #     print("You need to specify your name in lowercase (without quotes): 'michele', 'daniele', 'luca', 'vittorio' .")

    font_prop = font_manager.FontProperties(size=34)

    # data_dir = f'data/timeseries/benchmark/{my_name}'
    # TODO 'daniele', 'michele' was done
    data_dir = (f'data/timeseries/benchmark/luca')

    for filename in os.listdir(f'{data_dir}'):
        timeseries = json.load(open(os.path.join(data_dir, filename)))
        for fork_idx, fork in enumerate(timeseries):
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
            # plt.tight_layout()
            fig, ax = plt.subplots()
            tolerance = 10 # points
            plt.xticks(fontsize=34)
            plt.yticks(fontsize=34)
            ax.plot(range(len(fork)), fork, 'ro-', picker=tolerance)
            fig.canvas.callbacks.connect('pick_event', on_pick)
            fig.canvas.callbacks.connect('key_press_event', on_esc)
            plt.show()

