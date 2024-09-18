#!/usr/bin/env python3

"""Utility script for adding benchmark elements into JSON database"""

import json
import os
import simpleJDB
import matplotlib.pyplot as plt

data_dir = '../data'

# filename = 'cantaloupe-project__cantaloupe#edu.illinois.library.cantaloupe.perf.processor.codec.TIFFImageWriterPerformance.testWriteWithPlanarImage#.json'
# fork_idx = 1
#
# timeseries = json.load(open(os.path.join(data_dir, 'timeseries/all', filename)))[fork_idx]

db = simpleJDB.database('benchmark_database')


def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    # print('Artist picked:', event.artist
    # print('{} vertices picked'.format(len(ind)))
    # print('Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
    # print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    # print('Data point:', x[ind[0]], y[ind[0]])
    # print()
    print(int(xmouse))

    db.setkey(f'{filename}_{fork_idx}', int(xmouse))
    db.commit()


if __name__ == '__main__':
    for filename in os.listdir(f'{data_dir}/timeseries/all'):
        timeseries = json.load(open(os.path.join(data_dir, 'timeseries/all', filename)))
        for fork_idx, fork in enumerate(timeseries):
            print(f'{filename}: {fork_idx}')

            fork = timeseries[fork_idx]
            fig, ax = plt.subplots()
            tolerance = 10 # points
            ax.plot(range(len(fork)), fork, 'ro-', picker=tolerance)
            fig.canvas.callbacks.connect('pick_event', on_pick)
            plt.show()