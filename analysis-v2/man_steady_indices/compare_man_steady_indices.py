#!/usr/bin/env python3

"""The script to compare manual detection of steady states indices."""

import json
import simpleJDB
import matplotlib.pyplot as plt
import itertools


# Load the data
data_vittorio = simpleJDB.database('benchmark_database_steady_indices_vittorio')
data_michele = simpleJDB.database('benchmark_database_steady_indices_michele')
data_daniele = simpleJDB.database('benchmark_database_steady_indices_daniele')
data_luca = simpleJDB.database('benchmark_database_steady_indices_luca')
data_martin = simpleJDB.database('benchmark_database_steady_indices_martin')

data_sum = {}

for i, e in enumerate(data_vittorio.data['main']):
    print(e['value'], data_michele.getkey(e['keyname']), data_daniele.getkey(e['keyname']),
          data_luca.getkey(e['keyname']), data_martin.getkey(e['keyname']))

    # Remove the bad suffix
    fname = e['keyname'].rsplit('_', 1)[0]
    data_sum[e['keyname']] = {'idxs': [e['value'],
                                       data_michele.getkey(e['keyname']),
                                       data_daniele.getkey(e['keyname']),
                                       data_luca.getkey(e['keyname']),
                                       data_martin.getkey(e['keyname'])],
                              'series': json.load(open(f'data_st/{fname}'))}

    # # Create plots
    # plt.figure()
    # plt.plot(data_sum[e['keyname']]['series'])
    # plt.vlines([data_sum[e['keyname']]['idxs']],
    #            min(data_sum[e['keyname']]['series']),
    #            max(data_sum[e['keyname']]['series']),
    #            colors=['red', 'green', 'blue', 'yellow', 'orange'])
    # plt.title(fname.replace('#', ' '), wrap=True)
    # plt.savefig(f'plots/plot_{i}.png')
    # plt.close()

# Data indices
data_idxs = list(itertools.combinations(range(5), 2))

# Create histograms
diffs = {f'{e[0]}-{e[1]}': [] for e in data_idxs}

for k, e in data_sum.items():
    for idx_pair in data_idxs:
        diffs[f'{idx_pair[0]}-{idx_pair[1]}'].append(e['idxs'][idx_pair[0]] - e['idxs'][idx_pair[1]])

for k, e in diffs.items():
    plt.figure()
    plt.title(f'Differences idx {k}')
    plt.hist(e)
    plt.savefig(f'histograms/diffs_{k}.png')
    plt.close()

plt.figure()
plt.hist(diffs.values(), label=diffs.keys())
plt.title('SSD Differences')
plt.legend()
plt.savefig('histograms/all_diffs.png')
plt.close()

plt.figure()
plt.hist(diffs.values(), label=diffs.keys(), cumulative=True)
plt.title('SSD Cumulative differences')
plt.legend()
plt.savefig('histograms/all_diffs_cumulative.png')
plt.close()

# Create boxplots of differences' variance
plt.boxplot(diffs.values(), tick_labels=diffs.keys())
plt.savefig('boxplots/boxplot_all_diffs.png')
plt.close()
