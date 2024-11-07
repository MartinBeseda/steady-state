#!/usr/bin/env python3

"""The script to compare manual detection of steady states indices."""

import json
import numpy as np
import simpleJDB
import matplotlib.pyplot as plt
import itertools
import sklearn.cluster
from scipy.stats import median_abs_deviation

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
extreme_diffs = []
diff_stds = []
diff_mads = []
diff_max = []
no_cluster_dbscan_eps100 = 0
no_cluster_dbscan_eps50 = 0
no_cluster_dbscan_eps30 = 0
no_cluster_dbscan_eps20 = 0

dbscan_eps100 = sklearn.cluster.DBSCAN(eps=100, min_samples=3)
dbscan_eps50 = sklearn.cluster.DBSCAN(eps=50, min_samples=3)
dbscan_eps30 = sklearn.cluster.DBSCAN(eps=30, min_samples=3)
dbscan_eps20 = sklearn.cluster.DBSCAN(eps=20, min_samples=3)

for k, e in data_sum.items():
    for idx_pair in data_idxs:
        diffs[f'{idx_pair[0]}-{idx_pair[1]}'].append(e['idxs'][idx_pair[0]] - e['idxs'][idx_pair[1]])

    extreme_diffs.append(max(e['idxs']) - min(e['idxs']))
    diff_stds.append(np.std(e['idxs']))
    diff_mads.append(median_abs_deviation(e['idxs']))
    diff_max.append(max(e['idxs']))

    # Check, if there are any indices clustered
    no_cluster_dbscan_eps100 += (dbscan_eps100.fit(np.array(e['idxs']).reshape(-1, 1)).labels_ > -1).any()
    no_cluster_dbscan_eps50 += (dbscan_eps50.fit(np.array(e['idxs']).reshape(-1, 1)).labels_ > -1).any()
    no_cluster_dbscan_eps30 += (dbscan_eps30.fit(np.array(e['idxs']).reshape(-1, 1)).labels_ > -1).any()
    no_cluster_dbscan_eps20 += (dbscan_eps20.fit(np.array(e['idxs']).reshape(-1, 1)).labels_ > -1).any()


for k, e in diffs.items():
    plt.figure()
    plt.title(f'Differences idx {k}')
    plt.hist(e, bins=30)
    plt.savefig(f'histograms/diffs_{k}.png')
    plt.close()

plt.figure()
plt.hist(diffs.values(), label=diffs.keys(), bins=15)
plt.title('SSD Differences')
plt.legend()
plt.savefig('histograms/all_diffs.png')
plt.close()

plt.figure()
plt.hist(diffs.values(), label=diffs.keys(), cumulative=True, bins=15)
plt.title('SSD Cumulative differences')
plt.legend()
plt.savefig('histograms/all_diffs_cumulative.png')
plt.close()


plt.figure()
plt.hist(extreme_diffs, bins=30)
plt.title('Rightmost - leftmost idxs differences')
plt.savefig('histograms/extreme_idx_diffs.png')
plt.close()

plt.figure()
plt.hist(diff_stds, bins=30)
plt.title('SSD indexing standard deviations')
plt.savefig('histograms/idx_stds.png')
plt.close()

plt.figure()
plt.hist(diff_mads, bins=30)
plt.title('Median absolute deviations')
plt.savefig('histograms/idx_mads.png')
plt.close()

plt.figure()
plt.hist(diff_max, bins=30)
plt.title('Rightmost indices')
plt.savefig('histograms/idx_max.png')
plt.close()

# Create bar plots for numbers of clustered indices via DBSCAN
plt.figure()
plt.bar(range(1, 5),
        [no_cluster_dbscan_eps20, no_cluster_dbscan_eps30, no_cluster_dbscan_eps50, no_cluster_dbscan_eps100])
plt.title('Number of series with clustered indices via DBSCAN')
plt.xticks(range(1, 5), ['20', '30', '50', '100'])
plt.xlabel('DBSCAN eps')
plt.savefig('barplots/no_series_with_clusters.png')
plt.close()

# # Create boxplots of differences' variance
# plt.boxplot(diffs.values(), tick_labels=diffs.keys())
# plt.savefig('boxplots/boxplot_all_diffs.png')
# plt.close()
