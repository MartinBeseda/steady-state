#!/usr/bin/env python3

"""The script to compare manual detection of steady states indices."""

import json
import simpleJDB
import matplotlib.pyplot as plt


# Load the data
data_vittorio = simpleJDB.database('benchmark_database_steady_indices_vittorio')
data_michele = simpleJDB.database('benchmark_database_steady_indices_michele')
data_daniele = simpleJDB.database('benchmark_database_steady_indices_daniele')

data_sum = {}

for i, e in enumerate(data_vittorio.data['main']):
    print(e['value'], data_michele.getkey(e['keyname']), data_daniele.getkey(e['keyname']))

    # Remove the bad suffix
    fname = e['keyname'].rsplit('_', 1)[0]
    data_sum[e['keyname']] = {'idxs': [e['value'],
                                       data_michele.getkey(e['keyname']),
                                       data_daniele.getkey(e['keyname'])],
                              'series': json.load(open(f'data_st/{fname}'))}


    # # Create plots
    # plt.figure()
    # plt.plot(data_sum[e['keyname']]['series'])
    # plt.vlines([data_sum[e['keyname']]['idxs']],
    #            min(data_sum[e['keyname']]['series']),
    #            max(data_sum[e['keyname']]['series']),
    #            colors=['red', 'green', 'blue'])
    # plt.title(fname.replace('#', ' '), wrap=True)
    # plt.savefig(f'plots/plot_{i}.png')
    # plt.close()

# Create histograms
diffs_1_2 = []
diffs_2_3 = []
diffs_1_3 = []
for k, e in data_sum.items():
    diffs_1_2.append(e['idxs'][0] - e['idxs'][1])
    diffs_2_3.append(e['idxs'][1] - e['idxs'][2])
    diffs_1_3.append(e['idxs'][0] - e['idxs'][2])

# plt.figure()
# plt.hist(diffs_1_2)
# plt.savefig('histograms/diffs_1_2.png')
# plt.title('Differences idx 0 - idx 1')
# plt.close()
#
# plt.figure()
# plt.hist(diffs_2_3)
# plt.savefig('histograms/diffs_2_3.png')
# plt.title('Differences idx 1 - idx 2')
# plt.close()
#
# plt.figure()
# plt.hist(diffs_1_3)
# plt.savefig('histograms/diffs_1_3.png')
# plt.title('Differences idx 1 - idx 3')
# plt.close()
#
# plt.figure()
# plt.hist([diffs_1_2, diffs_1_3, diffs_2_3], label=['0-1', '0-2', '1-2'])
# plt.title('Differences (0 - 1, 0 - 2, 1 - 2)')
# plt.legend()
# plt.savefig('histograms/all_diffs.png')
# plt.close()

plt.figure()
plt.hist([diffs_1_2, diffs_1_3, diffs_2_3], label=['0-1', '0-2', '1-2'], cumulative=True)
plt.title('Cumulative differences (0 - 1, 0 - 2, 1 - 2)')
plt.legend()
plt.savefig('histograms/all_diffs_cumulative.png')
plt.close()

# # Create boxplots of differences' variance
# plt.boxplot([diffs_1_2, diffs_1_3, diffs_2_3], tick_labels=['0-1', '0-2', '1-2'])
# plt.savefig('boxplots/boxplot_all_diffs.png')
# plt.close()
