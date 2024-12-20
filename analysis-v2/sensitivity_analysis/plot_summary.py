#!/usr/bin/env python3

"""This script plots the summarization data from different parameter settings."""

import os
import matplotlib.pyplot as plt

res = []
for f in os.listdir('summaries'):
    params = f.rsplit('.', 1)[0].split('_', 1)[1]
    with open(f'summaries/{f}') as summary:
        for line in summary:
            if 'Clustered diffs' in line:
                tmp = [e.replace(',', '') for e in line.split()]
                std, mean = float(tmp[4]), float(tmp[7])
                res.append((params, std, mean))

plt.figure()
plt.title('Summary of STDs of differences w.r.t. different configurations\n(outliers_window_size, prob_win_size, t_crit, '
          'step_win_size, prob_threshold)')
plt.bar(x=range(1, len(res)+1), height=[e[1] for e in res], tick_label=[e[0] for e in res])
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig('barplots/summary_diffs_std.png')

plt.figure()
plt.title('Summary of means of differences w.r.t. different configurations\n(outliers_window_size, prob_win_size, t_crit, '
          'step_win_size, prob_threshold)')
plt.bar(x=range(1, len(res)+1), height=[e[2] for e in res], tick_label=[e[0] for e in res])
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig('barplots/summary_diffs_mean.png')
