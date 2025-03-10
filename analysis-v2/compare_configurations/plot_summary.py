#!/usr/bin/env python3

"""This script plots the summarization data from different parameter settings."""

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Function to generate the labels and values based on a descriptor change
def get_variations(fixed_position, fixed_values, data,param):
    variations = []
    for item in data:
        descriptor = item[0].split('_')
        # Check if all but the varying position are the same
        if all(descriptor[i] == fixed_values[i] for i in range(len(fixed_values)) if i != fixed_position):
            is_default = item[0] == default_label
            variations.append((descriptor[fixed_position], item[param], is_default, item[0]))

    # Sort variations by the varying descriptor (numerically)
    variations.sort(key=lambda x: float(x[0]))  # Sort by the changing descriptor (e.g., '120', '100')
    return variations

res = []
for f in os.listdir('summaries_new'):
    params = f.rsplit('.', 1)[0].split('_', 1)[1]
    with open(f'summaries_new/{f}') as summary:
        for line in summary:
            if 'Clustered diffs' in line:
                tmp = [e.replace(',', '') for e in line.split()]
                std, mean = float(tmp[4]), float(tmp[7])
                res.append((params, std, mean))


default_label = '100_500_4.5_80_0.85'
default_value = next(item[1] for item in res if item[0] == default_label)

param_names = ['outliers_window_size', 'prob_win_size', 't_crit', 'step_win_size', 'prob_threshold']



# Descriptor positions:
# [first, second, third, fourth, fifth]
fixed_values = ['100', '500', '4', '80', '0.85']
# Define the grid layout
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # Same height for both rows

# Plot the first row (3 subplots)
for i in [0,1]:
    ax = fig.add_subplot(gs[0, i])
    variations = get_variations(i, fixed_values, res,1)

    labels = [v[3] for v in variations]  # The entire label string
    values = [v[1] for v in variations]  # Values
    colors = ['navy' if v[2] else 'yellowgreen' for v in variations]  # Highlight default

    # Plotting
    ax.bar(labels, values, color=colors)
    ax.set_title(f'Varying {param_names[i]}')
    ax.tick_params(axis='x', rotation=90)

# Plot the centered bottom row with 2 subplots in the middle columns
for i in [2,4]:
    ax = fig.add_subplot(gs[1, int(i/2)-1])  # Shifted to middle columns
    variations = get_variations(i, fixed_values, res,1)

    labels = [v[3] for v in variations]  # The entire label string
    values = [v[1] for v in variations]  # Values
    colors = ['navy' if v[2] else 'yellowgreen' for v in variations]  # Highlight default

    # Plotting
    ax.bar(labels, values, color=colors)
    ax.set_title(f'Varying {param_names[i]}')
    ax.tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig('barplots/summary_std.png')
plt.close()




exit(-1)

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
