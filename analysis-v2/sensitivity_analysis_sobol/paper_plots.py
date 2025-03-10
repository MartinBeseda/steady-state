#!/usr/bin/env python3

"""The utility script for creating plots for a paper."""

from numpy import array, nan
import matplotlib.pyplot as plt

# Results obtained from the Sobol's analysis
results = {'S1': array([0.19718534, 0.00631332, 0.15670927, 0.1107751 , 0.15383209]),
           'S1_conf': array([0.04270919, 0.00693686, 0.04442212, 0.03822329, 0.03632412]),
           'ST': array([0.46349316, 0.01394031, 0.46206506, 0.35747568, 0.26101527]),
           'ST_conf': array([0.04872086, 0.00197441, 0.05458941, 0.04887354, 0.02257135]),
           'S2': array([[nan, -0.02788179,  0.07228866,  0.03960758,  0.00295189],
                        [nan,         nan, -0.00063416,  0.00097269,  0.00471581],
                        [nan,         nan,         nan,  0.13214761,  0.01748042],
                        [nan,         nan,         nan,         nan,  0.01248535],
                        [nan,         nan,         nan,         nan,         nan]]),
           'S2_conf': array([[nan, 0.06506981, 0.06946727, 0.06807653, 0.06549036],
                             [nan,        nan, 0.01147942, 0.01074953, 0.01052957],
                             [nan,        nan,        nan, 0.0931521 , 0.06706305],
                             [nan,        nan,        nan,        nan, 0.05786848],
                             [nan,        nan,        nan,        nan,        nan]])}

x_points = (0, 0.2, 0.4, 0.6, 0.8)#range(len(results['S1']))#(0, 0.3, 0.6, 0.9, 1.2)

# Barplot of 1st order indices
plt.figure(figsize=(2.3, 4))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.bar(x_points,
        results['S1'],
        yerr=results['S1_conf'],
        alpha=0.7,
        align='center',
        color='#4c72b0',
        edgecolor='#2A4D69',
        width=0.1,
        capsize=4)
plt.xticks(x_points,
           ('prob_win_size', 'step_win_size', 't_crit', 'prob_threshold', 'outlier_win_size'),
           rotation=90,
           fontsize=14)
plt.ylim(0, 0.25)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.savefig('barplots/s1.png')
plt.close()

# Barplot of total order indices
plt.figure(figsize=(2.3, 4))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.bar(x_points,
        results['ST'],
        yerr=results['ST_conf'],
        alpha=0.7,
        align='center',
        color='#4c72b0',
        edgecolor='#2A4D69',
        width=0.1,
        capsize=4)
plt.xticks(x_points,
           ('prob_win_size', 'step_win_size', 't_crit', 'prob_threshold', 'outlier_win_size'),
           rotation=90,
           fontsize=14)
plt.ylim(0, 0.6)
plt.yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.savefig('barplots/st.png')
plt.close()


# Barplot of 2nd order indices
x_points = (0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8)
plt.figure(figsize=(3.0, 6))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.bar(x_points,
        (-0.02788179,  0.07228866,  0.03960758,  0.00295189, -0.00063416,  0.00097269,  0.00471581, 0.13214761,
         0.01748042, 0.05786848),
        yerr=(0.06506981, 0.06946727, 0.06807653, 0.06549036, 0.01147942, 0.01074953, 0.01052957, 0.0931521 , 0.06706305, 0.05786848),
        alpha=0.7,
        align='center',
        color='#4c72b0',
        edgecolor='#2A4D69',
        width=0.1,
        capsize=4)
plt.xticks(x_points,
           ('(prob_win_size, step_win_size)',
            '(prob_win_size, t_crit)',
            '(prob_win_size, prob_threshold)',
            '(prob_win_size, outlier_win_size)',
            '(step_win_size, t_crit)',
            '(step_win_size, prob_threshold)',
            '(step_win_size, outlier_win_size)',
            '(t_crit, prob_threshold)',
            '(t_crit, outlier_win_size)',
            '(prob_threshold, outlier_win_size)'),
           rotation=90,
           fontsize=14)
plt.ylim(-0.1, 0.25)
plt.yticks((-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25))
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout()
plt.savefig('barplots/s2.png')
plt.close()