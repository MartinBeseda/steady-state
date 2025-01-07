#!/usr/bin/env python3

"""Utility script for plotting of results of different approaches dividing timeseries into similarity clasess"""

import matplotlib.pyplot as plt
import numpy as np

# Names of different methods
method_names = ['Raw data + MAE', 'Raw data + MSE',
                'Shifted data + MAE', 'Shifted data + MSE',
                'Normalized data + MAE', 'Normalized data + MSE',
                'Normalized + smooth (SG 50) + MAE', 'Normalized + smooth (SG 50) + MSE',
                'Normalized + smooth (SG 100) + MAE', 'Normalized + smooth (SG 100) + MSE',
                'Normalized + smooth (SG 150) + MAE', 'Normalized + smooth (SG 150) + MSE',
                'Normalized + smooth (Med 50) + MAE', 'Normalized + smooth (Med 50) + MSE',
                'Normalized + smooth (Med 100) + MAE', 'Normalized + smooth (Med 100) + MSE',
                'Normalized + smooth (Med 150) + MAE', 'Normalized + smooth (Med 150) + MSE']

# Number of detected classes
# TODO maybe +1?
no_classes = [201, 200, 194, 161, 34, 36, 45, 67, 56, 73, 59, 78, 73, 82, 85, 91, 89, 91]

# Number of "outlier" curves
no_outliers = np.array([2225, 3197, 2490, 3395, 5451, 5446, 5369, 5158, 5256, 5067, 5205, 5010, 5058, 5052, 4944, 4943,
                        4907, 4945])

# print(len(method_names))
# print(len(no_classes))
# print(len(no_outliers))

plt.figure()
plt.title('Number of classes detected')
plt.bar(range(len(method_names)), no_classes)
plt.xticks(range(len(method_names)), method_names, rotation=90)
plt.show()

plt.figure()
plt.title('Number of detected "outlier" curves')
plt.bar(range(len(method_names)), no_outliers)
plt.xticks(range(len(method_names)), method_names, rotation=90)
plt.show()

plt.figure()
plt.title('Percentage of detected "outlier" curves')
plt.bar(range(len(method_names)), (no_outliers / 5860)*100)
plt.xticks(range(len(method_names)), method_names, rotation=90)
plt.show()
