#!/usr/bin/env python3

"""Utility script for assigning timeseries into classes based on their similarity"""
import numpy as np
import scipy.stats
import similaritymeasures
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN
import os
import json
from scipy.ndimage import median_filter
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.cluster import KMeans

data_dir = '../data'


if __name__ == '__main__':
    # Load all timeseries
    print('Loading data')
    filenames = [fname for fname in os.listdir(f'{data_dir}/timeseries/all')][:20]
    print(f'Number of filenames: {len(filenames)}')

    # forknames = [f'fname_{i}' for fname in os.listdir(f'{data_dir}/timeseries/all') for i in range(10)]

    filenames = filenames[:-1]

    data = []
    og_data = []
    for filename in filenames:
        for fork in json.load(open(os.path.join(data_dir, 'timeseries/all', filename))):
            # Smoothen the fork via median filter
            # fork = median_filter(fork, size=200)
            og_data.append(fork)

            # Transform the timeseries
            yf = rfft(fork)
            xf = rfftfreq(len(fork))
            # new_yf = irfft(yf)

            # plt.figure()
            # plt.plot(xf, np.abs(yf))
            # plt.show()

            data.append(np.abs(yf))
        # exit(-1)

    # plt.figure()
    # for fork in og_data[60:70]:
    #     plt.plot(fork)
    # plt.show()
    # exit(-1)

    # Perform HDBSCAN
    print('Performing HDBSCAN with MAE:')
    hdb = HDBSCAN(metric=similaritymeasures.mae,
                  allow_single_cluster=True, store_centers='centroid').fit(data)
    classification = hdb.labels_
    centroids = hdb.centroids_
    print(classification)
    print(max(classification), min(classification))
    print(len(classification))
    print(sum(classification == -1))

    for j in range(max(classification)):
        for i, fork in enumerate(og_data):
            if classification[i] == j:
                plt.plot(fork)
        plt.title(f'Cluster no outs {j}')
        plt.savefig(f'cluster_no_outs_{j}')

        plt.close()

    # Indices of outliers
    outlier_ids = np.where(classification == -1)[0]

    # Run K-means to classify outliers to their respective closest centroids identified via HDBSCAN
    kmeans = KMeans(n_clusters=len(centroids), init=centroids).fit(np.array([data[idx] for idx in outlier_ids]))
    kmeans_outlier_labels = kmeans.labels_
    print(kmeans_outlier_labels)

    # Rewrite indices of outliers with new ones obtained via K-means
    classification[outlier_ids] = kmeans_outlier_labels

    print('new classification')
    print(classification)

    # exit(-1)

    for j in range(max(classification)):
        for i, fork in enumerate(og_data):
            if classification[i] == j:
                plt.plot(fork)
        plt.title(f'Cluster {j}')
        plt.savefig(f'cluster_{j}')

        plt.close()

    exit(-1)

    probs = hdb.probabilities_
    plt.figure()
    plt.hist(probs,bins=10)
    plt.title(f'Strength with which each sample is a member of its assigned cluster')
    plt.savefig('probs_after_first')
    plt.close()

    plt.figure()
    plt.hist(classification,bins=max(classification)+1)
    plt.title(f'Cluster distribution')
    plt.savefig(f'classification_one')
    plt.close()

    not_class = []
    not_class_og = []
    for i,x in enumerate(data):
        if classification[i] ==-1:
            not_class.append(x)
            not_class_og.append(og_data[i])
    hdb = HDBSCAN(min_cluster_size=5, metric=similaritymeasures.mae, allow_single_cluster=True).fit(not_class)
    classification = hdb.labels_
    # probs = hdb.probabilities_
    print(classification)
    # print(type(classification))
    print(max(classification), min(classification))
    print(len(classification))
    print(sum(classification == -1))

    # Run K-means


    # Plot randomly selected samples from clusters
    for j in range(10):  # range(max(classification)):
        for i, fork in enumerate(not_class_og):
            if classification[i] == j:
                plt.plot(fork)
        plt.title(f'Cluster in second run {j}')
        plt.savefig(f'cluster_second_{j}')

        plt.close()

    probs = hdb.probabilities_
    plt.figure()
    plt.hist(probs, bins=10)
    plt.title(f'Strength with which each sample is a member of its assigned cluster')
    plt.savefig('probs_after_second')
    plt.close()

    plt.figure()
    plt.hist(classification, bins=max(classification) + 1)
    plt.title(f'Cluster distribution in second run')
    plt.savefig(f'classification_second')
    plt.close()
    exit(-1)
