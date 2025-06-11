#!/usr/bin/env python3

"""Utility script for assigning timeseries into classes based on their similarity"""
import numpy as np
import scipy.stats
import similaritymeasures
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN
import os
import json
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.signal import welch

data_dir = '../../data'


if __name__ == '__main__':
    # Load all timeseries
    print('Loading data')
    # no_filenames = 20
    filenames = [fname for fname in os.listdir(f'{data_dir}/timeseries/all')]#[:no_filenames]
    print(f'Number of filenames: {len(filenames)}')

    # forknames = [f'fname_{i}' for fname in os.listdir(f'{data_dir}/timeseries/all') for i in range(10)]

    # filenames = filenames[:-1]

    method = 'fft'
    metrics = 'mae' #'cosine_distances' #'mae'
    dirname = f'{method}_{metrics}_{len(filenames)}'

    if not os.path.exists(f'{dirname}'):
        os.makedirs(f'{dirname}')

    data = []
    og_data = []
    for filename in filenames:
        for fork in json.load(open(os.path.join(data_dir, 'timeseries/all', filename))):
            og_data.append(fork)

            # Transform the timeseries
            yf = np.abs(rfft(fork))[:100]
            xf = rfftfreq(len(fork))

            data.append(np.abs(yf))


    with open(f'{dirname}/log.txt', 'w') as f:
        f.write(f'Total number of forks: {len(og_data)}\n')

        # Perform HDBSCAN
        print('Performing HDBSCAN with MAE:')
        hdb = HDBSCAN(metric=lambda x, y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1)), #similaritymeasures.mae,
                      allow_single_cluster=True, store_centers='centroid').fit(data)
        classification = hdb.labels_
        centroids = hdb.centroids_
        f.write(f'HDBSCAN classification: {classification}\n')
        f.write(f'Number of HDBSCAN-detected clusters: {len(centroids) - 1}\n')
        # print(len(classification))
        f.write(f'Number of HDBSCAN-detected outliers: {sum(classification == -1)}\n')

        # Compute the Silhouette score
        # print(classification != -1)
        # exit(-1)
        data_valid = np.array(data)[classification != -1]
        outlier_rate = np.mean(classification == -1)
        classification_valid = np.array(classification)[classification != -1]
        score_euclidean = silhouette_score(data_valid, classification_valid, metric='euclidean')
        score_cosine = silhouette_score(data_valid, classification_valid, metric='cosine')

        f.write(f'Silhouette score: {score_euclidean} (Euclidean), {score_cosine} (cosine)\n')
        f.write(f'Outlier rate: {outlier_rate}\n')

        for j in range(max(classification)):
            for i, fork in enumerate(og_data):
                if classification[i] == j:
                    plt.plot(fork)
            plt.title(f'Cluster no outs {j}')
            plt.savefig(f'{dirname}/cluster_no_outs_{j}')

            plt.close()

        # Indices of outliers
        outlier_ids = np.where(classification == -1)[0]

        # Run K-means to classify outliers to their respective closest centroids identified via HDBSCAN
        kmeans = KMeans(n_clusters=len(centroids), init=centroids).fit(np.array([data[idx] for idx in outlier_ids]))
        kmeans_outlier_labels = kmeans.labels_
        # print(kmeans_outlier_labels)

        # Rewrite indices of outliers with new ones obtained via K-means
        classification[outlier_ids] = kmeans_outlier_labels

        f.write(f'Classification after K-means: {classification}\n')

    # exit(-1)

    for j in range(max(classification)):
        for i, fork in enumerate(og_data):
            if classification[i] == j:
                plt.plot(fork)
        plt.title(f'Cluster {j}')
        plt.savefig(f'{dirname}/cluster_{j}')

        plt.close()

    # exit(-1)

    probs = hdb.probabilities_
    plt.figure()
    plt.hist(probs,bins=10)
    plt.title(f'Strength with which each sample is a member of its assigned cluster')
    plt.savefig(f'{dirname}/probs_after_clustering.png')
    plt.close()

    plt.figure()
    plt.hist(classification,bins=max(classification)+1)
    plt.title(f'Cluster distribution')
    plt.savefig(f'{dirname}/cluster_distribution.png')
    plt.close()

    exit(-1)

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
