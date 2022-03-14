import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.cluster import DBSCAN


def heatmap_distance(lhs, rhs):
    """
    Compute the distance between two heatmaps
    :param lhs: The first heatmap
    :param rhs: The second heatmap
    :return: The distance between the two heatmaps
    """
    # convert te contributions to grayscale if necessary
    lhs = tf.image.rgb_to_grayscale(lhs) if len(lhs.shape) > 2 and lhs.shape[-1] > 1 else lhs
    rhs = tf.image.rgb_to_grayscale(rhs) if len(rhs.shape) > 2 and rhs.shape[-1] > 1 else rhs
    # reduce to a 2D array
    lhs = np.squeeze(lhs)
    rhs = np.squeeze(rhs)

    # compute and return the distance between teh two matrices
    return np.sqrt(np.sum((lhs - rhs) ** 2))


def heatmap_distances(heatmaps, dist_func=heatmap_distance):
    """
    Compute the distance matrix for a list of heatmaps
    :param heatmaps: The list of heatmaps
    :param dist_func: The distance function to use (default is Euclidean distance)
    :return: The distance matrix NxN for the N heatmaps
    """
    # initialize the distances matrix to 0
    num_heatmaps = len(heatmaps)
    heatmaps_distances = np.zeros(shape=(num_heatmaps, num_heatmaps))
    # compute the heatmaps distances above the diagonal
    for row in range(0, num_heatmaps - 1):
        for col in range(row + 1, num_heatmaps):
            heatmaps_distances[row][col] = dist_func(heatmaps[row], heatmaps[col])
    # complete the rest of the distance matrix by summing it to its transposed
    heatmaps_distances = heatmaps_distances + heatmaps_distances.T

    return heatmaps_distances


def cluster_distance_matrix(distance_matrix):
    db_scan = DBSCAN(min_samples=6, eps=.7, metric='precomputed').fit(distance_matrix)

    labels = db_scan.labels_
    # associate one cluster label for each color
    colors = [plt.cm.Spectral(val) for val in np.linspace(0, 1, len(set(labels)))]
    colored_labels = dict(zip(labels, colors))
    # associate black with noise
    colored_labels[-1] = (0, 0, 0, 1)

    # get the mask for the core samples
    # initialize all to zeros
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    # set to true the core ones
    core_samples_mask[db_scan.core_sample_indices_] = True

    # prepare the general figure
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.despine()

    for label, color in colored_labels.items():
        # filter for the element belonging to the label
        labels_mask = labels == label
        # plot the core samples
        core_points = distance_matrix[labels_mask & core_samples_mask]
        plt.plot(
            core_points[:, 0], core_points[:, 1],
            'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=14
        )
        # plot the non-core samples
        non_core_points = distance_matrix[labels_mask & ~core_samples_mask]
        plt.plot(
            non_core_points[:, 0], non_core_points[:, 1],
            'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=6
        )
