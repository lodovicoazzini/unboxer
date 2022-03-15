import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from kneed import KneeLocator
from scipy.interpolate import UnivariateSpline
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


def cluster_distance_matrix(distance_matrix, min_samples=5, eps=.5, plot=False, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param distance_matrix: The distance matrix to use for the clusters
    :param min_samples: The minimum number of points to make a cluster
    :param eps: The distance for two points to be neighbors
    :param plot: Whether to visualize the clusters
    :param verbose: Print information about the clusters
    :return: The clusters' labels (+ fig, ax if plot == True)
    """
    # generate the clusters
    clusters = DBSCAN(metric='precomputed', min_samples=min_samples, eps=eps).fit_predict(distance_matrix)

    if verbose:
        print(f"""
Silhouette score = {silhouette_score(distance_matrix, clusters)}
    min_samples = {min_samples}
    eps = {eps} 
        """)
    if plot:
        # prepare the general figure
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.despine()

        # associate one cluster label for each color
        colors = [plt.cm.Spectral(val) for val in np.linspace(0, 1, len(set(clusters)))]
        colors_dict = {
            label: colors[idx]
            for idx, label in enumerate(set(clusters))
        }
        # associate black with noise
        colors_dict[-1] = (0, 0, 0, 1)

        for label, color in colors_dict.items():
            # plot the core samples
            points = distance_matrix[clusters == label]
            ax.plot(
                points[:, 0], points[:, 1],
                'x' if label == -1 else 'o', markerfacecolor=tuple(color), markeredgecolor='k',
                label='no cluster' if label == -1 else label
            )
            plt.legend()
            ax.set_xlim((0, ax.get_xlim()[1]))
            ax.set_ylim((0, ax.get_ylim()[1]))

        return clusters, fig, ax

    return clusters


def cluster_distance_matrix_optimize(distance_matrix, min_samples_list, eps=.5, plot=False, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param distance_matrix: The distance matrix to use for the clusters
    :param min_samples_list: The list of values to try for the minimum number of points in a cluster
    :param eps: The distance for two points to be neighbors
    :param plot: Whether to visualize the clusters
    :param verbose: Print information about the clusters
    :return: The clusters' labels (+ fig, ax if plot == True)
    """
    # compute the silhouette scores for the clustering configurations
    silhouette_scores = np.array([])
    for min_samples in min_samples_list:
        config_clusters = cluster_distance_matrix(distance_matrix, min_samples=min_samples, eps=eps)
        silhouette_scores = np.append(silhouette_scores, silhouette_score(distance_matrix, config_clusters))

    # get value corresponding to the minimum silhouette score
    min_samples = min_samples_list[np.argmax(silhouette_scores)]

    return cluster_distance_matrix(distance_matrix, min_samples=min_samples, eps=eps, plot=plot, verbose=verbose)


def get_elbow_point(distance_matrix, plot=False, smoothing_factor=0, degree=3, l_quantile=0, h_quantile=1):
    """
    Find the point of maximum curvature in the distances between elements
    :param distance_matrix: The matrix encoding the distances between the elements
    :param plot: Whether to plot the data
    :param smoothing_factor: Positive float determining the amount of applied to the data
    :param degree: The degree of the smoothing function
    :param l_quantile: The outliers filtered out on the left
    :param h_quantile: The outliers filtered out on the right
    :return: The distance corresponding to the maximum curvature (+ fig, ax if plot == True)
    """
    # initialize the matrix of the distances to the distance matrix
    triangular_distances = copy.deepcopy(distance_matrix)
    # set the lower triangular matrix to infinity
    triangular_distances[np.tril_indices(distance_matrix.shape[0])] = np.nan
    # order each row of the matrix
    triangular_distances = np.sort(triangular_distances)
    # remove the last row as all the distances have already been considered
    triangular_distances = triangular_distances[:-1, :]

    # start by considering the distances to the closest neighbor
    # skipping the last row as all the distances have already been considered (would be nan)
    distances = triangular_distances[:, 0]
    # # for each k consider the average distance to the k closest neighbors
    for k in range(2, distance_matrix.shape[0]):
        # slice the matrix to keep only the k closest neighbors to each point
        sliced = triangular_distances[:, :k]
        # compute the average
        avg_distances = np.nanmean(sliced, axis=1)
        distances = np.append(distances, avg_distances)

    # sort the distances
    distances = np.sort(distances)
    # filter out the outliers
    if l_quantile != 0:
        distances = distances[distances > np.quantile(distances, l_quantile)]
    if h_quantile != 1:
        distances = distances[distances < np.quantile(distances, h_quantile)]

    # get the data to plot
    x = np.arange(0, distances.shape[0], 1)
    data = np.array(list(zip(x, distances)))

    # compute the data approximation to smooth the curve
    spl = UnivariateSpline(x, distances, s=smoothing_factor, k=degree)

    data_smooth = np.array(list(zip(x, spl(x))))

    # find the point of maximum curvature
    knee_locator = KneeLocator(data_smooth[:, 0], data_smooth[:, 1])

    # plot the data
    if plot:
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.despine()
        # plot the original and smoothed data
        ax.plot(data[:, 0], data[:, 1], color='black', label='original')
        ax.plot(data_smooth[:, 0], data_smooth[:, 1], color='black', linestyle='dashed', label='smoothed')
        # show the intercept for the knee point
        ax.plot(
            [0, knee_locator.knee, knee_locator.knee],
            [knee_locator.knee_y, knee_locator.knee_y, 0],
            color='red', linestyle='dotted', label='knee point'
        )
        # add the tick for the elbow point
        plt.yticks(list(plt.yticks()[0]) + round(knee_locator.knee_y, 3))
        ax.set_xlim((0, np.max(x)))
        ax.set_ylim((0, np.max(data_smooth[:, 1])))
        # show the legend
        plt.legend()

        return knee_locator.knee_y, fig, ax

    return knee_locator.knee_y


def silhouette_score(distance_matrix, clusters):
    # compute the silhouette score for each point
    silhouette_scores = np.array([])
    for idx in range(0, distance_matrix.shape[0]):
        # compute the cohesion for the point (average intra-cluster distance)
        # find the label for the current point
        point_label = clusters[idx]
        # create a dataframe for the distances of the point
        point_distances = pd.DataFrame({
            'label': clusters,
            'distance': distance_matrix[1, :]
        }).reset_index()
        # filter for the points in the same clusters
        same_cluster_distances = point_distances[
            (point_distances['label'] == point_label) &
            (point_distances['index'] != idx)
            ]['distance'].values
        # compute the average intra distance, if empty (singleton) -> SI(i) = 1
        if point_label == -1 or len(same_cluster_distances) == 0:
            silhouette_scores = np.append(
                silhouette_scores,
                1
            )
            continue
        avg_intra_dist = np.average(same_cluster_distances) if len(same_cluster_distances) > 0 else 1
        # compute the separation for the point (minimum average inter-cluster distance)
        # initialize the minimum inter distance to infinite
        min_inter_dist = np.inf
        other_labels = set(clusters[clusters != point_label])
        for other_label in other_labels:
            # compute the average distance with the points of the cluster
            # compute the average distance with the points of the cluster
            other_cluster_distances = point_distances[point_distances['label'] == other_label]['distance'].values
            avg_other_dist = np.mean(other_cluster_distances)
            # update the value of the minimum distance
            min_inter_dist = avg_other_dist if avg_other_dist < min_inter_dist else min_inter_dist
        # append the silhouette score for the point to the list
        si_point = (min_inter_dist - avg_intra_dist) / max(avg_intra_dist, min_inter_dist)
        silhouette_scores = np.append(
            silhouette_scores,
            si_point
        )
    # compute the silhouette score for the clustering configuration (average silhouette for the points)
    return np.mean(silhouette_scores)
