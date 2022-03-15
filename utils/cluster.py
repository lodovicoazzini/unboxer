import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from kneebow.rotor import Rotor
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


def cluster_distance_matrix(distance_matrix, model=DBSCAN(metric='precomputed'), plot=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param distance_matrix: The distance matrix to use for the clusters
    :param model: The model to use to cluster the data
    :param plot: Whether to visualize the clusters
    :return: The clusters' labels (+ fig, ax if plot == True)
    """
    # cluster the distance matrix
    clusters = model.fit_predict(distance_matrix)

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
                label=label
            )
            plt.legend()

        return clusters, fig, ax

    return clusters


def get_elbow_point(distance_matrix, plot=False, smoothing_factor=.1, degree=3, l_quantile=0, h_quantile=1):
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
    rotor = Rotor()
    rotor.fit_rotate(data_smooth)
    knee_point = distances[rotor.get_elbow_index()]

    # plot the data
    if plot:
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.despine()
        ax.plot(data[:, 0], data[:, 1], color='red')
        ax.plot(data_smooth[:, 0], data_smooth[:, 1], color='blue')

        return knee_point, fig, ax

    return knee_point
