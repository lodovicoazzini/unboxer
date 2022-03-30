import copy

import numpy as np
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


def get_elbow_point(dist_matrix, plot=False, smoothing_factor=0, degree=3, l_quantile=0, h_quantile=1, curve='convex'):
    """
    Find the point of maximum curvature in the distances between elements
    :param dist_matrix: The matrix encoding the distances between the elements
    :param plot: Whether to plot the data
    :param smoothing_factor: Positive float determining the amount of applied to the data
    :param degree: The degree of the smoothing function
    :param l_quantile: The outliers filtered out on the left
    :param h_quantile: The outliers filtered out on the right
    :param curve: `concave` -> knee, `convex` -> elbows
    :return: The distance corresponding to the maximum curvature (+ fig, ax if plot == True)
    """
    # initialize the matrix of the distances to the distance matrix
    triangular_distances = copy.deepcopy(dist_matrix)
    # set the lower triangular matrix to infinity
    triangular_distances[np.tril_indices(dist_matrix.shape[0])] = np.nan
    # order each row of the matrix
    triangular_distances = np.sort(triangular_distances)
    # remove the last row as all the distances have already been considered
    triangular_distances = triangular_distances[:-1, :]

    # start by considering the distances to the closest neighbor
    # skipping the last row as all the distances have already been considered (would be nan)
    distances = triangular_distances[:, 0]
    # # for each k consider the average distance to the k closest neighbors
    for k in range(2, dist_matrix.shape[0]):
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
    knee_locator = KneeLocator(data_smooth[:, 0], data_smooth[:, 1], curve=curve)
    p_x = knee_locator.knee
    p_y = knee_locator.knee_y

    # plot the data
    if plot:
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.despine()
        # plot the original and smoothed data
        ax.plot(data[:, 0], data[:, 1], color='black', label='original')
        ax.plot(data_smooth[:, 0], data_smooth[:, 1], color='black', linestyle='dashed', label='smoothed')
        # show the intercept for the knee point
        ax.plot(
            [0, p_x, p_x],
            [p_y, p_y, 0],
            color='red', linestyle='dotted', label='knee point'
        )
        # add the tick for the elbow point
        plt.yticks(list(plt.yticks()[0]) + round(p_y, 3))
        ax.set_xlim((0, np.max(x)))
        ax.set_ylim((0, np.max(data_smooth[:, 1])))
        # show the legend
        plt.legend()

        return p_y, fig, ax

    return p_y
