import copy

import numpy as np
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


def get_elbow_point(
        dist_matrix: np.ndarray,
        plot: bool = False,
        smoothing_factor: int = 0,
        l_quantile: float = 0.,
        h_quantile: float = 1.,
        curve: str = 'convex'
):
    """
    Find the point of maximum curvature in the distances between elements
    :param dist_matrix: The matrix encoding the distances between the elements
    :param plot: Whether to plot the data
    :param smoothing_factor: Positive float determining the amount of smoothing applied to the data
    :param l_quantile: The outliers filtered out on the left
    :param h_quantile: The outliers filtered out on the right
    :param curve: `concave` -> knee, `convex` -> elbows
    :return: The distance corresponding to the maximum curvature (+ fig, ax if plot == True)
    """
    # Initialize tha matrix of the distances to the distance matrix
    triangular_distances = copy.deepcopy(dist_matrix)
    # Set the lower triangular matrix to infinity
    triangular_distances[np.tril_indices(dist_matrix.shape[0])] = np.nan
    # Order each row of the matrix -> sorted distances for each entry
    triangular_distances = np.sort(triangular_distances)
    # Remove the last row as all the distances are already taken into account in the previous ones
    triangular_distances = triangular_distances[:-1, :]

    # Start by considering the distances to the closest neighbor for each row -> first col
    distances = triangular_distances[:, 0]
    # Consider the distances to the k closest neighbors
    for k in range(2, dist_matrix.shape[0]):
        # Slice the matrix to keep only the k closest neighbors
        sliced = triangular_distances[:, :k]
        # Compute the average for each row
        avg_distances = np.nanmean(sliced, axis=1)
        # Append the values to the distances
        distances = np.append(distances, avg_distances)
    # Sort the distances
    distances = np.sort(distances)

    # Filter the data if quantiles are provided
    if l_quantile != 0:
        distances = distances[distances > np.quantile(distances, l_quantile)]
    if h_quantile != 1:
        distances = distances[distances < np.quantile(distances, h_quantile)]

    # Associate an index to the distances
    x = np.arange(0, distances.shape[0], 1)
    function_data = np.array(list(zip(x, distances)))
    # Compute the approximation of the curve for the distances
    spl = UnivariateSpline(x, distances, s=smoothing_factor, k=3)
    # Associate an index to the smoothed values
    function_smooth_data = np.array(list(zip(x, spl(x))))

    # Find the point of maximum curvature
    knee_locator = KneeLocator(function_smooth_data[:, 0], function_smooth_data[:, 1], curve=curve)
    knee_x = knee_locator.knee
    knee_y = knee_locator.knee_y

    # Plot the data for the knee point
    if plot:
        fig, ax = plt.subplots(figsize=(16, 9))
        # Plot the original and smoothed curves
        ax.plot(function_data[:, 0], function_data[:, 1], color='black', label='original')
        ax.plot(function_smooth_data[:, 0], function_smooth_data[:, 1], color='black', linestyle='dashed',
                label='smoothed')
        # Plot the intercept for the point of maximum curvature
        ax.plot(
            [0, knee_x, knee_x],
            [knee_y, knee_y, 0],
            color='red', linestyle='dotted', label='knee point'
        )
        # Add the ticks for the marked point
        plt.yticks(list(plt.yticks()[0]) + round(knee_y, 3))
        # Style
        ax.set_xlim((0, np.max(x)))
        ax.set_ylim((0, np.max(function_smooth_data[:, 1])))
        sns.despine()
        plt.legend()

        return knee_y, fig, ax

    return knee_y
