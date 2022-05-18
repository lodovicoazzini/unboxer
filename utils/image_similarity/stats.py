import numpy as np
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


def get_elbow_point(
        data: np.ndarray,
        smoothing: int = 0,
        plot: bool = False,
        curve: str = 'convex'
):
    """
        Find the point of maximum curvature in a list of values
        :param data: The input list of values
        :param smoothing: Positive float determining the amount of smoothing applied to the data
        :param plot: Whether to plot the data
        :param curve: `concave` -> knee, `convex` -> elbows
        :return: The value corresponding to the maximum curvature (+ fig, ax if plot == True)
        """
    # Sort the distances
    data = np.sort(data)

    # Associate an index to the distances
    x = np.arange(0, data.shape[0], 1)
    function_data = np.array(list(zip(x, data)))
    # Compute the approximation of the curve for the distances
    spl = UnivariateSpline(x, data, s=smoothing, k=3)
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
