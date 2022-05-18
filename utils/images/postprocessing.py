import warnings
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import numpy as np

from config.config_const import IMG_SIZE, GRID_SIZE
from utils.image_similarity.stats import get_elbow_point


def add_grid(ax: plt.Axes) -> plt.Axes:
    # Set the grid intervals
    grid_interval = IMG_SIZE / GRID_SIZE
    loc = plt_ticker.MultipleLocator(base=grid_interval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Find number of sections
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(grid_interval)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(grid_interval)))
    # Add the labels to the grid
    for j in range(ny):
        y = grid_interval / 2 + j * grid_interval
        for i in range(nx):
            x = grid_interval / 2. + float(i) * grid_interval
            ax.text(x, y, '{:d}'.format(i + j * nx), color='k', ha='center', va='center')

    # Show the grid
    ax.grid(b=True, which='major', axis='both', linestyle='-', color='k', zorder=10)

    return ax


def aggregate_activations(
        image: np.ndarray,
        threshold: float = 0,
        normalize: bool = False,
        aggregator: Callable = np.nanmax
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average the pixel values of an image
    :param image: The original image
    :param threshold: The threshold to ignore lower values
    :param normalize: Whether to normalize the values in [0, 1)
    :param aggregator: The aggregation function to use
    :return: The resized image
    """
    # Filter the warning in case of section with only nan values -> result is nan
    processed = image
    if normalize or not 0 == threshold:
        processed = processed * (1 - np.nanmin(processed)) / (np.nanmax(processed) - np.nanmin(processed))
    if threshold is None:
        hist_data, hist_idxs = np.histogram(processed, bins=10 ** 2)
        elbow_point = get_elbow_point(hist_data, smoothing=10 ** 3)
        threshold = hist_idxs[np.argmin(abs(hist_data - elbow_point))]
    processed = np.ma.masked_less_equal(processed, threshold).filled(np.nan)
    window_size = processed.shape[0] // GRID_SIZE
    reshaped = processed.reshape((1, GRID_SIZE, window_size, GRID_SIZE, window_size))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        aggregated = aggregator(aggregator(reshaped, axis=4), axis=2)
    aggregated = np.nan_to_num(aggregated)
    return aggregated, np.argsort(aggregated, axis=None)[::-1]
