import warnings
from typing import Callable

import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import numpy as np

from config.config_outputs import IMG_SIZE, GRID_SIZE
from utils.general import save_figure
from utils.images.image_similarity.stats import get_elbow_point


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
) -> tuple:
    """
    Average the pixel values of an image
    :param image: The original image
    :param threshold: The threshold to ignore lower values
    :param normalize: Whether to normalize the values in [0, 1)
    :param aggregator: The aggregation function to use
    :return: The resized image
    """
    processed = image
    # Normalize the activation values between [0, 1)
    if normalize or not 0 == threshold:
        processed = processed * (1 - np.nanmin(processed)) / (np.nanmax(processed) - np.nanmin(processed))
    # Filter the lower values based on the elbow point of the activations
    if threshold is None:
        hist_data, hist_idxs = np.histogram(processed, bins=10 ** 2)
        elbow_point, fig, ax = get_elbow_point(hist_data, smoothing=10 ** 3, plot=True)
        ax.set_title("Sorted histogram values for the heatmap's activations")
        ax.set_ylabel("histogram value")
        ax.set_xlabel("index")
        save_figure(fig, '/Users/lodovicoazzini/Repos/USI-MSDE/III/Thesis_no_sync/thesis_no_sync/imgs/aggregate_elbow')
        threshold = hist_idxs[np.argmin(abs(hist_data - elbow_point))]
        print('Threshold: ', threshold)
    processed = np.ma.masked_less_equal(processed, threshold).filled(np.nan)
    # Reshape the image to fit the grid size
    window_size = processed.shape[0] // GRID_SIZE
    reshaped = processed.reshape((1, GRID_SIZE, window_size, GRID_SIZE, window_size))
    # Filter the warning in case of section with only nan values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # Aggregate the values in each region using the given approach
        aggregated = aggregator(aggregator(reshaped, axis=4), axis=2)
    # Set the nan values to 0
    aggregated = np.nan_to_num(aggregated)
    aggregated = np.squeeze(aggregated)
    # Sort the regions by descending aggregated value
    sorted_regions = np.argsort(aggregated, axis=None)[::-1]
    return aggregated, sorted_regions, processed


def combine_images(lhs_path, rhs_path):
    # Create the general figure
    fig, ax = plt.subplots(1, 2)
    # Read and visualize the images
    lhs, rhs = plt_img.imread(lhs_path), plt_img.imread(rhs_path)
    ax[0].imshow(lhs)
    ax[1].imshow(rhs)
    # Remove the ticks
    for axx in ax.flatten():
        axx.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    return fig, ax
