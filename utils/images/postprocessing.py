import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import numpy as np

from config.config_const import IMG_SIZE, GRID_SIZE


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


def get_average_activations(
        image: np.ndarray,
        threshold: float = None,
        normalize: bool = False
):
    """
    Average the pixel values of an image
    :param image: The original image
    :param threshold: The threshold to ignore lower values
    :param normalize: Whether to normalize the values in [0, 1)
    :return: The resized image
    """
    processed = image
    if normalize or threshold is not None:
        processed = image * (1 - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
    if threshold is not None:
        processed = np.ma.masked_less_equal(processed, threshold).filled(np.nan)
        processed = processed * (1 - np.nanmin(processed)) / (np.nanmax(processed) - np.nanmin(processed))
    averaged = np.nanmean(
        np.nanmean(
            processed.reshape((GRID_SIZE, processed.shape[0] // GRID_SIZE, GRID_SIZE, -1)),
            axis=3),
        axis=1
    )
    return averaged, np.nanargmax(averaged)
