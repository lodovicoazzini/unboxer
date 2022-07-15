import os
import re

import matplotlib.pyplot as plt
import numpy as np


def beep():
    """
    Play a sound to notify the end of an execution
    """
    os.system('say "beep"')


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, transparent=True):
    """
    Save a figure to the given path, create the necessary path if not existing
    :param fig: The figure to save
    :param path: The path where to save the figure
    :param dpi: The resolution for the saved figure
    :param transparent: Whether to have a transparent background
    :return:
    """
    # Check if the path for the containing directory exists, if not -> create it
    try:
        containing_dir = re.search('.*/', path)[0]
        os.makedirs(containing_dir) if not os.path.exists(containing_dir) else None
        # Save the figure
        fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
    except IndexError:
        raise ValueError('Invalid path')


def scale_values_in_range(values_lists, min_size: float, max_size: float):
    # If more than one array is given to scale them together
    try:
        all_values = np.concatenate(values_lists)
    except ValueError:
        all_values = values_lists
    min_value = min(all_values)
    range_values = max(all_values) - min_value
    resized_list = [(values - min_value) / range_values for values in values_lists]
    range_sizes = max_size - min_size
    normalized_list = [resized * range_sizes + min_size for resized in resized_list]
    return normalized_list
