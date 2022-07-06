import os
import re

import matplotlib.pyplot as plt


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
