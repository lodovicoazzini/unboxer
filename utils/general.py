import math
import os

import matplotlib.pyplot as plt
import numpy as np


def beep():
    os.system('say "beep"')


def save_figure(fig: plt.Figure, path: str, dpi: int = 150, transparent=True):
    # check if the containing directory exists
    out_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save the figure
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')


def shorten_list(original: np.ndarray, size: int) -> np.ndarray:
    # compute the size of the window for the averaged elements
    window_size = math.ceil(len(original) / size)
    # compute the trimmed length so that the list can be reshaped correctly
    rounded_len = window_size * math.floor(len(original) / window_size)
    # average the list over windows to reduce its size
    return np.mean(original[:rounded_len].reshape(-1, window_size), axis=1)
