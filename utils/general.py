import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def weight_not_null(df: pd.DataFrame, group_by, agg_column: str, metric='mean') -> pd.DataFrame:
    """
    Weighting a column metric based on the number of non-null entries in the column.
    The weight is rages between [0, 1] and corresponds to the function ln( (e-1) * not_none/max_not_none + 1).
    """
    aggregated = df.groupby(group_by)[agg_column].agg(val=metric, not_none='count')
    max_not_none = aggregated['not_none'].max()
    aggregated['weighted_val'] = aggregated.apply(
        lambda row: row['val'] * math.log((math.e - 1) * row['not_none'] / max_not_none + 1),
        axis=1
    )
    return aggregated
