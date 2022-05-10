import math
import os
from itertools import combinations

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


def weight_values(values: np.array, weights: np.array) -> np.array:
    max_weight = weights.max()
    zipped = np.column_stack((values, weights))
    weighted = np.array([value * math.log((math.e - 1) * weight / max_weight + 1) for value, weight in zipped])
    return weighted


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


def weight_by_values(df: pd.DataFrame, column: str, weights: str) -> pd.DataFrame:
    result = pd.DataFrame.copy(df, deep=True)
    max_weight = result[weights].max()
    result[f'{column}_weighted'] = result.apply(
        lambda row: row[column] * math.log((math.e - 1) * row[weights] / max_weight + 1),
        axis=1
    )

    return result


def get_common_clusters(clusters_list, mask=None):
    # Filter based on the mask
    if mask is not None:
        mask_idxs = np.argwhere(mask).flatten()
        masked_clusters = [
            [
                [
                    element for element in cluster if element in mask_idxs
                ]
                for cluster in clusters
            ]
            for clusters in clusters_list
        ]
    else:
        masked_clusters = clusters_list
    # Flatten the list of clusters, remove singletons
    masked_clusters = [
        cluster for clusters in masked_clusters for cluster in clusters
        if len(cluster) > 1
    ]
    # Find all the combinations of clusters
    combined = list(combinations(masked_clusters, 2))
    # Find all the possible intersections between the clusters
    intersections = [set(lhs).intersection(set(rhs)) for lhs, rhs in combined]
    # Remove the intersections of one element
    intersections = [intersection for intersection in intersections if len(intersection) > 1]
    # Count the occurrences of the intersections
    intersections_counts = list(zip(*np.unique(intersections, return_counts=True)))
    # Remove the intersections occurring only once
    intersections_counts = [(list(intersection), count) for intersection, count in intersections_counts if count > 1]
    # Sort the intersections by descending count and descending size
    intersections_counts = sorted(intersections_counts, key=lambda entry: (-entry[1], -len(entry[0])))

    return intersections_counts


def get_balanced_samples(
        elements: np.array,
        sample_size: int,
        balanced_by: np.array,
        weights: np.array = None
) -> tuple[np.array, np.array]:
    # Get the weights to use for selecting the balanced samples
    if weights is not None:
        weighted_first = weight_values(balanced_by, weights)
        weighted_last = weight_values(1 - balanced_by, weights)
    else:
        weighted_first = balanced_by
        weighted_last = - balanced_by

    # Find the sizes for the samples
    sample_size = min(len(elements), sample_size)
    sample_size_first = math.ceil(sample_size / 2)
    sample_size_last = math.floor(sample_size / 2)

    # Get the indices for the first sample
    pure_idxs = weighted_first.argsort()[::-1][:sample_size_first]
    # Get the indices for the second sample
    impure_idxs = np.array([idx for idx in weighted_last.argsort()[::-1] if idx not in pure_idxs])[:sample_size_last]

    return elements[pure_idxs], elements[impure_idxs]
