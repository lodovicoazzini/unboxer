import math

import numpy as np

from utils.stats import weight_value


def shorten_list(original: np.ndarray, size: int) -> np.ndarray:
    """
    Shorten a list computing the average over subsets of entries
    :param original: The original list to shorten
    :param size: The size of the final shortened list
    :return: The shortened list
    """
    # Compute the size of the window for the averaged elements
    window_size = math.ceil(len(original) / size)
    # Compute the trimmed length so that the list can be reshaped correctly
    rounded_len = window_size * math.floor(len(original) / window_size)
    # Average the list over windows to reduce its size
    return np.mean(original[:rounded_len].reshape(-1, window_size), axis=1)


def weight_values(values: np.array, weights: np.array) -> np.array:
    """
    Compute the weighted values for a list of values and their corresponding weights
    :param values: The values
    :param weights: The weights
    :return: The weighted values
    """
    # Compute the maximum weight
    max_weight = weights.max()
    # Associate each value with its weight and compute teh weighted value
    zipped = np.column_stack((values, weights))
    weighted = np.array([weight_value(value=value, weight=weight, max_weight=max_weight) for value, weight in zipped])
    return weighted


def get_balanced_samples(
        elements: np.array,
        sample_size: int,
        balanced_by: np.array,
        weights: np.array = None
) -> tuple:
    """
    Sample the elements balances by some other (weighted) values
    :param elements: The elements to sample
    :param sample_size: The desired sample size
    :param balanced_by: The values to use to balance the samples
    :param weights: The weights to apply to the balancing values
    :return: The best and worst elements based on the (weighted) balancing values
    """
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
