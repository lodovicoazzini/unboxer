import math

import numpy as np


def euclidean_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the euclidean distance between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The euclidean distance between the two matrices
    """
    # Compute the distance between teh two matrices
    dist = np.sqrt(np.sum((lhs - rhs) ** 2))
    # Compute the maximum distance between the two matrices
    max_dist = np.sqrt(np.sum((np.ones_like(lhs) - np.zeros_like(rhs)) ** 2))
    # Map the values from [0:same, max:different) to [0:different, 1:same)
    sim = 1 - 1 / max_dist * dist
    return sim


def mean_squared_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the inverse of the mean squared error between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The mse normalized in [0, 1)
    """
    # Check that the two inputs have the same size
    # Compute the average of the squares of the differences for each pixel
    err = np.sum((lhs.astype('float') - rhs.astype('float')) ** 2)
    err /= float(lhs.shape[0] * rhs.shape[1])
    # Map the values from [0:same, 1:different) to [0:different, 1:same)
    sim = 1 - err
    return sim
