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
    # Map the values from [0:same, inf:different) to [0:different, 1:same)
    dist = math.exp(-dist)
    return dist


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
    # Map the values from [0:same, inf:different) to [0:different, 1:same)
    err = math.exp(-err)
    return err
