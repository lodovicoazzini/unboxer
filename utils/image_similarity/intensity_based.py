import math

import numpy as np


def euclidean_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the euclidean distance between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The euclidean distance between the two matrices
    """
    # compute and return the distance between teh two matrices
    return np.sqrt(np.sum((lhs - rhs) ** 2))


def mse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the inverse of the mean squared error between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The mse normalized in [0, 1)
    """
    # check that the two inputs have the same size
    # compute the average of the squares of the differences for each pixel
    err = np.sum((lhs.astype('float') - rhs.astype('float')) ** 2)
    err /= float(lhs.shape[0] * rhs.shape[1])
    # map the values from [0:same, inf:different) to [0:same, 1:different)
    return 1 - math.exp(-err)
