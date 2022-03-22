import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity


def emd(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the earth mover's distance between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The earth mover's distance between the two matrices
    """
    return wasserstein_distance(lhs.flatten(), rhs.flatten())


def ssim(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the SSIM distance between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The CW-SSIM distance between the two matrices
    """
    return 1 - structural_similarity(lhs, rhs)
