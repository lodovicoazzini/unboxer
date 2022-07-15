import numpy as np
from skimage.metrics import structural_similarity


def ssim(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """
    Compute the SSIM distance between two matrices
    :param lhs: The first matrix
    :param rhs: The second matrix
    :return: The CW-SSIM distance between the two matrices
    """
    return structural_similarity(np.squeeze(lhs), np.squeeze(rhs), win_size=3)
