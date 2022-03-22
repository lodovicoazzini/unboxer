import numpy as np
from scipy.stats import wasserstein_distance


def emd(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return wasserstein_distance(lhs, rhs)
