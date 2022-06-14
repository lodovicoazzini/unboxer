import math

import numpy as np
import pandas as pd
from cliffs_delta import cliffs_delta
from pingouin import compute_effsize
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

from utils.general import show_progress


def compare_distributions(lhs: list, rhs: list) -> tuple:
    """
    Compare two distributions by running the appropriate statistical test
    :param lhs: The first distribution
    :param rhs: The second distribution
    :return: [are different, p-value for the difference, effect size of the difference, magnitude of the difference]
    """
    # Check if the two distributions are normal
    _, p_val_lhs = shapiro(lhs)
    _, p_val_rhs = shapiro(rhs)
    if p_val_lhs < .05 and p_val_rhs < .05:
        # Normal distributions
        statistic, p_value = ttest_ind(lhs, rhs)
        eff_size = compute_effsize(lhs, rhs, eftype='cohen')
        if eff_size <= .3:
            eff_size_str = 'small'
        elif eff_size <= .5:
            eff_size_str = 'medium'
        else:
            eff_size_str = 'large'
    else:
        # Non-normal distributions
        statistic, p_value = mannwhitneyu(lhs, rhs)
        eff_size, eff_size_str = cliffs_delta(lhs, rhs)

    return p_value < .05, p_value, eff_size, eff_size_str


def get_effect_size(lhs: list, rhs: list) -> tuple:
    """
    Compare two distributions
    :return: The effect size if the difference is relevant, None otherwise
    """
    is_relevant, p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
    eff_size_str_to_val = {
        'small': 1,
        'medium': 2,
        'large': 3
    }
    return abs(eff_size), eff_size_str_to_val[eff_size_str] if is_relevant else None


def weight_value(value: float, weight: float, max_weight: float) -> float:
    """
    Compute the weighted value with a weight between 0, 1
    :param value: The value
    :param weight: The weight
    :param max_weight: The maximum value for the weights
    :return: The weighted value
    """
    return value * math.log((math.e - 1) * weight / max_weight + 1)


def compute_distance_matrix(
        values: list,
        index: list,
        dist_func: callable,
        remove_diagonal: bool = False,
        show_progress_bar: bool = False
):
    # Initialize the distance matrix to 0
    num_clusters = len(values)
    distance_matrix = np.zeros(shape=(num_clusters, num_clusters))

    # Compute the distances above the diagonal
    def execution(row):
        for col in range(row, num_clusters):
            lhs, rhs = values[row], values[col]
            distance_matrix[row][col] = dist_func(lhs, rhs)

    if show_progress_bar:
        show_progress(execution=execution, iterable=range(0, num_clusters))
    else:
        for row_idx in range(0, num_clusters):
            execution(row_idx)

    # Mirror on the diagonal to complete the rest of the matrix
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix[np.diag_indices_from(distance_matrix)] /= 2

    # Prepare the data for the image
    dist_matrix = pd.DataFrame(
        distance_matrix,
        columns=index,
        index=index
    )
    # Find the average value for each cell
    dist_matrix = dist_matrix.groupby(dist_matrix.columns, axis=1).mean()
    dist_matrix = dist_matrix.groupby(dist_matrix.index, axis=0).mean()

    # Remove the values on the diagonal
    if remove_diagonal:
        np.fill_diagonal(dist_matrix.values, np.nan)

    return dist_matrix.values
