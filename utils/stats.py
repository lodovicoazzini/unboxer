import math

from cliffs_delta import cliffs_delta
from pingouin import compute_effsize
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


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
