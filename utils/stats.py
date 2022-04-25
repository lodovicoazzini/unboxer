from cliffs_delta import cliffs_delta
from pingouin import compute_effsize
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


def compare_distributions(lhs, rhs):
    # Check if the two distributions are normal
    if shapiro(lhs)[1] < .05 and shapiro(rhs)[1] < .05:
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
