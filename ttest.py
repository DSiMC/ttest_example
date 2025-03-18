import numpy as np
from scipy import stats

def independent_ttest(data1, data2, alpha=0.05, verbose=True):
    """
    Performs an independent two-sample t-test.

    Args:
        data1 (list or numpy.ndarray): Sample 1 data.
        data2 (list or numpy.ndarray): Sample 2 data.
        alpha (float, optional): Significance level. Defaults to 0.05.
        verbose (bool, optional): Print results. Defaults to True.

    Returns:
        tuple: (t_statistic, p_value, degrees_of_freedom, reject_hypothesis)
    """

    data1 = np.array(data1)
    data2 = np.array(data2)

    t_statistic, p_value = stats.ttest_ind(data1, data2)
    degrees_of_freedom = len(data1) + len(data2) - 2

    reject_hypothesis = p_value < alpha

    if verbose:
        print("Independent Two-Sample T-test:")
        print(f"  Sample 1 Mean: {np.mean(data1):.4f}")
        print(f"  Sample 2 Mean: {np.mean(data2):.4f}")
        print(f"  T-statistic: {t_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Degrees of Freedom: {degrees_of_freedom}")
        print(f"  Alpha: {alpha}")
        if reject_hypothesis:
            print(f"  Reject the null hypothesis (p < {alpha}).")
        else:
            print(f"  Fail to reject the null hypothesis (p >= {alpha}).")

    return t_statistic, p_value, degrees_of_freedom, reject_hypothesis

def paired_ttest(data1, data2, alpha=0.05, verbose=True):
    """
    Performs a paired (dependent) two-sample t-test.

    Args:
        data1 (list or numpy.ndarray): Sample 1 data.
        data2 (list or numpy.ndarray): Sample 2 data.
        alpha (float, optional): Significance level. Defaults to 0.05.
        verbose (bool, optional): Print results. Defaults to True.

    Returns:
        tuple: (t_statistic, p_value, degrees_of_freedom, reject_hypothesis)
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    t_statistic, p_value = stats.ttest_rel(data1, data2)
    degrees_of_freedom = len(data1) - 1

    reject_hypothesis = p_value < alpha

    if verbose:
        print("Paired Two-Sample T-test:")
        print(f"  Sample 1 Mean: {np.mean(data1):.4f}")
        print(f"  Sample 2 Mean: {np.mean(data2):.4f}")
        print(f"  T-statistic: {t_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Degrees of Freedom: {degrees_of_freedom}")
        print(f"  Alpha: {alpha}")
        if reject_hypothesis:
            print(f"  Reject the null hypothesis (p < {alpha}).")
        else:
            print(f"  Fail to reject the null hypothesis (p >= {alpha}).")

    return t_statistic, p_value, degrees_of_freedom, reject_hypothesis

# Example Usage:
if __name__ == "__main__":
    # Independent t-test example
    group1 = [25, 30, 28, 35, 40]
    group2 = [20, 26, 32, 29, 33]
    independent_ttest(group1, group2)

    print("\n")

    # Paired t-test example
    before = [80, 75, 85, 90, 82]
    after = [78, 77, 83, 88, 85]
    paired_ttest(before, after)
