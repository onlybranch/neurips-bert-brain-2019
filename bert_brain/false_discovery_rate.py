import numpy as np


__all__ = ['fdr_correction']


def fdr_correction(p_values, alpha=0.05, method='by', axis=None):
    """
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Modified from the code at https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html

    Args:
        p_values: The p_values to correct.
        alpha: The error rate to correct the p-values with.
        method: one of by (for Benjamini/Yekutieli) or bh for Benjamini/Hochberg
        axis: Which axis of p_values to apply the correction along. If None, p_values is flattened.

    Returns:
        indicator_alternative: An boolean array with the same shape as p_values_corrected that is True where
            the null hypothesis should be rejected
        p_values_corrected: The p_values corrected for FDR. Sam