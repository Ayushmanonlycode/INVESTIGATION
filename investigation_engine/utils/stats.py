"""Statistical utility functions.

Shared statistical computations used across investigation modules.
All functions operate on numpy arrays or pandas Series for consistency
and performance.

These utilities are building blocks — investigation modules compose
them to produce findings. They do NOT produce findings directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats


def compute_z_scores(values: NDArray[np.floating[Any]] | pd.Series) -> NDArray[np.floating[Any]]:
    """Compute z-scores for a numeric array.

    Args:
        values: Numeric values (NaN values are excluded from computation).

    Returns:
        Array of z-scores, same shape as input. NaN positions remain NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(arr)

    if mask.sum() < 2:
        return np.full_like(arr, np.nan)

    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=1)

    if std == 0:
        return np.zeros_like(arr)

    z = np.full_like(arr, np.nan)
    z[mask] = (arr[mask] - mean) / std
    return z


def compute_iqr_bounds(
    values: NDArray[np.floating[Any]] | pd.Series,
    multiplier: float = 1.5,
) -> tuple[float, float]:
    """Compute IQR-based outlier bounds (Tukey's fences).

    Args:
        values: Numeric values (NaN values excluded).
        multiplier: IQR multiplier for fence computation.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[~np.isnan(arr)]

    if len(clean) < 4:
        return float("-inf"), float("inf")

    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = q3 - q1

    return q1 - multiplier * iqr, q3 + multiplier * iqr


def compute_effect_size_cohens_d(
    group_a: NDArray[np.floating[Any]] | pd.Series,
    group_b: NDArray[np.floating[Any]] | pd.Series,
) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group_a: First group of numeric values.
        group_b: Second group of numeric values.

    Returns:
        Cohen's d effect size. Positive means group_a > group_b.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    a_clean = a[~np.isnan(a)]
    b_clean = b[~np.isnan(b)]

    if len(a_clean) < 2 or len(b_clean) < 2:
        return 0.0

    n_a, n_b = len(a_clean), len(b_clean)
    var_a, var_b = float(np.var(a_clean, ddof=1)), float(np.var(b_clean, ddof=1))

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(a_clean) - np.mean(b_clean)) / pooled_std)


def compute_cramers_v(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """Compute Cramér's V association between two categorical variables.

    Args:
        x: First categorical series.
        y: Second categorical series.

    Returns:
        Cramér's V statistic (0.0 to 1.0).
    """
    contingency = pd.crosstab(x, y)

    if contingency.size == 0:
        return 0.0

    chi2 = float(scipy_stats.chi2_contingency(contingency)[0])
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1

    if min_dim == 0 or n == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy of a categorical series.

    Args:
        series: Categorical values (NaN values excluded).

    Returns:
        Shannon entropy in bits.
    """
    value_counts = series.dropna().value_counts(normalize=True)
    probabilities = value_counts.values.astype(np.float64)

    # Filter zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_coefficient_of_variation(
    values: NDArray[np.floating[Any]] | pd.Series,
) -> float:
    """Compute coefficient of variation (CV).

    Args:
        values: Numeric values (NaN values excluded).

    Returns:
        Coefficient of variation (std / mean). Returns inf if mean is zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[~np.isnan(arr)]

    if len(clean) < 2:
        return 0.0

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))

    if mean == 0:
        return float("inf") if std > 0 else 0.0

    return abs(std / mean)


def compute_skewness(values: NDArray[np.floating[Any]] | pd.Series) -> float:
    """Compute skewness of a numeric distribution.

    Args:
        values: Numeric values (NaN values excluded).

    Returns:
        Skewness statistic.
    """
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[~np.isnan(arr)]

    if len(clean) < 3:
        return 0.0

    return float(scipy_stats.skew(clean, bias=False))


def compute_kurtosis(values: NDArray[np.floating[Any]] | pd.Series) -> float:
    """Compute excess kurtosis of a numeric distribution.

    Args:
        values: Numeric values (NaN values excluded).

    Returns:
        Excess kurtosis (Fisher's definition, 0 for normal distribution).
    """
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[~np.isnan(arr)]

    if len(clean) < 4:
        return 0.0

    return float(scipy_stats.kurtosis(clean, bias=False))


def test_normality(
    values: NDArray[np.floating[Any]] | pd.Series,
    significance_level: float = 0.05,
) -> tuple[bool, float]:
    """Test if a distribution is approximately normal using Shapiro-Wilk.

    For samples > 5000, uses D'Agostino-Pearson test instead.

    Args:
        values: Numeric values (NaN values excluded).
        significance_level: P-value threshold for rejecting normality.

    Returns:
        Tuple of (is_normal, p_value).
    """
    arr = np.asarray(values, dtype=np.float64)
    clean = arr[~np.isnan(arr)]

    if len(clean) < 8:
        return True, 1.0

    if len(clean) > 5000:
        _, p_value = scipy_stats.normaltest(clean)
    else:
        _, p_value = scipy_stats.shapiro(clean)

    return bool(p_value > significance_level), float(p_value)


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 30,
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.

    Args:
        df: DataFrame with numeric columns.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        min_periods: Minimum observations required per pair.

    Returns:
        Correlation matrix as DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method, min_periods=min_periods)
