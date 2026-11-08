"""Inverse-volatility and equal-weight portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Compute trailing sample volatility for each stock and month.

    The volatility at month `t` is the sample standard deviation (`ddof=1`) of the returns
    over the `window` months `[t - window, t - 1]`. The current month is excluded, so the
    estimate uses only information available before the month it is applied to.

    Args:
        returns: Return panel indexed by month, one column per ticker.
        window: Length of the trailing estimation window in months.

    Returns:
        Volatilities indexed by month, one column per ticker. Months without a full window
        are `null`.
    """
    return returns.shift(1).rolling(window).std(ddof=1)


def inverse_vol_weights(volatility: pd.DataFrame) -> pd.DataFrame:
    """Compute inverse-volatility portfolio weights.

    The weight of stock `i` in month `t` is `(1 / sigma_{i,t})` divided by the sum of the
    inverse volatilities across stocks that month, so the weights sum to one.

    Args:
        volatility: Volatilities indexed by month, one column per ticker.

    Returns:
        Weights indexed by month, one column per ticker. Months in which any stock's
        volatility is missing are `null`.
    """
    inverse = 1.0 / volatility
    weights = inverse.div(inverse.sum(axis=1), axis=0)
    weights[volatility.isna().any(axis=1)] = np.nan
    return weights


def equal_weight_return(returns: pd.DataFrame) -> pd.Series:
    """Compute the equal-weight (1/N) portfolio return per month.

    Args:
        returns: Return panel indexed by month, one column per ticker.

    Returns:
        The cross-sectional mean return per month.
    """
    return returns.mean(axis=1)


def weighted_return(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute a weighted portfolio return per month.

    Args:
        weights: Portfolio weights indexed by month, one column per ticker.
        returns: Return panel indexed by month, one column per ticker.

    Returns:
        The weighted sum of returns per month, `null` before the first month with weights.
    """
    aligned = returns.reindex(index=weights.index, columns=weights.columns)
    return (weights * aligned).sum(axis=1, min_count=1)
