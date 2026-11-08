"""Momentum formation returns and decile portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def formation_returns(prices: pd.DataFrame, short_lag: int = 2, long_lag: int = 14) -> pd.DataFrame:
    """Compute the momentum formation-period return for each stock and month.

    The formation return at month `t` is `P[t - short_lag] / P[t - long_lag] - 1`. The
    default `(short_lag, long_lag) = (2, 14)` is a twelve-month return that skips the most
    recent month, which is the convention used in the original assignment.

    Args:
        prices: Price panel indexed by month, one column per ticker.
        short_lag: Months back to the end of the formation window.
        long_lag: Months back to the start of the formation window.

    Returns:
        Formation returns indexed by month, one column per ticker. Months without enough
        price history are `null`.
    """
    if long_lag <= short_lag:
        msg = f"long_lag ({long_lag}) must exceed short_lag ({short_lag})"
        raise ValueError(msg)
    return prices.shift(short_lag) / prices.shift(long_lag) - 1.0


def decile_returns(
    formation: pd.DataFrame, returns: pd.DataFrame, n_buckets: int = 10
) -> pd.DataFrame:
    """Compute equal-weight bucket returns and the top-minus-bottom spread.

    Each month, stocks are ranked by their formation return in ascending order and split
    into `n_buckets` groups of equal size. Each group return is the equal-weight mean of
    the holding-month returns of its stocks. Bucket 1 holds the lowest formation returns
    (losers) and bucket `n_buckets` the highest (winners).

    Args:
        formation: Formation returns indexed by month, one column per ticker.
        returns: Holding-month returns indexed by month, one column per ticker.
        n_buckets: Number of equal-size buckets (10 for deciles, 5 for quintiles).

    Returns:
        A frame indexed by month with columns `d1` ... `d{n_buckets}` and `spread`, where
        `spread` is the top bucket minus the bottom bucket.
    """
    months = formation.dropna(how="all").index.intersection(returns.index)
    records: dict[pd.Period, dict[str, float]] = {}
    for month in months:
        f = formation.loc[month]
        r = returns.loc[month]
        valid = f.notna() & r.notna()
        if int(valid.sum()) < n_buckets:
            continue
        order = f[valid].sort_values(kind="stable").index
        buckets = np.array_split(order, n_buckets)
        row = {
            f"d{position + 1}": float(r[tickers].mean()) for position, tickers in enumerate(buckets)
        }
        records[month] = row
    deciles = pd.DataFrame.from_dict(records, orient="index").sort_index()
    deciles["spread"] = deciles[f"d{n_buckets}"] - deciles["d1"]
    return deciles


def long_short_weights(formation: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Build the long-short portfolio weights of the top-minus-bottom bucket strategy.

    Each month, stocks are ranked by their formation return and split into `n_buckets`
    groups. The top group is held long with equal weights that sum to one, and the bottom
    group is held short with equal weights that sum to minus one. All other stocks get a
    weight of zero. These weights reproduce the bucket spread when multiplied by the
    holding-month returns, and they are the basis for the turnover calculation.

    Args:
        formation: Formation returns indexed by month, one column per ticker.
        n_buckets: Number of equal-size buckets.

    Returns:
        Weights indexed by month, one column per ticker, for months with enough stocks.
    """
    records: dict[pd.Period, pd.Series] = {}
    for month in formation.dropna(how="all").index:
        valid = formation.loc[month].dropna()
        if len(valid) < n_buckets:
            continue
        order = valid.sort_values(kind="stable").index
        buckets = np.array_split(order, n_buckets)
        weights = pd.Series(0.0, index=formation.columns)
        weights[list(buckets[-1])] = 1.0 / len(buckets[-1])
        weights[list(buckets[0])] = -1.0 / len(buckets[0])
        records[month] = weights
    return (
        pd.DataFrame.from_dict(records, orient="index")
        .reindex(columns=formation.columns)
        .sort_index()
    )
