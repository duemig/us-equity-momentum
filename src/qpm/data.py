"""Load prices and factor series and compute monthly returns."""

from __future__ import annotations

import pathlib

import pandas as pd


def load_prices(path: str | pathlib.Path) -> pd.DataFrame:
    """Load the monthly price panel.

    Args:
        path: CSV with a `date` column and one column of prices per stock.

    Returns:
        Prices indexed by calendar month (monthly `PeriodIndex`), one float column per
        ticker, sorted by month.
    """
    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    prices.index = prices.index.to_period("M")
    prices.index.name = "month"
    return prices.sort_index().astype(float)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute arithmetic monthly returns.

    Args:
        prices: Price panel indexed by month, one column per ticker.

    Returns:
        Returns `P_t / P_{t-1} - 1` indexed by month, one column per ticker. The first
        month, which has no prior price, is dropped.
    """
    return prices.pct_change().iloc[1:]


def load_factors(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a factor series.

    Args:
        path: CSV with a `date` column and one column per factor (decimal returns).

    Returns:
        Factors indexed by calendar month (monthly `PeriodIndex`), sorted by month.
    """
    factors = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    factors.index = factors.index.to_period("M")
    factors.index.name = "month"
    return factors.sort_index().astype(float)
