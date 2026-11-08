"""Portfolio turnover and transaction-cost adjustments."""

from __future__ import annotations

import pandas as pd


def long_short_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute per-month traded notional for a long-short book from target to target.

    The turnover in month `t` is the sum of the absolute changes in target weights from the
    previous month, which is the total notional bought and sold when the book is rebalanced
    to its new targets. Drift between rebalances is not modelled because the book is close to
    dollar-neutral, so a drift normalisation is not meaningful.

    Args:
        weights: Target weights indexed by month, one column per ticker.

    Returns:
        Traded notional per month, starting from the second month.
    """
    return weights.diff().abs().sum(axis=1).iloc[1:]


def long_only_turnover(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute per-month traded notional for a long-only book, accounting for drift.

    Between rebalances, last month's weights drift with the realised returns. The turnover in
    month `t` is the sum of the absolute differences between the new target weights and the
    drifted weights carried over from the previous month.

    Args:
        weights: Target weights indexed by month, one column per ticker, summing to one.
        returns: Return panel indexed by month, one column per ticker.

    Returns:
        Traded notional per month, starting from the second month.
    """
    months = weights.index
    traded: dict[pd.Period, float] = {}
    for position in range(1, len(months)):
        previous = weights.iloc[position - 1]
        held_return = returns.loc[months[position - 1]].reindex(previous.index)
        drifted = previous * (1.0 + held_return)
        total = drifted.sum()
        if total != 0:
            drifted = drifted / total
        traded[months[position]] = float((weights.iloc[position] - drifted).abs().sum())
    return pd.Series(traded)


def apply_cost(gross: pd.Series, traded_notional: pd.Series, cost_rate: float) -> pd.Series:
    """Subtract transaction costs from a gross return series.

    The cost in each month is `cost_rate` times the traded notional, so `cost_rate` is the
    cost per unit of notional traded (for example 0.001 for 10 basis points).

    Args:
        gross: Gross return series indexed by month.
        traded_notional: Traded notional per month from a turnover function.
        cost_rate: Cost per unit of traded notional, as a decimal.

    Returns:
        Net return series on the months of `gross`, with no cost in months without turnover.
    """
    aligned = traded_notional.reindex(gross.index).fillna(0.0)
    return gross - cost_rate * aligned
