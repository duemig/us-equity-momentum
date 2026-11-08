"""Tests for summary statistics and turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpm import stats, turnover


def test_summary_stats_basic() -> None:
    index = pd.period_range("2003-01", periods=120, freq="M")
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.01, 0.04, 120), index=index)

    summary = stats.summary_stats(returns)

    assert summary["n_months"] == 120
    assert summary["mean_monthly"] == pytest.approx(returns.mean())
    assert summary["mean_annual"] == pytest.approx(returns.mean() * 12)
    assert summary["cumulative"] == pytest.approx((1.0 + returns).prod() - 1.0)
    assert 0.0 <= summary["hit_rate"] <= 1.0


def test_long_only_turnover_zero_when_returns_equal() -> None:
    index = pd.period_range("2003-01", periods=3, freq="M")
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=index)
    returns = pd.DataFrame({"A": [0.1, 0.1, 0.1], "B": [0.1, 0.1, 0.1]}, index=index)

    traded = turnover.long_only_turnover(weights, returns)

    assert traded.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_long_short_turnover_counts_absolute_changes() -> None:
    index = pd.period_range("2004-01", periods=2, freq="M")
    weights = pd.DataFrame({"A": [0.5, -0.5], "B": [-0.5, 0.5]}, index=index)

    traded = turnover.long_short_turnover(weights)

    assert traded.loc[index[1]] == pytest.approx(2.0)


def test_apply_cost_reduces_return() -> None:
    index = pd.period_range("2004-01", periods=2, freq="M")
    gross = pd.Series([0.02, 0.03], index=index)
    traded = pd.Series([1.0], index=index[1:])

    net = turnover.apply_cost(gross, traded, cost_rate=0.001)

    assert net.loc[index[0]] == pytest.approx(0.02)
    assert net.loc[index[1]] == pytest.approx(0.03 - 0.001)
