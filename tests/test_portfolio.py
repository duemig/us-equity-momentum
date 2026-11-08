"""Tests for portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpm import portfolio


def test_rolling_volatility_uses_trailing_window() -> None:
    index = pd.period_range("2003-01", periods=14, freq="M")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"A": rng.normal(0.0, 0.05, 14)}, index=index)

    volatility = portfolio.rolling_volatility(returns, window=12)

    expected = returns["A"].iloc[0:12].std(ddof=1)
    assert volatility.loc[index[12], "A"] == pytest.approx(expected)
    assert volatility.iloc[:12].isna().all().all()


def test_inverse_vol_weights_sum_to_one() -> None:
    index = pd.period_range("2004-01", periods=1, freq="M")
    volatility = pd.DataFrame({"A": [0.1], "B": [0.2]}, index=index)

    weights = portfolio.inverse_vol_weights(volatility)

    assert weights.loc[index[0], "A"] == pytest.approx(10.0 / 15.0)
    assert weights.loc[index[0], "B"] == pytest.approx(5.0 / 15.0)
    assert weights.sum(axis=1).loc[index[0]] == pytest.approx(1.0)


def test_equal_weight_and_weighted_return() -> None:
    index = pd.period_range("2004-01", periods=1, freq="M")
    returns = pd.DataFrame({"A": [0.10], "B": [0.20]}, index=index)

    assert portfolio.equal_weight_return(returns).loc[index[0]] == pytest.approx(0.15)

    weights = pd.DataFrame({"A": [0.25], "B": [0.75]}, index=index)
    assert portfolio.weighted_return(weights, returns).loc[index[0]] == pytest.approx(0.175)
