"""Tests for momentum formation returns and portfolio formation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpm import momentum


def test_formation_skips_recent_month() -> None:
    index = pd.period_range("2002-01", periods=16, freq="M")
    steps = np.arange(16)
    prices = pd.DataFrame(
        {"A": 100.0 * 1.01**steps, "B": 100.0 * 1.02**steps},
        index=index,
    )

    formation = momentum.formation_returns(prices)

    # At month 16 the window runs from P[t-14] to P[t-2], twelve months of compounding.
    last = index[15]
    assert formation.loc[last, "A"] == pytest.approx(1.01**12 - 1.0)
    assert formation.loc[last, "B"] == pytest.approx(1.02**12 - 1.0)
    # The first fourteen months lack a full window.
    assert formation.iloc[:14].isna().all().all()


def test_formation_rejects_bad_window() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.period_range("2003-01", periods=2, freq="M"))
    with pytest.raises(ValueError, match="long_lag"):
        momentum.formation_returns(prices, short_lag=12, long_lag=2)


def test_decile_returns_and_spread() -> None:
    index = pd.period_range("2004-01", periods=1, freq="M")
    formation = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]}, index=index)
    returns = pd.DataFrame({"A": [0.10], "B": [0.20], "C": [0.30], "D": [0.40]}, index=index)

    buckets = momentum.decile_returns(formation, returns, n_buckets=2)

    month = index[0]
    assert buckets.loc[month, "d1"] == pytest.approx(0.15)
    assert buckets.loc[month, "d2"] == pytest.approx(0.35)
    assert buckets.loc[month, "spread"] == pytest.approx(0.20)


def test_long_short_weights_reproduce_spread() -> None:
    index = pd.period_range("2004-01", periods=1, freq="M")
    formation = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]}, index=index)
    returns = pd.DataFrame({"A": [0.10], "B": [0.20], "C": [0.30], "D": [0.40]}, index=index)

    weights = momentum.long_short_weights(formation, n_buckets=2)
    spread = (weights * returns).sum(axis=1)

    assert weights.loc[index[0], "D"] == pytest.approx(0.5)
    assert weights.loc[index[0], "A"] == pytest.approx(-0.5)
    assert spread.loc[index[0]] == pytest.approx(0.20)
