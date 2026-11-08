"""Tests for the return calculation."""

from __future__ import annotations

import pandas as pd
import pytest

from qpm import data


def test_compute_returns_matches_manual() -> None:
    index = pd.period_range("2003-01", periods=3, freq="M")
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 75.0]}, index=index)

    returns = data.compute_returns(prices)

    assert list(returns.index) == list(index[1:])
    assert returns.loc[index[1], "A"] == pytest.approx(0.10)
    assert returns.loc[index[2], "A"] == pytest.approx(99.0 / 110.0 - 1.0)
    assert returns.loc[index[1], "B"] == pytest.approx(0.0)
    assert returns.loc[index[2], "B"] == pytest.approx(0.50)


def test_compute_returns_drops_first_month() -> None:
    index = pd.period_range("2003-01", periods=4, freq="M")
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=index)

    returns = data.compute_returns(prices)

    assert len(returns) == 3
    assert index[0] not in returns.index
