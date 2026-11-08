"""Tests for the rolling factor model and factor regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qpm import factors, stats


def synthetic_factors(n: int, seed: int) -> pd.DataFrame:
    index = pd.period_range("2003-01", periods=n, freq="M")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0, 0.04, n),
            "smb": rng.normal(0.0, 0.02, n),
            "hml": rng.normal(0.0, 0.02, n),
        },
        index=index,
    )


def test_rolling_regression_recovers_known_betas() -> None:
    frame = synthetic_factors(24, seed=1)
    active = 0.001 + 0.5 * frame["mkt_rf"] - 0.3 * frame["smb"] + 0.2 * frame["hml"]

    model = factors.rolling_factor_model(active, frame, window=12)

    first = model.iloc[0]
    assert first["beta_mkt"] == pytest.approx(0.5, abs=1e-8)
    assert first["beta_smb"] == pytest.approx(-0.3, abs=1e-8)
    assert first["beta_hml"] == pytest.approx(0.2, abs=1e-8)
    assert first["intercept"] == pytest.approx(0.001, abs=1e-8)


def test_decomposition_adds_back_to_active() -> None:
    frame = synthetic_factors(30, seed=2)
    rng = np.random.default_rng(5)
    active = 0.001 + 0.4 * frame["mkt_rf"] + rng.normal(0.0, 0.01, len(frame))
    active.index = frame.index

    model = factors.rolling_factor_model(active, frame, window=12)
    rebuilt = model["systematic"] + model["idiosyncratic"]

    assert (rebuilt - active.reindex(model.index)).abs().max() == pytest.approx(0.0, abs=1e-12)


def test_factor_regression_recovers_alpha() -> None:
    frame = synthetic_factors(120, seed=4)
    strategy = 0.002 + 0.8 * frame["mkt_rf"]

    result = stats.factor_regression(strategy, frame, ("mkt_rf", "smb", "hml"))

    assert result["alpha_monthly"] == pytest.approx(0.002, abs=1e-9)
    assert result["betas"]["mkt_rf"] == pytest.approx(0.8, abs=1e-9)
    assert result["betas"]["smb"] == pytest.approx(0.0, abs=1e-9)
