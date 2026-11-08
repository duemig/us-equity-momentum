"""Summary statistics, Newey-West t-statistics, and factor-model alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

#: Months per year, used to annualise means, volatilities, and Sharpe ratios.
MONTHS_PER_YEAR = 12


def newey_west_lags(n_months: int) -> int:
    """Return the automatic Newey-West lag length `floor(4 (n / 100)^(2/9))`."""
    return int(np.floor(4.0 * (n_months / 100.0) ** (2.0 / 9.0)))


def mean_t_statistics(returns: pd.Series, lags: int | None = None) -> dict[str, float]:
    """Compute the t-statistic of a mean return under both plain and HAC standard errors.

    Args:
        returns: Return series.
        lags: Newey-West lag length; if `None`, the automatic rule is used.

    Returns:
        A mapping with the ordinary-least-squares `t_ols`, the Newey-West `t_nw`, and the
        `nw_lags` used.
    """
    values = returns.dropna().to_numpy()
    n = len(values)
    mean = values.mean()
    std = values.std(ddof=1)
    used_lags = newey_west_lags(n) if lags is None else lags
    hac = sm.OLS(values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": used_lags})
    return {
        "t_ols": float(mean / (std / np.sqrt(n))),
        "t_nw": float(hac.tvalues[0]),
        "nw_lags": used_lags,
    }


def summary_stats(returns: pd.Series) -> dict[str, float]:
    """Summarise a monthly return series.

    Args:
        returns: Monthly return series indexed by month.

    Returns:
        A mapping with the sample size, monthly and annualised mean and volatility, the
        annualised Sharpe ratio, plain and Newey-West t-statistics of the mean, the hit rate,
        the minimum and maximum months, and the cumulative return.
    """
    series = returns.dropna()
    values = series.to_numpy()
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    tstats = mean_t_statistics(series)
    return {
        "n_months": n,
        "mean_monthly": mean,
        "mean_annual": mean * MONTHS_PER_YEAR,
        "vol_monthly": std,
        "vol_annual": std * np.sqrt(MONTHS_PER_YEAR),
        "sharpe_annual": (mean / std) * np.sqrt(MONTHS_PER_YEAR) if std > 0 else float("nan"),
        "t_ols": tstats["t_ols"],
        "t_nw": tstats["t_nw"],
        "nw_lags": tstats["nw_lags"],
        "hit_rate": float((values > 0).mean()),
        "min": float(values.min()),
        "min_month": str(series.idxmin()),
        "max": float(values.max()),
        "max_month": str(series.idxmax()),
        "cumulative": float((1.0 + series).prod() - 1.0),
    }


def factor_regression(
    strategy: pd.Series,
    factors: pd.DataFrame,
    factor_columns: tuple[str, ...],
    lags: int | None = None,
) -> dict:
    """Regress a strategy return on factors with Newey-West standard errors.

    The intercept is the full-sample alpha. For a self-financing long-short strategy the raw
    return is used directly; no risk-free rate is subtracted.

    Args:
        strategy: Strategy return series indexed by month.
        factors: Factor returns indexed by month, including `factor_columns`.
        factor_columns: Names of the factor columns used as regressors.
        lags: Newey-West lag length; if `None`, the automatic rule is used.

    Returns:
        A mapping with the sample size, the monthly and annualised alpha and its t-statistic,
        the factor betas and their t-statistics, and the R-squared.
    """
    columns = list(factor_columns)
    data = pd.concat([strategy.rename("strategy"), factors[columns]], axis=1).dropna()
    n = len(data)
    used_lags = newey_west_lags(n) if lags is None else lags
    design = sm.add_constant(data[columns].to_numpy())
    fit = sm.OLS(data["strategy"].to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": used_lags}
    )
    names = ["alpha", *columns]
    params = dict(zip(names, fit.params, strict=False))
    tvalues = dict(zip(names, fit.tvalues, strict=False))
    return {
        "n_months": n,
        "nw_lags": used_lags,
        "alpha_monthly": float(params["alpha"]),
        "alpha_annual": float(params["alpha"] * MONTHS_PER_YEAR),
        "alpha_t": float(tvalues["alpha"]),
        "betas": {name: float(params[name]) for name in columns},
        "beta_t": {name: float(tvalues[name]) for name in columns},
        "r_squared": float(fit.rsquared),
    }
