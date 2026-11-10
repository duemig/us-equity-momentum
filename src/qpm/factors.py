"""Rolling factor-model regression and active-return decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Factor columns used as regressors, in the order they appear in the output beta names.
DEFAULT_FACTORS = ("mkt_rf", "smb", "hml")

#: Months per year, used to annualise the monthly alpha.
MONTHS_PER_YEAR = 12


def rolling_factor_model(
    active: pd.Series,
    factors: pd.DataFrame,
    window: int = 12,
    factor_columns: tuple[str, ...] = DEFAULT_FACTORS,
) -> pd.DataFrame:
    """Run a rolling multivariate regression and decompose the active return.

    For each month `t` with a full trailing window `[t - window, t - 1]`, the active return
    is regressed on the factors with an intercept by ordinary least squares, matching the
    Excel `LINEST` used in the original. The intercept and betas estimated on the trailing
    window are then applied to the current month's factor returns:

    - `active_alpha` = active minus the factor contribution (intercept plus residual),
    - `systematic` = intercept plus the factor contribution,
    - `idiosyncratic` = active minus `systematic`.

    Args:
        active: Active return indexed by month.
        factors: Factor returns indexed by month, including `factor_columns`.
        window: Length of the trailing estimation window in months.
        factor_columns: Names of the factor columns used as regressors.

    Returns:
        A frame indexed by month with one `beta_*` column per factor (the leading `_rf`
        suffix is dropped), plus `intercept`, `r2` (in-window), `factor_part`, `systematic`,
        `idiosyncratic`, `active_alpha`, and `annualised_alpha`.
    """
    columns = list(factor_columns)
    data = pd.concat([active.rename("active"), factors[columns]], axis=1).dropna()
    index = data.index
    records: dict[pd.Period, dict[str, float]] = {}
    for position in range(window, len(index)):
        month = index[position]
        estimation = data.iloc[position - window : position]
        y = estimation["active"].to_numpy()
        design = np.column_stack([np.ones(window), estimation[columns].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(coefficients[0])
        betas = coefficients[1:]

        residuals = y - design @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - (residuals**2).sum() / total) if total > 0 else np.nan

        current_factors = data.loc[month, columns].to_numpy()
        active_now = float(data.loc[month, "active"])
        factor_part = float(betas @ current_factors)
        systematic = intercept + factor_part
        active_alpha = active_now - factor_part

        record = {
            "intercept": intercept,
            "r2": r2,
            "factor_part": factor_part,
            "systematic": systematic,
            "idiosyncratic": active_now - systematic,
            "active_alpha": active_alpha,
            "annualised_alpha": (1.0 + active_alpha) ** MONTHS_PER_YEAR - 1.0,
        }
        for name, beta in zip(columns, betas, strict=True):
            record[f"beta_{name.removesuffix('_rf')}"] = float(beta)
        records[month] = record
    return pd.DataFrame.from_dict(records, orient="index").sort_index()
