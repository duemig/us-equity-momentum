"""Run the whole study and write the results and figures.

Rebuilds the momentum deciles, the inverse-volatility and equal-weight portfolios, and the
rolling factor model, then adds the robustness work: quintiles, alternative formation windows,
CAPM and Fama-French alphas with Newey-West t-statistics, turnover and after-cost returns, and
the correlation with the Kenneth French momentum factor. Writes CSV series and a metrics file
to ``results`` and figures to ``figures``.

Run from the repository root:

    python scripts/run_study.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from qpm import data, factors, momentum, portfolio, stats, turnover  # noqa: E402

DATA = REPO / "data"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

FF3 = ("mkt_rf", "smb", "hml")
CAPM = ("mkt_rf",)
FF4 = ("mkt_rf", "smb", "hml", "wml")
COST_LEVELS_BPS = [0, 5, 10, 20, 50]
FORMATION_VARIANTS = {
    "skip_12_1": (2, 14),
    "no_skip_12": (1, 13),
    "skip_6_1": (2, 8),
}


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as a PNG and close it."""
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_series(obj: pd.Series | pd.DataFrame, name: str) -> None:
    """Write a month-indexed series or frame to a CSV with a leading date column."""
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj.copy()
    frame.insert(0, "date", frame.index.to_timestamp(how="end").strftime("%Y-%m-%d"))
    frame.to_csv(RESULTS / name, index=False)


def net_mean_by_cost(gross: pd.Series, traded: pd.Series) -> dict[str, float]:
    """Return the mean net return for each cost level in basis points."""
    result = {}
    for bps in COST_LEVELS_BPS:
        net = turnover.apply_cost(gross, traded, bps / 10_000.0)
        result[str(bps)] = float(net.mean())
    return result


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    prices = data.load_prices(DATA / "prices.csv")
    factor_series = data.load_factors(DATA / "ff_factors_from_workbook.csv")
    returns = data.compute_returns(prices)

    metrics: dict = {
        "sample": {
            "n_stocks": prices.shape[1],
            "price_start": str(prices.index[0]),
            "price_end": str(prices.index[-1]),
            "return_start": str(returns.index[0]),
            "return_end": str(returns.index[-1]),
            "n_return_months": len(returns),
        }
    }

    # --- Momentum deciles (base strategy) ---
    formation = momentum.formation_returns(prices)
    deciles = momentum.decile_returns(formation, returns, n_buckets=10)
    spread = deciles["spread"]
    metrics["momentum"] = stats.summary_stats(spread)
    metrics["momentum"]["decile_means"] = {
        col: float(deciles[col].mean()) for col in deciles.columns if col != "spread"
    }
    write_series(deciles, "momentum_deciles.csv")

    # --- Quintiles ---
    quintiles = momentum.decile_returns(formation, returns, n_buckets=5)
    metrics["quintiles"] = stats.summary_stats(quintiles["spread"])
    write_series(quintiles, "momentum_quintiles.csv")

    # --- Formation-window robustness ---
    variant_rows = []
    for name, (short_lag, long_lag) in FORMATION_VARIANTS.items():
        variant_formation = momentum.formation_returns(prices, short_lag, long_lag)
        variant_spread = momentum.decile_returns(variant_formation, returns, n_buckets=10)["spread"]
        summary = stats.summary_stats(variant_spread)
        variant_rows.append(
            {
                "variant": name,
                "short_lag": short_lag,
                "long_lag": long_lag,
                "n_months": summary["n_months"],
                "mean_monthly": summary["mean_monthly"],
                "mean_annual": summary["mean_annual"],
                "t_nw": summary["t_nw"],
                "sharpe_annual": summary["sharpe_annual"],
            }
        )
    metrics["formation_variants"] = variant_rows
    pd.DataFrame(variant_rows).to_csv(RESULTS / "formation_variants.csv", index=False)

    # --- Factor alphas of the momentum spread ---
    alpha_specs = {"capm": CAPM, "ff3": FF3, "ff4": FF4}
    metrics["momentum_alphas"] = {
        name: stats.factor_regression(spread, factor_series, cols)
        for name, cols in alpha_specs.items()
    }

    # --- Correlation with the Kenneth French momentum factor ---
    aligned = pd.concat([spread.rename("spread"), factor_series["wml"]], axis=1).dropna()
    metrics["umd"] = {
        "correlation": float(aligned["spread"].corr(aligned["wml"])),
        "n_months": int(len(aligned)),
        "beta": metrics["momentum_alphas"]["ff4"]["betas"]["wml"],
        "beta_t": metrics["momentum_alphas"]["ff4"]["beta_t"]["wml"],
    }

    # --- Inverse-volatility vs equal-weight portfolio ---
    volatility = portfolio.rolling_volatility(returns)
    iv_weights = portfolio.inverse_vol_weights(volatility)
    iv = portfolio.weighted_return(iv_weights, returns).dropna()
    ew = portfolio.equal_weight_return(returns).reindex(iv.index)
    active = (iv - ew).dropna()
    write_series(pd.DataFrame({"iv": iv, "ew": ew, "active": active}), "portfolio_returns.csv")

    metrics["iv_portfolio"] = stats.summary_stats(iv)
    metrics["ew_benchmark"] = stats.summary_stats(ew)
    metrics["active"] = stats.summary_stats(active)
    metrics["active_alpha_ff3"] = stats.factor_regression(active, factor_series, FF3)

    model = factors.rolling_factor_model(active, factor_series)
    write_series(model, "rolling_factor_model.csv")
    metrics["rolling_model"] = {
        "beta_mkt_mean": float(model["beta_mkt"].mean()),
        "beta_smb_mean": float(model["beta_smb"].mean()),
        "beta_hml_mean": float(model["beta_hml"].mean()),
        "r2_mean": float(model["r2"].mean()),
        "n_months": int(len(model)),
    }

    # --- Turnover and after-cost returns ---
    ls_weights = momentum.long_short_weights(formation, n_buckets=10)
    ls_turnover = turnover.long_short_turnover(ls_weights)
    iv_turnover = turnover.long_only_turnover(iv_weights.loc[iv.index], returns)
    ew_weights = pd.DataFrame(1.0 / returns.shape[1], index=iv.index, columns=returns.columns)
    ew_turnover = turnover.long_only_turnover(ew_weights, returns)
    iv_net = {bps: turnover.apply_cost(iv, iv_turnover, bps / 10_000.0) for bps in COST_LEVELS_BPS}
    ew_net = {bps: turnover.apply_cost(ew, ew_turnover, bps / 10_000.0) for bps in COST_LEVELS_BPS}
    metrics["turnover"] = {
        "momentum_monthly_twoway": float(ls_turnover.mean()),
        "momentum_annual_twoway": float(ls_turnover.mean() * 12),
        "iv_monthly_twoway": float(iv_turnover.mean()),
        "ew_monthly_twoway": float(ew_turnover.mean()),
        "momentum_net_mean_by_bps": net_mean_by_cost(spread, ls_turnover),
        "active_net_mean_by_bps": {
            str(bps): float((iv_net[bps] - ew_net[bps]).mean()) for bps in COST_LEVELS_BPS
        },
    }

    # --- Downloaded vs workbook factor revision ---
    downloaded = data.load_factors(DATA / "factors_downloaded" / "ff_north_america_3factors.csv")
    revision = {}
    for col in ["mkt_rf", "smb", "hml"]:
        joined = pd.concat([downloaded[col], factor_series[col]], axis=1).dropna()
        revision[col] = float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().max())
    metrics["factor_revision_max_abs_diff"] = revision

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    make_figures(spread, factor_series, deciles, iv, ew, model)
    n_results = len(list(RESULTS.glob("*")))
    n_figures = len(list(FIGURES.glob("*.png")))
    print(f"wrote {n_results} result files and {n_figures} figures")


def make_figures(
    spread: pd.Series,
    factor_series: pd.DataFrame,
    deciles: pd.DataFrame,
    iv: pd.Series,
    ew: pd.Series,
    model: pd.DataFrame,
) -> None:
    """Draw and save all report figures."""
    plt.rcParams.update({"figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3})

    umd = factor_series["wml"].reindex(spread.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        spread.index.to_timestamp(), spread.to_numpy(), label="Decile 10 - Decile 1", linewidth=1.0
    )
    ax.plot(umd.index.to_timestamp(), umd.to_numpy(), label="Kenneth French UMD", linewidth=1.0)
    ax.set_title("Momentum strategy and the UMD factor")
    ax.set_ylabel("Monthly return")
    ax.legend()
    save_figure(fig, "momentum_vs_umd.png")

    decile_means = [deciles[f"d{i}"].mean() for i in range(1, 11)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 11), decile_means, color="#3b6ea5")
    ax.set_title("Average monthly return by formation decile")
    ax.set_xlabel("Decile (1 = losers, 10 = winners)")
    ax.set_ylabel("Mean monthly return")
    ax.set_xticks(range(1, 11))
    save_figure(fig, "decile_means.png")

    cumulative = (1.0 + spread).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative.index.to_timestamp(), cumulative.to_numpy(), color="#3b6ea5")
    ax.set_title("Cumulative return of the momentum long-short")
    ax.set_ylabel("Cumulative return")
    save_figure(fig, "momentum_cumulative.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for column, label in [("beta_mkt", "Market"), ("beta_smb", "Size"), ("beta_hml", "Value")]:
        ax.plot(model.index.to_timestamp(), model[column].to_numpy(), label=label, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Rolling 12-month factor betas of the active return")
    ax.set_ylabel("Beta")
    ax.legend()
    save_figure(fig, "rolling_betas.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        model.index.to_timestamp(),
        model["annualised_alpha"].to_numpy(),
        color="#4a7c3f",
        linewidth=1.0,
    )
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Annualised active alpha")
    ax.set_ylabel("Annualised alpha")
    save_figure(fig, "annualised_alpha.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        model.index.to_timestamp(),
        model["factor_part"].to_numpy(),
        label="Systematic",
        linewidth=1.0,
    )
    ax.plot(
        model.index.to_timestamp(),
        model["active_alpha"].to_numpy(),
        label="Idiosyncratic",
        linewidth=1.0,
    )
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Systematic and idiosyncratic active return")
    ax.set_ylabel("Monthly return")
    ax.legend()
    save_figure(fig, "systematic_idiosyncratic.png")

    iv_cum = (1.0 + iv).cumprod() - 1.0
    ew_cum = (1.0 + ew).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        iv_cum.index.to_timestamp(), iv_cum.to_numpy(), label="Inverse volatility", linewidth=1.0
    )
    ax.plot(ew_cum.index.to_timestamp(), ew_cum.to_numpy(), label="Equal weight", linewidth=1.0)
    ax.set_title("Inverse-volatility and equal-weight portfolios")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    save_figure(fig, "iv_vs_ew_cumulative.png")


if __name__ == "__main__":
    main()
