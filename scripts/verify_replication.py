"""Recompute every part of the study in Python and compare it to the original workbook.

Loads the extracted prices and workbook factor series, rebuilds the momentum deciles, the
inverse-volatility and equal-weight portfolios, and the rolling factor model, then compares
each series against the values pulled from the original spreadsheet (under
``verification/original``). Prints a summary table and writes the report in that folder.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from qpm import data, factors, momentum, portfolio  # noqa: E402

ORIG = REPO / "verification" / "original"


def to_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a CSV frame (with a `date` column) by calendar month."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M")
    return out.set_index("date").sort_index()


def compare(name: str, mine: pd.Series, original: pd.Series, tol: float) -> dict:
    """Align two series on their common months and summarise the difference."""
    joined = pd.concat([mine.rename("mine"), original.rename("orig")], axis=1).dropna()
    diff = (joined["mine"] - joined["orig"]).abs()
    max_diff = float(diff.max()) if len(diff) else float("nan")
    return {
        "component": name,
        "n": int(len(joined)),
        "max_abs_diff": max_diff,
        "verdict": "match" if max_diff <= tol else "DIFFERS",
    }


def main() -> None:
    prices = data.load_prices(REPO / "data" / "prices.csv")
    factor_series = data.load_factors(REPO / "data" / "ff_factors_from_workbook.csv")
    returns = data.compute_returns(prices)

    missing_prices = int(prices.isna().sum().sum())
    missing_returns = int(returns.isna().sum().sum())
    print(f"prices {prices.shape}, returns {returns.shape}")
    print(f"missing prices: {missing_prices}, missing returns: {missing_returns}")

    rows: list[dict] = []

    # Returns
    ret_orig = to_period(pd.read_csv(ORIG / "returns_workbook.csv"))
    common_cols = returns.columns.intersection(ret_orig.columns)
    rdiff = (returns[common_cols] - ret_orig[common_cols]).abs()
    rows.append(
        {
            "component": "monthly returns (all stocks)",
            "n": int(rdiff.notna().sum().sum()),
            "max_abs_diff": float(rdiff.max().max()),
            "verdict": "match" if rdiff.max().max() <= 1e-9 else "DIFFERS",
        }
    )

    # Momentum
    formation = momentum.formation_returns(prices)
    deciles = momentum.decile_returns(formation, returns)
    mom_orig = to_period(pd.read_csv(ORIG / "momentum.csv"))
    for col in [f"d{i}" for i in range(1, 11)] + ["spread"]:
        rows.append(compare(f"momentum {col}", deciles[col], mom_orig[col], 1e-6))
    rows.append(compare("momentum UMD factor (wml)", factor_series["wml"], mom_orig["umd"], 1e-9))

    # Portfolios
    volatility = portfolio.rolling_volatility(returns)
    weights = portfolio.inverse_vol_weights(volatility)
    iv = portfolio.weighted_return(weights, returns).dropna()
    ew = portfolio.equal_weight_return(returns)
    active = (iv - ew).dropna()
    q2 = to_period(pd.read_csv(ORIG / "q2_regression.csv"))
    rows.append(compare("IV portfolio return", iv, q2["iv_ret"], 1e-9))
    rows.append(compare("EW benchmark return", ew, q2["ew_ret"], 1e-9))
    rows.append(compare("active return (IV - EW)", active, q2["active"], 1e-9))

    # Rolling factor model
    model = factors.rolling_factor_model(active, factor_series)
    mapping = {
        "beta_mkt": "beta_mkt",
        "beta_smb": "beta_smb",
        "beta_hml": "beta_hml",
        "intercept": "intercept",
        "active_alpha": "active_alpha",
        "annualised_alpha": "annualised_alpha",
        "systematic": "total_predicted",
        "idiosyncratic": "idio_resid",
        "factor_part": "systematic_factor_return",
    }
    for mine_col, orig_col in mapping.items():
        rows.append(compare(f"regression {mine_col}", model[mine_col], q2[orig_col], 1e-7))

    # Cross-check downloaded factors vs workbook factors over the overlap
    downloaded_path = REPO / "data" / "factors_downloaded" / "ff_north_america_3factors.csv"
    if downloaded_path.exists():
        dl = data.load_factors(downloaded_path)
        for col in ["mkt_rf", "smb", "hml", "rf"]:
            rows.append(compare(f"downloaded vs workbook {col}", dl[col], factor_series[col], 1e-9))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n" + table.to_string(index=False))

    # Headline replicated numbers
    spread = deciles["spread"]
    ew_on_iv = ew.reindex(iv.index).mean()
    print("\nReplicated headline numbers:")
    print(
        f"  momentum spread: mean {spread.mean():.4%}/mo, min {spread.min():.4%} "
        f"on {spread.idxmin()}, n={spread.notna().sum()}"
    )
    print(f"  decile means: D1 {deciles['d1'].mean():.4%}  D10 {deciles['d10'].mean():.4%}")
    print(f"  IV {iv.mean():.4%}/mo, EW {ew_on_iv:.4%}/mo, active {active.mean():.4%}/mo")
    print(
        f"  avg betas: mkt {model['beta_mkt'].mean():.4f}, smb {model['beta_smb'].mean():.4f}, "
        f"hml {model['beta_hml'].mean():.4f}"
    )

    report = [
        "# Replication comparison",
        "",
        f"Prices {prices.shape[0]} months x {prices.shape[1]} stocks. "
        f"Missing prices: {missing_prices}; missing returns: {missing_returns}.",
        "",
        "| Component | Months compared | Max abs diff | Verdict |",
        "|---|---|---|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['component']} | {row['n']} | {row['max_abs_diff']:.2e} | {row['verdict']} |"
        )
    (REPO / "verification" / "replication_report.md").write_text("\n".join(report) + "\n")
    print("\nwrote verification/replication_report.md")


if __name__ == "__main__":
    main()
