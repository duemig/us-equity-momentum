"""Extract the stock prices, factor series, and original result columns from the workbook.

The raw assignment workbook is not distributed with this repository. This script documents how
the committed CSV files under ``data`` and ``verification/original`` were produced from it, so the
data can be regenerated if the workbook is available. Column positions are specific to that
workbook.

    python scripts/extract_from_workbook.py --workbook /path/to/QPM-Assignment_Final.xlsx
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Columns pulled from the "2b Factor Betas" sheet, by position, into the verification file.
Q2_COLUMNS = {
    "iv_ret": 1,
    "ew_ret": 2,
    "active": 3,
    "rf": 5,
    "mkt_rf": 6,
    "smb": 7,
    "hml": 8,
    "beta_hml": 9,
    "beta_smb": 10,
    "beta_mkt": 11,
    "intercept": 12,
    "active_alpha": 13,
    "annualised_alpha": 14,
    "total_predicted": 23,
    "idio_resid": 28,
    "systematic_factor_return": 29,
    "idiosyncratic_return": 30,
}


def extract_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract a date-by-ticker panel from a Bloomberg-style sheet.

    The ticker header row is found from the first stock name, the date column sits immediately
    to its left, and the data runs from the next row. Ticker suffixes are stripped.
    """
    location = np.argwhere(frame.values == "AXP UN Equity")
    header_row, first_column = int(location[0][0]), int(location[0][1])
    tickers = [frame.iat[header_row, first_column + offset] for offset in range(100)]
    body = frame.iloc[header_row + 1 :, :]
    dates = pd.to_datetime(body[first_column - 1], errors="coerce")
    valid = dates.notna()
    panel = body.loc[valid, [first_column + offset for offset in range(100)]].copy()
    panel.columns = [str(ticker).split(" ")[0] for ticker in tickers]
    panel.insert(0, "date", dates[valid].dt.strftime("%Y-%m-%d").to_numpy())
    return panel.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", required=True, type=pathlib.Path, help="Path to the source xlsx."
    )
    arguments = parser.parse_args()

    data_dir = REPO / "data"
    original_dir = REPO / "verification" / "original"
    data_dir.mkdir(exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(
        arguments.workbook,
        sheet_name=[
            "Price",
            "Return",
            "Fama French Factors",
            "1biv bv c Momentum",
            "2b Factor Betas",
        ],
        header=None,
    )

    extract_panel(sheets["Price"]).to_csv(data_dir / "prices.csv", index=False)
    extract_panel(sheets["Return"]).to_csv(original_dir / "returns_workbook.csv", index=False)

    factors = sheets["Fama French Factors"]
    rows = range(7, 196)
    factor_frame = pd.DataFrame(
        {
            "date": pd.to_datetime([factors.iat[row, 0] for row in rows]).strftime("%Y-%m-%d"),
            "mkt_rf": [factors.iat[row, 9] for row in rows],
            "smb": [factors.iat[row, 10] for row in rows],
            "hml": [factors.iat[row, 11] for row in rows],
            "rf": [factors.iat[row, 12] for row in rows],
            "wml": [factors.iat[row, 13] for row in rows],
        }
    )
    factor_frame.to_csv(data_dir / "ff_factors_from_workbook.csv", index=False)

    momentum_sheet = sheets["1biv bv c Momentum"].iloc[6:, :]
    momentum_dates = pd.to_datetime(momentum_sheet[0], errors="coerce")
    keep = momentum_dates.notna()
    momentum = pd.DataFrame({"date": momentum_dates[keep].dt.strftime("%Y-%m-%d").to_numpy()})
    for decile in range(1, 11):
        momentum[f"d{decile}"] = pd.to_numeric(
            momentum_sheet.loc[keep, decile], errors="coerce"
        ).to_numpy()
    momentum["spread"] = pd.to_numeric(momentum_sheet.loc[keep, 13], errors="coerce").to_numpy()
    momentum["umd"] = pd.to_numeric(momentum_sheet.loc[keep, 16], errors="coerce").to_numpy()
    momentum.to_csv(original_dir / "momentum.csv", index=False)

    betas_sheet = sheets["2b Factor Betas"].iloc[5:, :]
    betas_dates = pd.to_datetime(betas_sheet[0], errors="coerce")
    keep = betas_dates.notna()
    regression = pd.DataFrame({"date": betas_dates[keep].dt.strftime("%Y-%m-%d").to_numpy()})
    for name, position in Q2_COLUMNS.items():
        regression[name] = pd.to_numeric(
            betas_sheet.loc[keep, position], errors="coerce"
        ).to_numpy()
    regression.to_csv(original_dir / "q2_regression.csv", index=False)

    print(
        "wrote data/prices.csv, data/ff_factors_from_workbook.csv, and verification/original/*.csv"
    )


if __name__ == "__main__":
    main()
