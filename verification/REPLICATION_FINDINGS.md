# Replication findings

Every part of the assignment was rebuilt in Python from the extracted prices and the
workbook's own Fama-French factor series, then compared month by month against the result
columns in the original spreadsheet. Run `python scripts/verify_replication.py` to reproduce
the table.

## Result

The rebuild matches the original to floating-point precision everywhere except one isolated
spreadsheet error. Maximum absolute differences:

| Component | Months | Max abs diff | Verdict |
|---|---|---|---|
| Monthly returns (all 100 stocks) | 18,900 cells | 2e-16 | match |
| Momentum deciles d1, d2, d4...d10 | 176 each | <3e-17 | match |
| Momentum decile d3 | 176 | 2.7e-3 | one month differs (see below) |
| Momentum spread (d10 - d1) | 176 | 1e-16 | match |
| Ken French UMD factor | 189 | 1e-17 | match |
| IV portfolio return | 177 | 5e-17 | match |
| EW benchmark return | 177 | 1e-17 | match |
| Active return (IV - EW) | 177 | 6e-17 | match |
| Rolling betas, intercept, alpha, decomposition | 164 each | <4e-15 | match |

## Finding 1: decile-3 tie error in February 2010

In the workbook's sorted holding-return block, decile 3 for February 2010 lists ADM twice
and omits BMY. ADM and BMY have the same formation return that month (both 8.60%), and the
spreadsheet's rank-and-sort broke the tie by duplicating ADM and dropping BMY. Decile 3
therefore uses ADM's holding return (-2.04%) where BMY's (+0.62%) belongs, so the workbook's
decile-3 return is too low by `0.1 * (0.0062 - (-0.0204)) = 0.00266`. This matches the
observed difference. The error touches only decile 3 in that single month. The momentum
spread, which is the actual strategy, is not affected because neither ADM nor BMY sits in the
top or bottom decile. The Python rebuild ranks each stock once, so it does not reproduce the
duplication.

## Finding 2: Ken French factor revisions since 2018

The freshly downloaded North American 3-factor series (download date in
`data/factors_downloaded/download_metadata.json`) differs slightly from the series pasted into
the workbook in 2018 and 2019: up to 0.18 points on the market factor, 0.28 on size, and 0.73
on value in individual months. The risk-free rate is identical. These are routine revisions
that Kenneth French applies to the North American factor history, not an error. The
replication uses the workbook's factor series, as the brief requested; the downloaded series
is kept for the robustness work and as a record.

## Workbook artifacts navigated, not errors

- The regression area carries leftover columns that apply `(1+x)^12-1` five times in a row, so
  their values reach 1e138. Only the first annualization is meaningful, and the report's charts
  use it. The rebuild annualizes once.
- Two systematic-versus-idiosyncratic shares sit side by side, one from the R-squared of the
  active-return regression (about 0.61 / 0.39) and one from the IV regression (about 0.93 /
  0.07). The active-return decomposition is the relevant one.

## Methods pinned from the formulas

- Formation return: `P[t-2] / P[t-14] - 1`, a 12-month return that skips the most recent month.
- Volatility: sample standard deviation over the trailing 12 months `[t-12, t-1]`, excluding
  the current month.
- Rolling regression: multivariate OLS (Excel LINEST) of active return on market, size, and
  value over the trailing 12 months; the betas are then applied to the current month's factors.

## Replicated headline numbers (identical to the original)

- Universe: 100 US large-cap stocks, monthly, November 2002 to August 2018.
- Momentum D10 - D1: mean -0.45% per month, minimum -56.88% in April 2009, 176 months.
  Decile averages are not monotonic: losers (D1) returned 1.08% per month against winners
  (D10) at 0.63%, so momentum did not pay in this sample.
- Inverse-volatility portfolio 0.80% per month, equal-weight benchmark 0.87%, active -0.074%.
- Average rolling active betas: market -0.060, size -0.084, value -0.054.
