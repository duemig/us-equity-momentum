# US equity momentum and inverse-volatility study

A faithful Python rebuild of a quantitative portfolio study on 100 US large-cap stocks,
monthly from December 2002 to August 2018. It covers a decile momentum
strategy, an inverse-volatility portfolio against an equal-weight benchmark with a rolling
Fama-French factor decomposition, and a short volatility-decomposition identity, plus
robustness checks.

It started as an assignment for the Quantitative Portfolio Management course at Frankfurt School.

The full write-up, with method, results, robustness, and references, is in
[report/report.pdf](report/report.pdf).

## Main results

- Momentum (top decile minus bottom decile, equal weighted) returned -0.45% per
  month over 176 months, with a worst month of -56.88% in
  April 2009. The Newey-West t-statistic is -0.65, so the strategy is
  not distinguishable from zero. Average returns fall from 1.08% per month in the
  loser decile to 0.63% in the winner decile, so momentum did not pay in this
  sample of large caps.
- Against the Carhart four-factor model the strategy loads heavily on the momentum factor
  (beta 1.84, t 14.45) yet still shows a significant
  negative alpha of -0.72% per month (t -2.22): a 10/90
  equal-weighted long-short is a weaker version of the published factor.
- The inverse-volatility portfolio returned 0.80% per month (Sharpe
  0.69) against 0.87% for the equal-weight benchmark, so the
  active return was -0.07% per month and not significant (t -1.04).
- One tie-driven decile error in the original spreadsheet was found and documented under
  `verification/`. See `verification/REPLICATION_FINDINGS.md`.

| Portfolio | Mean per month | Sharpe (annual) | Cumulative |
|---|---|---|---|
| Momentum long-short | -0.45% | -0.18 | -80% |
| Inverse volatility | 0.80% | 0.69 | 265% |
| Equal weight | 0.87% | 0.66 | 301% |

The universe is a fixed set of survivors, so the long-only returns are likely biased upward; the
report covers this and the other caveats. Sharpe ratios for the long-only portfolios use returns in
excess of the risk-free rate.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

python scripts/download_factors.py     # refresh Kenneth French factors (records the download date)
python scripts/run_study.py            # write results/ and figures/
python scripts/verify_replication.py   # compare against the original spreadsheet series
bash report/build.sh                    # write report assets and build report/report.pdf
pytest                                  # run the test suite
```

The stock prices and the originally used factor series are in `data/`. The study reads those,
writes `results/` and `figures/`, and the report and this README take every number from
`results/metrics.json`.

## Layout

- `src/qpm/` the package: returns, momentum, portfolios, factor model, statistics, turnover.
- `scripts/` data extraction, factor download, the study runner, the report writer, verification.
- `tests/` unit tests on small synthetic data.
- `report/` the LaTeX report and its build script.
- `verification/` the comparison against the original spreadsheet.
