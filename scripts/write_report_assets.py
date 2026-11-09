"""Turn the study results into LaTeX inputs and the README.

Reads ``results/metrics.json`` and writes, under ``report/generated``, a macro file with every
scalar the report quotes and the data tables, and writes ``README.md`` at the repository root.
No number in the report or the README is typed by hand; they all come from this script.

Run after ``run_study.py``:

    python scripts/write_report_assets.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
GENERATED = REPO / "report" / "generated"

VARIANT_LABELS = {
    "skip_12_1": "12 months, skip most recent",
    "no_skip_12": "12 months, no skip",
    "skip_6_1": "6 months, skip most recent",
}


def pct(value: float, decimals: int = 2) -> str:
    """Format a fraction as a LaTeX percentage string, for example -0.45\\%."""
    return f"{value * 100:.{decimals}f}\\%"


def num(value: float, decimals: int = 2) -> str:
    """Format a number with a fixed number of decimals."""
    return f"{value:.{decimals}f}"


def month_name(year_month: str) -> str:
    """Convert a `YYYY-MM` string to a month name, for example April 2009."""
    return dt.datetime.strptime(year_month, "%Y-%m").strftime("%B %Y")


def build_macros(metrics: dict) -> dict[str, str]:
    """Assemble the named LaTeX macros the report quotes."""
    sample = metrics["sample"]
    mom = metrics["momentum"]
    alphas = metrics["momentum_alphas"]
    iv = metrics["iv_portfolio"]
    ew = metrics["ew_benchmark"]
    active = metrics["active"]
    active_alpha = metrics["active_alpha_ff3"]
    rolling = metrics["rolling_model"]
    turn = metrics["turnover"]
    revision = metrics["factor_revision_max_abs_diff"]
    return {
        "sampleStart": month_name(sample["return_start"]),
        "sampleEnd": month_name(sample["return_end"]),
        "nStocks": str(sample["n_stocks"]),
        "nMonths": str(sample["n_return_months"]),
        "momMean": pct(mom["mean_monthly"]),
        "momAnn": pct(mom["mean_annual"]),
        "momVol": pct(mom["vol_annual"]),
        "momSharpe": num(mom["sharpe_annual"]),
        "momTnw": num(mom["t_nw"]),
        "momMin": pct(mom["min"]),
        "momMinMonth": month_name(mom["min_month"]),
        "momHit": pct(mom["hit_rate"], 1),
        "momCum": pct(mom["cumulative"], 1),
        "momMonths": str(mom["n_months"]),
        "decLosers": pct(mom["decile_means"]["d1"]),
        "decWinners": pct(mom["decile_means"]["d10"]),
        "quintMean": pct(metrics["quintiles"]["mean_monthly"]),
        "quintTnw": num(metrics["quintiles"]["t_nw"]),
        "capmAlpha": pct(alphas["capm"]["alpha_monthly"]),
        "capmAlphaT": num(alphas["capm"]["alpha_t"]),
        "ffThreeAlpha": pct(alphas["ff3"]["alpha_monthly"]),
        "ffThreeAlphaT": num(alphas["ff3"]["alpha_t"]),
        "ffFourAlpha": pct(alphas["ff4"]["alpha_monthly"]),
        "ffFourAlphaAnn": pct(alphas["ff4"]["alpha_annual"]),
        "ffFourAlphaT": num(alphas["ff4"]["alpha_t"]),
        "ffFourWmlBeta": num(alphas["ff4"]["betas"]["wml"]),
        "ffFourWmlT": num(alphas["ff4"]["beta_t"]["wml"]),
        "ffFourRsq": num(alphas["ff4"]["r_squared"]),
        "umdCorr": num(metrics["umd"]["correlation"]),
        "ivMean": pct(iv["mean_monthly"]),
        "ivAnn": pct(iv["mean_annual"]),
        "ivSharpe": num(iv["sharpe_annual"]),
        "ivTnw": num(iv["t_nw"]),
        "ivCum": pct(iv["cumulative"], 0),
        "ewMean": pct(ew["mean_monthly"]),
        "ewSharpe": num(ew["sharpe_annual"]),
        "ewCum": pct(ew["cumulative"], 0),
        "activeMean": pct(active["mean_monthly"]),
        "activeTnw": num(active["t_nw"]),
        "activeAlpha": pct(active_alpha["alpha_monthly"]),
        "activeAlphaT": num(active_alpha["alpha_t"]),
        "betaMktMean": num(rolling["beta_mkt_mean"]),
        "betaSmbMean": num(rolling["beta_smb_mean"]),
        "betaHmlMean": num(rolling["beta_hml_mean"]),
        "rollRsq": num(rolling["r2_mean"]),
        "momTurnover": pct(turn["momentum_monthly_twoway"], 0),
        "ivTurnover": pct(turn["iv_monthly_twoway"], 1),
        "ewTurnover": pct(turn["ew_monthly_twoway"], 1),
        "momNetFifty": pct(turn["momentum_net_mean_by_bps"]["50"]),
        "revMkt": num(revision["mkt_rf"] * 100, 2),
        "revSmb": num(revision["smb"] * 100, 2),
        "revHml": num(revision["hml"] * 100, 2),
    }


def write_macros(macros: dict[str, str]) -> None:
    """Write the LaTeX macro definitions."""
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (GENERATED / "macros.tex").write_text("\n".join(lines) + "\n")


def tabular(header: list[str], rows: list[list[str]], spec: str) -> str:
    """Build a booktabs tabular environment as a string."""
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out.extend(" & ".join(row) + " \\\\" for row in rows)
    out.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(out) + "\n"


def write_tables(metrics: dict) -> None:
    """Write the data tables the report includes."""
    formation_rows = [
        [
            VARIANT_LABELS[row["variant"]],
            str(row["n_months"]),
            pct(row["mean_monthly"]),
            num(row["t_nw"]),
            num(row["sharpe_annual"]),
        ]
        for row in metrics["formation_variants"]
    ]
    (GENERATED / "table_formation.tex").write_text(
        tabular(
            ["Formation window", "Months", "Mean (per month)", "NW $t$", "Sharpe"],
            formation_rows,
            "lcccc",
        )
    )

    alphas = metrics["momentum_alphas"]
    alpha_rows = []
    for name, label in [("capm", "CAPM"), ("ff3", "Fama-French 3"), ("ff4", "Carhart 4")]:
        spec = alphas[name]
        betas = spec["betas"]
        alpha_rows.append(
            [
                label,
                pct(spec["alpha_monthly"]),
                num(spec["alpha_t"]),
                num(betas.get("mkt_rf", float("nan"))) if "mkt_rf" in betas else "--",
                num(betas["smb"]) if "smb" in betas else "--",
                num(betas["hml"]) if "hml" in betas else "--",
                num(betas["wml"]) if "wml" in betas else "--",
                num(spec["r_squared"]),
            ]
        )
    (GENERATED / "table_alphas.tex").write_text(
        tabular(
            [
                "Model",
                "$\\alpha$ (mo)",
                "$t(\\alpha)$",
                "$\\beta_{mkt}$",
                "$\\beta_{smb}$",
                "$\\beta_{hml}$",
                "$\\beta_{umd}$",
                "$R^2$",
            ],
            alpha_rows,
            "lccccccc",
        )
    )

    def portfolio_row(label: str, block: str) -> list[str]:
        stats = metrics[block]
        return [
            label,
            pct(stats["mean_monthly"]),
            pct(stats["vol_annual"]),
            num(stats["sharpe_annual"]),
            num(stats["t_nw"]),
            pct(stats["cumulative"], 0),
        ]

    portfolio_rows = [
        portfolio_row("Inverse volatility", "iv_portfolio"),
        portfolio_row("Equal weight", "ew_benchmark"),
        portfolio_row("Active (IV - EW)", "active"),
    ]
    (GENERATED / "table_portfolio.tex").write_text(
        tabular(
            ["Portfolio", "Mean (mo)", "Vol (yr)", "Sharpe", "NW $t$", "Cumulative"],
            portfolio_rows,
            "lccccc",
        )
    )

    turn = metrics["turnover"]
    mom_net = turn["momentum_net_mean_by_bps"]
    act_net = turn["active_net_mean_by_bps"]
    turnover_rows = [
        [
            "Momentum long-short",
            pct(turn["momentum_monthly_twoway"], 0),
            pct(mom_net["0"]),
            pct(mom_net["10"]),
            pct(mom_net["50"]),
        ],
        [
            "Active (IV - EW)",
            f"{pct(turn['iv_monthly_twoway'], 1)} / {pct(turn['ew_monthly_twoway'], 1)}",
            pct(act_net["0"]),
            pct(act_net["10"]),
            pct(act_net["50"]),
        ],
    ]
    (GENERATED / "table_turnover.tex").write_text(
        tabular(
            ["Book", "Turnover (mo)", "Net at 0 bps", "Net at 10 bps", "Net at 50 bps"],
            turnover_rows,
            "lcccc",
        )
    )


def write_readme(metrics: dict, macros: dict[str, str]) -> None:
    """Write the README with the main results drawn from the metrics."""

    def table_row(label: str, block: str) -> str:
        stats = metrics[block]
        return (
            f"| {label} | {stats['mean_monthly'] * 100:.2f}% | "
            f"{stats['sharpe_annual']:.2f} | {stats['cumulative'] * 100:.0f}% |"
        )

    results_table = "\n".join(
        [
            "| Portfolio | Mean per month | Sharpe (annual) | Cumulative |",
            "|---|---|---|---|",
            table_row("Momentum long-short", "momentum"),
            table_row("Inverse volatility", "iv_portfolio"),
            table_row("Equal weight", "ew_benchmark"),
        ]
    )
    readme = f"""# US equity momentum and inverse-volatility study

A faithful Python rebuild of a quantitative portfolio study on 100 US large-cap stocks,
monthly from {macros["sampleStart"]} to {macros["sampleEnd"]}. It covers a decile momentum
strategy, an inverse-volatility portfolio against an equal-weight benchmark with a rolling
Fama-French factor decomposition, and a short analytical proof, plus robustness checks.

It started as an assignment for the Quantitative Portfolio Management course at Frankfurt School.

The full write-up, with method, results, robustness, and references, is in
[report/report.pdf](report/report.pdf).

## Main results

- Momentum (top decile minus bottom decile, equal weighted) returned {macros["momMean"]} per
  month over {macros["momMonths"]} months, with a worst month of {macros["momMin"]} in
  {macros["momMinMonth"]}. The Newey-West t-statistic is {macros["momTnw"]}, so the strategy is
  not distinguishable from zero. Average returns fall from {macros["decLosers"]} per month in the
  loser decile to {macros["decWinners"]} in the winner decile, so momentum did not pay in this
  sample of large caps.
- Against the Carhart four-factor model the strategy loads heavily on the momentum factor
  (beta {macros["ffFourWmlBeta"]}, t {macros["ffFourWmlT"]}) yet still shows a significant
  negative alpha of {macros["ffFourAlpha"]} per month (t {macros["ffFourAlphaT"]}): a 10/90
  equal-weighted long-short is a weaker version of the published factor.
- The inverse-volatility portfolio returned {macros["ivMean"]} per month (Sharpe
  {macros["ivSharpe"]}) against {macros["ewMean"]} for the equal-weight benchmark, so the
  active return was {macros["activeMean"]} per month and not significant (t {macros["activeTnw"]}).
- One tie-driven decile error in the original spreadsheet was found and documented under
  `verification/`. See `verification/REPLICATION_FINDINGS.md`.

{results_table}

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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
"""
    (REPO / "README.md").write_text(readme)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    macros = build_macros(metrics)
    write_macros(macros)
    write_tables(metrics)
    write_readme(metrics, macros)
    print(f"wrote {len(macros)} macros, 4 tables, and README.md")


if __name__ == "__main__":
    main()
