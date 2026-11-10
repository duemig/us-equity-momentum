#!/usr/bin/env bash
# Regenerate the report inputs from the results, then build report/report.pdf.
# Run the study first (python scripts/run_study.py) so results/ exists.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python3 scripts/write_report_assets.py

cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
latexmk -c
echo "built report/report.pdf"
