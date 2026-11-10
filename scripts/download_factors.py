"""Download the Fama-French North American factors from the Kenneth French data library.

Fetches the North American 3 factors (market, size, value plus the risk-free rate) and
the North American momentum factor, parses the monthly blocks, converts the percentage
values to decimals, and records the download date. Run from the repository root:

    python -I scripts/download_factors.py
"""

from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import re
import ssl
import urllib.request
import zipfile

import certifi

#: TLS context using the certifi CA bundle, so verification works on machines whose
#: Python install does not ship with system root certificates (common on macOS).
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "factors_downloaded"

#: Each source: the zip on the library, and the value columns in its monthly block.
SOURCES = {
    "ff_north_america_3factors": {
        "zip": "North_America_3_Factors_CSV.zip",
        "columns": ["mkt_rf", "smb", "hml", "rf"],
    },
    "ff_north_america_mom": {
        "zip": "North_America_Mom_Factor_CSV.zip",
        "columns": ["wml"],
    },
}

MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,(.*)$")


def fetch_zip(zip_name: str) -> tuple[bytes, bytes]:
    """Download a zip from the data library and return the raw zip and its single CSV as bytes."""
    url = BASE + zip_name
    request = urllib.request.Request(url, headers={"User-Agent": "factor-download-script"})
    with urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT) as response:  # noqa: S310
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
    return payload, archive.read(csv_name)


def parse_monthly(csv_bytes: bytes, columns: list[str]) -> list[dict]:
    """Parse the monthly block of a Kenneth French CSV into decimal factor rows.

    Rows in the monthly block start with a six-digit ``YYYYMM`` date; annual rows start
    with a four-digit year and are skipped. Percentage values are divided by 100.
    """
    rows: list[dict] = []
    for line in csv_bytes.decode("utf-8", "replace").splitlines():
        match = MONTH_ROW.match(line)
        if match is None:
            continue
        values = [part.strip() for part in match.group(2).split(",") if part.strip() != ""]
        if len(values) != len(columns):
            continue
        month = match.group(1)
        record = {"date": f"{month[:4]}-{month[4:]}"}
        pairs = zip(columns, values, strict=True)
        record.update({name: float(value) / 100.0 for name, value in pairs})
        rows.append(record)
    return rows


def write_csv(path: pathlib.Path, rows: list[dict], columns: list[str]) -> None:
    """Write parsed factor rows to a CSV with a ``date`` column and the given columns."""
    header = ["date", *columns]
    lines = [",".join(header)]
    lines.extend(",".join(str(row[name]) for name in header) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    metadata = {
        "download_date": dt.date.today().isoformat(),
        "source": "Kenneth R. French Data Library",
        "base_url": BASE,
        "files": {},
    }
    for name, spec in SOURCES.items():
        payload, csv_bytes = fetch_zip(spec["zip"])
        (OUT / "raw" / spec["zip"]).write_bytes(payload)
        rows = parse_monthly(csv_bytes, spec["columns"])
        write_csv(OUT / f"{name}.csv", rows, spec["columns"])
        metadata["files"][name] = {
            "zip": spec["zip"],
            "rows": len(rows),
            "first_month": rows[0]["date"],
            "last_month": rows[-1]["date"],
        }
        print(f"{name}: {len(rows)} months, {rows[0]['date']} to {rows[-1]['date']}")
    (OUT / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"download date recorded: {metadata['download_date']}")


if __name__ == "__main__":
    main()
