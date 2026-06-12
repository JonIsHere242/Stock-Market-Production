"""Download Robert Shiller's long-history US market dataset.

Single Excel file with monthly S&P composite price, dividends, earnings,
CPI, long-term interest rates, and the derived CAPE (cyclically adjusted
P/E) ratio. Series start 1871.

Use for: long-horizon valuation context, CAPE-based regime flagging,
inflation-adjusted return baselines.

Refresh: Shiller updates this file ~monthly. Re-run to pull the latest.

URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, fmt_bytes, log, make_session

URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
OUT_DIR = DATA_ROOT / "Shiller"


def fetch() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    log(f"Downloading Shiller dataset from {URL}")
    r = session.get(URL, timeout=120)
    r.raise_for_status()
    xls_path = OUT_DIR / "ie_data.xls"
    xls_path.write_bytes(r.content)
    log(f"  wrote ie_data.xls ({fmt_bytes(len(r.content))})")

    # Also extract the 'Data' sheet to parquet for easy consumption.
    # The Shiller file has a 7-row header before the table starts.
    try:
        df = pd.read_excel(xls_path, sheet_name="Data", skiprows=7, engine="xlrd")
        # Coerce all non-date columns to numeric; rows with footnote strings
        # (e.g. "Sept price is Sept 1st close") become NaN and get dropped.
        date_col = df.columns[0]
        for c in df.columns:
            if c == date_col:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[date_col]).reset_index(drop=True)
        # Drop trailing rows where all numeric cols are NaN (footer)
        num_cols = [c for c in df.columns if c != date_col]
        df = df[df[num_cols].notna().any(axis=1)].reset_index(drop=True)
        out = OUT_DIR / "shiller_data.parquet"
        df.to_parquet(out, index=False)
        log(f"  parsed Data sheet → {out.name} ({len(df):,} rows, {fmt_bytes(out.stat().st_size)})")
    except Exception as e:
        log(f"  WARN: could not parse Data sheet to parquet ({e}); raw xls saved")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    fetch()


if __name__ == "__main__":
    main()
