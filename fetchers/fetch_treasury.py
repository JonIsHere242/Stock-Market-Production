"""Download US Treasury daily par yield curve — canonical primary source.

FRED's DGS series ultimately come from this Treasury table; we keep a
local copy as the authoritative source for cross-checking and because
Treasury publishes the *full curve* (all maturities) in one CSV per year.

Refresh: updates daily. Re-running pulls the current year fresh and
caches prior years.

URL pattern: https://home.treasury.gov/resource-center/data-chart-center/
  interest-rates/daily-treasury-rates.csv/{year}/all
    ?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, fmt_bytes, log, make_session

BASE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
OUT_DIR = DATA_ROOT / "Treasury"
START_YEAR = 1990


def _fetch_year(session, year: int) -> tuple[int, int]:
    url = BASE.format(year=year)
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return 0, 0
    r.raise_for_status()
    try:
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception:
        return 0, 0
    if df.empty:
        return 0, 0
    # Normalize date column
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date"}).dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dest = OUT_DIR / f"daily_par_yield_{year}.parquet"
    df.to_parquet(dest, index=False)
    return len(df), dest.stat().st_size


def fetch(start_year: int = START_YEAR, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    today = dt.date.today()
    years = range(start_year, today.year + 1)
    log(f"Fetching Treasury par yield curves for {start_year}–{today.year}")

    total_rows = total_bytes = 0
    for y in years:
        dest = OUT_DIR / f"daily_par_yield_{y}.parquet"
        if dest.exists() and not force and y < today.year:
            log(f"  {y} cached")
            continue
        try:
            n_rows, n_bytes = _fetch_year(session, y)
            total_rows += n_rows
            total_bytes += n_bytes
            log(f"  {y}  {n_rows:>4} rows  {fmt_bytes(n_bytes)}")
        except Exception as e:
            log(f"  {y}  FAIL: {e}")
    log(f"Done. {total_rows:,} rows total this run, {fmt_bytes(total_bytes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fetch(start_year=a.start_year, force=a.force)


if __name__ == "__main__":
    main()
