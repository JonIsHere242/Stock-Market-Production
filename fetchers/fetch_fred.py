"""Download FRED (St. Louis Fed) macroeconomic series via the official API.

Requires a free API key — register once at
  https://fredaccount.stlouisfed.org/apikey
then set the env var:
  PowerShell: $env:FRED_API_KEY = 'your_key_here'
  bash:       export FRED_API_KEY='your_key_here'
The script also reads C:/Users/Masam/Desktop/Stock-Market/.fred_api_key if
present (single line containing the key) — convenient for refresh runs.

~50 macro series covering: Treasury yields, yield curve spreads, credit
spreads, VIX, Fed Funds, financial conditions, employment, inflation, GDP,
oil/commodities, money supply, FX, mortgages, recession indicators.

Refresh: re-running pulls latest observations for every series. Idempotent.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, PROJECT_ROOT, RateLimiter, fmt_bytes, log, make_session

# FRED API allows ~120 req/min = 2/sec; we stay under to be safe.
_fred_limiter = RateLimiter(hz=1.5)

API = "https://api.stlouisfed.org/fred/series/observations"
OUT_DIR = DATA_ROOT / "FRED"
KEY_FILE = PROJECT_ROOT / ".fred_api_key"

SERIES = [
    # Treasury yields (constant maturity)
    ("DGS3MO", "Treasury 3-Month"),
    ("DGS6MO", "Treasury 6-Month"),
    ("DGS1", "Treasury 1-Year"),
    ("DGS2", "Treasury 2-Year"),
    ("DGS3", "Treasury 3-Year"),
    ("DGS5", "Treasury 5-Year"),
    ("DGS7", "Treasury 7-Year"),
    ("DGS10", "Treasury 10-Year"),
    ("DGS20", "Treasury 20-Year"),
    ("DGS30", "Treasury 30-Year"),
    # Yield curve slopes + inflation expectations
    ("T10Y2Y", "10Y-2Y Spread"),
    ("T10Y3M", "10Y-3M Spread"),
    ("T10YIE", "10Y Breakeven Inflation"),
    ("T5YIE", "5Y Breakeven Inflation"),
    ("T5YIFR", "5Y/5Y Forward Inflation"),
    # Credit spreads
    ("BAMLH0A0HYM2", "ICE BofA HY OAS"),
    ("BAMLC0A4CBBB", "ICE BofA BBB OAS"),
    ("BAMLC0A0CM", "ICE BofA IG Corporate OAS"),
    # Volatility / sentiment
    ("VIXCLS", "VIX"),
    ("VXVCLS", "VXV (3-month)"),
    # Fed policy
    ("DFF", "Effective Fed Funds (Daily)"),
    ("FEDFUNDS", "Effective Fed Funds (Monthly)"),
    ("DFEDTARU", "Fed Funds Target Upper"),
    ("DFEDTARL", "Fed Funds Target Lower"),
    # Financial conditions
    ("NFCI", "Chicago Fed NFCI"),
    ("ANFCI", "Chicago Fed Adjusted NFCI"),
    ("STLFSI4", "St Louis Fed Financial Stress Index"),
    # Labor
    ("UNRATE", "Unemployment Rate"),
    ("ICSA", "Initial Jobless Claims"),
    ("CCSA", "Continued Jobless Claims"),
    ("PAYEMS", "Total Nonfarm Payrolls"),
    # Inflation
    ("CPIAUCSL", "CPI All Urban"),
    ("CPILFESL", "Core CPI"),
    ("PCEPI", "PCE Price Index"),
    ("PCEPILFE", "Core PCE"),
    # Growth
    ("GDPC1", "Real GDP"),
    ("GDP", "Nominal GDP"),
    ("INDPRO", "Industrial Production"),
    # Commodities
    ("DCOILWTICO", "Crude Oil WTI"),
    ("DCOILBRENTEU", "Crude Oil Brent"),
    ("GOLDAMGBD228NLBM", "Gold London PM"),
    # Money supply
    ("M2SL", "M2 Money Stock"),
    ("BOGMBASE", "Monetary Base"),
    # FX
    ("DTWEXBGS", "Trade-Weighted USD (Broad)"),
    ("DEXUSEU", "USD/EUR"),
    ("DEXJPUS", "JPY/USD"),
    ("DEXCHUS", "CNY/USD"),
    ("DEXUSUK", "USD/GBP"),
    # Mortgages / housing
    ("MORTGAGE30US", "30Y Mortgage Rate"),
    ("HOUST", "Housing Starts"),
    # Recession indicators
    ("USREC", "NBER Recession Indicator"),
    ("USRECDM", "NBER Recession Indicator (Daily)"),
    # Sentiment
    ("UMCSENT", "U Michigan Consumer Sentiment"),
]


def _get_api_key() -> str | None:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        text = KEY_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    return None


def _fetch_one(session, series_id: str, api_key: str, force: bool = False) -> tuple[str, int, int]:
    dest = OUT_DIR / f"{series_id}.parquet"
    if dest.exists() and not force:
        n_bytes = dest.stat().st_size
        try:
            n_rows = len(pd.read_parquet(dest, columns=["date"]))
        except Exception:
            n_rows = 0
        return series_id, n_rows, n_bytes
    _fred_limiter.wait()
    r = session.get(API, params={
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        return series_id, 0, 0
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value", "realtime_start", "realtime_end"]]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_parquet(dest, index=False)
    return series_id, len(df), dest.stat().st_size


def fetch(workers: int = 2, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = _get_api_key()
    if not api_key:
        log("FRED_API_KEY not set — skipping FRED fetch.")
        log("  Register a free key at https://fredaccount.stlouisfed.org/apikey")
        log(f"  Then either set $env:FRED_API_KEY or write the key to {KEY_FILE}")
        return

    session = make_session()
    log(f"Fetching {len(SERIES)} FRED series via API ({workers} workers)")

    total_rows = total_bytes = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, session, sid, api_key, force): (sid, name)
                   for sid, name in SERIES}
        for fut in as_completed(futures):
            sid, name = futures[fut]
            try:
                series_id, n_rows, n_bytes = fut.result()
                if n_rows == 0:
                    failed.append(sid)
                    log(f"  FAIL {sid:18s} {name}  (empty)")
                else:
                    total_rows += n_rows
                    total_bytes += n_bytes
                    log(f"  ok   {sid:18s} {n_rows:>7,} rows  {fmt_bytes(n_bytes):>9s}  {name}")
            except Exception as e:
                failed.append(sid)
                log(f"  FAIL {sid:18s} {name}: {e}")

    log(f"Done. {len(SERIES) - len(failed)}/{len(SERIES)} series, "
        f"{total_rows:,} rows, {fmt_bytes(total_bytes)}")
    if failed:
        log(f"  failed: {failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="re-fetch already-cached series")
    a = ap.parse_args()
    fetch(workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
