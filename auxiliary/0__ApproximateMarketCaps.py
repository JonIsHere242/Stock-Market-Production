"""
Build an approximate point-in-time market cap table from existing data only.

Strategy:
    1. Read the static market-cap snapshot from ``Data/fundamental_cache.pkl``
       (a single Finviz dump, dated 2025-10-13 in the current file).
    2. For each ticker, compute the **21-day trailing median Close** ending
       at the anchor date. Median (not last close) eliminates single-day
       anomalies -- e.g., BYND's anchor landing on the first day of a 7x
       meme spike pushed naive implied-shares ~30% too high.
    3. Derive a constant approximate share count:
           shares_approx = market_cap_anchor / median_close_21d
    4. Multiply ``shares_approx`` by the full historical Close series to get
       a per-date market cap estimate. Splits are already adjusted in the
       price data (verified manually on NVDA 2024-06-10 10:1).

Limitations (acceptable for a $300M micro-cap filter):
    * Treats shares outstanding as constant -- ignores issuance, buybacks,
      and split changes that aren't reflected in adjusted closes.
    * Loses the leakage from using a future-dated snapshot ONLY for stocks
      whose share count actually changed materially in the window. For most
      large/mid caps over a 13-month window the error is <5%.

Output: ``Data/MarketCaps/historical_market_caps.parquet`` -- long format
columns ``Ticker, Date, MarketCap`` (USD, not millions).

For an exact PIT pull, see SEC EDGAR ``companyfacts.zip`` (research notes
saved separately). This script is the cheap-and-fast first cut.

Run:
    python 0__ApproximateMarketCaps.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- path shim: this script lives in auxiliary/, so add the project root to sys.path
# to keep `from Util import ...` working when run directly. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from Util import get_logger

logger = get_logger("ApproximateMarketCaps")

CACHE_PATH = Path("Data/fundamental_cache.pkl")
PRICE_DIR = Path("Data/PriceData")
OUT_DIR = Path("Data/MarketCaps")
OUT_FILE = OUT_DIR / "historical_market_caps.parquet"


def _anchor_date_from_cache() -> pd.Timestamp:
    """Use the cache file's mtime as the anchor (when the snapshot was taken)."""
    return pd.Timestamp(CACHE_PATH.stat().st_mtime, unit="s").normalize()


SMOOTH_WINDOW = 21


def _close_on_or_before(df: pd.DataFrame, anchor: pd.Timestamp) -> float | None:
    """Return the most recent Close price on or before ``anchor``, or None.

    Kept for callers that explicitly want the single-day value.
    """
    eligible = df[df["Date"] <= anchor]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["Close"])


def _smoothed_anchor_close(df: pd.DataFrame, anchor: pd.Timestamp,
                            window: int = SMOOTH_WINDOW) -> float | None:
    """Median Close over the trailing ``window`` trading days ending at anchor.

    Smooths out single-day anomalies (e.g., BYND's anchor day landing on the
    first day of a 7x meme spike, which made implied shares ~30% too high
    when divided into the static market cap). Trailing-only -- never peeks
    forward of the anchor. Falls back to single-day close when the window
    has insufficient data.
    """
    eligible = df[df["Date"] <= anchor].tail(window)
    if eligible.empty:
        return None
    return float(eligible["Close"].median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not CACHE_PATH.exists():
        logger.error(f"Cache not found: {CACHE_PATH}")
        return

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    if "market_cap_millions" not in cache:
        logger.error("'market_cap_millions' missing from fundamental_cache.pkl")
        return

    caps_millions = cache["market_cap_millions"]  # {ticker: float | None}
    anchor = _anchor_date_from_cache()
    logger.info(f"Cache anchor date: {anchor.date()}, tickers in cache: {len(caps_millions)}")

    rows = []
    skipped_no_cap = 0
    skipped_no_price = 0
    skipped_no_anchor_close = 0

    for ticker, cap_m in tqdm(caps_millions.items(), desc="Building PIT caps"):
        if cap_m is None or not np.isfinite(cap_m) or cap_m <= 0:
            skipped_no_cap += 1
            continue

        fp = PRICE_DIR / f"{ticker}.parquet"
        if not fp.exists():
            skipped_no_price += 1
            continue

        try:
            px = pd.read_parquet(fp, columns=["Date", "Close"])
        except Exception:
            skipped_no_price += 1
            continue

        if px.empty:
            skipped_no_price += 1
            continue
        px["Date"] = pd.to_datetime(px["Date"])
        px = px.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

        # 21-day trailing median Close eliminates single-day anchor anomalies
        # for high-vol names (e.g. BYND on its meme-spike day).
        anchor_close = _smoothed_anchor_close(px, anchor)
        if anchor_close is None or anchor_close <= 0:
            skipped_no_anchor_close += 1
            continue

        cap_anchor = cap_m * 1_000_000.0  # USD
        shares_approx = cap_anchor / anchor_close

        px["MarketCap"] = (px["Close"].astype("float64") * shares_approx).round(0)
        sub = px[["Date", "MarketCap"]].copy()
        sub["Ticker"] = ticker
        rows.append(sub)

    if not rows:
        logger.error("No rows produced; aborting.")
        return

    out = pd.concat(rows, ignore_index=True)
    out = out[["Ticker", "Date", "MarketCap"]]
    out.to_parquet(out_path, index=False)

    logger.info(f"Wrote {out_path} ({len(out):,} rows, {out['Ticker'].nunique()} tickers)")
    logger.info(f"Skipped: no_cap={skipped_no_cap}, no_price={skipped_no_price}, "
                f"no_anchor_close={skipped_no_anchor_close}")
    logger.info(f"Date coverage: {out['Date'].min().date()} -> {out['Date'].max().date()}")
    logger.info(f"Anchor cap quantiles (USD): "
                f"p10={out.groupby('Ticker')['MarketCap'].last().quantile(0.10)/1e6:.1f}M, "
                f"p50={out.groupby('Ticker')['MarketCap'].last().median()/1e6:.1f}M, "
                f"p90={out.groupby('Ticker')['MarketCap'].last().quantile(0.90)/1e6:.1f}M")


if __name__ == "__main__":
    main()
