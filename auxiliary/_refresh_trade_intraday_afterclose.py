#!/usr/bin/env python
"""
_refresh_trade_intraday_afterclose.py - wait for the 16:00 ET close, then pull the
finest-resolution (1-min) intraday data for every trade and re-run the fill sim.

IBKR throttles historical-data requests hard during RTH (the data lines compete with
live quotes), so a bulk 1-min pull mid-session times out constantly. After the close
those lines are free and the same pull runs clean - which is why the existing lake was
built after-hours. This launcher idles until 16:10 ET then runs:

  1. experimental/2.2__TradeHistoryIntradayDownloader.py  (merged trade set + SPY)
  2. 8__IntradayFillSim.py                                 (re-price on the fresh data)

Run detached:  python experimental/_refresh_trade_intraday_afterclose.py &
Log:           Data/_afterclose_intraday_refresh.log   (also stdout)
Safe to re-run - the downloader is coverage-tracked and resumable; NO orders are placed.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(ROOT, "Data", "trade_history_merged.parquet")
OUT_DIR = os.path.join(ROOT, "Data", "IntradayTradeSim")
LOG = os.path.join(ROOT, "Data", "_afterclose_intraday_refresh.log")
PY = sys.executable


def log(msg):
    line = f"[{datetime.now(ET):%Y-%m-%d %H:%M:%S ET}] {msg}"
    # ASCII-safe stdout (Windows console is cp1252; a stray non-ASCII char would crash us).
    try:
        print(line, flush=True)
    except Exception:
        print(line.encode("ascii", "replace").decode(), flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def seconds_until_close_plus(buffer_min=10):
    now = datetime.now(ET)
    target = now.replace(hour=16, minute=buffer_min, second=0, microsecond=0)
    if now >= target:                    # already past today's close
        if now.hour < 16:                # before close but after target somehow - run now
            return 0
        return 0                          # after close already -> run immediately
    return (target - now).total_seconds()


def main():
    # Build the merged trade history (root 1112 + Data/TradeHistory 929) so we cover
    # every trade old and new, then persist it where the downloader expects it.
    import pandas as pd
    root = pd.read_parquet(os.path.join(ROOT, "trade_history.parquet"))
    extra = os.path.join(ROOT, "Data", "TradeHistory.parquet")
    frames = [root]
    if os.path.exists(extra):
        frames.append(pd.read_parquet(extra))
    for d in frames:
        d["EntryDate"] = pd.to_datetime(d["EntryDate"])
        d["ExitDate"] = pd.to_datetime(d["ExitDate"])
    merged = (pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["Symbol", "EntryDate", "ExitDate"])
                .reset_index(drop=True))
    merged.to_parquet(MERGED, index=False)
    log(f"Merged trade history: {len(merged)} trades, {merged['Symbol'].nunique()} tickers "
        f"({merged['EntryDate'].min().date()} -> {merged['EntryDate'].max().date()}) -> {MERGED}")

    wait = seconds_until_close_plus(10)
    if wait > 0:
        eta = datetime.now(ET) + timedelta(seconds=wait)
        log(f"Waiting {wait/60:.0f} min for the close - will start the 1-min pull at {eta:%H:%M ET}.")
        # sleep in chunks so a machine clock change / interrupt is noticed within a minute
        end = time.time() + wait
        while time.time() < end:
            time.sleep(min(60, end - time.time()))
    log("Starting 1-min intraday download (post-close, lines free).")

    dl = [PY, os.path.join(ROOT, "experimental", "2.2__TradeHistoryIntradayDownloader.py"),
          "--trade-history", MERGED, "--out-dir", OUT_DIR,
          "--num-threads", "12", "--index-tickers", "SPY"]
    rc = subprocess.call(dl, cwd=ROOT)
    log(f"Downloader exit code {rc}.")

    # Fill in any interior timeout-holes left by the first pass.
    dl_holes = dl + ["--fill-holes"]
    subprocess.call(dl_holes, cwd=ROOT)
    log("Hole-fill pass done.")

    log("Re-running 8__IntradayFillSim on the refreshed data (merged trade set).")
    sim = [PY, os.path.join(ROOT, "8__IntradayFillSim.py"), "--trade-history", MERGED]
    rc2 = subprocess.call(sim, cwd=ROOT)
    log(f"Fill sim exit code {rc2}. Refresh complete. See analysis_output/intraday_fill_sim_report.md")


if __name__ == "__main__":
    main()
