#!/usr/bin/env python
"""
2.1_DeepPriceDownloader.py
==========================
Builds the deep-history price data lake in Data/PriceDataFull/ from IBKR --
the full available daily history for every ticker in Data/RFpredictions/.

Why it is built this way
------------------------
IBKR paces historical requests at ~60 per 10 minutes PER CONNECTION. The only
way to go faster is more connections. So this runs a pool of N independent IB
connections; each is a worker that pulls tickers from a shared queue and
self-paces (--pace seconds between request starts) to stay under the limit.

One request per ticker: a fixed large durationStr (default 60 Y) -- IBKR clips
it to whatever history exists. No reqHeadTimeStamp pre-pass: it counts against
the same pacing budget, so depth is read from the returned data instead.
Chained walk-back requests are NOT used -- they silently corrupt volume on
interior bars (see diagnose_stitch_mismatch.py).

Output
------
  Data/PriceDataFull/<TICKER>.parquet   one file per ticker
  Data/PriceDataFull/_catalog.parquet   summary: ticker, status, start, end, bars

Resumable: tickers already saved are skipped, so a re-run continues.

Usage
-----
  python 2.1_DeepPriceDownloader.py                  # full run, 8 connections
  python 2.1_DeepPriceDownloader.py --limit 150      # a 150-ticker sample
  python 2.1_DeepPriceDownloader.py --connections 8 --pace 11
  python 2.1_DeepPriceDownloader.py --ticker AAPL --force
"""

import argparse
import asyncio
import os
import re
import sys
import time

import pandas as pd
import nest_asyncio

try:
    from ib_insync import IB, Stock, util
except ImportError:
    sys.exit("ib_insync not found -- run from the same env as 2__PriceDownloader.py")

nest_asyncio.apply()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DATA_DIR = os.path.join(script_dir, "Data")
LAKE_DIR = os.path.join(DATA_DIR, "PriceDataFull")
RF_PREDICTIONS_DIR = os.path.join(DATA_DIR, "RFpredictions")
CATALOG = os.path.join(LAKE_DIR, "_catalog.parquet")
os.makedirs(LAKE_DIR, exist_ok=True)

DAILY = "1 day"
WHAT = "TRADES"        # matches 2__PriceDownloader.py
USE_RTH = True         # matches 2__PriceDownloader.py

_pacing_hits = []      # appended by the error handler -- should stay ~empty if pacing works


def _on_error(*args):
    msg = str(args[2]) if len(args) > 2 else ""
    if "pacing" in msg.lower():
        _pacing_hits.append(time.time())


def bars_to_df(bars):
    if not bars:
        return pd.DataFrame()
    try:
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                                "low": "Low", "close": "Close", "volume": "Volume",
                                "average": "Average", "barCount": "BarCount"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return pd.DataFrame()      # malformed bars (IBKR "Query failed") -> no data


def tidy(df, ticker):
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Average"]:
        if c in df.columns:
            df[c] = df[c].round(4)
    df["Ticker"] = ticker
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume",
                        "Average", "BarCount", "Ticker"] if c in df.columns]
    return df[keep]


def load_tickers(args):
    if args.ticker:
        return [args.ticker.upper().strip()]
    if not os.path.isdir(RF_PREDICTIONS_DIR):
        sys.exit(f"Ticker source not found: {RF_PREDICTIONS_DIR}")
    pat = re.compile(r"^(.+)\.parquet$")
    names = sorted({m.group(1) for f in os.listdir(RF_PREDICTIONS_DIR)
                    for m in [pat.match(f)] if m})
    if not names:
        sys.exit(f"No .parquet ticker files in {RF_PREDICTIONS_DIR}")
    if args.limit and 0 < args.limit < len(names):
        stride = len(names) / args.limit
        names = [names[int(i * stride)] for i in range(args.limit)]
    return names


class Pacer:
    """Limits one connection to >= `interval` seconds between request starts."""

    def __init__(self, interval):
        self.interval = interval
        self._next = 0.0

    async def wait(self):
        now = time.monotonic()
        if now < self._next:
            await asyncio.sleep(self._next - now)
        self._next = time.monotonic() + self.interval


async def connect_one(args, client_id):
    """Open one IB connection on a fixed clientId. Returns the IB or None."""
    for _ in range(3):
        ib = IB()
        ib.errorEvent += _on_error
        try:
            await ib.connectAsync(args.host, args.port, clientId=client_id, timeout=15)
            return ib
        except Exception:
            try:
                ib.disconnect()
            except Exception:
                pass
            await asyncio.sleep(2)
    return None


async def one_request(ib, pacer, contract, duration, timeout):
    """One pacing-gated reqHistoricalData. Returns (df, elapsed_s, err)."""
    await pacer.wait()
    t0 = time.perf_counter()
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting=DAILY, whatToShow=WHAT, useRTH=USE_RTH,
            formatDate=1, timeout=timeout)
        err = None
    except Exception as e:
        bars, err = None, str(e)
    return bars_to_df(bars), time.perf_counter() - t0, err


async def pull_ticker(ib, pacer, ticker, args):
    """Qualify + pull full history for one ticker. Returns a catalog record."""
    rec = {"ticker": ticker, "status": "fail", "start": pd.NaT, "end": pd.NaT,
           "bars": 0, "years": float("nan"), "secs": 0.0, "note": ""}
    out_path = os.path.join(LAKE_DIR, f"{ticker}.parquet")
    try:
        contract = Stock(ticker, "SMART", "USD")
        # qualify with retries -- a transient IBKR hiccup must not become a
        # permanent false 'notfound' (root cause of the first run's bad marks)
        qualified = None
        for attempt in range(3):
            try:
                qualified = await asyncio.wait_for(
                    ib.qualifyContractsAsync(contract), timeout=20)
            except Exception:
                qualified = None
            if qualified:
                break
            if attempt < 2:
                await asyncio.sleep(1.5)
        if not qualified:
            rec["status"] = "notfound"
            return rec

        # primary duration twice (transient retry), then descending fallbacks
        # for the "failed to compute time length" contracts
        attempts = [args.duration, args.duration, "30 Y", "12 Y"]
        total = 0.0
        last_err = None
        for dur in attempts:
            df, elapsed, err = await one_request(ib, pacer, contract, dur, args.timeout)
            total += elapsed
            last_err = err
            if not df.empty:
                df = tidy(df, ticker)
                df.to_parquet(out_path, index=False, compression="snappy")
                start, end = df["Date"].min(), df["Date"].max()
                rec.update(status="ok", start=start, end=end, bars=len(df),
                           years=round((end - start).days / 365.25, 1),
                           secs=round(total, 1), note=f"req={dur}")
                return rec
        rec.update(secs=round(total, 1), note=str(last_err or "empty")[:50])
        return rec
    except Exception as e:
        rec["note"] = f"error: {e}"[:60]
        return rec


def save_catalog(results):
    if not results:
        return
    cols = ["ticker", "status", "start", "end", "bars", "years", "secs", "note"]
    df = pd.DataFrame(results)[cols].sort_values("ticker").reset_index(drop=True)
    tmp = CATALOG + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, CATALOG)


def record(state, rec):
    state["results"].append(rec)
    state["counter"][0] += 1
    n, total = state["counter"][0], state["total"]
    if rec["status"] == "ok":
        line = (f"{rec['start'].date()}..{rec['end'].date()}  "
                f"{rec['bars']:>6,}b  {rec['years']:5.1f}y  {rec['secs']:5.1f}s")
    else:
        line = f"{rec['status']:<9} {rec['note']}"
    print(f"[{n:>4}/{total}] {rec['ticker']:<7} {line}")
    if n % 100 == 0 or n == total:
        save_catalog(state["results"])
        ok = sum(1 for r in state["results"] if r["status"] == "ok")
        mins = (time.time() - state["t0"]) / 60
        rate = n / mins if mins > 0 else 0
        eta = (total - n) / rate if rate > 0 else 0
        print(f"  --- {n}/{total}  ok={ok}  {mins:.0f}min elapsed  "
              f"~{eta:.0f}min left  ({rate:.0f}/min, {len(_pacing_hits)} pacing) ---")


async def worker(conn_idx, args, queue, state):
    client_id = args.client_id + conn_idx
    ib = await connect_one(args, client_id)
    if ib is None:
        print(f"  conn {conn_idx} (clientId {client_id}): could not connect -- idle")
        return
    pacer = Pacer(args.pace)
    try:
        while True:
            try:
                ticker = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not ib.isConnected():
                reconnected = False
                for _ in range(3):
                    try:
                        await ib.connectAsync(args.host, args.port,
                                              clientId=client_id, timeout=15)
                        reconnected = True
                        break
                    except Exception:
                        await asyncio.sleep(3)
                if not reconnected:
                    queue.put_nowait(ticker)          # hand the work back
                    print(f"  conn {conn_idx}: lost connection -- worker stopping")
                    break
            rec = await pull_ticker(ib, pacer, ticker, args)
            record(state, rec)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


async def main_async(args, tickers):
    todo = [t for t in tickers
            if args.force or not os.path.exists(os.path.join(LAKE_DIR, f"{t}.parquet"))]
    skipped = len(tickers) - len(todo)
    print(f"  {len(todo)} to pull, {skipped} already in lake")
    if not todo:
        return [], 0.0, skipped

    queue = asyncio.Queue()
    for t in todo:
        queue.put_nowait(t)
    state = {"results": [], "counter": [0], "total": len(todo), "t0": time.time()}
    n_conn = min(args.connections, len(todo))
    print(f"  opening {n_conn} connections, pace {args.pace}s/request each")
    print("-" * 78)
    await asyncio.gather(*[worker(i, args, queue, state) for i in range(n_conn)])
    save_catalog(state["results"])
    return state["results"], time.time() - state["t0"], skipped


def print_summary(results, elapsed, skipped):
    ok = [r for r in results if r["status"] == "ok"]
    notfound = [r for r in results if r["status"] == "notfound"]
    failed = [r for r in results if r["status"] == "fail"]
    print("\n" + "=" * 78)
    print("DEEP DOWNLOAD COMPLETE")
    print("=" * 78)
    print(f"  elapsed     : {elapsed / 3600:.2f} h  ({elapsed / 60:.0f} min)")
    print(f"  saved       : {len(ok)}")
    print(f"  skipped     : {skipped}  (already in lake)")
    print(f"  not on IBKR : {len(notfound)}")
    print(f"  failed      : {len(failed)}")
    if failed:
        names = ", ".join(r["ticker"] for r in failed[:50])
        print(f"     -> {names}{' ...' if len(failed) > 50 else ''}")
    if ok:
        total_bars = sum(r["bars"] for r in ok)
        yrs = sorted(r["years"] for r in ok if pd.notna(r["years"]))
        print(f"  total bars  : {total_bars:,}")
        if yrs:
            print(f"  depth       : median {yrs[len(yrs) // 2]:.1f}yr   "
                  f"deepest {yrs[-1]:.1f}yr")
    print(f"  pacing hits : {len(_pacing_hits)}")
    print(f"  catalog     : {CATALOG}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496, help="7496 live / 7497 paper")
    ap.add_argument("--client-id", type=int, default=100,
                    help="base clientId; connection i uses base+i")
    ap.add_argument("--connections", type=int, default=8,
                    help="IBKR connections in the pool (each ~60 req/10min)")
    ap.add_argument("--pace", type=float, default=11.0,
                    help="min seconds between requests on each connection")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap tickers from RFpredictions (0 = all; sampled evenly)")
    ap.add_argument("--ticker", type=str, help="single ticker (overrides --limit)")
    ap.add_argument("--duration", default="60 Y",
                    help="primary durationStr requested per ticker")
    ap.add_argument("--timeout", type=int, default=240, help="per-request timeout (s)")
    ap.add_argument("--force", action="store_true",
                    help="re-pull tickers already saved in Data/PriceDataFull/")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 78)
    print("2.1  DEEP PRICE DOWNLOADER  -  full-history pull -> Data/PriceDataFull/")
    print("=" * 78)
    tickers = load_tickers(args)
    print(f"  tickers     : {len(tickers)}")
    print(f"  connections : {args.connections}   pace : {args.pace}s")
    print(f"  port={args.port}  clientId base={args.client_id}  duration={args.duration}")

    results, elapsed, skipped = [], 0.0, 0
    try:
        results, elapsed, skipped = asyncio.run(main_async(args, tickers))
    except KeyboardInterrupt:
        print("\n  interrupted by user -- partial summary follows")
    print_summary(results, elapsed, skipped)


if __name__ == "__main__":
    main()
