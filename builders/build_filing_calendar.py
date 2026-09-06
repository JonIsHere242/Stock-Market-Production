"""
build_filing_calendar.py  --  EVERY SEC filing event, from Data/SEC/submissions.

WHY A SECOND SEC BUILDER
------------------------
build_filing_meta.py reads companyfacts, which only contains forms that carry XBRL FACTS
(10-K/10-Q/20-F/40-F). That structurally excludes the single most on-point event in the
reporting-delay literature:

    FORM NT 10-Q / NT 10-K  (Rule 12b-25, "Notification of Late Filing")

That IS the "we are going to miss the deadline" announcement Bartov/DeFond/Konchitchki measured
when they found post-notification drift running DOWN for months. It carries no XBRL, so
companyfacts cannot see it. submissions.zip can.

And once we are in submissions, three more veins open that companyfacts also cannot reach:

  * 8-K ITEM CODES. The `items` field gives the reason a firm filed an 8-K. The accounting-
    distress codes are exactly the "the books are wrong" events:
        4.02  Non-Reliance on Previously Issued Financial Statements  <- the LOUD restatement.
              (build_filing_meta's sfr_ family catches the QUIET one; this is its sibling.)
        4.01  Changes in Registrant's Certifying Accountant           <- auditor fired/resigned
        3.01  Notice of Delisting / Failure to Satisfy a Listing Rule
        1.03  Bankruptcy or Receivership
        2.06  Material Impairments
        5.02  Departure of Directors or Certain Officers
  * ACCEPTANCE TIMESTAMP. `acceptanceDateTime` is when EDGAR accepted the filing, to the second.
    EDGAR's cutoff is 17:30 ET; anything accepted after that is dated the next business day.
    Filing after the close, and filing on a Friday, are the classic bad-news-burial channels --
    and neither is visible anywhere in companyfacts.
  * FILING BURST. Raw count of filings in a trailing window: corporate distress (and corporate
    action) shows up as a spike in filing activity before it shows up in fundamentals.

POINT-IN-TIME
-------------
`filingDate` is the public date. Everything downstream is keyed on it and consumed through
FeatureTemplates/_filingcal.py, which resolves "days since the last event of type X" with a
searchsorted against dates strictly <= the trading day. `reportDate` (the period being reported)
is carried for reference but is NEVER used as a join key -- that would be future leakage.

INPUT   Data/SEC/submissions/CIK##########.json          (recent, up to 1000 filings)
        Data/SEC/submissions/CIK##########-submissions-*.json  (older shards, back to 1994)
        Data/Fundamentals/fundamentals_panel.parquet     (ticker <-> cik map only)
OUTPUT  Data/SEC/filing_calendar/{TICKER}.parquet        (one row per filing event)

USAGE   stock_env\\Scripts\\python.exe builders/build_filing_calendar.py
        ...\\python.exe builders/build_filing_calendar.py --tickers AAPL,WBA --jobs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # this script lives in builders/
SUB_DIR = ROOT / "Data" / "SEC" / "submissions"
PANEL = ROOT / "Data" / "Fundamentals" / "fundamentals_panel.parquet"
PRICE_DIR = ROOT / "Data" / "PriceData"
OUT_DIR = ROOT / "Data" / "SEC" / "filing_calendar"

# 8-K item codes worth naming. Everything else is folded into the generic 8-K count.
# Codes arrive as "4.02", sometimes bare "4" on very old filings -- we match on the prefix
# before the dot, then require the exact code, so a bare "4" never counts as a 4.02.
ITEM_FLAGS = {
    "it_nonreliance":   {"4.02"},   # Non-Reliance on Previously Issued Financial Statements
    "it_auditor":       {"4.01"},   # Changes in Certifying Accountant
    "it_delist":        {"3.01"},   # Notice of Delisting / listing-rule failure
    "it_bankruptcy":    {"1.03"},   # Bankruptcy or Receivership
    "it_impairment":    {"2.06"},   # Material Impairments
    "it_officer_exit":  {"5.02"},   # Departure of Directors or Certain Officers
    "it_results":       {"2.02"},   # Results of Operations (the earnings 8-K)
}

# EDGAR's daily cutoff. A filing accepted after this is dated the NEXT business day -- which is
# precisely why a firm with bad news files at 17:45.
EDGAR_CUTOFF_HOUR = 17.5

FLAG_COLS = ["is_nt", "is_amended", "is_periodic", "is_8k"] + list(ITEM_FLAGS)


def _to_et_hour(ts: str) -> float:
    """
    UTC acceptance stamp -> fractional hour in US Eastern.

    Uses a fixed-offset DST rule rather than zoneinfo: the tzdata package is not guaranteed in
    this env, and a one-hour error on the DST boundary days is irrelevant to a 17:30 cutoff test
    (nothing hinges on 16:30-vs-17:30 for two days a year), whereas a missing-tzdata crash would
    kill a 4000-ticker build.
    """
    if not ts:
        return np.nan
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return np.nan
    # US DST: second Sunday of March -> first Sunday of November.
    y = dt.year
    mar = datetime(y, 3, 8, tzinfo=timezone.utc)
    dst_start = mar + timedelta(days=(6 - mar.weekday()) % 7)
    nov = datetime(y, 11, 1, tzinfo=timezone.utc)
    dst_end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    offset = -4 if dst_start <= dt < dst_end else -5
    et = dt + timedelta(hours=offset)
    return et.hour + et.minute / 60.0


def _rows_from_block(r: dict) -> list[dict]:
    forms = r.get("form") or []
    n = len(forms)
    if not n:
        return []
    fdates = r.get("filingDate") or [None] * n
    rdates = r.get("reportDate") or [None] * n
    accept = r.get("acceptanceDateTime") or [None] * n
    items = r.get("items") or [None] * n
    out = []
    for i in range(n):
        form = str(forms[i] or "")
        if not form:
            continue
        codes = {c.strip() for c in str(items[i] or "").split(",") if c.strip()}
        row = {
            "filed_date": fdates[i],
            "report_date": rdates[i] if i < len(rdates) else None,
            "form": form,
            "accept_et_hour": _to_et_hour(accept[i] if i < len(accept) else None),
            "is_nt": float(form.startswith("NT ")),
            "is_amended": float("/A" in form),
            "is_periodic": float(form.startswith(("10-K", "10-Q", "20-F", "40-F"))),
            "is_8k": float(form.startswith("8-K")),
        }
        for flag, want in ITEM_FLAGS.items():
            row[flag] = float(bool(codes & want))
        out.append(row)
    return out


def build_one(cik: str, ticker: str) -> pd.DataFrame:
    main = SUB_DIR / f"CIK{cik}.json"
    if not main.exists():
        return pd.DataFrame()
    with open(main, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    filings = doc.get("filings") or {}
    rows = _rows_from_block(filings.get("recent") or {})
    # Older shards carry history back to 1994. Skipping them would silently truncate every
    # trailing-count feature for long-lived filers.
    for f in (filings.get("files") or []):
        p = SUB_DIR / str(f.get("name") or "")
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rows += _rows_from_block(json.load(fh))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["filed_date"]).sort_values("filed_date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Burial channel: accepted after EDGAR's 17:30 ET cutoff, or on a Friday.
    df["is_after_hours"] = (df["accept_et_hour"] >= EDGAR_CUTOFF_HOUR).astype(float)
    df.loc[df["accept_et_hour"].isna(), "is_after_hours"] = np.nan
    df["is_friday"] = (df["filed_date"].dt.dayofweek == 4).astype(float)
    # NEWS-BEARING forms only. Form 4 insider filings are routinely accepted after the close by
    # the filing agent and carry no news; including them makes an "after-hours rate" a proxy for
    # how many insiders a firm has (AAPL: 61% all-form, almost entirely Form 4). Every timing
    # rate must be measured on this subset.
    df["is_newsy"] = ((df["is_8k"] > 0) | (df["is_periodic"] > 0)).astype(float)

    df.insert(1, "ticker", ticker)
    return df[["filed_date", "ticker", "form", "report_date", "accept_et_hour",
               "is_after_hours", "is_friday", "is_newsy"] + FLAG_COLS]


def _job(task: tuple[str, str]) -> tuple[str, int, str]:
    ticker, cik = task
    try:
        df = build_one(cik, ticker)
    except Exception as exc:
        return ticker, -1, f"{type(exc).__name__}: {exc}"
    if df.empty:
        return ticker, 0, ""
    df.to_parquet(OUT_DIR / f"{ticker}.parquet", index=False)
    return ticker, len(df), ""


def ticker_cik_map() -> dict[str, str]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PANEL)
    out: dict[str, str] = {}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ticker", "cik"]).to_pandas().head(1)
        if t.empty:
            continue
        out[str(t["ticker"].iloc[0])] = str(int(t["cik"].iloc[0])).zfill(10)
    return out


def write_stamp(out_dir: Path, n_tickers: int) -> None:
    """Record build time + newest filing date -- see build_filing_meta.write_stamp for why."""
    import json
    latest = ""
    for q in sorted(out_dir.glob("*.parquet"))[:200]:
        try:
            d = pd.read_parquet(q, columns=["filed_date"])
        except Exception:
            continue
        if len(d):
            latest = max(latest, str(pd.to_datetime(d["filed_date"]).max().date()))
    (out_dir / "_BUILD_STAMP.json").write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tickers": n_tickers,
        "max_filed_date": latest,
    }, indent=1), encoding="utf-8")
    print(f"[stamp] {out_dir.name}: {n_tickers} tickers, newest filing {latest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    if not SUB_DIR.is_dir():
        print(f"FAIL: missing {SUB_DIR} (run fetchers/fetch_sec_submissions.py --extract)")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmap = ticker_cik_map()
    want = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    have_price = {p.stem.upper() for p in PRICE_DIR.glob("*.parquet")} if PRICE_DIR.is_dir() else set()

    tasks = []
    for ticker, cik in sorted(cmap.items()):
        if want and ticker.upper() not in want:
            continue
        if not want and have_price and ticker.upper() not in have_price:
            continue
        tasks.append((ticker, cik))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[tasks] {len(tasks)} tickers, jobs={args.jobs}")

    t0 = time.time()
    written = empty = failed = 0
    if args.jobs <= 1:
        results = (_job(t) for t in tasks)
        for ticker, n, err in results:
            if n > 0: written += 1
            elif n == 0: empty += 1
            else: failed += 1; print(f"  !! {ticker}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(_job, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                ticker, n, err = f.result()
                if n > 0: written += 1
                elif n == 0: empty += 1
                else: failed += 1; print(f"  !! {ticker}: {err}")
                if k % 500 == 0:
                    print(f"  {k}/{len(tasks)}  ({time.time() - t0:.0f}s)")

    write_stamp(OUT_DIR, written)
    print(f"[done] wrote {written}, empty {empty}, failed {failed}, "
          f"{time.time() - t0:.0f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
