"""Download Wikipedia daily pageviews per ticker.

For each ticker in the latest TickerCikData parquet, this:
  1. Looks up the company's Wikipedia article via the OpenSearch API
  2. Pulls daily pageviews from the Wikimedia REST pageview endpoint
  3. Caches the ticker -> article-title mapping so refresh runs skip step 1

Pageview counts are a useful proxy for retail attention and have small but
non-zero predictive value (see Da, Engelberg & Gao 2011, "In Search of
Attention"; FFR work on Wikipedia views).

Refresh: re-running pulls last 60 days of new data per article (cached
files are merged with the new range). Cap fully-historical at the API
limit (~daily counts available from 2015-07-01).

Endpoints:
  search:    https://en.wikipedia.org/w/api.php?action=opensearch&search=...
  pageviews: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
             en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, RateLimiter, fmt_bytes, log, make_session

OUT_DIR = DATA_ROOT / "Wikipedia"
PV_DIR = OUT_DIR / "pageviews"
MAPPING_FILE = OUT_DIR / "ticker_article_map.json"
TICKER_CIK_DIR = DATA_ROOT / "TickerCikData"

PV_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
SEARCH_API = "https://en.wikipedia.org/w/api.php"
START_DATE = "20150701"  # earliest pageview data
WIKI_UA = "StockMarketResearch/1.0 (masamunex9000@gmail.com)"

# Wikimedia REST: 200 req/sec hard cap; we cap at 50/sec which is plenty
_wiki_limiter = RateLimiter(hz=50.0)


def _load_universe() -> pd.DataFrame:
    files = sorted(TICKER_CIK_DIR.glob("TickerCIKs_*.parquet"))
    if not files:
        raise SystemExit(f"No TickerCikData parquets found in {TICKER_CIK_DIR}")
    latest = files[-1]
    log(f"Universe source: {latest.name}")
    df = pd.read_parquet(latest)
    return df


def _load_mapping() -> dict:
    if MAPPING_FILE.exists():
        return json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    return {}


def _save_mapping(m: dict) -> None:
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_FILE.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def _search_article(session, name: str) -> str | None:
    _wiki_limiter.wait()
    try:
        r = session.get(SEARCH_API, params={
            "action": "opensearch", "search": name, "limit": 1,
            "namespace": 0, "format": "json",
        }, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        titles = data[1] if len(data) > 1 else []
        return titles[0] if titles else None
    except Exception:
        return None


def _fetch_pageviews(session, title: str, start: str, end: str) -> pd.DataFrame | None:
    enc_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"{PV_API}/en.wikipedia/all-access/all-agents/{enc_title}/daily/{start}/{end}"
    _wiki_limiter.wait()
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None
        items = r.json().get("items", [])
        if not items:
            return None
        df = pd.DataFrame(items)
        df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
        df = df[["date", "views"]].sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def _process_ticker(session, ticker: str, name: str, mapping: dict,
                    start: str, end: str) -> tuple[str, int]:
    dest = PV_DIR / f"{ticker}.parquet"
    title = mapping.get(ticker)
    if title is None:
        title = _search_article(session, name)
        mapping[ticker] = title or ""  # cache miss to avoid re-search
        if not title:
            return f"{ticker} no-article", 0
    if not title:
        return f"{ticker} no-article", 0
    df = _fetch_pageviews(session, title, start, end)
    if df is None or df.empty:
        return f"{ticker} no-pv", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return f"{ticker} {len(df)} rows", len(df)


def fetch(workers: int = 16, max_tickers: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PV_DIR.mkdir(parents=True, exist_ok=True)
    universe = _load_universe()

    # Pick name columns conservatively
    name_col = next((c for c in ("name", "title", "companyName", "company_name")
                     if c in universe.columns), None)
    ticker_col = next((c for c in ("ticker", "symbol") if c in universe.columns),
                      universe.columns[0])
    if name_col is None:
        log("WARN: no name column found; using ticker as search term")
        universe["_name"] = universe[ticker_col].astype(str)
        name_col = "_name"
    rows = universe[[ticker_col, name_col]].dropna().drop_duplicates(subset=[ticker_col])
    if max_tickers:
        rows = rows.head(max_tickers)
    log(f"Processing {len(rows):,} tickers with {workers} workers")

    mapping = _load_mapping()
    session = make_session()
    session.headers["User-Agent"] = WIKI_UA

    end = dt.date.today().strftime("%Y%m%d")
    total_rows = ok = missing = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_ticker, session, str(r[0]).upper(),
                             str(r[1]), mapping, START_DATE, end): str(r[0])
                   for r in rows.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg, n = fut.result()
                total_rows += n
                if n > 0:
                    ok += 1
                else:
                    missing += 1
            except Exception as e:
                missing += 1
            if i % 250 == 0:
                _save_mapping(mapping)
                log(f"  progress: {i:,}/{len(rows):,}  ({ok} ok, {missing} missing, "
                    f"{total_rows:,} rows)")
    _save_mapping(mapping)
    log(f"Done. {ok:,} tickers with pageviews, {missing:,} missing, "
        f"{total_rows:,} total rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tickers", type=int, default=None,
                    help="limit for smoke-testing")
    a = ap.parse_args()
    fetch(workers=a.workers, max_tickers=a.max_tickers)


if __name__ == "__main__":
    main()
