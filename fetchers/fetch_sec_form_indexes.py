"""Download SEC quarterly form.idx files — index of every filing by form type.

Each quarter (since 1993) SEC publishes form.idx listing every filing
indexed by form type (10-K, 10-Q, 8-K, 4, 13F-HR, 13D, 13G, S-1, etc.)
with CIK, company name, filing date, and the relative archive path.

This is the bridge from "I want every Form 4 in 2024 Q1" to actual URLs.
Each file is ~10-30 MB. ~120 files for 1995-present.

Refresh: each quarter's idx is immutable after quarter-end; the *current*
quarter updates daily. We re-fetch the current quarter every run and skip
older quarters when already on disk.

URL pattern: https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx
"""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import DATA_ROOT, fmt_bytes, log, make_session, sec_get

BASE = "https://www.sec.gov/Archives/edgar/full-index"
OUT_DIR = DATA_ROOT / "SEC" / "form_indexes"

START_YEAR = 1995  # XBRL fundamentals start ~2009; pre-2009 useful for events/Form 4


def _quarters(start_year: int) -> list[tuple[int, int]]:
    today = dt.date.today()
    out = []
    for y in range(start_year, today.year + 1):
        for q in range(1, 5):
            q_end = dt.date(y, q * 3, 1)
            if q_end > today:
                break
            out.append((y, q))
    return out


def _is_current_quarter(y: int, q: int) -> bool:
    today = dt.date.today()
    cur_q = (today.month - 1) // 3 + 1
    return y == today.year and q == cur_q


def _fetch_one(session, y: int, q: int, force: bool) -> tuple[str, int]:
    dest = OUT_DIR / str(y) / f"QTR{q}_form.idx"
    if dest.exists() and not force and not _is_current_quarter(y, q):
        return f"{y}Q{q} cached", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{y}/QTR{q}/form.idx"
    r = sec_get(session, url)
    if r.status_code == 404:
        return f"{y}Q{q} 404", 0
    r.raise_for_status()
    tmp = dest.with_suffix(".idx.tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return f"{y}Q{q} {fmt_bytes(len(r.content))}", len(r.content)


def fetch(start_year: int = START_YEAR, force: bool = False, workers: int = 4) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session(host="www.sec.gov", sec=True)
    qs = _quarters(start_year)
    log(f"Fetching form.idx for {len(qs)} quarters ({start_year}–present), {workers} workers")

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, session, y, q, force): (y, q) for y, q in qs}
        for fut in as_completed(futures):
            msg, n = fut.result()
            total += n
            log(f"  {msg}")
    log(f"Total downloaded this run: {fmt_bytes(total)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    fetch(start_year=a.start_year, force=a.force, workers=a.workers)


if __name__ == "__main__":
    main()
