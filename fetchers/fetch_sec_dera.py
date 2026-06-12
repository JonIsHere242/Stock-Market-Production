"""Download SEC DERA Financial Statement Data Sets — pre-parsed XBRL by quarter.

Each quarter (2009 Q1 onward) the SEC's DERA publishes a clean tab-delimited
extract of every 10-K/10-Q/40-F/20-F filing's XBRL: sub.txt (submissions),
num.txt (numeric facts), pre.txt (presentation), tag.txt (taxonomy).

Compared to companyfacts.zip: DERA is *quarterly snapshots* with first-filed
values + restatement history per quarter, which is what you want for
point-in-time training (vs. the latest-restated values in companyfacts).

~3 GB per quarter compressed; ~60 GB for the full 2009-present archive.

URL pattern: https://www.sec.gov/files/dera/data/financial-statement-data-sets/{Y}q{Q}.zip
"""
from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

from common import (DATA_ROOT, fmt_bytes, log, make_session, sec_get,
                    stream_download)

BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{y}q{q}.zip"
RAW_DIR = DATA_ROOT / "SEC" / "dera" / "raw"
EXTRACT_DIR = DATA_ROOT / "SEC" / "dera" / "extracted"
START_YEAR = 2009


def _quarters(start_year: int) -> list[tuple[int, int]]:
    today = dt.date.today()
    out = []
    for y in range(start_year, today.year + 1):
        for q in range(1, 5):
            q_end_month = q * 3
            # DERA publishes about a quarter after the quarter ends
            release_est = dt.date(y, q_end_month, 1) + dt.timedelta(days=120)
            if release_est > today:
                break
            out.append((y, q))
    return out


def _is_current_dera_quarter(y: int, q: int) -> bool:
    today = dt.date.today()
    cur_q = (today.month - 1) // 3 + 1
    return y == today.year and q == cur_q


def fetch(start_year: int = START_YEAR, force: bool = False, extract: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session(host="www.sec.gov", sec=True)
    qs = _quarters(start_year)
    log(f"Fetching {len(qs)} DERA quarterly archives ({start_year}–present)")

    total_dl = total_ex = downloads = 0
    for y, q in qs:
        url = BASE.format(y=y, q=q)
        dest = RAW_DIR / f"{y}q{q}.zip"
        if dest.exists() and not force and not _is_current_dera_quarter(y, q):
            log(f"  {y}q{q} cached")
            continue
        try:
            n = stream_download(session, url, dest, rate_limited_get=sec_get)
            total_dl += n
            downloads += 1
            log(f"  {y}q{q}  {fmt_bytes(n)}")
        except Exception as e:
            log(f"  {y}q{q}  FAIL: {e}")
            continue
        if extract:
            qd = EXTRACT_DIR / f"{y}q{q}"
            qd.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(qd)
                ex_sz = sum(f.stat().st_size for f in qd.rglob("*") if f.is_file())
                total_ex += ex_sz
                log(f"    extracted to {qd.name} ({fmt_bytes(ex_sz)})")
    log(f"Done. {downloads} new archives, downloaded {fmt_bytes(total_dl)}"
        + (f", extracted {fmt_bytes(total_ex)}" if extract else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract", action="store_true",
                    help="unzip each quarter into Data/SEC/dera/extracted/{y}q{q}/")
    a = ap.parse_args()
    fetch(start_year=a.start_year, force=a.force, extract=a.extract)


if __name__ == "__main__":
    main()
