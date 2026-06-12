"""Download CFTC Commitments of Traders (COT) — weekly futures positioning.

Every Tuesday the CFTC releases positioning for commercial hedgers, large
speculators, and small traders across every futures market. Useful for
sector ETFs / index futures regime context. Free and downloadable as
annual zips of CSVs.

Available reports:
  - Legacy (Futures-Only and Combined Futures+Options) — 1986-present
  - Disaggregated Reports — 2006-present (split traders into more buckets)
  - TFF (Traders in Financial Futures) — 2010-present (for financial markets)

URL pattern: https://www.cftc.gov/files/dea/history/{report}_{year}.zip
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from common import DATA_ROOT, fmt_bytes, log, make_session, stream_download

OUT_DIR = DATA_ROOT / "CFTC_COT"
BASE = "https://www.cftc.gov/files/dea/history/{slug}_{year}.zip"

# (report_slug, human-name, start_year)
REPORTS = [
    ("deahistfo", "Legacy Futures-Only", 1986),
    ("deacot", "Legacy Combined (FO+Options)", 1986),
    ("fut_disagg_xls", "Disaggregated Futures-Only", 2006),
    ("com_disagg_xls", "Disaggregated Combined", 2006),
    ("fut_fin_xls", "TFF Futures-Only", 2010),
    ("com_fin_xls", "TFF Combined", 2010),
]


def fetch(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    today = dt.date.today()

    total_dl = downloads = 0
    for slug, name, start_year in REPORTS:
        rdir = OUT_DIR / slug
        rdir.mkdir(parents=True, exist_ok=True)
        log(f"=== {name} ({slug}) ===")
        for y in range(start_year, today.year + 1):
            dest = rdir / f"{slug}_{y}.zip"
            if dest.exists() and not force and y < today.year:
                continue
            url = BASE.format(slug=slug, year=y)
            try:
                n = stream_download(session, url, dest)
                total_dl += n
                downloads += 1
                log(f"  {y}  {fmt_bytes(n)}")
            except Exception as e:
                log(f"  {y}  FAIL: {e}")
    log(f"Done. {downloads} new archives, {fmt_bytes(total_dl)} this run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fetch(force=a.force)


if __name__ == "__main__":
    main()
