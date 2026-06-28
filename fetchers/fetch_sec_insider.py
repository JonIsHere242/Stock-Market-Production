"""Download SEC Insider Transactions Data Sets — pre-parsed Form 3/4/5 by quarter.

Each quarter (2006 Q1 onward) the SEC's Office of Structured Disclosure publishes a
clean tab-delimited extract of every ownership filing (Forms 3, 4, 5 and amendments):

  SUBMISSION.tsv        one row per filing  (ACCESSION_NUMBER, FILING_DATE,
                        PERIOD_OF_REPORT, DOCUMENT_TYPE, ISSUERCIK, ISSUERNAME,
                        ISSUERTRADINGSYMBOL, ...)
  REPORTINGOWNER.tsv    who the insider is (RPTOWNERCIK, RPTOWNERNAME,
                        RPTOWNER_RELATIONSHIP = isDirector/isOfficer/isTenPercentOwner,
                        RPTOWNER_TITLE, ...)
  NONDERIV_TRANS.tsv    non-derivative transactions (TRANS_DATE, TRANS_CODE [P/S/A/M/F/G],
                        TRANS_SHARES, TRANS_PRICEPERSHARE, TRANS_ACQUIRED_DISP_CD [A/D],
                        SHRS_OWND_FOLG_TRANS, ...)
  NONDERIV_HOLDING.tsv  non-derivative holdings (no transaction)
  DERIV_TRANS.tsv       derivative (option) transactions
  DERIV_HOLDING.tsv     derivative holdings
  FOOTNOTES.tsv / OWNER_SIGNATURE.tsv

These are the clean full-universe source for an insider buying/selling feature — no need
to parse millions of raw Form 4 XML filings. ~8-15 MB per quarter compressed (the entire
2006-present archive is well under ~1 GB).

THE PIT LINCHPIN: key everything on FILING_DATE, not TRANS_DATE. A Form 4 may be filed up
to two business days after the trade; FILING_DATE is the day it became public. (build_insider_panel.py
enforces this.)

URL pattern: https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip
"""
from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

from common import (DATA_ROOT, fmt_bytes, log, make_session, sec_get,
                    stream_download)

BASE = ("https://www.sec.gov/files/structureddata/data/"
        "insider-transactions-data-sets/{y}q{q}_form345.zip")
RAW_DIR = DATA_ROOT / "SEC" / "insider" / "raw"
EXTRACT_DIR = DATA_ROOT / "SEC" / "insider" / "extracted"
START_YEAR = 2006  # SEC insider data sets begin 2006 Q1


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


def fetch(start_year: int = START_YEAR, force: bool = False, extract: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session(host="www.sec.gov", sec=True)
    qs = _quarters(start_year)
    log(f"Fetching {len(qs)} SEC insider-transaction archives ({start_year}-present)")

    total_dl = downloads = 0
    for y, q in qs:
        url = BASE.format(y=y, q=q)
        dest = RAW_DIR / f"{y}q{q}_form345.zip"
        # Older quarters are immutable once published; the current quarter updates as new
        # filings arrive, so always re-fetch it.
        if dest.exists() and not force and not _is_current_quarter(y, q):
            log(f"  {y}q{q} cached")
            continue
        try:
            n = stream_download(session, url, dest, rate_limited_get=sec_get)
            total_dl += n
            downloads += 1
            log(f"  {y}q{q}  {fmt_bytes(n)}")
        except Exception as e:
            # 2006-era quarters or a not-yet-published current quarter -> skip quietly.
            log(f"  {y}q{q}  skip ({e})")
            continue
        if extract:
            qd = EXTRACT_DIR / f"{y}q{q}"
            qd.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(qd)
            log(f"    extracted -> {qd.name}")
    log(f"Done. {downloads} new archives, downloaded {fmt_bytes(total_dl)}")


def main():
    ap = argparse.ArgumentParser(description="Download SEC Insider Transactions Data Sets")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--extract", action="store_true",
                    help="unzip each quarter into Data/SEC/insider/extracted/{y}q{q}/")
    a = ap.parse_args()
    fetch(start_year=a.start_year, force=a.force, extract=a.extract)


if __name__ == "__main__":
    main()
