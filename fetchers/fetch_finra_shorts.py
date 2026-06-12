"""Download FINRA daily short sale volume files (Reg SHO).

For every trading day FINRA publishes per-ticker short sale volume across
its trade-reporting facilities. Predictive of forward returns per Diether/
Lee/Werner (2009) and many follow-ons.

Files: ~1-5 MB each, one per trading day. Available 2010-present.
  CNMSshvol  — consolidated NMS (the broad daily file)
  FNRAshvol  — FINRA ADF
  FNSQshvol  — NASDAQ TRF Carteret
  FNYXshvol  — NYSE TRF
  FORFshvol  — ORF off-exchange

Refresh: today's file becomes available next business morning. Cached
files are immutable; re-running pulls only what's new.

URL pattern: https://cdn.finra.org/equity/regsho/daily/{venue}shvol{YYYYMMDD}.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import DATA_ROOT, fmt_bytes, log, make_session

OUT_DIR = DATA_ROOT / "FINRA" / "short_volume"
BASE = "https://cdn.finra.org/equity/regsho/daily/{venue}shvol{date}.txt"
VENUES = ["CNMS", "FNRA", "FNSQ", "FNYX", "FORF"]
START_DATE = dt.date(2010, 5, 3)  # FINRA Reg SHO daily volume start


def _business_days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri (we'll let server 404 on holidays)
            yield d
        d += dt.timedelta(days=1)


def _fetch_one(session, venue: str, d: dt.date) -> tuple[str, int]:
    date_str = d.strftime("%Y%m%d")
    dest = OUT_DIR / venue / f"{date_str}.txt"
    if dest.exists():
        return f"{venue} {date_str} cached", 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(venue=venue, date=date_str)
    r = session.get(url, timeout=30)
    # FINRA's CDN returns 403 AccessDenied (not 404) for non-existent files.
    if r.status_code in (403, 404):
        return f"{venue} {date_str} missing", 0
    r.raise_for_status()
    if not r.content or r.content.startswith(b"<"):
        return f"{venue} {date_str} empty/html", 0
    tmp = dest.with_suffix(".txt.tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return f"{venue} {date_str} {fmt_bytes(len(r.content))}", len(r.content)


def fetch(start: dt.date = START_DATE, end: dt.date | None = None,
          venues: list[str] | None = None, workers: int = 8,
          verbose: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if end is None:
        end = dt.date.today()
    if venues is None:
        venues = VENUES
    session = make_session()

    jobs = [(v, d) for v in venues for d in _business_days(start, end)]
    log(f"Fetching FINRA shorts: {len(venues)} venues × {len(jobs) // len(venues)} days "
        f"= {len(jobs):,} files, {workers} workers")

    total_bytes = downloaded = cached = missing = errored = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_one, session, v, d) for v, d in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg, n = fut.result()
            except Exception as e:
                errored += 1
                if verbose:
                    log(f"  ERR: {e}")
                continue
            total_bytes += n
            if "cached" in msg:
                cached += 1
            elif "missing" in msg or "empty" in msg:
                missing += 1
            else:
                downloaded += 1
                if verbose:
                    log(f"  {msg}")
            if i % 500 == 0:
                log(f"  progress: {i:,}/{len(jobs):,}  "
                    f"({downloaded} new, {cached} cached, {missing} missing, {errored} err)")
    log(f"Done. {downloaded:,} new, {cached:,} cached, {missing:,} 404/empty, "
        f"{errored:,} errors, {fmt_bytes(total_bytes)} this run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=lambda s: dt.date.fromisoformat(s), default=START_DATE)
    ap.add_argument("--end", type=lambda s: dt.date.fromisoformat(s), default=None)
    ap.add_argument("--venues", nargs="+", default=None,
                    help=f"subset of {VENUES}; default all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    fetch(start=a.start, end=a.end, venues=a.venues, workers=a.workers, verbose=a.verbose)


if __name__ == "__main__":
    main()
