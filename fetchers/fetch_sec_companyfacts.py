"""Download SEC companyfacts.zip — XBRL fundamentals for every filer.

This is the bulk archive of every concept (Revenues, NetIncome, etc.) ever
filed via XBRL, one JSON per CIK. ~1-2 GB compressed; extracts to ~10+ GB.

Refresh: SEC regenerates the zip nightly. Re-running this script overwrites
the local copy. By default we keep it as a zip and stream-read at use time;
pass --extract to unzip into Data/SEC/companyfacts/.

URL: https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common import (DATA_ROOT, atomic_write_bytes, file_age_hours, fmt_bytes,
                    log, make_session, sec_get, stream_download)

URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
RAW_DIR = DATA_ROOT / "SEC" / "raw"
EXTRACT_DIR = DATA_ROOT / "SEC" / "companyfacts"
ZIP_PATH = RAW_DIR / "companyfacts.zip"


def fetch(force: bool = False, extract: bool = False, max_age_hours: float = 20.0) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    age = file_age_hours(ZIP_PATH)
    if not force and age < max_age_hours:
        log(f"companyfacts.zip is {age:.1f}h old (< {max_age_hours}h) — skipping. Use --force to override.")
    else:
        log(f"Downloading {URL} → {ZIP_PATH}")
        session = make_session(host="www.sec.gov", sec=True)
        n = stream_download(session, URL, ZIP_PATH, rate_limited_get=sec_get)
        log(f"  wrote {fmt_bytes(n)}")

    if extract:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        log(f"Extracting → {EXTRACT_DIR} (this can take several minutes)")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            members = zf.namelist()
            log(f"  {len(members):,} JSON files in archive")
            zf.extractall(EXTRACT_DIR)
        log("  extract complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redownload even if recent")
    ap.add_argument("--extract", action="store_true",
                    help="also unzip into Data/SEC/companyfacts/ (~10+ GB)")
    ap.add_argument("--max-age-hours", type=float, default=20.0)
    a = ap.parse_args()
    fetch(force=a.force, extract=a.extract, max_age_hours=a.max_age_hours)


if __name__ == "__main__":
    main()
