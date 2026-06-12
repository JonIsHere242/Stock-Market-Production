"""Download SEC submissions.zip — filing metadata for every CIK.

For each filer this contains a JSON list of every form they've filed
(form type, filing date, accession number, primary document, etc.).
This is the index needed to look up specific filings (10-K, 10-Q, 8-K,
Form 4, 13F-HR) later. ~600-900 MB compressed.

Refresh: nightly. Re-running this script overwrites the local copy.

URL: https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common import (DATA_ROOT, file_age_hours, fmt_bytes, log, make_session,
                    sec_get, stream_download)

URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
RAW_DIR = DATA_ROOT / "SEC" / "raw"
EXTRACT_DIR = DATA_ROOT / "SEC" / "submissions"
ZIP_PATH = RAW_DIR / "submissions.zip"


def fetch(force: bool = False, extract: bool = False, max_age_hours: float = 20.0) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    age = file_age_hours(ZIP_PATH)
    if not force and age < max_age_hours:
        log(f"submissions.zip is {age:.1f}h old — skipping. Use --force to override.")
    else:
        log(f"Downloading {URL} → {ZIP_PATH}")
        session = make_session(host="www.sec.gov", sec=True)
        n = stream_download(session, URL, ZIP_PATH, rate_limited_get=sec_get)
        log(f"  wrote {fmt_bytes(n)}")

    if extract:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        log(f"Extracting → {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            members = zf.namelist()
            log(f"  {len(members):,} JSON files in archive")
            zf.extractall(EXTRACT_DIR)
        log("  extract complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract", action="store_true",
                    help="also unzip into Data/SEC/submissions/")
    ap.add_argument("--max-age-hours", type=float, default=20.0)
    a = ap.parse_args()
    fetch(force=a.force, extract=a.extract, max_age_hours=a.max_age_hours)


if __name__ == "__main__":
    main()
