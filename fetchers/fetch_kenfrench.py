"""Download Ken French Data Library — canonical factor returns.

Daily and monthly factor returns (Mkt-RF, SMB, HML, MOM, RMW, CMA),
industry portfolios, and size/BM/momentum quintile returns. Goes back
to 1926-07-01 for the core 3-factor model.

Use these both as model features (regime conditioning) and as a clean
benchmark for attributing your daily-model alpha to known factors.

Refresh: monthly (typically updated within a week of month-end).

URL pattern: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{archive}
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, fmt_bytes, log, make_session

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
OUT_DIR = DATA_ROOT / "KenFrench"

# Each entry: (archive_name, human description). The archive contains a CSV.
ARCHIVES = [
    ("F-F_Research_Data_Factors_daily_CSV.zip", "FF 3-factor daily (Mkt-RF, SMB, HML, RF)"),
    ("F-F_Research_Data_Factors_CSV.zip", "FF 3-factor monthly"),
    ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "FF 5-factor daily (+ RMW, CMA)"),
    ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "FF 5-factor monthly"),
    ("F-F_Momentum_Factor_daily_CSV.zip", "Momentum daily"),
    ("F-F_Momentum_Factor_CSV.zip", "Momentum monthly"),
    ("F-F_ST_Reversal_Factor_daily_CSV.zip", "Short-term reversal daily"),
    ("F-F_LT_Reversal_Factor_daily_CSV.zip", "Long-term reversal daily"),
    ("10_Industry_Portfolios_daily_CSV.zip", "10-industry portfolio returns daily"),
    ("48_Industry_Portfolios_daily_CSV.zip", "48-industry portfolio returns daily"),
    ("25_Portfolios_5x5_Daily_CSV.zip", "25 size×BM portfolios daily"),
    ("25_Portfolios_ME_BETA_5x5_Daily_CSV.zip", "25 size×beta portfolios daily"),
    ("Portfolios_Formed_on_ME_Daily_CSV.zip", "Size deciles daily"),
    ("Portfolios_Formed_on_BE-ME_Daily_CSV.zip", "BM deciles daily"),
    ("Portfolios_Formed_on_OP_Daily_CSV.zip", "Profitability deciles daily"),
    ("Portfolios_Formed_on_INV_Daily_CSV.zip", "Investment deciles daily"),
]


def _fetch_one(session, archive: str) -> tuple[int, int, int]:
    url = f"{BASE}/{archive}"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return 0, 0, 0
    r.raise_for_status()
    zip_path = OUT_DIR / archive
    zip_path.write_bytes(r.content)
    extracted = 0
    n_bytes_extracted = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for member in zf.namelist():
            out_path = OUT_DIR / member
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(member)
            out_path.write_bytes(data)
            extracted += 1
            n_bytes_extracted += len(data)
    return len(r.content), extracted, n_bytes_extracted


def fetch() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    log(f"Fetching {len(ARCHIVES)} Ken French archives")

    total_dl = total_ex = 0
    for archive, desc in ARCHIVES:
        try:
            n_dl, n_files, n_ex = _fetch_one(session, archive)
            if n_dl == 0:
                log(f"  FAIL {archive}")
            else:
                total_dl += n_dl
                total_ex += n_ex
                log(f"  ok   {archive:55s} dl {fmt_bytes(n_dl):>8s}  "
                    f"extracted {n_files} files ({fmt_bytes(n_ex)})  {desc}")
        except Exception as e:
            log(f"  FAIL {archive}: {e}")
    log(f"Done. Downloaded {fmt_bytes(total_dl)}, extracted {fmt_bytes(total_ex)}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    fetch()


if __name__ == "__main__":
    main()
