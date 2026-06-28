"""Download & aggregate SEC EDGAR access logs to a per-CIK daily attention panel.

The SEC publishes its own web-server logs: every request for every filing,
timestamped. Abnormal pre-event download activity predicts returns and earnings
surprises (Drake/Roulstone/Thornock, "information demand"). This is the
"who's reading whose filings" signal -- fully orthogonal to OHLCV.

Coverage (SEC publishes only these windows):
  - 2003-01-01 .. 2017-06-30   (first data set)
  - 2020-05-19 .. 2025-06-30   (second data set)
  - GAP 2017-07-01 .. 2020-05-18: no data
  - Nothing after 2025-06-30 yet => ~1yr stale at the live edge.
  => TRAINING / RESEARCH signal, not a live same-day feature.

Each daily file is a large zip (250 MB .. 1.2 GB, growing over time) holding one
CSV of raw request records. We CANNOT keep the raw (~2 TB full). So per day we:
  stream the zip -> chunk-parse -> robot/index/error filter -> keep our universe
  CIKs -> aggregate to (date, cik, n_requests, n_unique_filings) -> write a tiny
  parquet -> DELETE the raw zip. Output for the whole history is a few hundred MB.

Resumable at day granularity: a day whose output parquet exists is skipped; a
confirmed-404 day is stamped under _missing/ so we never re-hit it.

URL: https://www.sec.gov/dera/data/Public-EDGAR-log-file-data/{Y}/Qtr{Q}/log{YYYYMMDD}.zip

The two data sets have DIFFERENT schemas (the second is FOIA-stripped):
  v1 (2003..2017): ip,date,time,zone,cik,accession,extention,code,size,idx,
                   norefer,noagent,find,crawler,browser   -> full robot filter.
  v2 (2020..2025): time,"uri_path"  ONLY. No cik/code/crawler. CIK+accession are
                   parsed from the path /Archives/edgar/data/{CIK}/{ACCESSION}/...
                   and robot/error filtering is impossible (documented limitation;
                   per-ticker abnormal-z in feature-build absorbs the bot baseline).
Output is identical for both: (date, cik, n_requests, n_unique_filings).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_ROOT,
    fmt_bytes,
    log,
    make_session,
    sec_get,
    stream_download,
)

BASE = "https://www.sec.gov/dera/data/Public-EDGAR-log-file-data/{y}/Qtr{q}/log{ymd}.zip"
OUT_DIR = DATA_ROOT / "EdgarAccessLogs"
DAILY_DIR = OUT_DIR / "daily"
MISSING_DIR = OUT_DIR / "_missing"
TMP_DIR = OUT_DIR / "_tmp"

# SEC-published coverage windows (inclusive). Days outside => skipped, no fetch.
COVERED = [
    (dt.date(2003, 1, 1), dt.date(2017, 6, 30)),
    (dt.date(2020, 5, 19), dt.date(2025, 6, 30)),
]

V1_USECOLS = ["cik", "accession", "code", "idx", "crawler"]
V1_DTYPES = {"cik": "float64", "accession": "string", "code": "float64",
             "idx": "float64", "crawler": "float64"}
# /Archives/edgar/data/{CIK}/{ACCESSION}/...  (accession may be dashed or not)
V2_PATH_RE = r"/edgar/data/(\d+)/([0-9-]+)"
CHUNK_ROWS = 3_000_000


def quarter(d: dt.date) -> int:
    return (d.month - 1) // 3 + 1


def in_coverage(d: dt.date) -> bool:
    return any(lo <= d <= hi for lo, hi in COVERED)


def day_paths(d: dt.date) -> tuple[Path, Path, Path]:
    ymd = d.strftime("%Y%m%d")
    out = DAILY_DIR / f"{d.year}" / f"log{ymd}.parquet"
    miss = MISSING_DIR / f"log{ymd}.flag"
    tmp = TMP_DIR / f"log{ymd}.zip"
    return out, miss, tmp


def load_universe_ciks(path: Path | None) -> set[int] | None:
    if path is None:
        cands = sorted(glob.glob(str(DATA_ROOT / "TickerCikData" / "TickerCIKs_*.parquet")))
        if not cands:
            return None
        path = Path(cands[-1])
    df = pd.read_parquet(path, columns=["cik"])
    ciks = pd.to_numeric(df["cik"], errors="coerce").dropna().astype("int64")
    log(f"universe: {ciks.nunique():,} unique CIKs from {Path(path).name}")
    return set(ciks.unique().tolist())


def iter_days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _detect_schema(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member) as fh:
        first = fh.readline().decode("utf-8", "replace").strip().lower()
    if "uri_path" in first:
        return "v2"
    if "cik" in first and ("crawler" in first or "idx" in first):
        return "v1"
    return "unknown"


def _v1_chunks(fh, universe):
    """2003-2017 structured logs -> (cik, accession) frames, robot/error filtered."""
    reader = pd.read_csv(fh, usecols=V1_USECOLS, dtype=V1_DTYPES,
                         chunksize=CHUNK_ROWS, on_bad_lines="skip")
    for chunk in reader:
        m = (chunk["code"] < 300) & (chunk["idx"] == 0.0) & (chunk["crawler"] != 1.0)
        chunk = chunk.loc[m, ["cik", "accession"]].dropna(subset=["cik"])
        if chunk.empty:
            continue
        chunk["cik"] = chunk["cik"].astype("int64")
        if universe is not None:
            chunk = chunk[chunk["cik"].isin(universe)]
        if not chunk.empty:
            yield chunk


def _v2_chunks(fh, universe):
    """2020-2025 stripped logs -> parse (cik, accession) from uri_path."""
    reader = pd.read_csv(fh, usecols=["uri_path"], dtype="string",
                         chunksize=CHUNK_ROWS, on_bad_lines="skip")
    for chunk in reader:
        ext = chunk["uri_path"].str.extract(V2_PATH_RE)
        ext.columns = ["cik", "accession"]
        ext = ext.dropna(subset=["cik", "accession"])
        if ext.empty:
            continue
        ext["cik"] = pd.to_numeric(ext["cik"], errors="coerce")
        ext = ext.dropna(subset=["cik"])
        ext["cik"] = ext["cik"].astype("int64")
        if universe is not None:
            ext = ext[ext["cik"].isin(universe)]
        if not ext.empty:
            yield ext[["cik", "accession"]]


def aggregate_zip(tmp: Path, date: dt.date, universe: set[int] | None):
    """Stream-parse one daily zip into a per-CIK aggregate.

    Returns (DataFrame|None, schema_str). Schema-adaptive across the two data sets.
    """
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(tmp) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            return None, "no-csv"
        member = members[0]
        schema = _detect_schema(zf, member)
        if schema == "unknown":
            return None, "unknown"
        with zf.open(member) as fh:
            chunks = _v1_chunks(fh, universe) if schema == "v1" else _v2_chunks(fh, universe)
            for chunk in chunks:
                g = chunk.groupby(["cik", "accession"], sort=False).size().rename("n").reset_index()
                parts.append(g)
    if not parts:
        return None, schema
    allg = pd.concat(parts, ignore_index=True)
    allg = allg.groupby(["cik", "accession"], sort=False)["n"].sum().reset_index()
    out = allg.groupby("cik").agg(
        n_requests=("n", "sum"),
        n_unique_filings=("accession", "nunique"),
    ).reset_index()
    out.insert(0, "date", pd.Timestamp(date))
    return out, schema


def process_day(session, date: dt.date, universe, force: bool = False) -> str:
    out, miss, tmp = day_paths(date)
    if out.exists() and not force:
        return "skip"
    if miss.exists() and not force:
        return "missing"
    url = BASE.format(y=date.year, q=quarter(date), ymd=date.strftime("%Y%m%d"))
    try:
        nbytes = stream_download(session, url, tmp, rate_limited_get=sec_get)
    except Exception as e:
        msg = str(e)
        if "404" in msg or "403" in msg:
            miss.parent.mkdir(parents=True, exist_ok=True)
            miss.write_text(msg[:200])
            return "missing"
        log(f"  ! download error {date}: {msg[:120]}")
        if tmp.exists():
            tmp.unlink()
        return "error"
    try:
        agg, schema = aggregate_zip(tmp, date, universe)
    finally:
        if tmp.exists():
            tmp.unlink()
    if agg is None or agg.empty:
        miss.parent.mkdir(parents=True, exist_ok=True)
        miss.write_text(f"empty-after-filter ({schema})")
        return "empty"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(".parquet.tmp")
    agg.to_parquet(tmp_out, index=False)
    tmp_out.replace(out)
    log(f"  {date} [{schema}] zip={fmt_bytes(nbytes):>9}  ciks={len(agg):>5}  "
        f"reqs={int(agg['n_requests'].sum()):>9,}")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--universe", default=None,
                    help="TickerCIKs parquet (default: latest in Data/TickerCikData)")
    ap.add_argument("--all-ciks", action="store_true",
                    help="keep ALL filers (default: filter to universe CIKs)")
    ap.add_argument("--force", action="store_true", help="re-process existing days")
    ap.add_argument("--limit-days", type=int, default=0,
                    help="smoke test: process at most N covered days then stop")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    universe = None if args.all_ciks else load_universe_ciks(
        Path(args.universe) if args.universe else None)

    session = make_session(sec=True)
    for d in (DAILY_DIR, MISSING_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    days = [d for d in iter_days(start, end) if in_coverage(d)]
    skipped_uncovered = sum(1 for d in iter_days(start, end) if not in_coverage(d))
    log(f"range {start}..{end}: {len(days):,} covered days "
        f"({skipped_uncovered:,} uncovered days skipped)")

    counts = {"ok": 0, "skip": 0, "missing": 0, "empty": 0, "error": 0}
    t0 = time.monotonic()
    processed = 0
    for i, d in enumerate(days, 1):
        status = process_day(session, d, universe, force=args.force)
        counts[status] = counts.get(status, 0) + 1
        if status == "ok":
            processed += 1
        if i % 25 == 0 or i == len(days):
            el = time.monotonic() - t0
            log(f"[{i}/{len(days)}] ok={counts['ok']} skip={counts['skip']} "
                f"missing={counts['missing']} err={counts['error']} "
                f"elapsed={el/60:.1f}m")
        if args.limit_days and processed >= args.limit_days:
            log(f"limit-days={args.limit_days} reached, stopping.")
            break
    log(f"DONE {counts}")


if __name__ == "__main__":
    main()
