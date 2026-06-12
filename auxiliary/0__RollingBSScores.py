"""
Rebuild enriched_ticker_scores rank-normalization at monthly anchor dates
using only the universe of tickers known to exist as of each anchor.

The underlying name embeddings are deterministic (input string is
"TICK -- COMPANY NAME", which doesn't change), so the raw concept scores
are static. The only PIT-sensitive piece is the rank-normalization base,
which currently uses the May-2026 universe of ~5,484 tickers regardless
of the trade date. This script restricts that base to the universe
present in the most recent ``TickerCIKs_*.parquet`` file with date <= anchor.

For anchors before the oldest ticker file (2025-09-13), the earliest
available file is used as a best-effort fallback.

Output: ``Data/BSScores/bs_scores_YYYYMM.parquet`` per anchor with columns
``ticker, bs_norm`` -- a drop-in replacement for the rank logic at
``5__NightlyBackTester.py:1046``.

Run:
    python 0__RollingBSScores.py --start 2025-04-01 --end 2026-05-01
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- path shim: this script lives in auxiliary/, so add the project root to sys.path
# to keep `from Util import ...` working when run directly. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from Util import get_logger

logger = get_logger("RollingBSScores")

SCORES_FILE = Path("Data/EDA/bullshit_stop_backtest/enriched_ticker_scores.parquet")
TICKER_DIR = Path("Data/TickerCikData")
OUT_DIR = Path("Data/BSScores")


def load_ticker_files() -> list[tuple[pd.Timestamp, Path]]:
    """Return list of (file_date, path) sorted ascending."""
    out = []
    for fp in TICKER_DIR.glob("TickerCIKs_*.parquet"):
        m = re.search(r"TickerCIKs_(\d{8})\.parquet$", fp.name)
        if not m:
            continue
        out.append((pd.to_datetime(m.group(1), format="%Y%m%d"), fp))
    out.sort()
    return out


def universe_for_anchor(ticker_files: list[tuple[pd.Timestamp, Path]],
                       anchor: pd.Timestamp) -> tuple[set[str], pd.Timestamp]:
    """Universe = tickers in latest file with date <= anchor; fallback to earliest."""
    eligible = [f for f in ticker_files if f[0] <= anchor]
    chosen = eligible[-1] if eligible else ticker_files[0]
    df = pd.read_parquet(chosen[1], columns=["ticker"])
    return set(df["ticker"].astype(str).str.upper()), chosen[0]


def build_anchor(scores_df: pd.DataFrame,
                 universe: set[str],
                 anchor: pd.Timestamp) -> pd.DataFrame:
    """Filter to PIT universe, then rank-normalize bs_max_score within it."""
    sub = scores_df[scores_df["ticker"].str.upper().isin(universe)].copy()
    sub["bs_norm"] = sub["bs_max_score"].rank(pct=True).astype("float32")
    return sub[["ticker", "bs_max_score", "bs_norm"]].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-04-01")
    p.add_argument("--end", default="2026-05-01")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SCORES_FILE.exists():
        logger.error(f"Source scores file not found: {SCORES_FILE}")
        return

    scores_df = pd.read_parquet(SCORES_FILE, columns=["ticker", "bs_max_score"])
    scores_df["ticker"] = scores_df["ticker"].astype(str)
    logger.info(f"Loaded {len(scores_df)} static scores from {SCORES_FILE.name}")

    ticker_files = load_ticker_files()
    if not ticker_files:
        logger.error(f"No TickerCIKs_*.parquet files in {TICKER_DIR}")
        return
    logger.info(f"Found {len(ticker_files)} ticker files (oldest: {ticker_files[0][0].date()}, "
                f"newest: {ticker_files[-1][0].date()})")

    anchors = pd.date_range(args.start, args.end, freq="MS")
    logger.info(f"Building {len(anchors)} anchor months")

    for anchor in tqdm(anchors, desc="Anchors"):
        out_path = out_dir / f"bs_scores_{anchor.strftime('%Y%m')}.parquet"
        if out_path.exists():
            continue
        universe, src_date = universe_for_anchor(ticker_files, anchor)
        df = build_anchor(scores_df, universe, anchor)
        df.to_parquet(out_path, index=False)
        logger.info(f"  {anchor.date()} -> {out_path.name} (universe={len(universe)} from {src_date.date()}, "
                    f"matched={len(df)})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
