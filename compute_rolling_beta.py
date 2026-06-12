#!/usr/bin/env python
"""
compute_rolling_beta.py
=======================
Precompute a LOOKAHEAD-SAFE rolling beta per (Ticker, Date) versus an
equal-weight universe market proxy, and write it to Data/RollingBeta.parquet.

Why this exists
---------------
The backtester's only diversification lever was a static, single-snapshot
cluster cap (Correlations.parquet, frozen in March). On synchronized risk-off
days realized correlations spike and that static cap gives a false sense of
diversification. This script produces a *rolling* beta the diversify-aware
selector in 5__NightlyBackTester_experimental.py uses to prefer lower-beta,
less market-coupled names when --diversify is on.

Lookahead safety
----------------
Beta as-of close of day t uses only returns through day t. The backtester
decides buys on bar t and fills at the t+1 open, so gating with beta-as-of-t
uses strictly past information. Safe.

Market proxy
------------
Equal-weight mean of daily log returns across every ticker that has data on a
given date. Robust, self-contained, no external index feed required.

Usage
-----
    python compute_rolling_beta.py                  # defaults: window=60, min_periods=40
    python compute_rolling_beta.py --window 90 --min_periods 60
    python compute_rolling_beta.py --pred_dir Data/RFpredictions --out Data/RollingBeta.parquet
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_close_matrix(pred_dir):
    """Read Date+Close from every prediction parquet into a wide (Date x Ticker) matrix."""
    files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {pred_dir}")

    series = {}
    skipped = 0
    for f in tqdm(files, desc="Loading closes", unit="file"):
        ticker = os.path.splitext(os.path.basename(f))[0]
        try:
            # read only the two columns we need -- much faster than full read
            tbl = pq.read_table(f, columns=["Date", "Close"])
            df = tbl.to_pandas()
            if df.empty:
                skipped += 1
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            s = (df.dropna(subset=["Date", "Close"])
                   .drop_duplicates(subset="Date", keep="last")
                   .set_index("Date")["Close"]
                   .astype("float64"))
            if len(s) >= 2:
                series[ticker] = s
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            continue

    if not series:
        raise SystemExit("No usable Close series loaded.")

    close = pd.DataFrame(series).sort_index()
    print(f"Loaded {close.shape[1]} tickers x {close.shape[0]} dates "
          f"({skipped} files skipped).")
    return close


def compute_rolling_beta(close, window=60, min_periods=40):
    """Vectorised rolling beta of every ticker vs an equal-weight market proxy.

    beta_i,t = Cov_t(r_i, r_m) / Var_t(r_m)   over a trailing `window` of days.
    Rolling aggregations skip NaN and require `min_periods` valid points, so
    tickers with short history simply produce NaN until they have enough data.
    """
    ret = np.log(close).diff()                       # (Date x Ticker) log returns
    mkt = ret.mean(axis=1)                            # equal-weight market proxy (NaN-skipping)

    # E[x] terms (rolling, NaN-aware)
    mean_i = ret.rolling(window, min_periods=min_periods).mean()
    mean_m = mkt.rolling(window, min_periods=min_periods).mean()
    mean_im = ret.mul(mkt, axis=0).rolling(window, min_periods=min_periods).mean()
    mean_mm = (mkt * mkt).rolling(window, min_periods=min_periods).mean()

    cov_im = mean_im.sub(mean_i.mul(mean_m, axis=0))  # Cov(r_i, r_m)
    var_m = mean_mm - (mean_m * mean_m)               # Var(r_m)
    var_m = var_m.replace(0.0, np.nan)

    beta = cov_im.div(var_m, axis=0)
    return beta


def main():
    ap = argparse.ArgumentParser(description="Precompute rolling beta vs equal-weight universe.")
    ap.add_argument("--pred_dir", default="Data/RFpredictions",
                    help="Folder of per-ticker prediction parquets (need Date, Close).")
    ap.add_argument("--out", default="Data/RollingBeta.parquet",
                    help="Output parquet path (long format: Date, Ticker, Beta).")
    ap.add_argument("--window", type=int, default=60, help="Rolling window in trading days.")
    ap.add_argument("--min_periods", type=int, default=40, help="Minimum valid days in window.")
    args = ap.parse_args()

    close = load_close_matrix(args.pred_dir)
    beta = compute_rolling_beta(close, window=args.window, min_periods=args.min_periods)

    long = (beta.reset_index()
                .melt(id_vars="Date", var_name="Ticker", value_name="Beta")
                .dropna(subset=["Beta"]))
    long = long[np.isfinite(long["Beta"])]
    long = long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    long.to_parquet(args.out, index=False)

    print(f"\nWrote {len(long):,} (Ticker, Date) rows for "
          f"{long['Ticker'].nunique():,} tickers to {args.out}")
    print(f"Beta summary: min={long['Beta'].min():.2f}  "
          f"p25={long['Beta'].quantile(.25):.2f}  "
          f"median={long['Beta'].median():.2f}  "
          f"p75={long['Beta'].quantile(.75):.2f}  "
          f"max={long['Beta'].max():.2f}")
    print(f"Window={args.window}d, min_periods={args.min_periods}. "
          f"Beta is as-of each date's close (lookahead-safe for t+1 entries).")


if __name__ == "__main__":
    main()
