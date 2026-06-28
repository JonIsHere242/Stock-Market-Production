"""
Overnight vs intraday jump decomposition.

Decomposes price jumps by session:
  - Overnight jump: |ln(Open/prevClose)| > 4x local vol
  - Intraday jump: |ln(Close/Open)| > 4x local vol
Produces rolling 60d counts and overnight share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_overnight_jump",
    "description": (
        "Decomposes jumps by session using a 4x-local-vol threshold. "
        "Local vol is estimated as 20-day rolling std of log-returns (close-to-close). "
        "Overnight jump = |ln(Open/prevClose)| > 4*sigma; "
        "Intraday jump = |ln(Close/Open)| > 4*sigma. "
        "Produces rolling-60d overnight count, intraday count, and overnight share "
        "(overnight / total jumps). Per-ticker OHLCV proxy -- no cross-sectional step. "
        "Extends / relates to xdom_allan_variance decomposition."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext4_overnight_jump_on_count60",
        "ext4_overnight_jump_id_count60",
        "ext4_overnight_jump_on_share60",
    ],
    "tags": ["jump", "overnight", "intraday", "session", "volatility", "decomposition"],
    "version": "1.0",
    "author": "Round-5 expansion (xdom_allan_variance)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Log-return series (close-to-close) for local vol estimation
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=np.float64)
    open_ = df["Open"].to_numpy(dtype=np.float64)

    # Safe log-return of close-to-close
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    # Guard divide-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        cc_ret = np.where(prev_close > 0, np.log(np.where(prev_close > 0, close / prev_close, np.nan)), np.nan)
        cc_ret = cc_ret.astype(np.float64)

    # 20-day rolling std of cc returns (Bessel-corrected, min_periods=10)
    cc_series = pd.Series(cc_ret, index=df.index)
    sigma20 = cc_series.rolling(window=20, min_periods=10).std()  # pandas std = sample std

    sigma_np = sigma20.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 2. Overnight log-move: ln(Open_t / Close_{t-1})
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        on_move = np.where(
            (prev_close > 0) & ~np.isnan(prev_close),
            np.abs(np.log(np.where(prev_close > 0, open_ / prev_close, np.nan))),
            np.nan,
        )

    # ------------------------------------------------------------------ #
    # 3. Intraday log-move: ln(Close_t / Open_t)
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        id_move = np.where(
            open_ > 0,
            np.abs(np.log(np.where(open_ > 0, close / open_, np.nan))),
            np.nan,
        )

    # ------------------------------------------------------------------ #
    # 4. Jump flags: move > 4 * sigma (threshold applied point-in-time)
    # ------------------------------------------------------------------ #
    threshold = 4.0 * sigma_np  # shape (n,)

    on_jump = np.where(~np.isnan(on_move) & ~np.isnan(threshold), (on_move > threshold).astype(np.float64), np.nan)
    id_jump = np.where(~np.isnan(id_move) & ~np.isnan(threshold), (id_move > threshold).astype(np.float64), np.nan)

    # ------------------------------------------------------------------ #
    # 5. Rolling 60-day sums (min_periods=20 so early rows stay NaN)
    # ------------------------------------------------------------------ #
    on_series = pd.Series(on_jump, index=df.index)
    id_series = pd.Series(id_jump, index=df.index)

    on_count60 = on_series.rolling(window=60, min_periods=20).sum()
    id_count60 = id_series.rolling(window=60, min_periods=20).sum()

    # ------------------------------------------------------------------ #
    # 6. Overnight share = on_count / (on_count + id_count)
    # ------------------------------------------------------------------ #
    total60 = on_count60 + id_count60
    with np.errstate(divide="ignore", invalid="ignore"):
        on_share60 = np.where(total60 > 0, on_count60 / total60, np.nan)

    # ------------------------------------------------------------------ #
    # 7. Assign to dataframe
    # ------------------------------------------------------------------ #
    df["ext4_overnight_jump_on_count60"] = on_count60.values
    df["ext4_overnight_jump_id_count60"] = id_count60.values
    df["ext4_overnight_jump_on_share60"] = on_share60

    return df
