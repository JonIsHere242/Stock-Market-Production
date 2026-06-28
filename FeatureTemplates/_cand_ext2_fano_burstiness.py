"""
Fano factor / index of dispersion (burstiness) — per-ticker causal proxy.

From counting statistics: over a rolling 80-day window split into 16 blocks of
5 days, count the number of large-move days per block (|return| > window 80th
percentile) and produce Fano factor = variance(counts) / mean(counts).
Values >1 indicate bursty/clustered behaviour; =1 Poisson; <1 regular.
Same metric applied to volume spikes for an orthogonal second axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext2_fano_burstiness",
    "description": (
        "Fano factor (index of dispersion) over a rolling 80-day window split "
        "into 16 non-overlapping 5-day blocks. Counts large-move days per block "
        "(|ret| > 80th-pct of window) and large-volume days per block "
        "(volume > 80th-pct of window), then computes Fano = var(counts)/mean(counts). "
        "Fano>1 = bursty/clustered extremes, =1 Poisson, <1 regular. "
        "Orthogonal to Allan-variance features (different statistic: count dispersion, "
        "not mean-level fluctuation). Produces return-Fano, volume-Fano, and "
        "their ratio as a conditioning signal."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext2_fano_burstiness_ret",
        "ext2_fano_burstiness_vol",
        "ext2_fano_burstiness_ratio",
    ],
    "tags": ["burstiness", "fano", "counting", "dispersion", "volume", "volatility"],
    "version": "1.0.0",
    "author": "Round-3 deep exploration of xdom_allan_variance winner vein",
}

# Window configuration
_WIN = 80          # total rolling window (days)
_NBLOCKS = 16      # number of sub-blocks
_BLOCK = 5         # days per block  (80 / 16 = 5)
_PCTILE = 80       # percentile threshold for "large" event


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Fano factor features on a single-ticker ascending DataFrame."""

    n = len(df)
    # Pre-allocate output arrays as float64 (NaN by default)
    fano_ret = np.full(n, np.nan, dtype=np.float64)
    fano_vol = np.full(n, np.nan, dtype=np.nan.__class__)
    fano_vol = np.full(n, np.nan, dtype=np.float64)

    # Log returns (today vs yesterday) – shift(1) is fully causal
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Absolute log return per day (causal: today's close vs yesterday's)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[:-1] > 0) & np.isfinite(close[:-1]) & np.isfinite(close[1:]),
            np.abs(np.log(close[1:] / close[:-1])),
            np.nan,
        )
    # Align: ret[i] corresponds to row i+1; pad with NaN at front
    abs_ret = np.empty(n, dtype=np.float64)
    abs_ret[0] = np.nan
    abs_ret[1:] = log_ret

    # Guard volume zeros / negatives
    safe_vol = np.where(volume > 0, volume, np.nan)

    # We need _WIN rows to compute one Fano estimate.
    # We also need _NBLOCKS * _BLOCK == _WIN rows split into _NBLOCKS blocks.
    # Loop from index (_WIN - 1) onward; window = rows [i-_WIN+1 .. i] inclusive.
    for i in range(_WIN - 1, n):
        w_ret = abs_ret[i - _WIN + 1 : i + 1]   # shape (80,)
        w_vol = safe_vol[i - _WIN + 1 : i + 1]  # shape (80,)

        # ---- return Fano ------------------------------------------------
        # 80th percentile threshold (uses only past+current window -> causal)
        valid_ret = w_ret[np.isfinite(w_ret)]
        if len(valid_ret) < _NBLOCKS:
            # not enough data to fill even one block per non-NaN; leave NaN
            pass
        else:
            thresh_ret = np.nanpercentile(w_ret, _PCTILE)
            # large-move indicator (1 if |ret| > threshold, else 0), NaN→0
            large_ret = np.where(
                np.isfinite(w_ret) & (w_ret > thresh_ret), 1.0, 0.0
            )
            # reshape into (16, 5) blocks and count per block
            counts_ret = large_ret.reshape(_NBLOCKS, _BLOCK).sum(axis=1)
            mu_ret = counts_ret.mean()
            if mu_ret > 0:
                fano_ret[i] = counts_ret.var(ddof=0) / mu_ret
            # if mu_ret == 0 all blocks empty -> leave NaN (undefined Fano)

        # ---- volume Fano ------------------------------------------------
        valid_vol = w_vol[np.isfinite(w_vol)]
        if len(valid_vol) >= _NBLOCKS:
            thresh_vol = np.nanpercentile(w_vol, _PCTILE)
            large_vol = np.where(
                np.isfinite(w_vol) & (w_vol > thresh_vol), 1.0, 0.0
            )
            counts_vol = large_vol.reshape(_NBLOCKS, _BLOCK).sum(axis=1)
            mu_vol = counts_vol.mean()
            if mu_vol > 0:
                fano_vol[i] = counts_vol.var(ddof=0) / mu_vol

    # ---- ratio: return burstiness / volume burstiness -------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        fano_ratio = np.where(
            np.isfinite(fano_ret) & np.isfinite(fano_vol) & (fano_vol != 0),
            fano_ret / fano_vol,
            np.nan,
        )

    df["ext2_fano_burstiness_ret"] = fano_ret
    df["ext2_fano_burstiness_vol"] = fano_vol
    df["ext2_fano_burstiness_ratio"] = fano_ratio

    return df
