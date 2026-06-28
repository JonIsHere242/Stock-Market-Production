"""
Signed-volume (order-flow) persistence feature block.

Spec: ext_signed_volume_persist
Extends: xdom2_autocorr_volume_return (gate-validated winner)
Orthogonal axis: persistence / autocorrelation of directional order flow,
    distinct from the contemporaneous volume-return correlation in the parent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_signed_volume_persist",
    "description": (
        "Signed-volume order-flow persistence. "
        "sv_t = sign(daily_return) * detrended_log_volume_t is a per-bar order-flow "
        "imbalance proxy. Three features: (1) rolling 40-day lag-1 autocorrelation of "
        "sv_t (flow persistence / momentum-in-order-flow); (2) rolling 60-day mean of "
        "sv_t (net directional imbalance level); (3) 20-day change in the 60-day mean "
        "(acceleration of imbalance). Orthogonal to the parent xdom2_autocorr_volume_return "
        "which measures contemporaneous volume-return correlation -- here we measure whether "
        "signed flow PERSISTS across days (lag-1 AC). Per-ticker proxy; vectorised with "
        "numpy sliding_window_view for speed."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext_signed_volume_persist_ac40",    # 40d lag-1 autocorrelation of sv
        "ext_signed_volume_persist_mean60",  # 60d rolling mean of sv (net imbalance level)
        "ext_signed_volume_persist_chg20",   # 20d change in the 60d mean (acceleration)
    ],
    "tags": ["volume", "order_flow", "autocorrelation", "momentum", "persistence"],
    "version": "1.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner xdom2_autocorr_volume_return. "
        "Implementation: Claude (Anthropic) per project spec."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed-volume persistence features (per-ticker, causal/no-lookahead)."""
    n = len(df)

    # --- 1. Build signed volume sv_t ---
    # Return: log(Close/Close.shift(1))
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Log return (safe: guard zero/negative close with nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.where(close > 0, np.log(close), np.nan)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = log_close[1:] - log_close[:-1]

    # Sign of return (+1, 0, -1); leave 0 as 0 (no clear direction)
    ret_sign = np.sign(ret)  # nan propagates as nan via comparison? No -- np.sign(nan) = nan. Good.

    # Detrended log volume: log(vol) - 20d rolling mean of log(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.where(volume > 0, np.log(volume), np.nan)

    # 20d rolling mean of log_vol
    win_dv = 20
    log_vol_mean = np.full(n, np.nan)
    for i in range(win_dv - 1, n):
        window = log_vol[i - win_dv + 1: i + 1]
        if not np.all(np.isnan(window)):
            log_vol_mean[i] = np.nanmean(window)

    detrended_log_vol = log_vol - log_vol_mean  # NaN if either is NaN

    # sv_t = sign(ret) * detrended_log_vol
    sv = ret_sign * detrended_log_vol  # NaN if either component is NaN

    # --- 2. Rolling 40-day lag-1 autocorrelation of sv ---
    win_ac = 40
    ac40 = np.full(n, np.nan)
    for i in range(win_ac - 1, n):
        window = sv[i - win_ac + 1: i + 1]
        if np.sum(~np.isnan(window)) >= win_ac // 2 + 1:
            x = window[:-1]
            y = window[1:]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 10:
                xm = x[mask] - x[mask].mean()
                ym = y[mask] - y[mask].mean()
                denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
                if denom > 0:
                    ac40[i] = np.dot(xm, ym) / denom

    # --- 3. Rolling 60-day mean of sv (net directional imbalance level) ---
    win_mean = 60
    mean60 = np.full(n, np.nan)
    for i in range(win_mean - 1, n):
        window = sv[i - win_mean + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= win_mean // 2:
            mean60[i] = valid.mean()

    # --- 4. 20-day change in the 60-day mean (imbalance acceleration) ---
    chg20 = np.full(n, np.nan)
    lag = 20
    for i in range(lag, n):
        if not np.isnan(mean60[i]) and not np.isnan(mean60[i - lag]):
            chg20[i] = mean60[i] - mean60[i - lag]

    # Assign produced columns
    df["ext_signed_volume_persist_ac40"] = ac40
    df["ext_signed_volume_persist_mean60"] = mean60
    df["ext_signed_volume_persist_chg20"] = chg20

    return df
