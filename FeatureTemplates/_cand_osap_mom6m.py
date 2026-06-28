"""
osap_mom6m — 6-month price momentum (OSAP standard definition).

Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers:
Implications for Stock Market Efficiency", Journal of Finance 48(1), 65-91.
Also catalogued in Chen & Zimmermann (2022) Open Source Asset Pricing (OSAP)
as signal "mom6m": cumulative log-return over the past 6 months (approximately
months t-7 through t-2), skipping the most recent month to avoid the
short-term reversal effect documented in Jegadeesh (1990).

Per-ticker proxy: standard calendar-month momentum, measured in trading days
(21 days ≈ 1 month), look-back [d-147 .. d-22] (7 months back, skip 1 month).
Also produces a slope variant (linear fit over the 6-month window) and a
consistency variant (fraction of rolling 21-day windows that were positive)
to capture smoothness of the momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_mom6m",
    "description": (
        "6-month price momentum following Jegadeesh & Titman (1993) / OSAP mom6m. "
        "Cumulative log-return over [t-147 .. t-22] trading days (≈ months t-7 to t-2), "
        "skipping the most recent ~1 month to avoid short-term reversal. "
        "Three variants: (1) osap_mom6m_ret — raw cumulative log-return; "
        "(2) osap_mom6m_slope — OLS slope of log-price over the same window, "
        "normalised by price level, capturing trend strength; "
        "(3) osap_mom6m_consist — fraction of the six non-overlapping 21-day "
        "sub-windows with positive return (0-1 score), capturing breadth of gains. "
        "Pure per-ticker OHLCV proxy; no cross-sectional ranking applied here."
    ),
    "requires": ["Close"],
    "produces": ["osap_mom6m_ret", "osap_mom6m_slope", "osap_mom6m_consist"],
    "tags": ["momentum", "price", "osap", "medium-term"],
    "version": "1.0",
    "author": "Jegadeesh & Titman (1993 JF); OSAP catalog Chen & Zimmermann (2022)",
}

# ── constants ────────────────────────────────────────────────────────────────
_SKIP = 21        # trading days skipped (≈ 1 month reversal skip)
_WINDOW = 126     # 6 × 21 = 126 trading days for the momentum look-back
_TOTAL = _SKIP + _WINDOW   # 147 — index from which cumret is measured


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    ret_out = np.full(n, np.nan)
    slope_out = np.full(n, np.nan)
    consist_out = np.full(n, np.nan)

    # Need at least _TOTAL bars before we can compute anything.
    if n < _TOTAL + 1:
        df["osap_mom6m_ret"] = ret_out
        df["osap_mom6m_slope"] = slope_out
        df["osap_mom6m_consist"] = consist_out
        return df

    log_close = np.where(close > 0, np.log(close), np.nan)

    # Pre-compute slope denominator: OLS X matrix for _WINDOW points.
    x = np.arange(_WINDOW, dtype=np.float64)
    x_mean = x.mean()
    x_dm = x - x_mean
    ss_x = (x_dm ** 2).sum()           # sum of squared deviations

    for t in range(_TOTAL, n):
        # look-back window is [t - _TOTAL .. t - _SKIP - 1]  (inclusive)
        start = t - _TOTAL   # oldest bar in window
        end   = t - _SKIP    # bar just before the skip zone; NOT included in ret
        # cumulative log-return: log(P[end-1]) - log(P[start])
        #   end-1 = t - _SKIP - 1  (last bar of 6-month window)
        #   start = t - _TOTAL     (first bar of 6-month window)
        p_start = log_close[start]
        p_end   = log_close[end - 1]   # end is exclusive → end-1

        if np.isnan(p_start) or np.isnan(p_end):
            continue

        ret_out[t] = p_end - p_start

        # ── slope variant ─────────────────────────────────────────────────
        # OLS slope of log-price over the _WINDOW-bar window
        window_lp = log_close[start:end]   # length = _WINDOW
        if np.any(np.isnan(window_lp)):
            pass   # leave slope as nan
        else:
            y_mean = window_lp.mean()
            y_dm   = window_lp - y_mean
            cov_xy = (x_dm * y_dm).sum()
            raw_slope = cov_xy / ss_x   # log-price units per day
            # normalise by average log-price magnitude to get a scale-free signal
            ref_price = close[t - _SKIP - 1]
            if ref_price > 0:
                slope_out[t] = raw_slope / (ref_price / _WINDOW)
            # else leave nan

        # ── consistency variant ───────────────────────────────────────────
        # fraction of the 6 non-overlapping 21-day sub-windows with +ve return
        pos_count = 0
        valid_count = 0
        for k in range(6):
            w_start = start + k * 21
            w_end   = w_start + 21
            lp_s = log_close[w_start]
            lp_e = log_close[w_end - 1]
            if not (np.isnan(lp_s) or np.isnan(lp_e)):
                valid_count += 1
                if lp_e > lp_s:
                    pos_count += 1
        if valid_count > 0:
            consist_out[t] = pos_count / valid_count

    df["osap_mom6m_ret"]     = ret_out
    df["osap_mom6m_slope"]   = slope_out
    df["osap_mom6m_consist"] = consist_out
    return df
