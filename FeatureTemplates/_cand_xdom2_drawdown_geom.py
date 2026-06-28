"""
xdom2_drawdown_geom — Drawdown geometry: depth, duration, recovery
Spec ID: xdom2_drawdown_geom
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_drawdown_geom",
    "description": (
        "Rolling 120-bar drawdown geometry per ticker: "
        "(1) xdom2_drawdown_geom_depth — current drawdown from the rolling 120-bar "
        "running high (0 = at high, -1 = fully collapsed); "
        "(2) xdom2_drawdown_geom_duration — number of bars since the last 120-bar "
        "running high was set (0 = at high today); "
        "(3) xdom2_drawdown_geom_recovery_slope — OLS slope of Close from the trough "
        "to today within the 120-bar window, normalised by the trough price. "
        "Captures where in the drawdown/recovery cycle the stock sits. "
        "Per-ticker proxy — inherently single-series, no cross-sectional component needed."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_drawdown_geom_depth",
        "xdom2_drawdown_geom_duration",
        "xdom2_drawdown_geom_recovery_slope",
    ],
    "tags": ["drawdown", "recovery", "price_structure", "cross_domain"],
    "version": "1.0",
    "author": (
        "Cross-domain / practitioner method transfer (batch 2): "
        "'Drawdown geometry: depth, duration, recovery'"
    ),
}

_WINDOW = 120


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    depth = np.full(n, np.nan)
    duration = np.full(n, np.nan)
    recovery_slope = np.full(n, np.nan)

    # Need at least 2 bars to do anything meaningful; first bar with full window
    # starts at index _WINDOW-1 but we emit partial windows too.
    for i in range(1, n):
        start = max(0, i - _WINDOW + 1)
        window = close[start : i + 1]  # inclusive current bar

        # --- running high within window ---
        running_high = np.maximum.accumulate(window)
        curr_high = running_high[-1]

        # depth: (Close - RunningHigh) / RunningHigh  (always <= 0)
        if curr_high > 0:
            depth[i] = (window[-1] - curr_high) / curr_high
        # else: leave as NaN (price <= 0, invalid)

        # duration: bars since the last bar where price == running high
        # i.e. where price touched the rolling max
        at_high = (window == running_high)
        # last index within window where at_high is True
        last_high_rel = len(window) - 1 - np.argmax(at_high[::-1])
        duration[i] = (len(window) - 1) - last_high_rel  # 0 = at high right now

        # recovery slope: OLS slope from trough to now, normalised by trough price
        trough_idx_rel = int(np.argmin(window))
        trough_price = window[trough_idx_rel]
        # Need at least 2 points from trough to current bar
        segment_len = len(window) - trough_idx_rel  # >= 1
        if segment_len >= 2 and trough_price > 0:
            x = np.arange(segment_len, dtype=float)
            y = window[trough_idx_rel:]
            # OLS slope via formula: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
            s_len = float(segment_len)
            sx = x.sum()
            sy = y.sum()
            sxy = (x * y).sum()
            sx2 = (x * x).sum()
            denom = s_len * sx2 - sx * sx
            if denom != 0.0:
                slope = (s_len * sxy - sx * sy) / denom
                recovery_slope[i] = slope / trough_price  # normalised
        # else: leave NaN (trough is current bar, or invalid price)

    df["xdom2_drawdown_geom_depth"] = depth
    df["xdom2_drawdown_geom_duration"] = duration
    df["xdom2_drawdown_geom_recovery_slope"] = recovery_slope

    return df
