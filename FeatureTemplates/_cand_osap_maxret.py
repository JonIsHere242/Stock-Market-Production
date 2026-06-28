"""
osap_maxret — Maximum daily return (lottery demand / MAX anomaly)
Bali, Cakici & Whitelaw (2011), Journal of Financial Economics.

Predicted sign: -1  (high MAX → overpriced lottery stock → low future returns)
SOURCE: OpenSourceAP (Chen-Zimmermann), category: Price / Lottery demand
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_maxret",
    "description": (
        "Maximum single-day return (MAX) over the past 21 trading days "
        "(Bali, Cakici & Whitelaw 2011). "
        "High-MAX stocks attract lottery-demand buyers and subsequently underperform "
        "(predicted sign: -1, cross-sectional). "
        "Per-ticker proxy: rolling max of daily Close return over 21-day window. "
        "Also produces avg of top-5 daily returns (osap_maxret_avg5) for a more robust "
        "lottery-demand measure, and a 63-day vs 21-day spread (osap_maxret_chg) that "
        "captures regime change in lottery demand for this stock."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_maxret",       # max daily return, prior 21 trading days
        "osap_maxret_avg5",  # mean of top-5 daily returns, prior 21 trading days
        "osap_maxret_chg",   # 21-day MAX minus 63-day MAX (recent vs longer-run lottery demand)
    ],
    "tags": ["price", "reversal", "lottery", "max_return", "cross_sectional_proxy"],
    "version": "1.0",
    "author": "Bali, Cakici & Whitelaw (2011) JFE; OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAX-return lottery-demand features for a single ticker.

    All windows use only past data (shift(1) so current bar is excluded from
    the look-back, making the feature known at close of day t for prediction
    of day t+1).
    """
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Daily log return: r_t = log(C_t / C_{t-1})
    # Using log returns is more robust to extreme moves; consistent with paper.
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    prev = close[:-1]
    curr = close[1:]
    # Guard: if prev == 0, set to nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(prev > 0, np.log(curr / prev), np.nan)

    # ---- window parameters ----
    WIN_SHORT = 21   # ~1 month
    WIN_LONG  = 63   # ~3 months

    # We shift by 1 so that on day t the window covers [t-WIN, t-1]:
    # i.e. exclude the current bar to avoid any forward leakage.
    # Implementation: at position i, window covers log_ret[i-WIN .. i-1]
    # which is index range [i-WIN, i) — achieved via rolling on the shifted series.

    ret_series = pd.Series(log_ret)

    # Shift forward by 1 so rolling(...).max() at position i looks at [i-WIN, i-1]
    ret_shifted = ret_series.shift(1)

    # max daily return over last 21 days (positions i-21 to i-1 inclusive)
    maxret_21 = ret_shifted.rolling(window=WIN_SHORT, min_periods=max(5, WIN_SHORT // 2)).max()

    # max daily return over last 63 days
    maxret_63 = ret_shifted.rolling(window=WIN_LONG, min_periods=max(10, WIN_LONG // 3)).max()

    # avg of top-5 daily returns over last 21 days
    def top5_mean(x: np.ndarray) -> float:
        """Return mean of the top-5 values; nan if fewer than 5 valid values."""
        valid = x[~np.isnan(x)]
        if len(valid) < 5:
            return np.nan
        return float(np.mean(np.partition(valid, -5)[-5:]))

    avg5_21 = ret_shifted.rolling(window=WIN_SHORT, min_periods=5).apply(
        top5_mean, raw=True
    )

    # Change in lottery demand: recent MAX vs longer-run MAX
    # Positive value → lottery demand has intensified recently (even stronger sell signal)
    # Guard against inf/-inf (shouldn't occur but be safe)
    maxret_chg = maxret_21 - maxret_63
    maxret_chg = maxret_chg.replace([np.inf, -np.inf], np.nan)
    maxret_21  = maxret_21.replace([np.inf, -np.inf], np.nan)
    avg5_21    = avg5_21.replace([np.inf, -np.inf], np.nan)

    df["osap_maxret"]      = maxret_21.values
    df["osap_maxret_avg5"] = avg5_21.values
    df["osap_maxret_chg"]  = maxret_chg.values

    return df
