from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_amihud_dispersion",
    "description": (
        "Per-ticker Amihud illiquidity level, dispersion, and trend (OHLCV-only). "
        "Daily illiquidity = |return| / (Close * Volume). "
        "Produces: 60-day rolling mean (level), 60-day coefficient of variation (dispersion), "
        "and 20-day change of the 60-day mean (slope/trend). "
        "Captures whether a stock is becoming more or less liquid, and how variable its "
        "illiquidity regime is -- both are per-ticker proxies; cross-sectional ranking is "
        "handled externally. High amihud_cv signals unstable liquidity regimes. "
        "Negative slope signals improving liquidity (tailwind for momentum). "
        "Source: Amihud (2002) illiquidity measure; dispersion and slope extensions from "
        "cross-domain signal-processing / econophysics / HRV literature."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "xdom_amihud_dispersion_level60",
        "xdom_amihud_dispersion_cv60",
        "xdom_amihud_dispersion_slope60_20",
    ],
    "tags": ["microstructure", "liquidity", "amihud", "cross-domain", "ohlcv"],
    "version": "1.0.0",
    "author": (
        "Amihud illiquidity dispersion & convexity (microstructure proxy, OHLCV-only). "
        "SOURCE: Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Base measure: Amihud, Y. (2002). Illiquidity and stock return: cross-section and "
        "time-series effects. Journal of Financial Markets 5(1), 31-56."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Amihud illiquidity level, CV, and slope for a single ticker.

    illiq_t = |r_t| / (Close_t * Volume_t)
    where r_t = log(Close_t / Close_{t-1})

    Columns added (never modify existing):
      xdom_amihud_dispersion_level60   : 60-day rolling mean of illiq
      xdom_amihud_dispersion_cv60      : 60-day rolling std / mean (coefficient of variation)
      xdom_amihud_dispersion_slope60_20: level60.diff(20) -- 20-day change in rolling mean
    """
    # --- daily log return (past only, shift(1) is safe) ---
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).abs()

    # --- dollar volume; guard zero ---
    dollar_vol = df["Close"] * df["Volume"]
    dollar_vol = dollar_vol.replace(0, np.nan)

    # --- raw Amihud daily illiquidity ---
    illiq = log_ret / dollar_vol

    # guard: replace inf/-inf produced by 0 dollar volume before they propagate
    illiq = illiq.replace([np.inf, -np.inf], np.nan)

    # --- 60-day rolling statistics (min_periods=30 to avoid junk at startup) ---
    roll60 = illiq.rolling(window=60, min_periods=30)

    level60 = roll60.mean()
    roll60_std = roll60.std(ddof=1)

    # coefficient of variation = std / mean; guard zero mean
    level60_safe = level60.replace(0, np.nan)
    cv60 = roll60_std / level60_safe

    # --- 20-day change in the rolling mean (slope proxy) ---
    slope60_20 = level60.diff(20)

    df["xdom_amihud_dispersion_level60"] = level60
    df["xdom_amihud_dispersion_cv60"] = cv60
    df["xdom_amihud_dispersion_slope60_20"] = slope60_20

    return df
