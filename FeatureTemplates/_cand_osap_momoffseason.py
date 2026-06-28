"""
Off-season momentum / long-term reversal (Heston & Sadka 2008, JFE).

MomOffSeason: average monthly return in months that are NOT the same
calendar month as the current observation, measured over the preceding
2-5 years.  Sign is NEGATIVE (reversal) — high off-season past returns
predict lower future returns (t=5.6).

A companion "year 1" variant (Mom12mOffSeason) uses only the prior 12
months and has a POSITIVE sign (momentum component, t=4.2).

Per-ticker proxy: we can compute these faithfully because the signal is
purely time-series — no cross-sectional rank is needed.  We compute
daily returns, label each day by its calendar month, and average returns
in "off-season" months (≠ current month) over the specified look-back
windows.  Portfolio formation is monthly but the daily granularity here
gives a smoothly-updating version, which is valid for a daily predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momoffseason",
    "description": (
        "Off-season momentum / long-term reversal (Heston & Sadka 2008, JFE). "
        "osap_momoffseason_2_5: average monthly return in months ≠ current calendar "
        "month over the preceding 2-5 years (negative predictor, reversal, t=5.6). "
        "osap_momoffseason_1: same metric over the preceding 1 year (positive predictor, "
        "momentum component, Mom12mOffSeason, t=4.2). "
        "osap_momoffseason_seas_ratio: ratio of same-season to off-season avg returns "
        "(seasonality concentration proxy). "
        "Per-ticker OHLCV proxy; no cross-sectional component needed as the original "
        "signal is purely a time-series average."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momoffseason_2_5",
        "osap_momoffseason_1",
        "osap_momoffseason_seas_ratio",
    ],
    "tags": ["momentum", "seasonality", "reversal", "heston_sadka", "osap"],
    "version": "1.0.0",
    "author": "Heston & Sadka (2008) 'Seasonality in the Cross-Section of Stock Returns', JFE. Implemented as per-ticker proxy by Claude.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute off-season momentum features per Heston & Sadka (2008).

    For each row t:
      - current_month = month of Date[t]
      - daily_ret[s] = daily log return on day s
      - off_season_days: days in window where month(s) != current_month
      - avg off-season monthly return = mean of daily_ret on off-season days
        (approximates the monthly-frequency average used in the paper)

    Windows:
      Year 1:   [252, 21) trading days ago  (approx months t-12 to t-1)
      Years 2-5: [1260, 252) trading days ago (approx months t-60 to t-13)

    Lookahead safety: all windows reference PAST rows only (positive lags).
    """
    # ------------------------------------------------------------------ #
    # 1. Daily log returns (NaN for first row)
    # ------------------------------------------------------------------ #
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # log returns; guard zero/negative close
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # align: ret[i] is the return on day i (uses close[i-1] and close[i])
    daily_ret = np.empty(n, dtype=np.float64)
    daily_ret[0] = np.nan
    daily_ret[1:] = log_ret

    # ------------------------------------------------------------------ #
    # 2. Calendar month of each row
    # ------------------------------------------------------------------ #
    dates = pd.to_datetime(df["Date"])
    months = dates.dt.month.values.astype(np.int8)  # 1..12

    # ------------------------------------------------------------------ #
    # 3. Trading-day look-back windows (approximate business days)
    #    Year 1  : lags [21, 252)  -- skip most-recent month (~21 days)
    #    Years 2-5: lags [252, 1260)
    # ------------------------------------------------------------------ #
    SKIP_LAG = 21          # ~1 month lag skip (avoid reversal bias)
    YEAR_DAYS = 252        # trading days per year
    LAG_1Y_LO = SKIP_LAG  # year-1 window start (inclusive lag)
    LAG_1Y_HI = YEAR_DAYS  # year-1 window end   (exclusive lag)
    LAG_25_LO = YEAR_DAYS  # years 2-5 start
    LAG_25_HI = 5 * YEAR_DAYS  # years 2-5 end

    # Minimum observations required to emit a non-NaN value
    MIN_OBS_1Y = 30    # at least 30 off-season trading days in year-1 window
    MIN_OBS_25 = 120   # at least 120 off-season trading days in years 2-5

    # ------------------------------------------------------------------ #
    # 4. Iterate over rows and compute running off-season averages
    #    We use numpy vectorised slicing per row — O(n * window_size / n)
    #    amortised via the fixed window bounds.  For n~700 and windows up
    #    to 1260, this is fast enough (<100 ms).
    # ------------------------------------------------------------------ #
    feat_1y  = np.full(n, np.nan, dtype=np.float64)
    feat_25  = np.full(n, np.nan, dtype=np.float64)
    feat_rat = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        cur_month = months[i]

        # ---- Year-1 window ---- #
        lo1 = max(0, i - LAG_1Y_HI)
        hi1 = max(0, i - LAG_1Y_LO)
        if hi1 > lo1:
            r1 = daily_ret[lo1:hi1]
            m1 = months[lo1:hi1]
            off_mask1 = (m1 != cur_month) & np.isfinite(r1)
            off_count1 = off_mask1.sum()
            if off_count1 >= MIN_OBS_1Y:
                feat_1y[i] = r1[off_mask1].mean()

        # ---- Years 2-5 window ---- #
        lo25 = max(0, i - LAG_25_HI)
        hi25 = max(0, i - LAG_25_LO)
        if hi25 > lo25:
            r25 = daily_ret[lo25:hi25]
            m25 = months[lo25:hi25]
            off_mask25 = (m25 != cur_month) & np.isfinite(r25)
            off_count25 = off_mask25.sum()
            if off_count25 >= MIN_OBS_25:
                feat_25[i] = r25[off_mask25].mean()

        # ---- Seasonality ratio: same-season / off-season (years 2-5) ---- #
        if hi25 > lo25:
            r25_slice = daily_ret[lo25:hi25]
            m25_slice = months[lo25:hi25]
            seas_mask  = (m25_slice == cur_month) & np.isfinite(r25_slice)
            off_mask_s = (m25_slice != cur_month) & np.isfinite(r25_slice)
            if seas_mask.sum() >= 5 and off_mask_s.sum() >= MIN_OBS_25:
                seas_avg = r25_slice[seas_mask].mean()
                off_avg  = r25_slice[off_mask_s].mean()
                if np.isfinite(off_avg) and off_avg != 0.0:
                    feat_rat[i] = seas_avg / off_avg

    # Guard against any stray inf/-inf (shouldn't occur but defensive)
    feat_1y  = np.where(np.isfinite(feat_1y),  feat_1y,  np.nan)
    feat_25  = np.where(np.isfinite(feat_25),  feat_25,  np.nan)
    feat_rat = np.where(np.isfinite(feat_rat), feat_rat, np.nan)

    df["osap_momoffseason_1"]         = feat_1y
    df["osap_momoffseason_2_5"]       = feat_25
    df["osap_momoffseason_seas_ratio"] = feat_rat

    return df
