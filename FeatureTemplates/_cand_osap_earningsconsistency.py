"""
Earnings Consistency (Alwathainani 2009) — per-ticker PIT proxy
via SEC fundamentals (eps_basic_ttm).

Source: OpenSourceAP (Chen-Zimmermann), Alwathainani 2009.
Cross-sectional signal adapted to per-ticker time-series.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_earningsconsistency",
    "description": (
        "Per-ticker earnings consistency proxy following Alwathainani (2009). "
        "Uses PIT EPS (eps_basic_ttm) from SEC fundamentals. "
        "For each bar we compute rolling 12-month EPS growth = (EPS_t - EPS_t-12m) / "
        "avg(|EPS_t-12m|, |EPS_t-24m|), then cap/filter per the original spec "
        "(exclude |growth|>6, require sign-consistency with prior-year growth). "
        "osap_earningsconsistency_avg: average of valid monthly EPS-growth readings "
        "over the trailing 48 months (the original signal). "
        "osap_earningsconsistency_sign_streak: count of consecutive sign-consistent "
        "growth observations (captures streak momentum). "
        "osap_earningsconsistency_valid_frac: fraction of 48-month windows with valid "
        "(non-excluded) growth readings (data quality / coverage flag). "
        "NOTE: this is a per-ticker time-series proxy; the original is cross-sectional. "
        "Price<5 filter applied at row level: output is NaN when Close<5."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_earningsconsistency_avg",
        "osap_earningsconsistency_sign_streak",
        "osap_earningsconsistency_valid_frac",
    ],
    "tags": ["fundamentals", "earnings", "quality", "growth", "accounting"],
    "version": "1.0",
    "author": "Alwathainani 2009 / OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Pull PIT EPS fundamentals
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["eps_basic_ttm"])

    eps_col = "fund_eps_basic_ttm"

    # ------------------------------------------------------------------
    # 2. Build a daily EPS series aligned to the df index
    # ------------------------------------------------------------------
    eps = df[eps_col].copy()  # NaN where coverage missing

    # We need EPS approximately 12 months ago and 24 months ago.
    # The df is one stock, ascending by Date. Use calendar-based lag:
    # find the row whose Date is closest to (current_date - 365d) etc.
    # For vectorised approach: use integer row shifts as an approximation
    # then refine with merge_asof on date offsets.

    # Better approach: build a date-indexed series, then use reindex
    # with nearest-backward to get 12m and 24m lags.

    dates = pd.to_datetime(df["Date"])
    eps_series = pd.Series(eps.values, index=dates, name="eps")

    # For each date t, look up the eps value whose date is closest but
    # not after (t - 12 months) and (t - 24 months).
    # We'll vectorise by creating offset date arrays and using searchsorted.

    sorted_dates = eps_series.index  # already ascending (per-ticker)
    n = len(sorted_dates)

    date_arr = sorted_dates.values  # numpy datetime64[ns]
    eps_arr = eps_series.values      # float64

    # offsets in ns
    _12M_NS = np.timedelta64(365, "D")
    _24M_NS = np.timedelta64(730, "D")
    _30D_NS = np.timedelta64(30, "D")   # tolerance for finding the lag bar

    target_12 = date_arr - _12M_NS
    target_24 = date_arr - _24M_NS

    def _lag_eps(target_dates: np.ndarray) -> np.ndarray:
        """For each date[i], return eps at the latest date <= target_dates[i]."""
        idx = np.searchsorted(date_arr, target_dates, side="right") - 1
        # idx == -1 means no prior data
        out = np.where(idx >= 0, eps_arr[np.clip(idx, 0, n - 1)], np.nan)
        out = np.where(idx >= 0, out, np.nan)
        return out.astype(float)

    eps_12m = _lag_eps(target_12)   # EPS ~12 months ago
    eps_24m = _lag_eps(target_24)   # EPS ~24 months ago

    # ------------------------------------------------------------------
    # 3. Compute 12-month EPS growth at each bar
    # ------------------------------------------------------------------
    # growth_t = (eps_t - eps_12m) / avg(|eps_12m|, |eps_24m|)
    denom = (np.abs(eps_arr) + np.abs(eps_12m)) / 2.0
    denom_safe = np.where(denom == 0.0, np.nan, denom)

    growth = (eps_arr - eps_12m) / denom_safe  # NaN where denom==0 or missing

    # Prior-year growth: use eps_12m vs eps_24m
    denom2 = (np.abs(eps_12m) + np.abs(eps_24m)) / 2.0
    denom2_safe = np.where(denom2 == 0.0, np.nan, denom2)
    growth_prior = (eps_12m - eps_24m) / denom2_safe

    # ------------------------------------------------------------------
    # 4. Apply exclusion filters
    # ------------------------------------------------------------------
    # Exclude if |growth| > 6 (600%)
    valid_mag = np.abs(growth) <= 6.0

    # Exclude if growth and growth_prior have different signs
    # (treat zero as neither positive nor negative → exclude)
    same_sign = (
        (growth > 0) & (growth_prior > 0)
    ) | (
        (growth < 0) & (growth_prior < 0)
    )

    # Combined valid flag (also requires non-NaN)
    valid = (
        valid_mag
        & same_sign
        & np.isfinite(growth)
        & np.isfinite(growth_prior)
    )

    growth_valid = np.where(valid, growth, np.nan)

    # ------------------------------------------------------------------
    # 5. Rolling 48-month window statistics
    # ------------------------------------------------------------------
    # We want the average over the prior 48 months.
    # Build a rolling window based on calendar offset (approx 1460 days).
    # Use a fixed row window as approximation: median ~252 rows/year → 48m ≈ 1008 rows
    # But data could be sparse (monthly eps updates); use date-based rolling.

    # Convert to pandas for date-aware rolling:
    gv_series = pd.Series(growth_valid, index=sorted_dates)
    valid_series = pd.Series(valid.astype(float), index=sorted_dates)

    # Rolling 1460-day (48-month) window
    win = "1460D"

    # Average of valid growth readings in the window
    roll_sum = gv_series.rolling(win, min_periods=1).sum()
    roll_count = valid_series.rolling(win, min_periods=1).sum()
    roll_count_safe = roll_count.replace(0, np.nan)

    earningsconsistency_avg = roll_sum / roll_count_safe

    # Valid fraction (what share of bars in 48m window had valid growth)
    total_obs = valid_series.rolling(win, min_periods=1).count()
    earningsconsistency_valid_frac = roll_count / total_obs.replace(0, np.nan)

    # Sign streak: consecutive bars where current growth_valid is the same sign
    # as the rolling-average sign (proxy for "consistent" earnings trajectory)
    avg_sign = np.sign(earningsconsistency_avg.values)
    cur_sign = np.sign(growth_valid)

    streaks = np.zeros(n, dtype=float)
    for i in range(n):
        if not np.isfinite(cur_sign[i]) or not np.isfinite(avg_sign[i]):
            streaks[i] = np.nan
        elif cur_sign[i] == avg_sign[i] and avg_sign[i] != 0:
            streaks[i] = (streaks[i - 1] + 1) if (i > 0 and np.isfinite(streaks[i - 1])) else 1.0
        else:
            streaks[i] = 0.0

    # ------------------------------------------------------------------
    # 6. Price < 5 filter — set outputs to NaN
    # ------------------------------------------------------------------
    close_arr = df["Close"].values.astype(float)
    price_ok = close_arr >= 5.0

    ea_avg = earningsconsistency_avg.values.copy()
    ea_streak = streaks.copy()
    ea_vfrac = earningsconsistency_valid_frac.values.copy()

    ea_avg = np.where(price_ok, ea_avg, np.nan)
    ea_streak = np.where(price_ok, ea_streak, np.nan)
    ea_vfrac = np.where(price_ok, ea_vfrac, np.nan)

    # ------------------------------------------------------------------
    # 7. Assign to df and clean up scratch columns
    # ------------------------------------------------------------------
    df["osap_earningsconsistency_avg"] = ea_avg
    df["osap_earningsconsistency_sign_streak"] = ea_streak
    df["osap_earningsconsistency_valid_frac"] = ea_vfrac

    # Drop scratch fundamentals column
    df.drop(columns=[eps_col], errors="ignore", inplace=True)

    # Guard: replace any inf/-inf
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
