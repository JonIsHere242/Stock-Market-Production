"""
Candidate feature block: osap_zerotrade6m
Liu (2006) Zero-Trade Days liquidity measure — 6-month rolling average.

Per-ticker proxy notes:
  - "Zero trades" is proxied by Volume == 0 (daily bars with no reported volume).
  - Turnover = Volume / shares_outstanding (PIT via fundamentals helper).
    When shares_outstanding is unavailable (ETFs, foreign), turnover-based term
    falls back to NaN, leaving only the zero-day-count term (scaled).
  - The formula per Liu (2006): for each month m,
        zt_m = (zero_days_m + (sum_turnover_m / 48e5)) * 21 / trading_days_m
    zerotrade6m = average of zt_m over the trailing 6 calendar months.
  - This is inherently cross-sectional in the original paper (ranks stocks);
    here we compute the raw scalar per ticker, which captures the same
    economic signal (less-liquid stocks have higher values).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional: PIT fundamentals (shares_outstanding for turnover denominator)
# ---------------------------------------------------------------------------
try:
    _s2 = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _HAS_FUND = False

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_zerotrade6m",
    "description": (
        "Liu (2006) zero-trading-days liquidity measure, 6-month rolling average. "
        "For each calendar month: zt = (zero_volume_days + monthly_turnover_sum/48e5) "
        "* 21 / trading_days. zerotrade6m is the trailing 6-month mean of zt. "
        "Higher values = less liquid. Per-ticker proxy: zero trades = Volume==0; "
        "turnover = Volume/shares_outstanding (PIT fundamentals, degrades to NaN if "
        "unavailable). Slope variant captures trend in illiquidity. "
        "Cross-sectional ranking is NOT applied here — raw scalar per ticker."
    ),
    "requires": ["Volume", "Close"],
    "produces": [
        "osap_zerotrade6m_val",    # 6-month average zerotrade (level)
        "osap_zerotrade6m_slope",  # linear slope of monthly zt over trailing 6m
        "osap_zerotrade6m_zvol",   # zero-volume-day fraction (trailing 6m), standalone
    ],
    "tags": ["liquidity", "turnover", "zero_trade", "liu2006", "osap"],
    "version": "1.0",
    "author": "Liu 2006 (OpenSourceAP / Chen-Zimmermann); implemented as per-ticker OHLCV proxy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_zt(grp: pd.DataFrame, shrout_series: pd.Series | None) -> float:
    """Compute the Liu (2006) zt statistic for a single month group."""
    n_trading = len(grp)
    if n_trading == 0:
        return np.nan

    zero_days = int((grp["Volume"] == 0).sum())

    # Turnover component: sum(vol/shrout) over the month
    if shrout_series is not None:
        # Align shrout to this group's index
        shrout_vals = shrout_series.reindex(grp.index)
        valid_shrout = shrout_vals.replace(0, np.nan)
        turnover_sum = (grp["Volume"] / valid_shrout).sum()
        if np.isnan(turnover_sum):
            turnover_term = np.nan
        else:
            turnover_term = turnover_sum / 48e5
    else:
        turnover_term = np.nan

    # zt_m = (zero_days + turnover_term) * 21 / n_trading
    # If turnover_term is nan, use zero_days only (partial signal)
    if np.isnan(turnover_term):
        zt = zero_days * (21.0 / n_trading)
    else:
        zt = (zero_days + turnover_term) * (21.0 / n_trading)

    return zt


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Need at least a minimal history; initialise output columns to NaN
    df["osap_zerotrade6m_val"] = np.nan
    df["osap_zerotrade6m_slope"] = np.nan
    df["osap_zerotrade6m_zvol"] = np.nan

    if len(df) < 5:
        return df

    # -----------------------------------------------------------------------
    # Optionally attach PIT shares_outstanding
    # -----------------------------------------------------------------------
    shrout_col: pd.Series | None = None
    if _HAS_FUND:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_fund = _fundamentals.as_of(df.copy(), fields=["shares_outstanding"])
            if "fund_shares_outstanding" in df_fund.columns:
                raw = df_fund["fund_shares_outstanding"].replace(0, np.nan)
                # Forward-fill within the block (PIT safe — already backward-merged)
                shrout_col = raw.ffill()
        except Exception:
            shrout_col = None

    # -----------------------------------------------------------------------
    # Build a year-month grouper on the existing index
    # -----------------------------------------------------------------------
    dates = pd.to_datetime(df["Date"])
    df_work = df[["Volume"]].copy()
    df_work.index = dates
    if shrout_col is not None:
        df_work["_shrout"] = shrout_col.values

    ym_key = dates.dt.to_period("M")

    # Compute zt for every calendar month present
    unique_months = ym_key.unique().sort_values()
    month_zt: dict = {}

    for m in unique_months:
        mask = ym_key == m
        grp = df_work.loc[mask.values]
        s_col = grp["_shrout"] if "_shrout" in grp.columns else None
        month_zt[m] = _monthly_zt(grp, s_col)

    zt_series = pd.Series(month_zt)  # index = Period[M]

    # -----------------------------------------------------------------------
    # For each row, compute the 6-month trailing average (months t-5 … t)
    # and slope of zt over those 6 months
    # -----------------------------------------------------------------------
    val_out = np.full(len(df), np.nan)
    slope_out = np.full(len(df), np.nan)
    zvol_out = np.full(len(df), np.nan)

    # Zero-volume fraction: trailing 126 trading days (~6m) rolling
    vol_arr = df["Volume"].to_numpy(dtype=float)
    window_6m = 126  # approx 6 calendar months of trading days

    for i in range(len(df)):
        m_i = ym_key.iloc[i]

        # 6-month trailing window of monthly zt values
        months_window = pd.period_range(end=m_i, periods=6, freq="M")
        zt_vals = zt_series.reindex(months_window).to_numpy(dtype=float)
        valid_mask = ~np.isnan(zt_vals)
        n_valid = valid_mask.sum()

        if n_valid >= 3:
            val_out[i] = np.nanmean(zt_vals)

            # Linear slope of zt over available months (x = 0..5)
            x = np.arange(6)[valid_mask].astype(float)
            y = zt_vals[valid_mask]
            if len(x) >= 2:
                x_m = x - x.mean()
                denom = (x_m ** 2).sum()
                if denom > 0:
                    slope_out[i] = (x_m * y).sum() / denom

        # Zero-volume fraction: rolling 126-bar window ending at row i
        start = max(0, i - window_6m + 1)
        window_vol = vol_arr[start: i + 1]
        if len(window_vol) >= 5:
            zvol_out[i] = float((window_vol == 0).sum()) / len(window_vol)

    df["osap_zerotrade6m_val"] = val_out
    df["osap_zerotrade6m_slope"] = slope_out
    df["osap_zerotrade6m_zvol"] = zvol_out

    return df
