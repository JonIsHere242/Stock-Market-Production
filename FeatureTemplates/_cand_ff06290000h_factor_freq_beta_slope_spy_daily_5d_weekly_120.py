"""
Frequency-dependent beta slope vs SPY (Epps-effect signature).

Over a rolling 120-day window compute OLS betas of ticker returns on SPY
returns at three return horizons h=1 (daily), h=5 (5-day overlapping), and
h=10 (10-day overlapping).  beta_h = cov(r_h, m_h) / var(m_h).  The feature
is the OLS slope of [beta_1, beta_5, beta_10] on log(h=1,5,10).

A positive slope means beta grows as the horizon lengthens -- consistent with
diffusion / gradual information incorporation (Epps effect in reverse).
A negative slope (beta shrinks with horizon) flags mean-reversion / noise.

Per-ticker proxy; uses SPY index via _indexes helper.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120",
    "description": (
        "Frequency-dependent beta slope vs SPY over 120-day rolling window. "
        "Computes OLS betas at return horizons h=1,5,10; regresses them on "
        "log(h) to capture the Epps-effect signature (how sensitivity to SPY "
        "changes across observation frequencies). Positive slope = beta grows "
        "with horizon (diffusion); negative = mean-reversion / noise-dominant. "
        "Per-ticker OHLCV proxy; causal/no-lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_beta1",
        "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_beta10",
        "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_slope",
    ],
    "tags": ["beta", "frequency", "epps", "spy", "factor", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06290000h",
}

# log(h) values for the three horizons -- fixed, used in OLS
_LOG_H = np.log(np.array([1.0, 5.0, 10.0]))
# OLS design matrix [1, log(h)]  shape (3,2)
_X_OLS = np.column_stack([np.ones(3), _LOG_H])
# precompute (X'X)^{-1} X'  shape (2,3)
_XtX_inv_Xt = np.linalg.lstsq(_X_OLS, np.eye(3), rcond=None)[0]  # shape (2,3)


def _ols_beta_on_market(r_ticker: np.ndarray, r_market: np.ndarray) -> float:
    """OLS beta: cov(r_t, r_m) / var(r_m); returns nan if var too small."""
    if len(r_ticker) < 2:
        return np.nan
    var_m = np.var(r_market, ddof=1)
    if var_m < 1e-12:
        return np.nan
    cov_tm = np.cov(r_ticker, r_market, ddof=1)[0, 1]
    return cov_tm / var_m


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_beta1 = "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_beta1"
    col_beta10 = "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_beta10"
    col_slope = "ff06290000h_factor_freq_beta_slope_spy_daily_5d_weekly_120_slope"

    # Initialise all produced columns to NaN up front (required on all paths)
    df[col_beta1] = np.nan
    df[col_beta10] = np.nan
    df[col_slope] = np.nan

    if len(df) < 20:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY daily closes and align with ticker dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    work = work.set_index(work.index).reindex(df.index, method=None)
    # Safer: just use the merged close values aligned by position after sort
    # Re-merge preserving original df index order
    work2 = df[["Date", "Close"]].copy().reset_index(drop=False)
    work2["Date"] = pd.to_datetime(work2["Date"])
    work2 = pd.merge_asof(
        work2.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # work2 is sorted by Date; restore df row order via original index
    work2 = work2.set_index("index").reindex(df.index)

    ticker_close = work2["Close"].values.astype(float)
    spy_close_vals = work2["spy_close"].values.astype(float)
    n = len(ticker_close)

    # ------------------------------------------------------------------
    # Compute log returns for each horizon using np.log differences
    # Overlapping returns: r_h[i] = log(close[i]/close[i-h])
    # ------------------------------------------------------------------
    WINDOW = 120
    MIN_OBS = 30

    slopes = np.full(n, np.nan)
    b1_arr = np.full(n, np.nan)
    b10_arr = np.full(n, np.nan)

    # Precompute log-price series
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ticker = np.where(ticker_close > 0, np.log(ticker_close), np.nan)
        log_spy = np.where(spy_close_vals > 0, np.log(spy_close_vals), np.nan)

    horizons = [1, 5, 10]

    for i in range(WINDOW - 1, n):
        # Window indices: [i-WINDOW+1 .. i]  (inclusive, length=WINDOW)
        start = i - WINDOW + 1
        end = i + 1  # exclusive

        betas = np.full(3, np.nan)
        for hi, h in enumerate(horizons):
            # Overlapping h-day log returns within the window
            # r[j] = log_price[j] - log_price[j-h]  for j in [start+h .. end-1]
            seg_end_idx = np.arange(start + h, end)
            seg_start_idx = seg_end_idx - h

            if len(seg_end_idx) < MIN_OBS:
                continue

            r_t = log_ticker[seg_end_idx] - log_ticker[seg_start_idx]
            r_m = log_spy[seg_end_idx] - log_spy[seg_start_idx]

            # Drop rows where either is nan
            mask = np.isfinite(r_t) & np.isfinite(r_m)
            if mask.sum() < MIN_OBS:
                continue

            betas[hi] = _ols_beta_on_market(r_t[mask], r_m[mask])

        # OLS: slope of betas ~ a + b*log(h)
        valid = np.isfinite(betas)
        if valid.sum() < 2:
            continue

        x_sub = _LOG_H[valid]
        y_sub = betas[valid]
        # Simple OLS slope with at least 2 points
        xm = x_sub.mean()
        ym = y_sub.mean()
        denom = np.sum((x_sub - xm) ** 2)
        if denom < 1e-12:
            continue
        slope = np.sum((x_sub - xm) * (y_sub - ym)) / denom

        slopes[i] = slope
        if np.isfinite(betas[0]):
            b1_arr[i] = betas[0]
        if np.isfinite(betas[2]):
            b10_arr[i] = betas[2]

    df[col_beta1] = b1_arr
    df[col_beta10] = b10_arr
    df[col_slope] = slopes

    return df
