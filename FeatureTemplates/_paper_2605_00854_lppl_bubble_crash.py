"""
_paper_2605_00854_lppl_bubble_crash.py  --  Cheap per-ticker LPPL bubble/crash proxies.

Based on: "Dynamics of Periodic Bubbles and Crashes: Modeling Market Overheating
and Panic Selling" (arXiv 2605.00854).

The full LPPL (Log-Periodic Power Law) model requires a nonlinear 7-parameter fit
per row — far too heavy for a 700-row rolling block. This file implements CHEAP
CLOSED-FORM PROXIES of the LPPL bubble/crash signature:

  1. Super-exponential (quadratic log-price) acceleration — the defining LPPL fingerprint.
  2. Oscillation energy around the quadratic trend — rough log-periodic stand-in.
  3. Distance from trailing peak / trough — drawdown / panic strength.
  4. Combined overheating score (acceleration + stretch above SMA).
  5. Mirror panic score (deceleration + stretch below SMA).

No IO, no randomness, no nonlinear optimiser, no external data.
Column prefix: lppl_
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "lppl_bubble_crash",
    "description": (
        "Cheap per-ticker LPPL bubble/crash proxies: quadratic log-price "
        "acceleration, oscillation energy, drawdown, overheating and panic scores "
        "(no full nonlinear LPPL fit — closed-form rolling regression only)."
    ),
    "requires": ["Close"],
    "produces": [
        "lppl_log_accel_60",       # quadratic coefficient of log-price over 60-day window
        "lppl_log_accel_120",      # same, 120-day window
        "lppl_osc_energy_60",      # variance of log-price residuals around quadratic fit (60d)
        "lppl_drawdown_peak_60",   # distance below 60-day rolling peak (0..1)
        "lppl_drawdown_trough_60", # distance above 60-day rolling trough (bounce proxy)
        "lppl_overheat_60",        # combined: positive accel + log-price above 60d mean
        "lppl_panic_60",           # combined: negative accel + log-price below 60d mean
        "lppl_log_accel_z",        # z-score of lppl_log_accel_60 over its own 252-day history
    ],
    "tags": ["momentum", "volatility", "experimental", "paper"],
    "version": "1.0",
    "author": "arXiv 2605.00854 — LPPL bubble/crash proxies (cheap, closed-form)",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols3_rolling(y: np.ndarray, window: int) -> tuple:
    """
    Rolling OLS of y on [1, t, t^2] over a trailing window.

    Returns three arrays (same length as y), each NaN for the first
    (window - 1) rows:
        b0  — intercept
        b1  — linear coefficient
        b2  — quadratic coefficient  (positive = super-exponential / accelerating)

    Uses closed-form normal equations on the fixed Vandermonde design matrix
    for integer t = 0, 1, ..., window-1. The design is the same for every
    window position, so we pre-invert (X'X)^{-1} X' once and apply it as a
    dot product with the sliding window — O(n) total cost.
    """
    n = len(y)
    t = np.arange(window, dtype=np.float64)
    # Design matrix columns: [1, t, t^2]
    X = np.column_stack([np.ones(window), t, t * t])
    # Pre-compute the pseudo-inverse: (3 x window) matrix
    XtX = X.T @ X
    try:
        pinv = np.linalg.solve(XtX, X.T)   # shape (3, window)
    except np.linalg.LinAlgError:
        nan3 = np.full(n, np.nan)
        return nan3, nan3, nan3

    b0 = np.full(n, np.nan)
    b1 = np.full(n, np.nan)
    b2 = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg = y[i - window + 1: i + 1]
        if not np.all(np.isfinite(seg)):
            continue
        coeffs = pinv @ seg          # shape (3,)
        b0[i] = coeffs[0]
        b1[i] = coeffs[1]
        b2[i] = coeffs[2]

    return b0, b1, b2


def _quad_residual_var(y: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                       b2: np.ndarray, window: int) -> np.ndarray:
    """
    Variance of residuals (y - fitted) inside each trailing window.
    Uses the same window the OLS was fit on; recomputes residuals cheaply.
    """
    n = len(y)
    t = np.arange(window, dtype=np.float64)
    var_arr = np.full(n, np.nan)

    for i in range(window - 1, n):
        if not np.isfinite(b0[i]):
            continue
        seg = y[i - window + 1: i + 1]
        fitted = b0[i] + b1[i] * t + b2[i] * t * t
        resid = seg - fitted
        var_arr[i] = np.var(resid)   # population variance is fine here

    return var_arr


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LPPL bubble/crash proxy columns to df (one stock, ascending by Date).

    Windows used: 60-day (primary) and 120-day (secondary for confirmation).
    Leading NaNs are expected for the first (window-1) rows.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # Replace zero/negative closes with NaN so log is safe
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.where(close > 0, np.log(close), np.nan)

    # -----------------------------------------------------------------------
    # 1. QUADRATIC LOG-PRICE FIT  (super-exponential acceleration)
    # -----------------------------------------------------------------------
    WIN60 = 60
    WIN120 = 120

    _, _, b2_60 = _ols3_rolling(log_c, WIN60)
    _, _, b2_120 = _ols3_rolling(log_c, WIN120)

    # b2 is in units of (log-price per squared bar-index). A positive value
    # means log-price is curving upward — the LPPL bubble fingerprint.
    # Scale by window^2 so the two windows produce comparable magnitudes.
    # (b2 * W^2 ~ total quadratic deviation in log-price units)
    lppl_log_accel_60 = b2_60 * (WIN60 ** 2)
    lppl_log_accel_120 = b2_120 * (WIN120 ** 2)

    # -----------------------------------------------------------------------
    # 2. OSCILLATION ENERGY around the quadratic fit (60-day)
    #    High residual variance on a super-exponential trend ≈ log-periodic
    #    oscillations as described in LPPL.
    # -----------------------------------------------------------------------
    b0_60, b1_60, _ = _ols3_rolling(log_c, WIN60)
    # We re-compute b2_60 inside the helper through the same call — reuse
    # what we already have by passing it directly.
    # Build b0/b1 via a second dedicated call (b2 already done above).
    osc_energy_60 = _quad_residual_var(log_c, b0_60, b1_60, b2_60, WIN60)

    # -----------------------------------------------------------------------
    # 3. DRAWDOWN / CRASH PROXIMITY  (60-day rolling peak / trough)
    # -----------------------------------------------------------------------
    close_s = pd.Series(close, index=df.index)
    roll_max = close_s.rolling(WIN60, min_periods=WIN60).max().to_numpy()
    roll_min = close_s.rolling(WIN60, min_periods=WIN60).min().to_numpy()

    # Drawdown from peak: 0 = at peak, 1 = wiped out (practically impossible)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_peak = np.where(
            (roll_max > 0) & np.isfinite(roll_max),
            (roll_max - close) / roll_max,
            np.nan,
        )
        # Distance above trough: 0 = at trough, positive = bounced above it
        dd_trough = np.where(
            (roll_min > 0) & np.isfinite(roll_min) & (roll_max > roll_min),
            (close - roll_min) / (roll_max - roll_min),
            np.nan,
        )

    # -----------------------------------------------------------------------
    # 4. COMBINED OVERHEATING SCORE
    #    = clip(b2, 0, ∞)  +  clip(log_price_above_SMA60, 0, ∞)
    #    Both terms are zero or positive; both rising together = overheating.
    # -----------------------------------------------------------------------
    sma60_log = pd.Series(log_c, index=df.index).rolling(WIN60, min_periods=WIN60).mean().to_numpy()

    log_above_sma = np.where(
        np.isfinite(sma60_log) & np.isfinite(log_c),
        np.maximum(log_c - sma60_log, 0.0),
        np.nan,
    )

    accel_pos = np.where(np.isfinite(lppl_log_accel_60), np.maximum(lppl_log_accel_60, 0.0), np.nan)

    overheat = np.where(
        np.isfinite(log_above_sma) & np.isfinite(accel_pos),
        accel_pos + log_above_sma,
        np.nan,
    )

    # -----------------------------------------------------------------------
    # 5. MIRROR PANIC SCORE
    #    = clip(-b2, 0, ∞)  +  clip(log_price_below_SMA60, 0, ∞)
    # -----------------------------------------------------------------------
    log_below_sma = np.where(
        np.isfinite(sma60_log) & np.isfinite(log_c),
        np.maximum(sma60_log - log_c, 0.0),
        np.nan,
    )

    accel_neg = np.where(np.isfinite(lppl_log_accel_60), np.maximum(-lppl_log_accel_60, 0.0), np.nan)

    panic = np.where(
        np.isfinite(log_below_sma) & np.isfinite(accel_neg),
        accel_neg + log_below_sma,
        np.nan,
    )

    # -----------------------------------------------------------------------
    # 6. Z-SCORE OF 60-DAY ACCELERATION over its own trailing 252-day history
    # -----------------------------------------------------------------------
    accel_s = pd.Series(lppl_log_accel_60, index=df.index)
    roll_mean_a = accel_s.rolling(252, min_periods=60).mean()
    roll_std_a = accel_s.rolling(252, min_periods=60).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        accel_z = np.where(
            roll_std_a.notna() & (roll_std_a > 0),
            (accel_s - roll_mean_a) / roll_std_a,
            np.nan,
        )

    # -----------------------------------------------------------------------
    # Replace any inf values with NaN (contract requirement)
    # -----------------------------------------------------------------------
    def _no_inf(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.where(np.isnan(arr[~np.isfinite(arr)]), np.nan, np.nan)
        # Simpler: set all non-finite (inf / -inf) to NaN
        arr = np.where(np.isinf(arr), np.nan, arr)
        return arr

    df["lppl_log_accel_60"]       = _no_inf(lppl_log_accel_60)
    df["lppl_log_accel_120"]      = _no_inf(lppl_log_accel_120)
    df["lppl_osc_energy_60"]      = _no_inf(osc_energy_60)
    df["lppl_drawdown_peak_60"]   = _no_inf(dd_peak)
    df["lppl_drawdown_trough_60"] = _no_inf(dd_trough)
    df["lppl_overheat_60"]        = _no_inf(overheat)
    df["lppl_panic_60"]           = _no_inf(panic)
    df["lppl_log_accel_z"]        = _no_inf(np.asarray(accel_z, dtype=np.float64))

    return df
