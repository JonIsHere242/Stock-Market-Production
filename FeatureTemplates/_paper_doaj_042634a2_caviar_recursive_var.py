"""
_paper_doaj_042634a2_caviar_recursive_var.py

Per-ticker proxy for Conditional Autoregressive Value at Risk (CAViaR),
inspired by:
  "A Comparative Analysis of Green and Brown Stocks: The Impact of
   Uncertainty Indices on Tail-Risk Forecasting"
  DOAJ: 042634a2... (uses the Realized-ES-CAViaR framework)

Original formulation: Engle & Manganelli (2004) — CAViaR models a return
quantile RECURSIVELY rather than as a rolling window empirical percentile.
The quantile at time t is a function of its own lag and of recent return shocks:

    VaR[t] = b0 + b1 * VaR[t-1] + b2 * f(return[t-1])

This gives a dynamically-updating tail estimate that responds faster to
volatility clustering than rolling empirical quantiles.

DISTINCTION from existing blocks:
  - qdd_* block (2604.12927): rolling EMPIRICAL quantiles of the distribution
    shape; these are static snapshots of the window.
  - cvar_* block (2606.13338): rolling empirical Expected Shortfall; again
    simple rolling means of the worst observations.
  - THIS block: RECURSIVE state-space quantile — VaR is propagated forward one
    step at a time, with exponential-decay memory via the b1 AR coefficient.
    The recursion makes the feature sensitive to CURRENT volatility clustering,
    not just the past-N-days average.

COEFFICIENT STRATEGY (no lookahead):
  Coefficients are calibrated on a TRAILING 252-day burn-in window using a
  simple grid search that minimises the quantile regression pinball loss over
  that window alone. The calibration is re-run every 63 trading days (quarterly)
  using only past data. Between re-calibration dates coefficients are held fixed
  (frozen AR system), so the recursion itself uses only information available at
  each t. Fallback to hardcoded sensible defaults if the burn-in window
  is too short.

  Default coefficients (Engle & Manganelli SAV-CAViaR, Table 1 estimates):
    Symmetric Absolute Value (SAV):  b0=-0.02, b1=0.90, b2=0.15
    Asymmetric Slope (AS):           b0_neg=-0.03, b0_pos=-0.01,
                                     b1=0.88, b2_neg=0.20, b2_pos=0.05

NOTE: This is an UNPROVEN candidate block (leading underscore). Features are
honest per-ticker recursive CAViaR proxies. No cross-sectional or look-ahead
information is used.
"""

import math
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_doaj_042634a2_caviar_recursive_var",
    "description": (
        "Recursive CAViaR tail-risk feature pack (Engle & Manganelli): "
        "VaR[t] = b0 + b1*VaR[t-1] + b2*|ret[t-1]| (SAV) and asymmetric-slope "
        "variant; trailing-window coefficient calibration every 63 bars; "
        "implied ES proxy, VaR exceedance rate, tail-pressure distance."
    ),
    "requires": ["Close"],
    "produces": [
        "cav_sav_var5",       # SAV-CAViaR 5% VaR (left-tail, always <= 0 after warmup)
        "cav_sav_var1",       # SAV-CAViaR 1% VaR (deeper left-tail)
        "cav_as_var5",        # Asymmetric-slope CAViaR 5% VaR
        "cav_es_proxy",       # ES proxy: mean of returns on exceedance days (trailing 63-bar window)
        "cav_exceedance_rate",# Fraction of last 63 returns that breached cav_sav_var5
        "cav_tail_pressure",  # (return[t] - cav_sav_var5[t]) / abs(cav_sav_var5[t]); >0 = inside VaR
        "cav_vol_regime",     # SAV-VaR z-score vs its own 63-bar rolling mean (regime indicator)
    ],
    "tags": ["volatility", "tail_risk", "distribution", "experimental"],
    "version": "1.0",
    "author": "paper doaj:042634a2 — CAViaR recursive tail-risk proxy",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Hard-coded sensible defaults (Engle & Manganelli SAV, 5% quantile estimates)
_DEFAULT_SAV5  = (0.02,  0.90, 0.15)   # (|b0|, b1, b2); b0 negated for left tail
_DEFAULT_SAV1  = (0.04,  0.88, 0.18)
_DEFAULT_AS5   = (0.03, 0.01, 0.88, 0.22, 0.04)  # (|b0_neg|, |b0_pos|, b1, b2_neg, b2_pos)

# Calibration schedule: calibrate on first BURN bars, then every RECAL bars
_BURN  = 252
_RECAL = 63
# Minimum observations for a valid calibration fit
_MIN_FIT = 63
# Grid search resolution for b1 and b2 (b0 is solved analytically given b1,b2)
_B1_GRID = np.array([0.80, 0.85, 0.88, 0.90, 0.92, 0.95])
_B2_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25])


def _pinball_loss(q: float, errors: np.ndarray) -> float:
    """Quantile regression (pinball / tick) loss for scalar q in (0,1)."""
    return float(np.where(errors >= 0, q * errors, (q - 1.0) * errors).mean())


def _sav_recursive(rets: np.ndarray, b0: float, b1: float, b2: float,
                   var_init: float) -> np.ndarray:
    """
    Run the SAV-CAViaR recursion over `rets` (1-D, already shifted).

    VaR[t] = -b0 + b1 * VaR[t-1] - b2 * |rets[t]|

    Note: b0, b1, b2 are all positive; the negative sign embeds the
    left-tail convention so VaR stays <= 0.

    Parameters
    ----------
    rets     : 1-D array of returns for the period (r[t] = shock at position t).
    b0, b1, b2 : positive coefficients.
    var_init  : starting value for VaR (negative).

    Returns
    -------
    var_out : 1-D array same length as rets, VaR estimate AT each t
              (uses |rets[t]| as the shock, which equals |ret[t-1]| in the
               caller's aligned frame).
    """
    n = len(rets)
    var_out = np.empty(n, dtype=np.float64)
    v = var_init
    for i in range(n):
        v = -b0 + b1 * v - b2 * abs(rets[i])
        # Clip: VaR should never be positive (left tail)
        if v > 0.0:
            v = 0.0
        var_out[i] = v
    return var_out


def _as_recursive(rets: np.ndarray, b0_neg: float, b0_pos: float,
                  b1: float, b2_neg: float, b2_pos: float,
                  var_init: float) -> np.ndarray:
    """
    Asymmetric-slope CAViaR:
      VaR[t] = -b0_neg - b1*VaR[t-1] + b2_neg*max(rets[t], 0) - b2_pos*min(rets[t], 0)

    Negative shocks (rets < 0) increase tail risk more than positive shocks.
    """
    n = len(rets)
    var_out = np.empty(n, dtype=np.float64)
    v = var_init
    for i in range(n):
        r = rets[i]
        pos_part = max(r, 0.0)
        neg_part = min(r, 0.0)  # <= 0
        v = -b0_neg + b1 * v + b2_neg * pos_part - b2_pos * neg_part
        if v > 0.0:
            v = 0.0
        var_out[i] = v
    return var_out


def _calibrate_sav(rets_window: np.ndarray, q: float,
                   default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calibrate SAV-CAViaR on `rets_window` by grid search over (b1, b2).
    b0 is set to (1 - b1) * empirical_q so the stationary mean = empirical_q.

    Returns (b0, b1, b2) as positive magnitudes (left-tail sign convention
    is applied inside _sav_recursive).
    """
    if len(rets_window) < _MIN_FIT:
        return default

    emp_q = float(np.quantile(rets_window, q))   # typically negative
    if emp_q >= 0.0:
        return default

    best_loss = math.inf
    best = default

    # Initialise VaR at empirical quantile for the trial run
    var_init = emp_q

    for b1 in _B1_GRID:
        for b2 in _B2_GRID:
            b0 = abs(emp_q) * (1.0 - b1)   # stationarity: E[VaR] ≈ emp_q
            if b0 < 0.0:
                continue
            var_series = _sav_recursive(rets_window, b0, b1, b2, var_init)
            residuals  = rets_window - var_series
            loss = _pinball_loss(q, residuals)
            if loss < best_loss:
                best_loss = loss
                best = (b0, b1, b2)

    return best


def _calibrate_as(rets_window: np.ndarray, q: float,
                  default: Tuple) -> Tuple:
    """
    Calibrate asymmetric-slope CAViaR on `rets_window`.
    Grid over b1 and b2_neg only (b2_pos fixed at b2_neg/4, b0_neg analytical).
    """
    if len(rets_window) < _MIN_FIT:
        return default

    emp_q = float(np.quantile(rets_window, q))
    if emp_q >= 0.0:
        return default

    best_loss = math.inf
    best = default
    var_init = emp_q

    b2_pos_ratio = 0.2   # b2_pos = b2_neg * ratio (leverage: neg shocks hurt more)

    for b1 in _B1_GRID:
        for b2_neg in _B2_GRID:
            b2_pos  = b2_neg * b2_pos_ratio
            b0_neg  = abs(emp_q) * (1.0 - b1)
            if b0_neg < 0.0:
                continue
            b0_pos = b0_neg * 0.4   # smaller intercept on positive-shock side
            var_series = _as_recursive(rets_window, b0_neg, b0_pos,
                                       b1, b2_neg, b2_pos, var_init)
            residuals  = rets_window - var_series
            loss = _pinball_loss(q, residuals)
            if loss < best_loss:
                best_loss = loss
                best = (b0_neg, b0_pos, b1, b2_neg, b2_pos)

    return best


# ---------------------------------------------------------------------------
# Public compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the CAViaR recursive tail-risk feature pack.

    Causality contract
    ------------------
    At row t, the recursive VaR uses only rets[0..t-1] (via the pre-shift).
    Coefficient calibration uses only rets[0..calibration_end-1], where
    calibration_end <= t for all rows in the subsequent segment.
    Re-running compute() on df[:k] will produce identical values for rows
    0..k-1 (bit-exact by construction — the recursive state is deterministic).
    """
    n = len(df)

    # Output arrays — initialised to NaN
    sav5_arr  = np.full(n, np.nan)
    sav1_arr  = np.full(n, np.nan)
    as5_arr   = np.full(n, np.nan)
    es_arr    = np.full(n, np.nan)
    exc_arr   = np.full(n, np.nan)
    pres_arr  = np.full(n, np.nan)
    reg_arr   = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Log returns; r[t] = log(Close[t] / Close[t-1])
    # r[0] = NaN by construction; we shift so the CAViaR shock at t
    # is r[t-1] (already available at time t, no lookahead).
    # ------------------------------------------------------------------
    close = df["Close"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ret = np.empty(n, dtype=np.float64)
        log_ret[0] = np.nan
        for i in range(1, n):
            if close[i - 1] > 0.0 and close[i] > 0.0:
                log_ret[i] = math.log(close[i] / close[i - 1])
            else:
                log_ret[i] = np.nan

    # Shock at position t is log_ret[t] (the return JUST completed = ret_{t-1}
    # in the econometric notation where the CAViaR equation is for Q_{t|t-1}).
    # We therefore align: shock[t] = log_ret[t], VaR[t] depends on VaR[t-1]
    # and shock[t] = log_ret[t] (which uses Close[t-1] known at t).
    shocks = log_ret   # shape (n,), shocks[0]=NaN

    # ------------------------------------------------------------------
    # Segment-by-segment processing with recalibration
    # ------------------------------------------------------------------

    # Calibration state (coeff tuples for SAV-5%, SAV-1%, AS-5%)
    coeff_sav5 = _DEFAULT_SAV5
    coeff_sav1 = _DEFAULT_SAV1
    coeff_as5  = _DEFAULT_AS5

    # We need at least _BURN rows before we start emitting values
    if n < _BURN:
        # Still run but emit NaN everywhere except a partial CAViaR trace
        # that starts from row 1.  This handles short-history tickers.
        warmup_end = max(20, n // 4)
    else:
        warmup_end = _BURN

    # Initialise VaR from empirical quantile of warmup period
    valid_warmup = shocks[1:warmup_end]
    valid_warmup = valid_warmup[np.isfinite(valid_warmup)]

    if len(valid_warmup) < 10:
        # Cannot initialise; emit all NaN
        df["cav_sav_var5"]        = np.nan
        df["cav_sav_var1"]        = np.nan
        df["cav_as_var5"]         = np.nan
        df["cav_es_proxy"]        = np.nan
        df["cav_exceedance_rate"] = np.nan
        df["cav_tail_pressure"]   = np.nan
        df["cav_vol_regime"]      = np.nan
        return df

    var5_init = float(np.quantile(valid_warmup, 0.05))
    var1_init = float(np.quantile(valid_warmup, 0.01))

    # Clamp initials to be negative
    if var5_init >= 0.0:
        var5_init = -abs(np.std(valid_warmup)) * 1.645
    if var1_init >= 0.0:
        var1_init = -abs(np.std(valid_warmup)) * 2.326

    # Calibrate on the warmup window
    coeff_sav5 = _calibrate_sav(valid_warmup, 0.05, _DEFAULT_SAV5)
    coeff_sav1 = _calibrate_sav(valid_warmup, 0.01, _DEFAULT_SAV1)
    coeff_as5  = _calibrate_as(valid_warmup, 0.05, _DEFAULT_AS5)

    # ------------------------------------------------------------------
    # Run the full recursive pass from warmup_end onward.
    # Rows 1..warmup_end-1 get "in-warmup" VaR (do NOT emit to output —
    # those would come from partially-fitted coefficients on the short
    # window; we use them only to carry recursive state into the live zone).
    # ------------------------------------------------------------------

    # Carry states
    v_sav5 = var5_init
    v_sav1 = var1_init
    v_as5  = var5_init  # start asymmetric VaR at same initial level

    b0_s5, b1_s5, b2_s5 = coeff_sav5
    b0_s1, b1_s1, b2_s1 = coeff_sav1
    b0_an, b0_ap, b1_as, b2_an, b2_ap = coeff_as5

    # Track next recalibration index
    next_recal = warmup_end + _RECAL

    for t in range(1, n):
        shock_t = shocks[t]

        # ------ Recalibrate if scheduled ----------------------------------
        if t >= next_recal and t >= warmup_end:
            # Use all returns available STRICTLY before t
            hist = shocks[1:t]
            hist = hist[np.isfinite(hist)]
            if len(hist) >= _MIN_FIT:
                coeff_sav5 = _calibrate_sav(hist, 0.05, coeff_sav5)
                coeff_sav1 = _calibrate_sav(hist, 0.01, coeff_sav1)
                coeff_as5  = _calibrate_as(hist,  0.05, coeff_as5)
            b0_s5, b1_s5, b2_s5 = coeff_sav5
            b0_s1, b1_s1, b2_s1 = coeff_sav1
            b0_an, b0_ap, b1_as, b2_an, b2_ap = coeff_as5
            next_recal = t + _RECAL

        # ------ Recursive VaR update at t ---------------------------------
        # Shock is shock_t = log_ret[t] = log(Close[t]/Close[t-1])
        # This is fully known by the close of bar t; the equation reads:
        #   VaR_{t+1|t} depends on |shock_t|
        # so VaR stored in arr[t] is the estimate FOR bar t using shock at t.
        if not math.isfinite(shock_t):
            # Carry forward previous VaR state; don't emit
            v_sav5 = v_sav5
            v_sav1 = v_sav1
            v_as5  = v_as5
            # Mark output NaN
            if t >= warmup_end:
                sav5_arr[t] = np.nan
                sav1_arr[t] = np.nan
                as5_arr[t]  = np.nan
            continue

        abs_shock = abs(shock_t)

        # SAV recursion
        v_sav5_new = -b0_s5 + b1_s5 * v_sav5 - b2_s5 * abs_shock
        if v_sav5_new > 0.0:
            v_sav5_new = 0.0
        v_sav5 = v_sav5_new

        v_sav1_new = -b0_s1 + b1_s1 * v_sav1 - b2_s1 * abs_shock
        if v_sav1_new > 0.0:
            v_sav1_new = 0.0
        v_sav1 = v_sav1_new

        # AS recursion
        pos_part = shock_t if shock_t > 0.0 else 0.0
        neg_part = shock_t if shock_t < 0.0 else 0.0
        v_as5_new = -b0_an + b1_as * v_as5 + b2_an * pos_part - b2_ap * neg_part
        if v_as5_new > 0.0:
            v_as5_new = 0.0
        v_as5 = v_as5_new

        # Emit only after warmup
        if t >= warmup_end:
            sav5_arr[t] = v_sav5
            sav1_arr[t] = v_sav1
            as5_arr[t]  = v_as5

    # ------------------------------------------------------------------
    # Derived features: ES proxy, exceedance rate, tail pressure,
    # vol regime.  All use only PAST values (trailing 63-bar window).
    # ------------------------------------------------------------------
    WIN = _RECAL  # 63-bar trailing window for derived features

    for t in range(warmup_end, n):
        start = max(1, t - WIN + 1)
        # Past returns in window [start, t-1] (exclude t itself → no lookahead)
        past_end = t         # exclusive — does NOT include t
        past_start = max(1, past_end - WIN)

        past_rets  = log_ret[past_start:past_end]
        past_var5  = sav5_arr[past_start:past_end]

        # Filter to finite values
        mask = np.isfinite(past_rets) & np.isfinite(past_var5)
        if mask.sum() < 5:
            continue

        pr = past_rets[mask]
        pv = past_var5[mask]

        # Exceedance: how often did past returns breach the recursive VaR?
        breach = pr < pv
        exc_arr[t] = float(breach.sum()) / float(len(pr))

        # ES proxy: conditional mean of returns on breach days
        if breach.sum() > 0:
            es_arr[t] = float(pr[breach].mean())
        else:
            es_arr[t] = np.nan

        # Tail pressure at t: (log_ret[t] - sav5[t]) / |sav5[t]|
        var_t   = sav5_arr[t]
        ret_now = log_ret[t]
        if math.isfinite(var_t) and math.isfinite(ret_now) and var_t != 0.0:
            pres_arr[t] = (ret_now - var_t) / abs(var_t)

    # Vol regime: z-score of sav5 vs its own trailing 63-bar mean/std
    sav5_s = pd.Series(sav5_arr)
    roll_mean = sav5_s.shift(1).rolling(WIN, min_periods=max(10, WIN // 4)).mean()
    roll_std  = sav5_s.shift(1).rolling(WIN, min_periods=max(10, WIN // 4)).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        reg_arr = np.where(
            (roll_std.notna()) & (roll_std > 0),
            (sav5_s - roll_mean) / roll_std,
            np.nan
        ).astype(np.float64)

    # Replace inf with NaN for safety
    sav5_arr = np.where(np.isfinite(sav5_arr), sav5_arr, np.nan)
    sav1_arr = np.where(np.isfinite(sav1_arr), sav1_arr, np.nan)
    as5_arr  = np.where(np.isfinite(as5_arr),  as5_arr,  np.nan)
    es_arr   = np.where(np.isfinite(es_arr),   es_arr,   np.nan)
    exc_arr  = np.where(np.isfinite(exc_arr),  exc_arr,  np.nan)
    pres_arr = np.where(np.isfinite(pres_arr), pres_arr, np.nan)
    reg_arr  = np.where(np.isfinite(reg_arr),  reg_arr,  np.nan)

    # ------------------------------------------------------------------
    # Assign to df — ONLY the produced columns
    # ------------------------------------------------------------------
    df["cav_sav_var5"]        = sav5_arr
    df["cav_sav_var1"]        = sav1_arr
    df["cav_as_var5"]         = as5_arr
    df["cav_es_proxy"]        = es_arr
    df["cav_exceedance_rate"] = exc_arr
    df["cav_tail_pressure"]   = pres_arr
    df["cav_vol_regime"]      = reg_arr

    return df
