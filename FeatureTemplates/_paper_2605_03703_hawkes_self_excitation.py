"""
_paper_2605_03703_hawkes_self_excitation.py
───────────────────────────────────────────
Per-ticker Hawkes self-excitation proxy computed from daily OHLCV data only.

Reference: "Scaling Limits of Bivariate Nearly-Unstable Hawkes Processes and
Applications to Rough Volatility" arXiv:2605.03703.

CONCEPT
-------
A Hawkes (self-exciting) process models a stream of events where each event
temporarily raises the probability of future events via an exponential kernel:

    λ(t) = μ + α Σ_{t_i < t} exp(−β (t − t_i))

In a "nearly-unstable" regime the branching ratio n = α/β approaches 1, meaning
each event triggers close to one additional child event on average → self-
sustaining volatility clusters.  The paper connects this near-critical branching
to rough-volatility scaling (Hurst H → 0).

WHAT THIS BLOCK COMPUTES
------------------------
"Events" are large absolute log-returns (|r_t| > θ × causal trailing vol).
We track two event streams — positive shocks (up-events) and negative shocks
(down-events) — giving a bivariate Hawkes flavour.

All intensities are updated RECURSIVELY (O(n), causal, shift-free) using the
standard Hawkes exponential-decay update:

    λ[t] = μ + (λ[t-1] − μ) × e^{−β} + α × event[t-1]

This is causal: the intensity at t is the decayed legacy of ALL past events,
updated each bar with yesterday's event indicator.  No lookahead.

PRODUCED COLUMNS (prefix hwk_)
-------------------------------
hwk_intensity        : total (bivariate) Hawkes intensity — sum of up + down streams
hwk_intensity_up     : self-exciting intensity driven by positive large-return events
hwk_intensity_dn     : self-exciting intensity driven by negative large-return events
hwk_branching_proxy  : (intensity − baseline) / intensity — fraction of intensity
                       that is "excited" (approaches 1 near criticality)
hwk_asym             : (intensity_up − intensity_dn) / (intensity_up + intensity_dn + ε)
                       signed bias of recent self-excitation (+1 = all up-shocks)
hwk_intensity_chg_5d : 5-bar log-change in total intensity (momentum of excitation)
hwk_abs_ret_ac5      : rolling 21-bar autocorrelation of |log-returns| at lag 5
                       (direct measure of volatility clustering / memory)

NOTE: this block is DISTINCT from jmp_* (jump diffusion):
  - jmp_*  asks "did a jump occur, how big, how often?"
  - hwk_*  asks "how self-excited is the system, is it near criticality,
            is excitation directionally biased, is it accelerating?"
No raw jump-intensity fraction is emitted here.

All columns prefixed `hwk_`.  Leading NaNs are expected.  No inf emitted.
UNPROVEN candidate — kept hidden from auto-discovery (leading underscore).
"""

import math
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
METADATA = {
    "name": "paper_2605_03703_hawkes_self_excitation",
    "description": (
        "Per-ticker Hawkes self-excitation proxy: recursive bivariate intensity "
        "(up/down large-return streams), branching-ratio proxy, excitation "
        "asymmetry, intensity momentum, and volatility-clustering autocorrelation "
        "(daily OHLCV proxy — arXiv:2605.03703 nearly-unstable Hawkes)."
    ),
    "requires": ["Close"],
    "produces": [
        "hwk_intensity",         # total bivariate Hawkes intensity at each bar
        "hwk_intensity_up",      # intensity stream driven by positive shock events
        "hwk_intensity_dn",      # intensity stream driven by negative shock events
        "hwk_branching_proxy",   # (intensity − μ) / intensity → near 1 = near-critical
        "hwk_asym",              # signed up-vs-down excitation asymmetry
        "hwk_intensity_chg_5d",  # 5-bar log-change in total intensity
        "hwk_abs_ret_ac5",       # rolling 21-bar autocorr of |log-returns| at lag 5
    ],
    "tags": ["volatility", "experimental", "hawkes", "self_excitation"],
    "version": "1.0",
    "author": "paper arXiv:2605.03703 — nearly-unstable bivariate Hawkes proxy",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────
_VOL_WINDOW   = 21      # causal rolling window (bars) for local-vol threshold
_VOL_MIN_OBS  = 10      # min obs before threshold is valid
_EVENT_K      = 1.5     # event threshold: |r| > k × local_vol  (lower than jmp_'s 3.0)
_BETA         = 0.3     # exponential decay rate per bar  (τ ≈ 1/β ≈ 3.3 bars)
_ALPHA        = 0.25    # excitation amplitude per event  (branching ratio α/β ≈ 0.83)
_MU           = 0.10    # baseline (unconditional) intensity
_DECAY        = math.exp(-_BETA)   # precomputed: e^{-β}
_AC_WINDOW    = 21      # rolling window for |ret| autocorrelation
_AC_LAG       = 5       # lag for autocorrelation (volatility clustering memory)
_CHG_LAG      = 5       # bars for intensity change


# ─────────────────────────────────────────────────────────────────────────────
def _causal_local_vol(log_ret: np.ndarray) -> np.ndarray:
    """
    Rolling std of log-returns using only PAST bars (shift-by-1).
    Returns array of same length; NaN during warmup.
    """
    s = pd.Series(log_ret, dtype=float)
    return (
        s.shift(1)
        .rolling(_VOL_WINDOW, min_periods=_VOL_MIN_OBS)
        .std()
        .to_numpy()
    )


def _hawkes_recursive(
    event_up: np.ndarray,
    event_dn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bivariate Hawkes intensities recursively.

    Each stream has its own intensity driven by its own events (diagonal
    excitation).  The shared total intensity is the sum.

    Update rule per bar t (using only information up to t-1):
        λ_up[t] = μ + (λ_up[t-1] − μ) × decay + α × event_up[t-1]
        λ_dn[t] = μ + (λ_dn[t-1] − μ) × decay + α × event_dn[t-1]

    event_up / event_dn are 0.0 (no event) or 1.0 (event) per bar.
    NaN in either event array (warmup) is treated as 0.0 for the update
    but the intensity for that bar is tagged NaN in intensity_up/dn.

    Returns:
        lam_up : (n,) float64, NaN during warmup
        lam_dn : (n,) float64, NaN during warmup
        lam_tot: (n,) float64, NaN during warmup
    """
    n = len(event_up)
    lam_up  = np.full(n, np.nan)
    lam_dn  = np.full(n, np.nan)
    lam_tot = np.full(n, np.nan)

    # Track whether we've exited warmup (first bar where BOTH events are valid)
    started = False
    lu = _MU   # running intensity: up stream
    ld = _MU   # running intensity: down stream

    for t in range(n):
        eu = event_up[t]
        ed = event_dn[t]

        # Still in warmup — events are NaN
        if not np.isfinite(eu) or not np.isfinite(ed):
            # Decay the intensity but don't record it (still warmup)
            lu = _MU + (lu - _MU) * _DECAY
            ld = _MU + (ld - _MU) * _DECAY
            continue

        # First valid bar — mark started BEFORE recording
        started = True

        # Record intensity AT this bar (driven by history up to t-1)
        lam_up[t]  = lu
        lam_dn[t]  = ld
        lam_tot[t] = lu + ld

        # Update for next bar: decay + excitation from THIS bar's event
        lu = _MU + (lu - _MU) * _DECAY + _ALPHA * eu
        ld = _MU + (ld - _MU) * _DECAY + _ALPHA * ed

    return lam_up, lam_dn, lam_tot


def _rolling_autocorr_lag(x: np.ndarray, window: int, lag: int) -> np.ndarray:
    """
    Rolling Pearson autocorrelation of x at fixed lag, using only past data.

    For each position i (0-indexed), uses the window ending at i:
        corr(x[i-window+1 : i-lag+1],  x[i-window+1+lag : i+1])

    Returns NaN where fewer than window observations are available.
    This is causal: at time t, only x[0..t] is used.
    """
    n = len(x)
    min_obs = max(window // 2, lag + 2)
    out = np.full(n, np.nan)

    for i in range(window - 1, n):
        w_start = i - window + 1
        # x-series and lagged x-series within the window
        xi  = x[w_start : i - lag + 1]      # length = window - lag
        xil = x[w_start + lag : i + 1]      # length = window - lag

        # Both must have valid (finite) values at paired positions
        valid = np.isfinite(xi) & np.isfinite(xil)
        nv = valid.sum()
        if nv < min_obs:
            continue

        a = xi[valid]
        b = xil[valid]

        # Pearson correlation
        mean_a = a.mean()
        mean_b = b.mean()
        num  = ((a - mean_a) * (b - mean_b)).sum()
        den  = math.sqrt(((a - mean_a) ** 2).sum() * ((b - mean_b) ** 2).sum())
        if den < 1e-12:
            continue
        out[i] = num / den

    return out


# ─────────────────────────────────────────────────────────────────────────────
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Hawkes self-excitation proxy columns (prefix `hwk_`) to df.

    All features are causal: value at row t uses only rows <= t.
    Leading NaNs (warmup period) are left as-is.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ── 1. Daily log-returns ─────────────────────────────────────────────────
    log_ret = np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        valid_idx = (close[:-1] > 0) & (close[1:] > 0)
        log_ret[1:] = np.where(valid_idx, np.log(close[1:] / close[:-1]), np.nan)
    log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

    # ── 2. Causal local volatility ────────────────────────────────────────────
    local_vol = _causal_local_vol(log_ret)

    # ── 3. Event indicators: large absolute return, split up / down ──────────
    # An event occurs when |r_t| > threshold.  Up-event = r_t > 0.  Down = r_t < 0.
    # NaN where local_vol is not yet valid.
    threshold = _EVENT_K * local_vol       # shape (n,)
    is_event = np.where(
        np.isfinite(local_vol) & np.isfinite(log_ret),
        (np.abs(log_ret) > threshold).astype(np.float64),
        np.nan,
    )
    event_up = np.where(
        np.isfinite(is_event),
        np.where((is_event == 1.0) & (log_ret > 0), 1.0, 0.0),
        np.nan,
    )
    event_dn = np.where(
        np.isfinite(is_event),
        np.where((is_event == 1.0) & (log_ret < 0), 1.0, 0.0),
        np.nan,
    )

    # ── 4. Recursive Hawkes intensities ─────────────────────────────────────
    lam_up, lam_dn, lam_tot = _hawkes_recursive(event_up, event_dn)

    # ── 5. Branching proxy: how much of intensity is "excited" ───────────────
    # (lam_tot − 2μ) / lam_tot  — approaches 1 when baseline is swamped by excitation
    baseline_total = 2.0 * _MU   # sum of both stream baselines
    with np.errstate(divide="ignore", invalid="ignore"):
        branching_proxy = np.where(
            np.isfinite(lam_tot) & (lam_tot > 0),
            (lam_tot - baseline_total) / lam_tot,
            np.nan,
        )
    branching_proxy = np.where(np.isfinite(branching_proxy), branching_proxy, np.nan)

    # ── 6. Up-vs-down excitation asymmetry ──────────────────────────────────
    denom = lam_up + lam_dn + 1e-12
    asym = np.where(
        np.isfinite(lam_up) & np.isfinite(lam_dn),
        (lam_up - lam_dn) / denom,
        np.nan,
    )

    # ── 7. Intensity momentum: 5-bar log-change in total intensity ───────────
    eps_intensity = 1e-9
    lam_lag = np.full(n, np.nan)
    lam_lag[_CHG_LAG:] = lam_tot[: n - _CHG_LAG]
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity_chg = np.where(
            np.isfinite(lam_tot) & np.isfinite(lam_lag) & (lam_lag > eps_intensity),
            np.log(lam_tot / lam_lag),
            np.nan,
        )
    intensity_chg = np.where(np.isfinite(intensity_chg), intensity_chg, np.nan)

    # ── 8. Volatility-clustering autocorrelation of |log-returns| ────────────
    abs_ret = np.abs(log_ret)
    abs_ret_ac5 = _rolling_autocorr_lag(abs_ret, _AC_WINDOW, _AC_LAG)

    # ── 9. Assign columns to df ──────────────────────────────────────────────
    df["hwk_intensity"]        = pd.array(lam_tot,         dtype="Float64")[:n]
    df["hwk_intensity_up"]     = pd.array(lam_up,          dtype="Float64")[:n]
    df["hwk_intensity_dn"]     = pd.array(lam_dn,          dtype="Float64")[:n]
    df["hwk_branching_proxy"]  = pd.array(branching_proxy, dtype="Float64")[:n]
    df["hwk_asym"]             = pd.array(asym,            dtype="Float64")[:n]
    df["hwk_intensity_chg_5d"] = pd.array(intensity_chg,   dtype="Float64")[:n]
    df["hwk_abs_ret_ac5"]      = pd.array(abs_ret_ac5,     dtype="Float64")[:n]

    return df
