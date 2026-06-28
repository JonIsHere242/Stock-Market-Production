"""
_paper_2604_06068_jump_diffusion_features.py
─────────────────────────────────────────────
Merton jump-diffusion decomposition proxy computed from daily OHLCV data only.

Reference: "Beyond Black-Scholes: Pricing Options with Heston, GARCH, and Jump
Diffusion Models" arXiv:2604.06068 (Merton branch).

The Merton model splits log-returns into:
    r_t = (μ - ½σ²) dt  +  σ dW_t   (continuous diffusion)
                         +  J_t dN_t  (jump component, Poisson arrivals)

We proxy this decomposition from daily log-returns using a causal
jump-detection threshold (|r_t| > k × rolling_vol_t) then build rolling
statistics that separate the jump vs continuous regimes.

NOTE: this is an honest daily-OHLCV Merton/bipower proxy — true high-freq
bipower variation requires tick/minute data.  The continuous-vol and jump-var
estimates here are indicative, not exact.

All columns prefixed `jmp_`.  Leading NaNs are expected.  No inf emitted.
"""

import math
import warnings

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
METADATA = {
    "name": "paper_2604_06068_jump_diffusion",
    "description": (
        "Merton jump-diffusion decomposition: causal jump flags from daily log-returns, "
        "then rolling jump intensity, mean jump size, signed asymmetry, jump-variation share, "
        "time-since-last-jump, continuous-component vol, and EWMA conditional vol of the "
        "continuous component (daily OHLCV proxy — arXiv:2604.06068)."
    ),
    "requires": ["Close"],
    "produces": [
        "jmp_flag",                # 1 = jump day, 0 = normal, NaN in warmup
        "jmp_intensity_21d",       # fraction of days flagged as jumps (trailing 21d)
        "jmp_mean_abs_size_21d",   # mean |log-return| on jump days (trailing 21d)
        "jmp_signed_asym_21d",     # (up-jump count - down-jump count) / total jumps (trailing 21d)
        "jmp_var_share_21d",       # jump-variation share of total RV (trailing 21d)
        "jmp_days_since_last",     # days elapsed since most-recent jump (causal)
        "jmp_continuous_vol_21d",  # annualised vol of non-jump log-returns (trailing 21d)
        "jmp_ewma_cont_vol",       # EWMA conditional vol of the continuous component
    ],
    "tags": ["volatility", "experimental", "jump_diffusion"],
    "version": "1.0",
    "author": "paper arXiv:2604.06068 — Merton jump-diffusion branch",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────
_VOL_WINDOW = 21        # causal rolling window for local vol estimate
_JUMP_K     = 3.0       # jump threshold: flag when |r| > k × local_vol
_ROLL_W     = 21        # rolling window for all summary statistics
_EWMA_SPAN  = 21        # EWMA span for conditional continuous vol
_ANNUALISE  = math.sqrt(252)


def _rolling_jump_stats(
    log_ret: np.ndarray,
    flag: np.ndarray,
    window: int,
) -> tuple:
    """
    Compute all rolling statistics in one pass over a causal window.

    Returns six arrays (same length as inputs), all float64 with np.nan
    wherever fewer than (window // 2) observations are available:
        intensity, mean_abs_size, signed_asym, var_share, cont_vol_ann, dummy
    """
    n = len(log_ret)
    min_periods = max(window // 2, 5)

    intensity        = np.full(n, np.nan)
    mean_abs_size    = np.full(n, np.nan)
    signed_asym      = np.full(n, np.nan)
    var_share        = np.full(n, np.nan)
    cont_vol_ann     = np.full(n, np.nan)

    for i in range(window - 1, n):
        w_start = i - window + 1
        r_w = log_ret[w_start: i + 1]   # shape (window,)
        f_w = flag[w_start: i + 1]       # shape (window,), 0/1/nan

        # Only rows where flag is valid (not nan from warmup)
        valid_mask = np.isfinite(f_w) & np.isfinite(r_w)
        n_valid = valid_mask.sum()
        if n_valid < min_periods:
            continue

        r_v = r_w[valid_mask]
        f_v = f_w[valid_mask]

        jump_mask = f_v == 1.0
        cont_mask = f_v == 0.0
        n_jump = jump_mask.sum()

        # Jump intensity: fraction of days that are jumps
        intensity[i] = n_jump / n_valid

        # Mean absolute jump size
        if n_jump > 0:
            mean_abs_size[i] = np.abs(r_v[jump_mask]).mean()
        # else stays NaN (no jumps in window)

        # Signed asymmetry: (up_jumps - down_jumps) / total_jumps
        if n_jump > 0:
            up_j   = (r_v[jump_mask] > 0).sum()
            down_j = (r_v[jump_mask] < 0).sum()
            signed_asym[i] = (up_j - down_j) / n_jump
        # else stays NaN

        # Jump-variation share: sum(r_jump²) / sum(r_all²)
        total_rv = np.sum(r_v ** 2)
        if total_rv > 0:
            jump_rv = np.sum(r_v[jump_mask] ** 2) if n_jump > 0 else 0.0
            var_share[i] = jump_rv / total_rv
        else:
            var_share[i] = 0.0

        # Continuous-component vol: std of non-jump log-returns, annualised
        n_cont = cont_mask.sum()
        if n_cont >= 2:
            cont_vol_ann[i] = np.std(r_v[cont_mask], ddof=1) * _ANNUALISE
        # else stays NaN

    return intensity, mean_abs_size, signed_asym, var_share, cont_vol_ann


def _ewma_continuous_vol(
    log_ret: np.ndarray,
    flag: np.ndarray,
    span: int,
) -> np.ndarray:
    """
    EWMA conditional variance of the continuous component only.
    On jump days the squared return is replaced by NaN (ignored), then
    pandas ewm(span, min_periods) propagates the last variance forward.
    Returns annualised vol (sqrt of variance × 252).
    """
    n = len(log_ret)
    cont_sq = np.where(
        (flag == 0.0) & np.isfinite(log_ret),
        log_ret ** 2,
        np.nan,
    )
    # Use pandas EWM ignoring NaN so jump days don't pollute
    s = pd.Series(cont_sq)
    ewm_var = s.ewm(span=span, min_periods=span // 2, ignore_na=True).mean()
    out = np.sqrt(ewm_var.to_numpy() * 252.0)
    # Guard: no inf
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def _days_since_last_jump(flag: np.ndarray) -> np.ndarray:
    """
    For each position i, return the number of rows since the most-recent jump
    flag==1.  Returns NaN before the first jump has occurred.
    Uses only past information (causal).
    """
    n = len(flag)
    out = np.full(n, np.nan)
    last_jump = -1  # -1 = no jump seen yet

    for i in range(n):
        if not np.isfinite(flag[i]):
            # Still in warmup — no history yet
            continue
        if flag[i] == 1.0:
            # This bar IS a jump
            if last_jump >= 0:
                out[i] = float(i - last_jump)
            # Update AFTER recording distance (so distance to self = 0 only
            # if we set it before; we leave it as distance-from-previous)
            last_jump = i
        else:
            if last_jump >= 0:
                out[i] = float(i - last_jump)
            # else: still haven't seen any jump → NaN

    return out


# ─────────────────────────────────────────────────────────────────────────────
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Merton jump-diffusion proxy columns (prefix `jmp_`) to df.

    All computations are causal: each value at row t depends only on rows
    at indices <= t.  Leading NaNs during the warmup period are expected and
    left as-is.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ── 1. Daily log-returns ─────────────────────────────────────────────────
    log_ret = np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret[1:] = np.log(close[1:] / close[:-1])
    # Guard: if close has zeros or negatives, ratio can be inf/nan
    log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

    # ── 2. Causal local volatility (rolling std of log-returns) ──────────────
    # Use pandas rolling on Series for cleanliness; shift(1) ensures only
    # PAST bars enter the threshold at each time t.
    ret_series = pd.Series(log_ret, dtype=float)
    local_vol = (
        ret_series.shift(1)                          # past returns only
        .rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW // 2)
        .std()
        .to_numpy()
    )

    # ── 3. Jump flag ─────────────────────────────────────────────────────────
    # A bar is a JUMP if |log_ret| > k × local_vol AND local_vol is valid.
    # Result is 1.0 (jump) / 0.0 (continuous) / NaN (insufficient history).
    flag = np.where(
        np.isfinite(local_vol) & np.isfinite(log_ret),
        (np.abs(log_ret) > _JUMP_K * local_vol).astype(np.float64),
        np.nan,
    )

    # ── 4. Rolling summary statistics ─────────────────────────────────────────
    (
        intensity,
        mean_abs_sz,
        signed_asym,
        var_share,
        cont_vol,
    ) = _rolling_jump_stats(log_ret, flag, _ROLL_W)

    # ── 5. Days-since-last-jump ───────────────────────────────────────────────
    days_since = _days_since_last_jump(flag)

    # ── 6. EWMA conditional continuous vol ────────────────────────────────────
    ewma_cont = _ewma_continuous_vol(log_ret, flag, _EWMA_SPAN)

    # ── 7. Assign columns to df ───────────────────────────────────────────────
    idx = df.index

    df["jmp_flag"]               = pd.array(flag,        dtype="Float64")[: n]
    df["jmp_intensity_21d"]      = pd.array(intensity,   dtype="Float64")[: n]
    df["jmp_mean_abs_size_21d"]  = pd.array(mean_abs_sz, dtype="Float64")[: n]
    df["jmp_signed_asym_21d"]    = pd.array(signed_asym, dtype="Float64")[: n]
    df["jmp_var_share_21d"]      = pd.array(var_share,   dtype="Float64")[: n]
    df["jmp_days_since_last"]    = pd.array(days_since,  dtype="Float64")[: n]
    df["jmp_continuous_vol_21d"] = pd.array(cont_vol,    dtype="Float64")[: n]
    df["jmp_ewma_cont_vol"]      = pd.array(ewma_cont,   dtype="Float64")[: n]

    return df
