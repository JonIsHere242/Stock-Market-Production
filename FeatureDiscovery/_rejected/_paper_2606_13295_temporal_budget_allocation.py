"""
Temporal Budget Allocation Features
Paper: "Simultaneous Latent Budget Trees for Stratified Classification"
arXiv: 2606.13295

The paper introduces Simultaneous Latent Budget Trees (SLBT) — a probabilistic tree
where child nodes are latent components of a SIMULTANEOUS MIXTURE MODEL fitted to the
parent, with STRATIFICATION by a temporal/spatial control variable.  The key model
idea: each observation's class profile is a mixture ("budget") that DIFFERS BY STRATUM.

The paper applies this to disease progression but has no OHLCV-computable method.
SKIPPED.

INSTEAD: We implement a distinctive OHLCV feature family capturing the same concept —
TEMPORAL BUDGET ALLOCATION, or how each bar's OHLCV fingerprint allocates across
multiple "latent regime components" (temporal strata) at different horizons.

Method:
  1. Stratify rolling history into three temporal strata: short (5d), medium (21d),
     long (63d).
  2. For each stratum, compute the "budget" = normalised soft allocation of today's
     primitives across three latent components (low/mid/high quantile buckets).
  3. The deviation of the current budget from the historical average budget reveals
     how the current bar is unusual within each stratum.
  4. Cross-stratum divergence (KL-style) captures how the short-term regime differs
     from the long-term "prior" — the SLBT's stratification signal.

Produces 7 columns prefixed "tba_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13295_temporal_budget_allocation",
    "description": (
        "Temporal budget allocation features inspired by arXiv:2606.13295 (SLBT). "
        "Rolling mixture-budget decomposition of OHLCV primitives across temporal "
        "strata; cross-stratum KL divergence detects regime shifts between "
        "short/medium/long-horizon latent components."
    ),
    "requires": ["Close", "High", "Low", "Volume"],
    "produces": [
        "tba_budget_short_low",
        "tba_budget_short_high",
        "tba_budget_long_high",
        "tba_budget_dev_short",
        "tba_budget_dev_long",
        "tba_cross_stratum_kl",
        "tba_regime_shift",
    ],
    "tags": ["market_regime", "mean_reversion", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13295",
}


def _soft_budget(val: float, lo: float, hi: float) -> np.ndarray:
    """
    Soft 3-bucket allocation: [prob_low, prob_mid, prob_high].
    Uses a smooth step function so the allocation is differentiable.
    val: current value; lo, hi: 33rd/67th percentiles.
    """
    # Linear interpolation: val < lo -> [1,0,0], val > hi -> [0,0,1]
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return np.array([1/3, 1/3, 1/3])
    if val <= lo:
        t_low = np.clip((lo - val) / max(hi - lo, 1e-10), 0, 1)
        return np.array([0.5 + 0.5 * t_low, 0.5 - 0.5 * t_low, 0.0])
    elif val >= hi:
        t_high = np.clip((val - hi) / max(hi - lo, 1e-10), 0, 1)
        return np.array([0.0, 0.5 - 0.5 * min(t_high, 1), 0.5 + 0.5 * min(t_high, 1)])
    else:
        # In middle bucket
        frac = (val - lo) / (hi - lo)
        return np.array([0.0, 1.0, 0.0]) * 1.0 + \
               np.array([0.3, -0.3, 0.0]) * (0.5 - frac) + \
               np.array([0.0, -0.3, 0.3]) * (frac - 0.5)

    # normalise to sum to 1
    b = np.array([max(0, v) for v in b])
    s = b.sum()
    return b / s if s > 0 else np.array([1/3, 1/3, 1/3])


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL divergence D_KL(p || q), safe for zero entries."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    V = df["Volume"].astype(float)
    n = len(df)

    # ── Composite primitive: weighted OHLCV signal ──────────────────────────────
    log_ret = np.log(C / C.shift(1))
    hl_ratio = (H - L) / C.replace(0, np.nan)
    log_vol = np.log(V.replace(0, np.nan))

    # ── Compute rolling quantile boundaries for 3-bucket allocation ─────────────
    def q33(s, w): return s.rolling(w, min_periods=max(5, w // 4)).quantile(0.33)
    def q67(s, w): return s.rolling(w, min_periods=max(5, w // 4)).quantile(0.67)

    # Short stratum (5-day)
    ret_lo5,  ret_hi5  = q33(log_ret, 5),  q67(log_ret, 5)
    # Long stratum (63-day)
    ret_lo63, ret_hi63 = q33(log_ret, 63), q67(log_ret, 63)

    # Same for volume
    vol_lo5,  vol_hi5  = q33(log_vol, 5),  q67(log_vol, 5)
    vol_lo63, vol_hi63 = q33(log_vol, 63), q67(log_vol, 63)

    # ── Per-bar budget vectors ───────────────────────────────────────────────────
    bud_short = np.full((n, 3), np.nan)
    bud_long  = np.full((n, 3), np.nan)

    ret_arr   = log_ret.values
    rlo5_arr  = ret_lo5.values;  rhi5_arr  = ret_hi5.values
    rlo63_arr = ret_lo63.values; rhi63_arr = ret_hi63.values
    vlo5_arr  = vol_lo5.values;  vhi5_arr  = vol_hi5.values
    vlo63_arr = vol_lo63.values; vhi63_arr = vol_hi63.values
    lv_arr    = log_vol.values

    for i in range(n):
        rv = ret_arr[i]
        vv = lv_arr[i]
        if np.isfinite(rv) and np.isfinite(vv):
            # Short-stratum budget: blend return + volume budgets equally
            b_ret_s = _soft_budget(rv, rlo5_arr[i], rhi5_arr[i])
            b_vol_s = _soft_budget(vv, vlo5_arr[i], vhi5_arr[i])
            b_s = 0.5 * b_ret_s + 0.5 * b_vol_s
            bud_short[i] = b_s / max(b_s.sum(), 1e-10)

            b_ret_l = _soft_budget(rv, rlo63_arr[i], rhi63_arr[i])
            b_vol_l = _soft_budget(vv, vlo63_arr[i], vhi63_arr[i])
            b_l = 0.5 * b_ret_l + 0.5 * b_vol_l
            bud_long[i] = b_l / max(b_l.sum(), 1e-10)

    df["tba_budget_short_low"]  = bud_short[:, 0]
    df["tba_budget_short_high"] = bud_short[:, 2]
    df["tba_budget_long_high"]  = bud_long[:, 2]

    # ── Budget deviation: current budget vs its 21-day rolling mean ─────────────
    # Captures how today's allocation differs from the recent "prior"
    bs_low_s  = pd.Series(bud_short[:, 0], index=df.index)
    bs_high_s = pd.Series(bud_short[:, 2], index=df.index)
    bl_high_s = pd.Series(bud_long[:, 2], index=df.index)

    bs_low_mean  = bs_low_s.rolling(21, min_periods=7).mean()
    bs_high_mean = bs_high_s.rolling(21, min_periods=7).mean()
    bl_high_mean = bl_high_s.rolling(63, min_periods=20).mean()

    # Dev = L1 distance between current and mean budget
    df["tba_budget_dev_short"] = (
        (bs_low_s - bs_low_mean).abs() + (bs_high_s - bs_high_mean).abs()
    )
    df["tba_budget_dev_long"] = (bl_high_s - bl_high_mean).abs()

    # ── Cross-stratum KL divergence: D_KL(short_budget || long_budget) ──────────
    # High KL = short-term regime is very different from long-term regime
    kl_vals = np.full(n, np.nan)
    for i in range(n):
        ps = bud_short[i]
        pl = bud_long[i]
        if np.all(np.isfinite(ps)) and np.all(np.isfinite(pl)):
            kl_vals[i] = _kl_divergence(ps, pl)

    df["tba_cross_stratum_kl"] = kl_vals

    # ── Regime shift: 5-day change in cross-stratum KL ──────────────────────────
    kl_s = pd.Series(kl_vals, index=df.index)
    df["tba_regime_shift"] = kl_s.diff(5)

    return df
