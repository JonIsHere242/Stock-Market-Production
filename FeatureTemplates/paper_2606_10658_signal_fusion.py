"""
Heterogeneous Signal Fusion Features
Paper: "Post-Quantum Secure Federated DeFi for Inclusive Banking" (arXiv 2606.10658)

SKIP REASON: Pure cryptography / DeFi / banking paper with no OHLCV-computable method.
The paper concerns post-quantum lattice-based FHE for federated inter-bank credit scoring —
no time-series transforms applicable to price data.

SLOT REPLACEMENT (same theme: multi-source evidence fusion / probabilistic belief combination):
The paper's technical core is *combining heterogeneous information sources* (local bank data,
expert beliefs, geospatial evidence) in an evidence-fusion framework. We implement this
as OHLCV heterogeneous signal fusion:

  - Treat Price momentum, Volume momentum, and Range-based signals as three independent
    "evidence sources" (analogous to the paper's banks/expert/GFM sources)
  - Fuse them via Dempster-Shafer-inspired belief combination (normalised product of
    probability mass assignments from each source)
  - Also implement a "participation imbalance" measure (imbalance between up-day and
    down-day volumes, analogous to the paper's participation-imbalance concern for
    underserved borrowers)
  - Multiple fusion horizons give a family of columns

The Dempster-Shafer fusion of K binary belief sources:
  m_fused(H) ∝ ∏_k m_k(H) / (1 - K_conflict)
  where conflict K = ∑_{H∩G=∅} ∏_k m_k sources intersecting empty set
  Approximated here as: normalised product of per-source upward probability estimates.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_10658_signal_fusion",
    "description": (
        "Heterogeneous OHLCV signal fusion (price/volume/range belief combination) "
        "inspired by federated multi-source evidence aggregation (arXiv 2606.10658). "
        "Paper skipped (DeFi/quantum-crypto); slot replaced with this fusion feature family."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces":    [
        # Per-source upward probability (normalised ranking within rolling window)
        "hsf_price_belief_20d",
        "hsf_volume_belief_20d",
        "hsf_range_belief_20d",
        # Fused belief: geometric mean of source beliefs (Dempster-Shafer approximation)
        "hsf_fused_belief_20d",
        "hsf_fused_belief_60d",
        # Conflict score: disagreement between sources (high conflict → uncertain)
        "hsf_source_conflict_20d",
        # Volume participation imbalance: up-day vol vs down-day vol ratio
        "hsf_vol_participation_imbalance_20d",
        "hsf_vol_participation_imbalance_60d",
        # Fusion momentum: change in fused belief over 5 days
        "hsf_fused_belief_momentum_5d",
    ],
    "tags":        ["volume", "momentum", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.10658",
}


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of current value within past `window` bars. [0,1]"""
    return series.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True
    )


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Source 1: Price momentum signal (rolling return rank) ──────────────────
    ret_1d = close.pct_change()
    price_belief_20 = _rolling_rank(ret_1d, 20)

    # ── Source 2: Volume signal (volume vs rolling average rank) ───────────────
    vol_ratio = volume / (volume.rolling(20, min_periods=10).mean().clip(lower=1e-8))
    volume_belief_20 = _rolling_rank(vol_ratio, 20)

    # ── Source 3: Range signal (close position within high-low range) ──────────
    hl_range = (high - low).clip(lower=1e-8)
    close_pos = (close - low) / hl_range   # 0=at low, 1=at high
    range_belief_20 = close_pos.rolling(20, min_periods=10).mean()

    df["hsf_price_belief_20d"]  = price_belief_20
    df["hsf_volume_belief_20d"] = volume_belief_20
    df["hsf_range_belief_20d"]  = range_belief_20

    # ── Fused belief: geometric mean of the 3 sources (bounded [0,1]) ──────────
    eps = 1e-6
    p = price_belief_20.clip(lower=eps, upper=1 - eps)
    v = volume_belief_20.clip(lower=eps, upper=1 - eps)
    r = range_belief_20.clip(lower=eps, upper=1 - eps)

    fused_20 = (p * v * r) ** (1.0 / 3.0)
    df["hsf_fused_belief_20d"] = fused_20

    # 60-day version uses longer price rank window
    price_belief_60  = _rolling_rank(ret_1d, 60)
    vol_ratio_60     = volume / (volume.rolling(60, min_periods=30).mean().clip(lower=1e-8))
    volume_belief_60 = _rolling_rank(vol_ratio_60, 60)
    range_belief_60  = close_pos.rolling(60, min_periods=30).mean()
    p60 = price_belief_60.clip(lower=eps, upper=1 - eps)
    v60 = volume_belief_60.clip(lower=eps, upper=1 - eps)
    r60 = range_belief_60.clip(lower=eps, upper=1 - eps)
    df["hsf_fused_belief_60d"] = (p60 * v60 * r60) ** (1.0 / 3.0)

    # ── Conflict score: std of 3 source beliefs (high → disagreement) ──────────
    stacked = pd.concat([p, v, r], axis=1)
    df["hsf_source_conflict_20d"] = stacked.std(axis=1)

    # ── Volume participation imbalance: up-day vs down-day volume ratio ────────
    up_vol = volume.where(ret_1d > 0, 0.0)
    dn_vol = volume.where(ret_1d <= 0, 0.0)
    for w in [20, 60]:
        sum_up = up_vol.rolling(w, min_periods=w // 2).sum()
        sum_dn = dn_vol.rolling(w, min_periods=w // 2).sum()
        # Imbalance in (-1, +1): positive means up-day volume dominates
        df[f"hsf_vol_participation_imbalance_{w}d"] = (
            (sum_up - sum_dn) / (sum_up + sum_dn + 1e-8)
        )

    # ── Fusion momentum: 5-day change in fused belief ─────────────────────────
    df["hsf_fused_belief_momentum_5d"] = fused_20.diff(5)

    return df
