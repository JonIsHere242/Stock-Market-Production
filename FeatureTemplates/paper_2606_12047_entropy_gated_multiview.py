"""
Entropy-Gated Multi-View OHLCV Features  —  arxiv:2606.12047
"Metadata-Aware Multi-Prompt Reasoning for Zero-Shot Accident Understanding"

PAPER NOTE: The paper is computer vision / video accident detection using
VLMs — no OHLCV-computable method.  However the core ALGORITHMIC ideas are
directly portable:
  1. MULTI-VIEW DECOMPOSITION: 5 complementary views of the same signal
     (baseline, motion, geometry, contrast, tiebreaker).
  2. ENTROPY-GATED ADJUDICATION: when views disagree (high entropy over
     the vote distribution), trigger a tiebreaker / conflict resolution signal.
  3. SCORE-WEIGHTED CENTROID: aggregate detection scores across keyframes
     using a score-weighted centroid to locate the "impact event".

Applied to OHLCV as 5 complementary return views + entropy-gated combination:
  VIEW 1 — Baseline:  raw log-return
  VIEW 2 — Motion:    absolute acceleration (2nd diff of log-price)
  VIEW 3 — Geometry:  position within daily high-low range (range ratio)
  VIEW 4 — Contrast:  return vs rolling median (contrast = deviation from typical)
  VIEW 5 — Tiebreaker: volume-weighted return (breaks directional ties)

Entropy gate: compute Shannon entropy of the sign distribution of the 5 views.
  - Low entropy → views agree → strong signal
  - High entropy → views disagree → use tiebreaker view as adjudicator

Score-weighted centroid: exponentially weighted composite of views, weights
  proportional to |IC-like correlation with recent realized return|.

Produces 8 columns prefixed "egv_".  Fully vectorised; no Python loops.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12047_entropy_gated_multiview",
    "description": (
        "Entropy-gated multi-view OHLCV features inspired by arxiv:2606.12047 — "
        "5 complementary views (baseline/motion/geometry/contrast/volume tiebreaker), "
        "Shannon entropy of view disagreement, entropy-gated composite, and "
        "score-weighted centroid aggregation across multiple horizons."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "egv_view_motion_5d",
        "egv_view_geometry_5d",
        "egv_view_contrast_21d",
        "egv_entropy_disagree_5d",
        "egv_entropy_disagree_21d",
        "egv_gated_composite_21d",
        "egv_gated_composite_63d",
        "egv_swc_signal_21d",
    ],
    "tags": ["momentum", "volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12047",
}


def _view_entropy(v1, v2, v3, v4, v5, w: int = 5, mp: int = 3) -> pd.Series:
    """
    Shannon entropy of the sign distribution across 5 views in a rolling window.
    Uses rolling mean of each binarised view, then computes H = -Σ p*log(p).
    Low H → views agree (strong signal).  High H → views disagree.
    """
    # Binarise each view: 1 if positive, 0 if non-positive
    b1 = (v1 > 0).astype(float)
    b2 = (v2 > 0).astype(float)
    b3 = (v3 > 0).astype(float)
    b4 = (v4 > 0).astype(float)
    b5 = (v5 > 0).astype(float)

    # Fraction of "positive" views in each rolling window
    frac_pos = (b1 + b2 + b3 + b4 + b5).rolling(w, min_periods=mp).mean() / 5.0
    frac_neg = 1.0 - frac_pos

    # Shannon entropy: H = -p*log(p) - q*log(q)
    eps = 1e-9
    H = -(frac_pos * np.log(frac_pos + eps) + frac_neg * np.log(frac_neg + eps))
    return H


def _score_weighted_centroid(views: list, w: int = 21, mp: int = 7) -> pd.Series:
    """
    Score-weighted centroid of views.
    Weight for each view = rolling |autocorrelation| of that view at lag 1
    (proxy for IC: views with more persistent signal get higher weight).
    """
    n = len(views[0])
    idx = views[0].index

    weights = []
    for v in views:
        # Rolling lag-1 autocorrelation as weight proxy
        v_shift = v.shift(1)
        rolling_cov = v.rolling(w, min_periods=mp).cov(v_shift)
        rolling_var = v.rolling(w, min_periods=mp).var()
        acf1 = rolling_cov / (rolling_var + 1e-12)
        weights.append(acf1.abs())

    # Normalise weights to sum to 1
    total_w = sum(weights) + 1e-9
    swc = sum(v * (wt / total_w) for v, wt in zip(views, weights))
    return swc


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].replace(0, np.nan).astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    V = df["Volume"].replace(0, np.nan).astype(float)
    O = df["Open"].astype(float)

    # ── Build 5 views ──────────────────────────────────────────────────────────

    # VIEW 1: Baseline — log-return
    v1 = np.log(C / C.shift(1))

    # VIEW 2: Motion — acceleration = 2nd difference of log price
    log_p = np.log(C)
    v2_raw = log_p.diff().diff()   # 2nd difference
    # Smooth over 5 bars to reduce noise
    v2 = v2_raw.rolling(5, min_periods=2).mean()
    df["egv_view_motion_5d"] = v2

    # VIEW 3: Geometry — position within daily high-low range
    # = (Close - Low) / (High - Low), then de-mean: 0.5 = midpoint
    hl_range = (H - L).replace(0, np.nan)
    v3_raw = (C - L) / hl_range - 0.5   # centred: positive = upper half
    v3 = v3_raw.rolling(5, min_periods=2).mean()
    df["egv_view_geometry_5d"] = v3

    # VIEW 4: Contrast — return vs rolling median (deviation from typical)
    ret = np.log(C / C.shift(1))
    roll_med = ret.rolling(21, min_periods=7).median()
    v4 = ret - roll_med
    df["egv_view_contrast_21d"] = v4

    # VIEW 5: Tiebreaker — volume-weighted return (breaks directional ties)
    log_vol = np.log(V)
    vol_z = (log_vol - log_vol.rolling(21, min_periods=7).mean()) / (
        log_vol.rolling(21, min_periods=7).std() + 1e-9
    )
    v5 = ret * vol_z   # sign of return magnified/reversed by volume anomaly

    # ── Entropy-gated disagreement ─────────────────────────────────────────────
    df["egv_entropy_disagree_5d"]  = _view_entropy(v1, v2, v3, v4, v5, w=5,  mp=3)
    df["egv_entropy_disagree_21d"] = _view_entropy(v1, v2, v3, v4, v5, w=21, mp=7)

    # ── Entropy-gated composite ────────────────────────────────────────────────
    # When entropy is LOW (agreement), use simple average of views.
    # When entropy is HIGH (disagreement), use the tiebreaker view (v5).
    # Implemented as a smooth blend: composite = (1-H_norm)*avg_view + H_norm*v5
    H_norm_21 = df["egv_entropy_disagree_21d"]
    # Normalise entropy to [0,1] (max entropy for 2-class = log(2) ≈ 0.693)
    H_max = float(np.log(2))
    H_weight_21 = (H_norm_21 / H_max).clip(0, 1)

    # Normalise each view to comparable scale before blending
    def _zs(s, w=63, mp=20):
        mu = s.rolling(w, min_periods=mp).mean()
        sd = s.rolling(w, min_periods=mp).std()
        return (s - mu) / (sd + 1e-9)

    v1z = _zs(v1); v2z = _zs(v2); v3z = _zs(v3); v4z = _zs(v4); v5z = _zs(v5)
    avg_views = (v1z + v2z + v3z + v4z) / 4.0

    df["egv_gated_composite_21d"] = (1.0 - H_weight_21) * avg_views + H_weight_21 * v5z

    # 63d horizon: use 63d entropy normalisation
    H_norm_63 = _view_entropy(v1, v2, v3, v4, v5, w=63, mp=20)
    H_weight_63 = (H_norm_63 / H_max).clip(0, 1)
    df["egv_gated_composite_63d"] = (1.0 - H_weight_63) * avg_views + H_weight_63 * v5z

    # ── Score-weighted centroid ────────────────────────────────────────────────
    df["egv_swc_signal_21d"] = _score_weighted_centroid(
        [v1z, v2z, v3z, v4z, v5z], w=21, mp=7
    )

    return df
