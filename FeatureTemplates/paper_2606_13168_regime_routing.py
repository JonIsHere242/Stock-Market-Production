"""
Regime-routing affinity features for OHLCV.

Paper: "When Does Routing Become Interpretable? Causal Probes on Block Attention
        Residuals" (arXiv 2606.13168)

The paper studies LEARNED ROUTING WEIGHTS in neural networks: at each layer, the
network soft-routes to one of several "source" depth-levels with learned weights
(routing mass). The key findings are:
  1. Routing mass ≠ causal importance (descriptive ≠ mechanistic)
  2. Routing motifs: embedding-source pathway, current-state pathway, older-history pathway
  3. Structured depth routing emerges ONLY when trained — it reflects regime in the data

OHLCV adaptation — "historical regime routing affinity":
  Map the three routing motifs to three OHLCV "depth" levels:
    - Motif A (embedding / long-term memory): regime defined by 120d feature state
    - Motif B (current state): recent 10d return/vol fingerprint
    - Motif C (older history): intermediate 40d state

  For each bar t, compute SOFTMAX ROUTING WEIGHTS over the three historical "regimes"
  based on how similar today's state vector is to each regime's centroid.
  The AFFINITY (routing weight) to each regime is the feature.

  Regime centroids are estimated as rolling window cluster prototypes:
    - Regime A prototype: rolling 120d median of [return, vol, range] feature vector
    - Regime B prototype: rolling 10d median
    - Regime C prototype: rolling 40d median
  Similarity: negative Euclidean distance → softmax → routing weights.

Produces 7 features:
  rrm_affinity_shortterm_10d   : softmax routing mass to 10d (current-state) regime
  rrm_affinity_medterm_40d     : softmax routing mass to 40d (older-history) regime
  rrm_affinity_longterm_120d   : softmax routing mass to 120d (embedding) regime
  rrm_dominant_regime          : argmax regime (0=short, 1=med, 2=long)
  rrm_routing_entropy          : entropy of routing weight distribution (regime ambiguity)
  rrm_affinity_shift_10v40d    : change in short-term vs medium-term affinity
  rrm_regime_stability_20d     : rolling std of dominant_regime (how stable the routing is)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13168_regime_routing",
    "description": (
        "Softmax regime-routing affinity features: similarity of current OHLCV state "
        "to short/medium/long-term historical regime prototypes, inspired by learned "
        "depth-routing motifs in Block Attention Residuals (arXiv 2606.13168)."
    ),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "rrm_affinity_shortterm_10d",
        "rrm_affinity_medterm_40d",
        "rrm_affinity_longterm_120d",
        "rrm_dominant_regime",
        "rrm_routing_entropy",
        "rrm_affinity_shift_10v40d",
        "rrm_regime_stability_20d",
    ],
    "tags":        ["market_regime", "momentum", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13168",
}

_SOFTMAX_TEMP = 10.0  # temperature for softmax (higher = crisper routing)


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    z = x * temp
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-14)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].clip(lower=1e-8)
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Feature vector: [log_return, realised_vol_5d, hl_range_norm] ────────
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # Realised vol = rolling std of log returns (5d), normalised by 20d std
    rv5  = log_ret.rolling(5,  min_periods=2).std()
    rv20 = log_ret.rolling(20, min_periods=5).std().clip(lower=1e-8)
    rv5_norm = (rv5 / rv20).fillna(1.0)

    # High-low range normalised by 20d average range
    hl_range = (high - low) / close
    hl_mean20 = hl_range.rolling(20, min_periods=5).mean().clip(lower=1e-8)
    hl_norm = (hl_range / hl_mean20).fillna(1.0)

    # Volume normalised by 20d average
    vol_mean20 = volume.rolling(20, min_periods=5).mean().clip(lower=1.0)
    vol_norm = (volume / vol_mean20).fillna(1.0)

    # State vector at each bar t: [log_ret, rv5_norm, hl_norm, vol_norm]
    state_mat = np.column_stack([
        log_ret.values,
        rv5_norm.values,
        hl_norm.values,
        vol_norm.values,
    ]).astype(np.float64)

    n = len(df)
    aff_short  = np.full(n, np.nan)
    aff_med    = np.full(n, np.nan)
    aff_long   = np.full(n, np.nan)
    dom_regime = np.full(n, np.nan)
    rout_ent   = np.full(n, np.nan)
    aff_shift  = np.full(n, np.nan)

    W_short = 10
    W_med   = 40
    W_long  = 120

    for i in range(W_long, n):
        cur = state_mat[i]
        if np.any(np.isnan(cur)):
            continue

        # Prototype = rolling median of state vectors over each window
        # (median is robust to outliers, unlike mean)
        s_short = state_mat[max(0, i - W_short): i]
        s_med   = state_mat[max(0, i - W_med):   i]
        s_long  = state_mat[max(0, i - W_long):  i]

        # Filter rows with any nan
        s_short = s_short[~np.any(np.isnan(s_short), axis=1)]
        s_med   = s_med[~np.any(np.isnan(s_med),   axis=1)]
        s_long  = s_long[~np.any(np.isnan(s_long), axis=1)]

        if len(s_short) < 3 or len(s_med) < 5 or len(s_long) < 15:
            continue

        proto_short = np.median(s_short, axis=0)
        proto_med   = np.median(s_med,   axis=0)
        proto_long  = np.median(s_long,  axis=0)

        # Normalise by the spread in the long window to make distances comparable
        # Use MAD of each feature over the long window
        mad_long = np.median(np.abs(s_long - proto_long), axis=0).clip(min=1e-8)

        cur_norm       = (cur - proto_long) / mad_long
        proto_short_n  = (proto_short - proto_long) / mad_long
        proto_med_n    = (proto_med   - proto_long) / mad_long
        proto_long_n   = np.zeros(cur.shape[0])  # long-term is the reference (all zeros)

        # Distances: negative squared Euclidean
        dist_short = -np.sum((cur_norm - proto_short_n) ** 2)
        dist_med   = -np.sum((cur_norm - proto_med_n)   ** 2)
        dist_long  = -np.sum((cur_norm - proto_long_n)  ** 2)

        dists = np.array([dist_short, dist_med, dist_long])
        weights = _softmax(dists, temp=_SOFTMAX_TEMP)

        aff_short[i]  = weights[0]
        aff_med[i]    = weights[1]
        aff_long[i]   = weights[2]
        dom_regime[i] = float(np.argmax(weights))

        # Routing entropy
        w_clip = weights.clip(min=1e-10)
        rout_ent[i] = float(-np.sum(w_clip * np.log(w_clip)))

        # Affinity shift: short vs medium routing mass difference
        aff_shift[i] = weights[0] - weights[1]

    df["rrm_affinity_shortterm_10d"]  = aff_short
    df["rrm_affinity_medterm_40d"]    = aff_med
    df["rrm_affinity_longterm_120d"]  = aff_long
    df["rrm_dominant_regime"]         = dom_regime
    df["rrm_routing_entropy"]         = rout_ent
    df["rrm_affinity_shift_10v40d"]   = aff_shift

    # Regime stability: rolling std of dominant_regime over 20d
    dom_s = pd.Series(dom_regime, index=df.index)
    df["rrm_regime_stability_20d"] = dom_s.rolling(20, min_periods=5).std()

    return df
