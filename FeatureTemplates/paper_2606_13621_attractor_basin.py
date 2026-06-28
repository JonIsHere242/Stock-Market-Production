"""
Attractor basin and defensibility features derived from:
  "Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis
   for Adversarial Networks"
  arXiv 2606.13621

The paper uses ATTRACTOR COMPUTATION and WINNING-REGION EXTRACTION from
two-player safety games to characterize how "defensible" a network topology is.
Key concepts: attractor (set of states the adversary can force the system into),
winning region (states from which the defender guarantees safety),
defensibility fingerprint (formal + operational security metrics).

We translate to OHLCV:
  - Price levels act as "states"; support/resistance zones are "safe regions".
  - The ATTRACTOR is the rolling price cluster center — approximated via
    kernel density mode over a trailing window (fast histogram approach).
  - "Defender" = buy-side support proximity (rolling low).
  - "Adversary" = sell-side pressure proximity (rolling high).
  - WINNING REGION metric: fraction of recent days price > attractor (price
    in the "defender-favorable" zone above the gravitational center).
  - DEFENSIBILITY = how easily price can be "defended" from downward pressure.
  - BASIN DEPTH = potential energy: (Close - attractor)^2 / rolling variance.

Feature family (7 cols):
  ab_attractor_dist       (Close - attractor) / ATR-like scale
  ab_support_dist_20      distance from rolling 20-day low / rolling range
  ab_resist_dist_20       distance from rolling 20-day high / rolling range
  ab_basin_depth_20       (Close - attractor)^2 / 20d variance
  ab_winning_region_20    fraction of past 20 days where price > attractor
  ab_defensibility_ratio  support_dist / resist_dist (defender vs attacker strength)
  ab_attractor_momentum   rate of change of attractor level (5-day diff / ATR)
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13621_attractor_basin",
    "description": (
        "Price-level attractor basin and defensibility metrics: rolling histogram-mode "
        "as attractor, winning region, and defender/adversary balance; "
        "from arXiv 2606.13621 (Shield Synthesis / Defensibility Analysis)."
    ),
    "requires": ["Close", "High", "Low"],
    "produces": [
        "ab_attractor_dist",
        "ab_support_dist_20",
        "ab_resist_dist_20",
        "ab_basin_depth_20",
        "ab_winning_region_20",
        "ab_defensibility_ratio",
        "ab_attractor_momentum",
    ],
    "tags": ["mean_reversion", "market_regime", "experimental"],
    "version": "1.1",
    "author": "paper:2606.13621",
}


def _histogram_mode(prices: np.ndarray, n_bins: int = 8) -> float:
    """
    Fast histogram-mode: price level with the highest frequency count.
    Uses n_bins equal-width bins; returns midpoint of the densest bin.
    O(m) — much faster than KDE.
    """
    mask = np.isfinite(prices)
    p = prices[mask]
    m = len(p)
    if m < 3:
        return np.nan
    pmin, pmax = p.min(), p.max()
    if pmax - pmin < 1e-10:
        return float(p.mean())
    counts, edges = np.histogram(p, bins=n_bins, range=(pmin, pmax))
    best_bin = int(np.argmax(counts))
    return float(0.5 * (edges[best_bin] + edges[best_bin + 1]))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(np.float64)
    high  = df["High"].values.astype(np.float64)
    low   = df["Low"].values.astype(np.float64)
    n = len(df)

    # ---- Rolling attractor via histogram mode ----------------------------------
    window = 30
    attractor = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = max(0, i - window + 1)
        attractor[i] = _histogram_mode(close[start: i + 1])

    attractor_s = pd.Series(attractor, index=df.index)
    close_s     = pd.Series(close, index=df.index)

    # ---- ATR proxy for scaling ------------------------------------------------
    atr_proxy = (
        close_s.rolling(14, min_periods=5).max()
        - close_s.rolling(14, min_periods=5).min()
    ).replace(0.0, np.nan)

    # ---- Attractor distance (mean-reversion signal) ---------------------------
    ab_attractor_dist = (close_s - attractor_s) / atr_proxy

    # ---- Support / resistance distances (fully vectorised) --------------------
    roll20_low  = pd.Series(low,  index=df.index).rolling(20, min_periods=5).min()
    roll20_high = pd.Series(high, index=df.index).rolling(20, min_periods=5).max()
    roll20_range = (roll20_high - roll20_low).replace(0.0, np.nan)

    ab_support_dist_20 = (close_s - roll20_low)  / roll20_range
    ab_resist_dist_20  = (roll20_high - close_s) / roll20_range

    # ---- Basin depth: (Close - attractor)^2 / 20d variance -------------------
    var20 = close_s.rolling(20, min_periods=5).var().replace(0.0, np.nan)
    ab_basin_depth_20 = (close_s - attractor_s) ** 2 / var20

    # ---- Winning region: fraction of past 20d where price > attractor ---------
    above_attractor = (close_s > attractor_s).astype(float)
    ab_winning_region_20 = above_attractor.rolling(20, min_periods=5).mean()

    # ---- Defensibility ratio: support_dist / resist_dist ----------------------
    ab_defensibility_ratio = (
        ab_support_dist_20 / ab_resist_dist_20.replace(0.0, np.nan)
    )

    # ---- Attractor momentum: 5-day change / ATR --------------------------------
    ab_attractor_momentum = attractor_s.diff(5) / atr_proxy

    # ---- Assign ---------------------------------------------------------------
    df["ab_attractor_dist"]      = ab_attractor_dist
    df["ab_support_dist_20"]     = ab_support_dist_20
    df["ab_resist_dist_20"]      = ab_resist_dist_20
    df["ab_basin_depth_20"]      = ab_basin_depth_20
    df["ab_winning_region_20"]   = ab_winning_region_20
    df["ab_defensibility_ratio"] = ab_defensibility_ratio
    df["ab_attractor_momentum"]  = ab_attractor_momentum

    return df
