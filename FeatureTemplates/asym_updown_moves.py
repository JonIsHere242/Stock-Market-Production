"""
asym_updown_moves.py — Up/down move asymmetry of recent returns (Tier-2).

Up- vs down-move asymmetry is a classic discriminator of names about to lead.
Two sound constructs:
  * Gain/Pain ratio (Schwager, "Hedge Fund Market Wizards", 2012): the ratio of
    summed gains to the absolute value of summed losses over a window — a robust
    payoff-asymmetry measure that is monotone in the right thing without assuming
    Gaussianity.
  * Semivariance split (Markowitz 1959; Sortino & Price 1994): the share of total
    return variation contributed by up days vs down days. A name whose recent
    variance is dominated by UP-day moves has a different forward distribution
    than one whose variance is downside-driven.

Over a rolling window W of daily simple returns r (g = max(r,0), l = min(r,0)):
    gain_pain   = Σg / |Σl|                       payoff asymmetry  (>1 = up-heavy)
    upmove_size = mean(g | r>0) / mean(|l| | r<0)  avg up vs avg down move size
    up_var_frac = Σg² / (Σg² + Σl²) in [0,1]       semivariance share that is upside
    var_asym    = (Σg² - Σl²)/(Σg² + Σl²) in [-1,1] signed variance asymmetry

Pure per-ticker, vectorised. Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "asym_updown_moves",
    "description": "Up/down move asymmetry of rolling daily returns: gain/pain ratio, avg up-vs-down move size, and upside semivariance share/asymmetry at 21d/63d (Schwager gain-pain; Markowitz/Sortino semivariance).",
    "requires":    ["Close"],
    "produces":    [
        f"asy_{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("gainpain", "upmovesize", "upvarfrac", "varasym")
    ],
    "tags":        ["volatility", "tail", "distributional", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 dist-shape build (Schwager / Markowitz-Sortino)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change()
    g = r.clip(lower=0)            # gains (>=0)
    l = (-r).clip(lower=0)         # losses as positive magnitudes (>=0)
    up = (r > 0).astype(float)
    dn = (r < 0).astype(float)

    for w, mp in _WINDOWS:
        sum_g = g.rolling(w, min_periods=mp).sum()
        sum_l = l.rolling(w, min_periods=mp).sum()
        n_up = up.rolling(w, min_periods=mp).sum()
        n_dn = dn.rolling(w, min_periods=mp).sum()
        s_gp = (g ** 2).rolling(w, min_periods=mp).sum()
        s_ln = (l ** 2).rolling(w, min_periods=mp).sum()

        # mean up-move size and mean down-move size (conditional on direction)
        mean_up = sum_g / n_up.replace(0, np.nan)
        mean_dn = sum_l / n_dn.replace(0, np.nan)

        denom_var = (s_gp + s_ln).replace(0, np.nan)

        df[f"asy_gainpain_{w}"] = (sum_g / sum_l.clip(lower=1e-9)).clip(0, 50).values
        df[f"asy_upmovesize_{w}"] = (mean_up / mean_dn.clip(lower=1e-9)).clip(0, 50).values
        df[f"asy_upvarfrac_{w}"] = (s_gp / denom_var).values
        df[f"asy_varasym_{w}"] = ((s_gp - s_ln) / denom_var).clip(-1, 1).values

    return df
