"""
_p625_illiq_reversal.py - Illiquidity-conditioned short-horizon (5d) reversal pressure.

Avramov, Chordia & Goyenko (2006) JF 61(5):2365-2394 show short-term reversal profits are
concentrated in ILLIQUID stocks; Amihud (2002) JFM 5(1) gives the illiquidity measure
(|ret| / dollar-volume); Nagel (2012) RFS 25(7) frames reversal as compensation for liquidity
provision. We standardise a 5-day displacement by the stock's own short-run return vol, then
scale it by the stock's OWN trailing Amihud percentile (how illiquid it currently is relative to
its 252d history). The signed output is the implied next-day reversion pressure (more illiquid =>
stronger expected snap-back).

  daily_illiq        = |ret| / dollar_volume                         (Amihud, per day)
  illiq_pctile_252   = trailing 252d percentile-rank of daily_illiq  (own-history illiquidity)
  g_illiq            = clip( (pctile-0.5)/0.5, 0, 1 )                 (soft top-half hinge in [0,1])
  ret5               = 5-day return
  sd5                = 21d daily-return std * sqrt(5)                 (5d-scaled vol)
  rev_disp           = clip( ret5 / (sd5+1e-9), -5, 5 )              (vol-standardised displacement)
  ilrev_pressure     = clip( -rev_disp * g_illiq, -5, 5 )            (signed reversion pressure)
  ilrev_strength_60  = clip( rolling-mean_60( g_illiq*|rev_disp| ), 0, 5 )

Distinct from the _ilr block: that block uses a 40d log-price displacement gated by a 60d
inverse-DOLLAR-VOLUME thinness percentile; this block uses a 5d return displacement gated by a
252d AMIHUD percentile -- a different horizon AND a different (return-numerator) conditioner.
Strictly trailing / causal; all divisions guarded. Vectorised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_illiq_reversal",
    "description": "Illiquidity-conditioned short-horizon (5d) reversal: a vol-standardized 5d displacement scaled by the stock's own trailing 252d Amihud percentile, emitting signed reversion pressure (Avramov-Chordia-Goyenko 2006 JF 61(5); Amihud 2002 JFM 5(1); Nagel 2012 RFS 25(7)).",
    "requires":    [],
    "produces":    ["ilrev_pressure", "ilrev_strength_60", "ilrev_illiq_pctile_252"],
    "tags":        ["mean_reversion", "liquidity", "experimental"],
    "version":     "1.0",
    "author":      "paper:Avramov-Chordia-Goyenko (2006) JF 61(5):2365-2394; Amihud (2002) JFM 5(1); Nagel (2012) RFS 25(7)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    ret = close.pct_change()
    abs_ret = ret.abs()
    dvol = (close * vol).replace(0, np.nan)

    # Amihud daily illiquidity, guarded against inf
    daily_illiq = (abs_ret / dvol).replace([np.inf, -np.inf], np.nan)

    # Own-history trailing illiquidity percentile (strictly trailing)
    illiq_pctile_252 = daily_illiq.rolling(252, min_periods=120).rank(pct=True)

    # Soft top-half illiquidity hinge in [0, 1]
    g_illiq = ((illiq_pctile_252 - 0.5) / 0.5).clip(0, 1)

    # Vol-standardised 5d displacement
    ret5 = close.pct_change(5)
    sd5 = ret.rolling(21, min_periods=10).std() * np.sqrt(5)
    rev_disp = (ret5 / (sd5 + 1e-9)).clip(-5, 5)

    # Signed reversion pressure (more illiquid => stronger expected snap-back)
    df["ilrev_pressure"] = (-rev_disp * g_illiq).clip(-5, 5)

    # Conditioned displacement magnitude, smoothed
    df["ilrev_strength_60"] = (g_illiq * rev_disp.abs()).rolling(60, min_periods=20).mean().clip(0, 5)

    df["ilrev_illiq_pctile_252"] = illiq_pctile_252

    return df
