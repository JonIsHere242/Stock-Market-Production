"""
loss_aversion_overhang.py - Loss-aversion VALUE of the unrealized-P&L distribution (Tier-2).

The screen's single best feature was the cumulative-prospect-theory LOSS component over the
return distribution. This applies the same loss-aversion value function to a different, more
structural reference object: the distribution of holders' UNREALIZED gains/losses. Using the
turnover-survival cost-basis weights (Grinblatt-Han), each past purchase carries an unrealized
return g = P_today/P_basis - 1; a loss-averse investor values it through v(g) = g^a (g>=0),
-lambda*(-g)^a (g<0) with a=0.88, lambda=2.25. The aggregate is the prospect value of the
overhang -- dominated by underwater holders (loss aversion magnifies the pain), who form
sticky overhead supply.

  lax_overhang_pt : sum_i w_i v(g_i)   -- loss-averse prospect value of the overhang
  lax_loss_mass   : sum_i w_i (-g_i)[g_i<0]  -- weighted unrealized-loss mass (trapped sellers)
  lax_gl_asym     : gain mass - loss mass    -- net realized-supply asymmetry

A loss-averse aggregation of a reference-relative distribution (non-monotone), so distinct
from the CGO level and volatility-orthogonal. Vectorised via the strided survival-weight matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_L = 252
_TURN = 0.005
_A = 0.88
_LAM = 2.25

METADATA = {
    "name":        "loss_aversion_overhang",
    "description": "Loss-aversion (Tversky-Kahneman) value function applied to the turnover-survival-weighted distribution of holders' unrealized gains/losses: prospect value of the overhang, weighted unrealized-loss mass (trapped sellers), and gain/loss supply asymmetry. Extends the screen-winning CPT-loss idea to the cost-basis distribution.",
    "requires":    ["Close", "Volume"],
    "produces":    ["lax_overhang_pt", "lax_loss_mass", "lax_gl_asym"],
    "tags":        ["behavioral", "mean_reversion", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 behavioral build (loss aversion on cost-basis distribution)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    c = df["Close"].astype(float).values
    vol = df["Volume"].astype(float)

    pv = np.full(n, np.nan); lm = np.full(n, np.nan); gl = np.full(n, np.nan)

    if n >= _L + 1:
        volma = vol.rolling(_L, min_periods=_L // 2).mean()
        tn = (vol / volma.replace(0, np.nan) * _TURN).clip(0, 0.2).fillna(0.0).values
        g_surv = 1.0 - tn

        cw = sliding_window_view(c, _L)
        tnw = sliding_window_view(tn, _L)
        gw = sliding_window_view(g_surv, _L)
        rev = np.cumprod(gw[:, ::-1], axis=1)[:, ::-1]
        w = tnw * (rev / gw)
        wsum = np.where(w.sum(1) > 0, w.sum(1), np.nan)
        wn = w / wsum[:, None]

        use = n - _L
        wn_u, cw_u = wn[:use], cw[:use]
        cur = c[_L:]
        g = (cur[:, None] / cw_u - 1.0).clip(-0.95, 5.0)          # unrealized return per basis
        v = np.where(g >= 0, np.power(np.clip(g, 0, None), _A),
                     -_LAM * np.power(np.clip(-g, 0, None), _A))
        loss = g < 0
        pv[_L:] = (wn_u * v).sum(1)
        lm[_L:] = (wn_u * (-g) * loss).sum(1)
        gl[_L:] = (wn_u * g * (g > 0)).sum(1) - (wn_u * (-g) * loss).sum(1)

    df["lax_overhang_pt"] = np.clip(pv, -3, 3)
    df["lax_loss_mass"] = np.clip(lm, 0, 2)
    df["lax_gl_asym"] = np.clip(gl, -2, 2)
    return df
