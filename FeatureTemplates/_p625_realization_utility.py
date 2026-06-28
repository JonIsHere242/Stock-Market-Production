"""
_p625_realization_utility.py - Realization-utility FLOW (Barberis-Xiong).

Barberis & Xiong (2012, JFE) "Realization utility"; Frydman, Barberis, Camerer,
Bossaerts & Rangel (2014, JF) "Using neural data to test a theory of investor
behavior". Investors derive a BURST of utility at the MOMENT a position is
realized, and that realized utility is concave over gains and convex/steeper
over losses (a CPT-style kink at the cost basis). Realization is gated by
trading activity: a name only generates realization-utility flow when shares
actually turn over. So the flow is rel_turn * u(g), where g is the embedded
gain/loss vs a Grinblatt-Han turnover-survival-weighted reference price RP and
u(.) is a kinked power utility (lambda=2.25 on the loss side, exponent 0.88).

This is a TIME-VARYING FLOW of disposition-driven realized utility, distinct
from (a) the static overhang LEVEL (CGO, which is just g) and (b) the CPT value
of raw returns. We emit a 63d smoothed total flow, a 21d gain-only realization
flow, and a 63d gain-minus-loss kink asymmetry. Turnover V is proxied from
volume (no shares outstanding). All windows are trailing => strictly causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_L = 252          # cost-basis lookback (trading days), as in capital_gains_overhang
_TURN = 0.005     # nominal daily turnover scale applied to relative volume
_ALPHA = 0.88     # CPT power exponent
_LAMBDA = 2.25    # CPT loss aversion

METADATA = {
    "name":        "_p625_realization_utility",
    "description": "Realization-utility flow (Barberis-Xiong 2012 JFE; Frydman et al 2014 JF): volume-gated kinked-power utility of embedded gain vs Grinblatt-Han reference price, as 63d total flow, 21d gain-realization flow, and 63d gain-loss kink asymmetry.",
    "requires":    [],
    "produces":    ["rzu_flow_63", "rzu_gain_realize_21", "rzu_kink_asym_63"],
    "tags":        ["behavioral", "mean_reversion", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "paper:Barberis & Xiong (2012) JFE; Frydman et al. (2014) JF",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    c = close.values

    g = np.full(n, np.nan)   # embedded gain/loss vs reference price

    if n >= _L + 1:
        # ---- Grinblatt-Han turnover-survival-weighted reference price RP ----
        volma = vol.rolling(_L, min_periods=_L // 2).mean()
        tn = (vol / volma.replace(0, np.nan) * _TURN).clip(0, 0.2).fillna(0.0).values
        surv = 1.0 - tn                                # per-day survival (in [0.8, 1])

        cw = sliding_window_view(c, _L)                # (m, L): window k covers days k..k+L-1
        tnw = sliding_window_view(tn, _L)
        gw = sliding_window_view(surv, _L)

        rev = np.cumprod(gw[:, ::-1], axis=1)[:, ::-1]  # rev[:,j] = prod(gw[:, j:])
        surv_excl = rev / gw                            # prod(gw[:, j+1:])  (gw>=~0.8, safe)
        wgt = tnw * surv_excl
        wsum = wgt.sum(1)
        rp = (wgt * cw).sum(1) / np.where(wsum > 0, wsum, np.nan)

        # output day t (t>=L) uses the window covering days t-L..t-1 -> sliding row index t-L
        rp_full = np.full(n, np.nan)
        rp_full[_L:] = rp[: n - _L]

        g = (c - rp_full) / np.where(rp_full != 0, rp_full, np.nan)

    # ---- kinked CPT power utility u(g) = sign(g)*|g|^0.88, loss side * lambda ----
    g_s = pd.Series(g, index=df.index)
    absg = np.abs(g)
    u = np.sign(g) * np.power(absg, _ALPHA)            # nan-safe: nan propagates
    u = np.where(g < 0, u * _LAMBDA, u)                # steepen losses
    u = pd.Series(u, index=df.index)

    # ---- volume gating: relative turnover clipped to [0, 5] ----
    sma21_vol = vol.rolling(21, min_periods=21).mean()
    rel_turn = (vol / sma21_vol.replace(0, np.nan)).clip(0, 5)

    gain_mask = (g_s > 0).astype(float)
    loss_mask = (g_s < 0).astype(float)

    flow = rel_turn * u                                # signed realization-utility flow
    gain_flow = rel_turn * u * gain_mask               # gain-side flow only (>=0)
    loss_flow = rel_turn * (-u) * loss_mask            # loss-side magnitude (>=0)

    df["rzu_flow_63"] = np.clip(
        flow.rolling(63, min_periods=63).mean().values, -3, 3
    )
    df["rzu_gain_realize_21"] = np.clip(
        gain_flow.rolling(21, min_periods=21).mean().values, -3, 3
    )
    df["rzu_kink_asym_63"] = np.clip(
        (gain_flow.rolling(63, min_periods=63).mean()
         - loss_flow.rolling(63, min_periods=63).mean()).values, -3, 3
    )

    return df
