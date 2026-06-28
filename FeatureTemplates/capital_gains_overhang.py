"""
capital_gains_overhang.py - Grinblatt-Han unrealized capital-gains overhang (Tier-2).

Grinblatt & Han (2005, JFE) "Prarefactory disposition effect and momentum". Investors
are reluctant to realize losses and quick to realize gains, so a stock trading ABOVE its
aggregate cost basis faces underreaction-driven upward drift (holders won't sell winners
cheaply) and one BELOW faces overhang. The reference price RP is a turnover-weighted
average of past prices, where each past day's weight is its turnover times the survival
probability that those shares were NOT retraded since:

    RP_t = sum_i [ V_{t-i} * prod_{j<i}(1 - V_{t-j}) * P_{t-i} ] / (normalizer)
    CGO_t = (P_t - RP_t) / P_t      (>0 = embedded gain -> positive expected drift)

This is a SIGNED price-vs-anchor distance: ~0 at the basis, and non-monotone in recent
return magnitude (a name can be far above basis on a slow grind or near it after a violent
round-trip), so it is volatility-orthogonal and distinct from 52-week-high anchoring. We
lack shares outstanding, so turnover V is proxied from volume. Emitted as the raw CGO, a
within-ticker time-series z-score (the proven `_tsz` discrimination lever), and a no-survival
volume-weighted-price-distance twin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_L = 252          # cost-basis lookback (trading days)
_TURN = 0.005     # nominal daily turnover scale applied to relative volume
_TSZ = 252        # window for the within-ticker z-score

METADATA = {
    "name":        "capital_gains_overhang",
    "description": "Grinblatt-Han (2005) unrealized capital-gains overhang: turnover-survival-weighted reference price vs current price, as raw CGO, a 252d within-ticker z-score, and a no-survival volume-weighted-price-distance twin.",
    "requires":    ["Close", "Volume"],
    "produces":    ["xdm_cgo", "xdm_cgo_tsz", "xdm_vwap_dist"],
    "tags":        ["behavioral", "mean_reversion", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (Grinblatt-Han 2005)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    c = close.values

    cgo = np.full(n, np.nan)
    vwdist = np.full(n, np.nan)

    if n >= _L + 1:
        volma = vol.rolling(_L, min_periods=_L // 2).mean()
        tn = (vol / volma.replace(0, np.nan) * _TURN).clip(0, 0.2).fillna(0.0).values
        g = 1.0 - tn                                   # per-day survival (in [0.8, 1])

        cw = sliding_window_view(c, _L)                # (m, L): window k covers days k..k+L-1
        tnw = sliding_window_view(tn, _L)
        gw = sliding_window_view(g, _L)
        vw = sliding_window_view(vol.values, _L)

        rev = np.cumprod(gw[:, ::-1], axis=1)[:, ::-1]  # rev[:,j] = prod(gw[:, j:])
        surv_excl = rev / gw                            # prod(gw[:, j+1:])  (gw>=~0.8, safe)
        wgt = tnw * surv_excl
        wsum = wgt.sum(1)
        rp = (wgt * cw).sum(1) / np.where(wsum > 0, wsum, np.nan)

        vsum = vw.sum(1)
        vwap = (vw * cw).sum(1) / np.where(vsum > 0, vsum, np.nan)

        # output day t (t>=L) uses the window covering days t-L..t-1 -> sliding row index t-L
        rp_full = np.full(n, np.nan); rp_full[_L:] = rp[: n - _L]
        vw_full = np.full(n, np.nan); vw_full[_L:] = vwap[: n - _L]

        cgo = (c - rp_full) / np.where(c != 0, c, np.nan)
        vwdist = (c - vw_full) / np.where(vw_full != 0, vw_full, np.nan)

    df["xdm_cgo"] = np.clip(cgo, -1, 1)
    df["xdm_vwap_dist"] = np.clip(vwdist, -1, 1)

    cgo_s = pd.Series(cgo, index=df.index)
    mu = cgo_s.rolling(_TSZ, min_periods=120).mean()
    sd = cgo_s.rolling(_TSZ, min_periods=120).std().replace(0, np.nan)
    df["xdm_cgo_tsz"] = np.clip(((cgo_s - mu) / sd).values, -4, 4)

    return df
