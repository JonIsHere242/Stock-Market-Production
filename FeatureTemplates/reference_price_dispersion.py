"""
reference_price_dispersion.py - Cost-basis overhang HETEROGENEITY (Tier-2).

Capital-gains overhang (Grinblatt-Han) compresses the whole cost-basis distribution to a
single reference price. But the SHAPE of that distribution carries extra, orthogonal
information: when holders bought at WIDELY DIFFERENT prices there is more latent disagreement
and more price levels at which disposition-driven supply/demand will trigger -- the
"heterogeneous overhang" idea (cf. Grinblatt-Han; Frazzini 2006 disposition; volume-profile
support/resistance). We reuse the turnover-survival weighting to estimate the trailing
distribution of holders' purchase prices and read its DISPERSION and GAIN/LOSS BALANCE:

  rdd_ref_disp     : turnover-survival-weighted dispersion of cost basis / price (disagreement)
  rdd_gain_frac    : weighted fraction of holders sitting in a gain (overhang balance; centred)
  rdd_overhang_skew: weighted skew of (cost basis - price) -- few far-underwater vs many small

These are signed/standardised shape statistics of a reference-relative distribution, ~0 at a
tight consensus basis and non-monotone in recent move size, so volatility-orthogonal and
distinct from the CGO level. Vectorised via the same strided survival-weight matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_L = 252
_TURN = 0.005

METADATA = {
    "name":        "reference_price_dispersion",
    "description": "Heterogeneity of the turnover-survival-weighted cost-basis distribution: weighted dispersion (holder disagreement), weighted gain fraction (overhang balance), and weighted skew of cost-basis-minus-price. Extends Grinblatt-Han CGO to the distribution shape.",
    "requires":    ["Close", "Volume"],
    "produces":    ["rdd_ref_disp", "rdd_overhang_skew"],
    "tags":        ["behavioral", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 behavioral build (Grinblatt-Han overhang heterogeneity)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    c = df["Close"].astype(float).values
    vol = df["Volume"].astype(float)

    disp = np.full(n, np.nan); oskew = np.full(n, np.nan)

    if n >= _L + 1:
        volma = vol.rolling(_L, min_periods=_L // 2).mean()
        tn = (vol / volma.replace(0, np.nan) * _TURN).clip(0, 0.2).fillna(0.0).values
        g = 1.0 - tn

        cw = sliding_window_view(c, _L)                    # row k covers days k..k+L-1
        tnw = sliding_window_view(tn, _L)
        gw = sliding_window_view(g, _L)
        rev = np.cumprod(gw[:, ::-1], axis=1)[:, ::-1]
        surv = rev / gw
        w = tnw * surv
        wsum = np.where(w.sum(1) > 0, w.sum(1), np.nan)
        wn = w / wsum[:, None]                             # normalized cost-basis weights

        # output day t (t>=L) uses window row j=t-L (covers t-L..t-1); current price = c[t]
        use = n - _L
        wn_u, cw_u = wn[:use], cw[:use]
        cur = c[_L:]                                        # cur[j] = price at output day t=L+j
        rp = (wn_u * cw_u).sum(1)                           # weighted reference price
        var = (wn_u * (cw_u - rp[:, None]) ** 2).sum(1)
        sd = np.sqrt(np.clip(var, 0, None))
        m3 = (wn_u * (cw_u - rp[:, None]) ** 3).sum(1)
        sk = m3 / np.power(np.where(sd > 0, sd, np.nan), 3)

        disp[_L:] = sd / np.where(np.abs(cur) > 0, np.abs(cur), np.nan)
        oskew[_L:] = sk

    df["rdd_ref_disp"] = np.clip(disp, 0, 2)
    df["rdd_overhang_skew"] = np.clip(oskew, -5, 5)
    return df
