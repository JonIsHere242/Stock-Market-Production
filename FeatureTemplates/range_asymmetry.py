"""
range_asymmetry.py — Intraday range asymmetry (upside vs downside) (Tier-2).

Where range_volatility.py measures the SIZE of the intraday range, this block
measures its DIRECTIONAL SHAPE — how the bar's range is split between buying
pressure (close pushed toward the high) and selling pressure (close pushed toward
the low). The Rogers-Satchell estimator decomposes naturally into an upside leg
(h-c)(h-o) and a downside leg (l-c)(l-o); the ratio of the two is a clean
intraday demand-asymmetry signal. Asymmetric/positively-skewed intraday ranges
are the lottery-like names Bali-Cakici-Whitelaw and Boyer-Mitton-Vorkink show
under-perform — so this is a complementary Tier-2 tail discriminator built purely
from the bar geometry.

Per-bar pieces (raw prices):
    upside_range   = High - Close                    (room given back from the high)
    downside_range = Close - Low                     (recovery off the low)
    clpos          = (Close - Low)/(High - Low)      close location in the bar [0,1]
    rs_up = (h-c)(h-o), rs_dn = (l-c)(l-o)  on log prices (Rogers-Satchell legs)

Produces over 21d/63d windows:
    rng_hl_close_ratio_<w>  : rolling mean of (High-Close)/(Close-Low) -> >1 means
                              closes tend to sit nearer the LOW (weak/lottery-down)
    rng_close_loc_<w>       : rolling mean close-location-in-bar [0,1]
    rng_rs_up_share_<w>     : rs_up / (rs_up + rs_dn) rolling -> upside variation share
    rng_updown_vol_ratio_<w>: rolling vol of up-days / vol of down-days (range based)

Pure per-ticker, vectorised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "range_asymmetry",
    "description": "Intraday range asymmetry: High-Close vs Close-Low ratio, close-location, Rogers-Satchell upside share, and up/down range-vol ratio at 21d/63d.",
    "requires":    ["Open", "High", "Low", "Close"],
    "produces":    [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("rng_hl_close_ratio", "rng_close_loc",
                  "rng_rs_up_share", "rng_updown_vol_ratio")
    ],
    "tags":        ["volatility", "range", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Rogers-Satchell 1991 decomposition)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    O, H, L, C = df["Open"], df["High"], df["Low"], df["Close"]

    hl_span = (H - L).replace(0, np.nan)
    upside_range = (H - C)
    downside_range = (C - L).replace(0, np.nan)

    # (High-Close)/(Close-Low): >1 -> close sits nearer the low.
    hl_close = (upside_range / downside_range).clip(0, 20)
    # Close location in bar: 1 = closed on the high, 0 = closed on the low.
    clpos = ((C - L) / hl_span).clip(0, 1)

    # Rogers-Satchell legs on log prices.
    o = np.log(O.where(O > 0))
    h = np.log(H.where(H > 0))
    l = np.log(L.where(L > 0))
    c = np.log(C.where(C > 0))
    rs_up = (h - c) * (h - o)        # >= 0 by construction
    rs_dn = (l - c) * (l - o)        # >= 0 by construction

    # Range proxy per bar for up/down-day vol split. We accumulate the squared
    # range only on up- (resp. down-) days via masking to ZERO (not NaN) and
    # divide by the rolling COUNT of such days, so min_periods on the full
    # window still applies (a NaN-masked series would starve min_periods since
    # only ~half the window is non-NaN).
    park_d = (h - l) ** 2
    ret = C.pct_change()
    up_mask = (ret > 0).astype(float)
    dn_mask = (ret < 0).astype(float)
    park_up_sum = (park_d * up_mask)
    park_dn_sum = (park_d * dn_mask)

    for w, mp in _WINDOWS:
        df[f"rng_hl_close_ratio_{w}"] = (
            hl_close.rolling(w, min_periods=mp).mean().values
        )
        df[f"rng_close_loc_{w}"] = (
            clpos.rolling(w, min_periods=mp).mean().values
        )

        up_sum = rs_up.rolling(w, min_periods=mp).sum()
        dn_sum = rs_dn.rolling(w, min_periods=mp).sum()
        denom = (up_sum + dn_sum).replace(0, np.nan)
        df[f"rng_rs_up_share_{w}"] = (up_sum / denom).clip(0, 1).values

        up_cnt = up_mask.rolling(w, min_periods=mp).sum().replace(0, np.nan)
        dn_cnt = dn_mask.rolling(w, min_periods=mp).sum().replace(0, np.nan)
        mean_up = park_up_sum.rolling(w, min_periods=mp).sum() / up_cnt
        mean_dn = park_dn_sum.rolling(w, min_periods=mp).sum() / dn_cnt
        vol_up = np.sqrt(mean_up.clip(lower=0))
        vol_dn = np.sqrt(mean_dn.clip(lower=0))
        df[f"rng_updown_vol_ratio_{w}"] = (
            (vol_up / vol_dn.replace(0, np.nan)).clip(0, 20).values
        )

    return df
