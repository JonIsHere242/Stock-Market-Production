"""
informed_drift.py - Post-event drift / under-reaction proxies (Tier-2).

Bernard & Thomas (1989) post-earnings-announcement drift and the gap-continuation
literature (e.g. the "gap-and-go vs gap-fill" effect): prices UNDER-react to genuine
information, so moves that happen on abnormal VOLUME or as overnight GAPS tend to keep
drifting in the same direction. We have no earnings dates, so we proxy the "information
event" from price/volume structure alone -- and deliberately key off RELATIVE volume and
SIGN, not return magnitude, to stay off the volatility tail.

  csa_pead_score          : EWM(span 10) of sign(ret) * excess relative volume
                            -- recency-weighted "informed flow" drift score. Positive when
                            recent up-moves came on above-average volume (good-news drift).
  csa_gap_cont_60         : rolling 60d correlation between the overnight gap and the
                            following intraday move. >0 = this name is a gap-and-go
                            (continuation) stock; <0 = a gap-fill (reversal) stock.
                            A behavioral fingerprint of the name.
  csa_gap_drift           : today's overnight gap * csa_gap_cont_60 -- the directional
                            drift the name's own gap behavior implies for the next session.

Per-ticker, fully vectorized. csa_gap_drift uses today's open vs prior close, fully
observed by the bar's close -- no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "informed_drift",
    "description": "Post-event under-reaction drift proxies: EWM informed-flow score from sign(ret)*excess relative volume, 60d gap-continuation (gap-and-go vs gap-fill) correlation, and the implied next-session gap drift. Bernard-Thomas PEAD / gap-continuation literature.",
    "requires":    ["Open", "Close", "Volume"],
    "produces":    ["csa_pead_score", "csa_gap_cont_60", "csa_gap_drift"],
    "tags":        ["momentum", "volume", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Bernard-Thomas 1989 / gap continuation)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    openp = df["Open"]
    vol = df["Volume"].astype(float)

    ret = np.log(close / close.shift(1))

    # Excess relative volume: how far above its own 20d average today's volume is.
    vol_ma = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
    excess_relvol = (vol / vol_ma - 1.0).clip(lower=0, upper=4)   # only above-average days count

    # Informed-flow drift: recency-weighted signed move on abnormal volume.
    informed = np.sign(ret) * excess_relvol
    df["csa_pead_score"] = informed.ewm(span=10, min_periods=10).mean().clip(-4, 4).values

    # Gap (overnight) vs intraday continuation fingerprint.
    prev_close = close.shift(1).replace(0, np.nan)
    gap = np.log(openp.replace(0, np.nan) / prev_close)
    intraday = np.log(close / openp.replace(0, np.nan))
    gap_cont = gap.rolling(60, min_periods=40).corr(intraday)
    df["csa_gap_cont_60"] = gap_cont.clip(-1, 1).values

    # Directional drift implied by today's gap given the name's gap behavior.
    df["csa_gap_drift"] = (gap * gap_cont).clip(-0.5, 0.5).values

    return df
