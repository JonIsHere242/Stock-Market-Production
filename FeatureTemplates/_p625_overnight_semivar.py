import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_overnight_semivar",
    "description": "Overnight (close-to-open) signed gap semivariance, downside-gap share and large-gap intensity (BNKS 2010; Bondarenko-Bernard 2021; Patton-Sheppard 2015)",
    "requires":    [],
    "produces":    [
        "osv_gap_signed_21", "osv_gap_dnshare_21", "osv_gap_intensity_21",
        "osv_gap_signed_63", "osv_gap_dnshare_63", "osv_gap_intensity_63",
    ],
    "tags":        ["volatility", "overnight", "semivariance", "experimental"],
    "version":     "1.0",
    "author":      "paper:Barndorff-Nielsen,Kinnebrock&Shephard(2010);Bondarenko&Bernard(2021);Patton&Sheppard(2015)",
}

_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    # Overnight gap return: today's open vs yesterday's close.
    # Both are known pre-decision (open is observable at the decision bar,
    # prior close is strictly past) -> causal.
    prev_close = close.shift(1)
    g = open_ / prev_close.replace(0.0, np.nan) - 1.0

    gp2 = g.clip(lower=0.0) ** 2          # positive (good) gap squared
    gn2 = g.clip(upper=0.0) ** 2          # negative (bad) gap squared
    abs_g = g.abs()
    prev_g = g.shift(1)                    # strictly past gaps for the threshold

    for W, mp in ((21, 15), (63, 40)):
        rv_p = gp2.rolling(W, min_periods=mp).sum()
        rv_n = gn2.rolling(W, min_periods=mp).sum()
        rv_g = rv_p + rv_n
        denom = rv_g + _EPS

        df[f"osv_gap_signed_{W}"] = (rv_p - rv_n) / denom        # in [-1, 1]
        df[f"osv_gap_dnshare_{W}"] = rv_n / denom                # in [0, 1]

        # Large-gap intensity: fraction of recent gaps exceeding 2.5x the
        # trailing std of (strictly past) gaps. std uses prev_g so the current
        # gap never enters its own threshold.
        thr = 2.5 * prev_g.rolling(W, min_periods=mp).std()
        exceed = (abs_g > thr).astype(float)
        # rows where the threshold is undefined cannot count as an exceedance
        exceed = exceed.where(thr.notna(), np.nan)
        df[f"osv_gap_intensity_{W}"] = exceed.rolling(W, min_periods=mp).mean()

    return df
