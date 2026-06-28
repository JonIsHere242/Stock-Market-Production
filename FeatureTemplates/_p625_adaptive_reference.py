import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_adaptive_reference",
    "description": "Adaptive reference point: asymmetric EWMA anchor (fast adaptation in gains, slow in losses), signed distance, and rolling above-anchor regime share. Arkes, Hirshleifer, Jiang & Lim (2008) OBHDP; Baucells, Weber & Welfens (2011) Mgmt Sci; Wang, Yan & Yu (2017).",
    "requires":    [],
    "produces":    ["arp_dist", "arp_above_frac_63", "arp_anchor_gap_252"],
    "tags":        ["behavioral", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper:Arkes-Hirshleifer-Jiang-Lim 2008 OBHDP; Baucells-Weber-Welfens 2011 Mgmt Sci; Wang-Yan-Yu 2017",
}

# Asymmetric adaptation speeds (EWMA alphas).
_ALPHA_GAIN = 2.0 / 21.0    # fast update toward new highs in gains
_ALPHA_LOSS = 2.0 / 121.0   # slow update in losses


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float).to_numpy()
    n = close.shape[0]

    rp = np.full(n, np.nan, dtype="float64")
    if n > 0:
        # Causal recursion: RP_0 = Close_0; alpha depends ONLY on RP_{t-1} (past info).
        prev = close[0]
        rp[0] = prev
        for i in range(1, n):
            c = close[i]
            if not np.isfinite(prev):
                # carry through a NaN seed without contaminating with current close
                prev = c if np.isfinite(c) else prev
                rp[i] = prev
                continue
            if not np.isfinite(c):
                rp[i] = prev
                continue
            alpha = _ALPHA_GAIN if c >= prev else _ALPHA_LOSS
            prev = prev + alpha * (c - prev)
            rp[i] = prev

    rp_s = pd.Series(rp, index=df.index)

    # arp_dist = clip((Close - RP) / RP, -1, 1); guard RP == 0 -> NaN.
    denom_rp = rp_s.replace(0.0, np.nan)
    df["arp_dist"] = ((df["Close"].astype(float) - rp_s) / denom_rp).clip(-1.0, 1.0)

    # arp_above_frac_63 = SMA63( 1[Close > RP] ) - 0.5 . Rolling mean is causal.
    above = (df["Close"].astype(float) > rp_s).astype("float64")
    df["arp_above_frac_63"] = above.rolling(window=63, min_periods=63).mean() - 0.5

    # arp_anchor_gap_252 = clip((RP - SMA252(Close)) / SMA252(Close), -1, 1); guard SMA == 0.
    sma252 = df["Close"].astype(float).rolling(window=252, min_periods=252).mean()
    denom_sma = sma252.replace(0.0, np.nan)
    df["arp_anchor_gap_252"] = ((rp_s - sma252) / denom_sma).clip(-1.0, 1.0)

    return df
