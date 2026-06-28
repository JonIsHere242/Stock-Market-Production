import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_amivest_depth",
    "description": "Amivest liquidity ratio (dollar volume absorbed per unit absolute return on non-zero-return days) with trend and resilience z; the depth/resilience side Amihud-dense lakes ignore. Amihud, Mendelson & Lauterbach (1997) JFE 45(3); Goyenko, Holden & Trzcinka (2009) JFE 92(2):153-181.",
    "requires":    [],
    "produces":    ["amvst_ratio_21", "amvst_ratio_63", "amvst_trend_63", "amvst_resilience_z_60"],
    "tags":        ["liquidity", "volume", "experimental"],
    "version":     "1.0",
    "author":      "paper:Amihud-Mendelson-Lauterbach(1997) JFE 45(3); Goyenko-Holden-Trzcinka(2009) JFE 92(2):153-181",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # Daily Amivest = dollar volume absorbed per unit absolute return, only on
    # meaningfully non-zero-return days (zero/near-zero return days are excluded
    # to avoid divide-by-tiny explosions). Strictly trailing throughout.
    ret = close.pct_change()
    abs_ret = ret.abs()
    dvol = (close * vol).clip(lower=0.0)

    mask = abs_ret > 1e-4
    dv_eff = dvol.where(mask)
    ar_eff = abs_ret.where(mask)

    daily = (dv_eff / ar_eff).replace([np.inf, -np.inf], np.nan)

    # Rolling means of the daily Amivest ratio (trailing windows).
    roll21 = daily.rolling(21, min_periods=21 // 2).mean()
    roll63 = daily.rolling(63, min_periods=63 // 2).mean()
    roll60 = daily.rolling(60, min_periods=60 // 2)
    roll60_mean = roll60.mean()
    roll60_std = roll60.std()

    log21 = np.log1p(roll21)
    log63 = np.log1p(roll63)

    df["amvst_ratio_21"] = log21.clip(lower=0.0, upper=60.0)
    df["amvst_ratio_63"] = log63.clip(lower=0.0, upper=60.0)

    # Trend: short minus long log-Amivest (rising depth -> positive).
    df["amvst_trend_63"] = (log21 - log63).clip(lower=-10.0, upper=10.0)

    # Resilience z: how far the recent 21d depth sits above its own 60d level,
    # scaled by 60d dispersion. Guarded denominator.
    resilience = (roll21 - roll60_mean) / (roll60_std + 1e-9)
    df["amvst_resilience_z_60"] = resilience.replace([np.inf, -np.inf], np.nan).clip(lower=-6.0, upper=6.0)

    return df
