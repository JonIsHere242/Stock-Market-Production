import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_high_volume_premium",
    "description": "Gervais-Kaniel-Mingelgrin high-volume-return premium: extreme-rank volume events, recency, signed shock drift and dry-up coil (GKM 2001 JF; Kang arXiv:2512.14134).",
    "requires":    [],
    "produces":    [
        "hvd_volmult_20",
        "hvd_shock_intensity_20",
        "hvd_recency_50",
        "hvd_count_50",
        "hvd_shock_drift_10",
        "hvd_dryup_20",
    ],
    "tags":        ["volume", "event", "experimental"],
    "version":     "1.0",
    "author":      "paper:Gervais Kaniel Mingelgrin (2001) JF 56(3):877-919; Kaniel Ozoguz Starks (2012) JFE; Kang (2024-2025) arXiv:2512.14134",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # ---- Abnormal volume: v / trailing-20d mean EXCLUDING the current bar -----
    # mean over t-1..t-20 == (rolling-20 sum shifted by 1) / 20-equivalent count.
    prior_vol = vol.shift(1)
    mean20_excl = prior_vol.rolling(window=20, min_periods=20).mean()
    abn = vol / mean20_excl.replace(0.0, np.nan)
    abn = abn.replace([np.inf, -np.inf], np.nan)

    # ---- hvd_volmult_20: log abnormal volume, clipped ------------------------
    df["hvd_volmult_20"] = np.log(abn).clip(-3.0, 3.0)

    # ---- hvd_shock_intensity_20: excess abnormal volume above 5x -------------
    df["hvd_shock_intensity_20"] = (abn - 5.0).clip(lower=0.0)

    # ---- High-volume day flag: v == trailing-50d max (inclusive) -------------
    roll_max50 = vol.rolling(window=50, min_periods=30).max()
    is_hvd = (vol == roll_max50) & roll_max50.notna()

    # ---- hvd_recency_50: days since last HVD, clipped to [0,50], scaled ------
    # idx of bars; carry forward the index of the most recent HVD.
    n = len(df)
    pos = np.arange(n, dtype=float)
    hvd_pos = np.where(is_hvd.to_numpy(), pos, np.nan)
    last_hvd_pos = pd.Series(hvd_pos, index=df.index).ffill()
    days_since = pos - last_hvd_pos.to_numpy()  # NaN before any HVD has occurred
    df["hvd_recency_50"] = np.clip(days_since, 0.0, 50.0) / 50.0

    # ---- hvd_count_50: number of HVD events in trailing 50 bars --------------
    df["hvd_count_50"] = is_hvd.astype(float).rolling(window=50, min_periods=30).sum()

    # ---- hvd_shock_drift_10: count of big shocks (abn>5) last 10d, signed ----
    shock = (abn > 5.0).astype(float)
    shock_cnt10 = shock.rolling(window=10, min_periods=10).sum()
    ret10 = close / close.shift(10) - 1.0
    ret10 = ret10.replace([np.inf, -np.inf], np.nan)
    df["hvd_shock_drift_10"] = shock_cnt10 * np.sign(ret10)

    # ---- hvd_dryup_20: log ratio of recent short-MA volume to longer median --
    ma5 = vol.rolling(window=5, min_periods=5).mean()
    med50 = vol.rolling(window=50, min_periods=30).median()
    dryup = np.log((ma5 + 1.0) / (med50 + 1.0))
    df["hvd_dryup_20"] = dryup.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)

    return df
