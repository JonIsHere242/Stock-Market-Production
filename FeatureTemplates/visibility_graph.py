import numpy as np
import networkx as nx
import pandas as pd
from ts2vg import NaturalVG

METADATA = {
    "name":        "visibility_graph",
    "description": "Natural Visibility Graph path-length asymmetry between bullish and bearish price structure",
    "requires":    ["Close", "Volume"],
    "produces":    [
        "vg_pos", "vg_neg", "vg_diff", "vg_abs_diff",
        "volume_trend", "volume_trend_signal",
        "vg_long_signal", "vg_short_signal", "vg_combined_signal",
    ],
    "tags":        ["momentum", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.add_vg_indicators",
}

_LOOKBACK               = 12
_VOLUME_TREND_THRESHOLD = 0.70
_VOLUME_LOOKBACK        = 50


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].fillna(0).to_numpy(dtype=float)
    n      = len(close)

    # ---- detrend close prices for stationarity --------------------------------
    log_prices = np.log(close)
    x          = np.arange(n, dtype=float)
    coeffs     = np.polyfit(x, log_prices, 1)
    detrended  = log_prices - (coeffs[0] * x + coeffs[1])

    # ---- volume trend correlation (rolling 50-bar) ----------------------------
    vol_trend  = np.full(n, np.nan)
    vol_signal = np.ones(n, dtype=float)
    time_idx   = np.arange(_VOLUME_LOOKBACK + 1, dtype=float)

    for i in range(_VOLUME_LOOKBACK, n):
        window = volume[i - _VOLUME_LOOKBACK : i + 1]
        if window.std() > 0:
            corr = np.corrcoef(time_idx, window)[0, 1]
            vol_trend[i] = corr
            if corr > _VOLUME_TREND_THRESHOLD:
                vol_signal[i] = 0.0

    # ---- visibility graph path lengths ----------------------------------------
    pos_path = np.full(n, np.nan)
    neg_path = np.full(n, np.nan)

    for i in range(_LOOKBACK, n):
        window = detrended[i - _LOOKBACK + 1 : i + 1]
        try:
            pos_vg = NaturalVG()
            pos_vg.build(window)
            pos_path[i] = nx.average_shortest_path_length(pos_vg.as_networkx())

            neg_vg = NaturalVG()
            neg_vg.build(-window)
            neg_path[i] = nx.average_shortest_path_length(neg_vg.as_networkx())
        except Exception:
            pass  # leaves NaN

    # ---- assemble columns -----------------------------------------------------
    long_mask  = (pos_path > neg_path) & (vol_signal == 1)
    short_mask = (pos_path < neg_path) & (vol_signal == 1)
    vg_long    = long_mask.astype(float)
    vg_short   = np.where(short_mask, -1.0, 0.0)

    new_cols = {
        "vg_pos":              pos_path,
        "vg_neg":              neg_path,
        "vg_diff":             pos_path - neg_path,
        "vg_abs_diff":         np.abs(pos_path - neg_path),
        "volume_trend":        vol_trend,
        "volume_trend_signal": vol_signal,
        "vg_long_signal":      vg_long,
        "vg_short_signal":     vg_short,
        "vg_combined_signal":  vg_long + vg_short,
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
