import numpy as np
import pandas as pd

METADATA = {
    "name":        "significant_indicators",
    "description": "High-Close-ratio / RSI / Lyapunov regime composite ('significant indicators') ensemble ported from 3__AlphaSensitivity.calculate_significant_indicators",
    "requires":    ["Close", "High"],
    "produces":    [
        "hc_ratio",
        "pattern_indicator",
        "rsi",
        "rsi_hc_composite",
        "rsi_hc_composite_norm",
        "lyapunov_scaled",
        "hc_predict_regime",
        "hc_predict_regime_norm",
        "pattern_indicator_norm",
        "hc_ratio_norm",
        "hc_gp_composite",
        "significant_indicators_ensemble",
    ],
    "tags":        ["momentum", "mean_reversion", "gp", "experimental"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.calculate_significant_indicators",
}

_EPSILON        = 1e-10
_MIN_MAX_WINDOW = 10


def _norm(series: pd.Series) -> pd.Series:
    """Source normalization: (x - x.shift(1).rolling(50).mean()) / x.shift(1).rolling(50).std(), clipped to [-3, 3]."""
    shifted = series.shift(1)
    out = (series - shifted.rolling(50).mean()) / shifted.rolling(50).std()
    return out.clip(-3, 3)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- high-close ratio (always computed here as hc_ratio) ------------------
    hc_ratio = (df["High"] - df["Close"]) / (df["Close"] + _EPSILON)

    n = len(df)
    close_values = df["Close"].to_numpy(dtype=float)

    # ---- distance from nearest historical peak / trough (causal) --------------
    # df is ONE ticker. The historical-window argrelextrema causal logic is
    # preserved exactly: 50-bar history window max(0, i-50):i+1, peaks/troughs
    # detected with order=min_max_window, and the `< i` guard so the current
    # bar is never counted as its own peak/trough.
    #
    # Equivalence note (vectorized replacement for the per-bar double
    # argrelextrema call): scipy's argrelextrema(order=W, mode='clip') marks a
    # window-local index j as a maximum iff data[j] is strictly greater than
    # take(j+/-s, clip) for s=1..W. With mode='clip' the out-of-window neighbors
    # collapse onto the window's boundary values (already inside the window
    # range), so they never change the strict-max test -- EXCEPT at the window's
    # leftmost local index, where the negative-shift take clamps onto the point
    # itself (data[0] > data[0] is False), so the window's left-edge bar can
    # never be a peak/trough. We therefore find, for each bar i, the LAST index
    # g in (w, i) -- w = max(0, i-50), g strictly > w -- whose Close is strictly
    # greater (peak) / less (trough) than every Close over the clamped neighbor
    # span [max(w, g-W), min(i, g+W)]. The inner search walks g downward from
    # i-1 and stops at the first hit (the most recent extremum), matching the
    # source's local_*_df_indices[-1] selection bit-for-bat.
    distance_from_peak = np.zeros(n)
    distance_from_trough = np.zeros(n)
    W = _MIN_MAX_WINDOW

    for i in range(W, n):
        w = i - 50 if i - 50 > 0 else 0
        current_price = close_values[i]

        peak_idx = -1
        for g in range(i - 1, w, -1):
            cg = close_values[g]
            left_start = g - W if g - W > w else w
            right_end = i if g + W > i else g + W
            left = close_values[left_start:g]
            right = close_values[g + 1:right_end + 1]
            if (left.size == 0 or cg > left.max()) and (right.size == 0 or cg > right.max()):
                peak_idx = g
                break
        if peak_idx >= 0:
            peak_price = close_values[peak_idx]
            distance_from_peak[i] = (current_price - peak_price) / peak_price

        trough_idx = -1
        for g in range(i - 1, w, -1):
            cg = close_values[g]
            left_start = g - W if g - W > w else w
            right_end = i if g + W > i else g + W
            left = close_values[left_start:g]
            right = close_values[g + 1:right_end + 1]
            if (left.size == 0 or cg < left.min()) and (right.size == 0 or cg < right.min()):
                trough_idx = g
                break
        if trough_idx >= 0:
            trough_price = close_values[trough_idx]
            distance_from_trough[i] = (current_price - trough_price) / trough_price

    distance_from_peak_series = pd.Series(distance_from_peak, index=df.index)

    # ---- pattern indicator ----------------------------------------------------
    pattern_indicator = -1 * distance_from_peak_series * hc_ratio

    # ---- RSI (always computed here as rsi) ------------------------------------
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # ---- RSI / HC composite ---------------------------------------------------
    rsi_hc_composite = hc_ratio * (100 - rsi) / 50
    rsi_hc_composite_norm = _norm(rsi_hc_composite)

    # ---- Lyapunov scaled (one-ticker body of the source loop) -----------------
    close_volatility = df["Close"].pct_change().rolling(50, min_periods=10).std()
    rolling_mean = close_volatility.shift(1).rolling(50, min_periods=10).mean()
    rolling_std = close_volatility.shift(1).rolling(50, min_periods=10).std()
    lyapunov_scaled = (close_volatility - rolling_mean) / rolling_std.replace(0, 1)
    lyapunov_scaled = lyapunov_scaled.clip(-3, 3) / 3

    # ---- HC predict regime (lagged-Lyapunov gated hc_ratio) -------------------
    hc_predict_regime = pd.Series(
        np.where(
            lyapunov_scaled.shift(1) < -0.3,
            hc_ratio * 1.5,
            np.where(
                lyapunov_scaled.shift(1) > 0.3,
                hc_ratio * 0.5,
                hc_ratio,
            ),
        ),
        index=df.index,
    )
    hc_predict_regime_norm = _norm(hc_predict_regime)

    new_cols = {
        "hc_ratio":               hc_ratio,
        "pattern_indicator":      pattern_indicator,
        "rsi":                    rsi,
        "rsi_hc_composite":       rsi_hc_composite,
        "rsi_hc_composite_norm":  rsi_hc_composite_norm,
        "lyapunov_scaled":        lyapunov_scaled,
        "hc_predict_regime":      hc_predict_regime,
        "hc_predict_regime_norm": hc_predict_regime_norm,
    }

    # ---- the remaining *_norm columns from the source's norm loop -------------
    pattern_indicator_norm = _norm(pattern_indicator)
    hc_ratio_norm = _norm(hc_ratio)
    new_cols["pattern_indicator_norm"] = pattern_indicator_norm
    new_cols["hc_ratio_norm"] = hc_ratio_norm

    # ---- soft GP dependency ---------------------------------------------------
    if (
        "g_momentum_confluence_indicator" in df.columns
        and "g_lagged_price_volume_convergence" in df.columns
    ):
        new_cols["hc_gp_composite"] = (
            hc_ratio
            * df["g_momentum_confluence_indicator"]
            * (1 + df["g_lagged_price_volume_convergence"])
        )

    # ---- ensemble = mean of the available *_norm columns ----------------------
    # Mirrors the source's norm_columns list built from
    # [pattern_indicator, rsi_hc_composite, hc_ratio, hc_predict_regime].
    norm_map = {
        "pattern_indicator":  pattern_indicator_norm,
        "rsi_hc_composite":   rsi_hc_composite_norm,
        "hc_ratio":           hc_ratio_norm,
        "hc_predict_regime":  hc_predict_regime_norm,
    }
    norm_series = [norm_map[k] for k in ["pattern_indicator", "rsi_hc_composite", "hc_ratio", "hc_predict_regime"]]
    if len(norm_series) > 0:
        new_cols["significant_indicators_ensemble"] = pd.concat(norm_series, axis=1).mean(axis=1)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
