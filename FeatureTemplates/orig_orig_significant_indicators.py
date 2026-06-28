"""
orig_orig_significant_indicators.py

Faithful port of 3__AlphaSensitivity.calculate_significant_indicators (backup
_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py L2763-2920), emitting the
ORIGINAL CapCase column names verbatim.

This is the CapCase sibling of the already-parity-tested FeatureTemplates/significant_indicators.py
(lowercase). The peak/trough causal logic is reproduced exactly from that vetted block; the only
additions here are:
  * BOTH HC_Ratio AND High_Close_Ratio (in the full monolith High_Close_Ratio was created by an
    earlier function so calculate_significant_indicators saw it existing and emitted HC_Ratio from
    the same (High-Close)/(Close+eps) formula; both ground-truth columns are bit-identical).
  * HC_GP_Composite, which needs G_Momentum_Confluence_Indicator and
    G_Lagged_Price_Volume_Convergence. Those two genetic features are computed inline here
    (formulas copied verbatim from calculate_genetic_indicators L2537 / L2579) so the block is
    self-contained and does not require a prereq.

Contract: df is ONE ticker, ascending by Date. Add only the produced columns; stateless.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_significant_indicators",
    "description": "High-Close-ratio / RSI / Lyapunov regime composite ('significant indicators') ensemble, original CapCase names, ported from 3__AlphaSensitivity.calculate_significant_indicators",
    "requires":    ["High", "Close", "Open", "Low"],
    "produces":    [
        "HC_Ratio",
        "HC_Ratio_norm",
        "High_Close_Ratio",
        "High_Close_Ratio_norm",
        "Pattern_Indicator",
        "Pattern_Indicator_norm",
        "RSI",
        "RSI_HC_Composite",
        "RSI_HC_Composite_norm",
        "Lyapunov_Scaled",
        "HC_Predict_Regime",
        "HC_Predict_Regime_norm",
        "HC_GP_Composite",
        "Significant_Indicators_Ensemble",
    ],
    "tags":        ["momentum", "mean_reversion", "gp", "experimental"],
    "version":     "1.0",
    "author":      "alphasens port",
}

_EPSILON        = 1e-10
_MIN_MAX_WINDOW = 10


# --- helpers copied verbatim from the AlphaSensitivity backup (L199, L215) -----------------
def _safe_divide(a, b, fill_value=0):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, fill_value=fill_value)
    elif isinstance(a, pd.Series):
        b = pd.Series(b, index=a.index)
    elif isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        if isinstance(result, pd.Series):
            result = result.where((b != 0) & (b.notna()), fill_value)
        else:
            result = np.where((b != 0) & (~np.isnan(b)), result, fill_value)
    return result


def _norm(series: pd.Series) -> pd.Series:
    """Source normalization: (x - x.shift(1).rolling(50).mean()) / x.shift(1).rolling(50).std(), clipped [-3,3]."""
    shifted = series.shift(1)
    out = (series - shifted.rolling(50).mean()) / shifted.rolling(50).std()
    return out.clip(-3, 3)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- high-close ratio: emit BOTH CapCase names (identical in ground truth) ------------
    hc_ratio = (df["High"] - df["Close"]) / (df["Close"] + _EPSILON)

    n = len(df)
    close_values = df["Close"].to_numpy(dtype=float)

    # ---- distance from nearest historical peak / trough (causal) --------------------------
    # df is ONE ticker. Reproduces the source's per-bar argrelextrema(order=10) over the
    # max(0, i-50):i+1 history window, picking the MOST RECENT extremum strictly before i.
    # This is the exact vetted logic from FeatureTemplates/significant_indicators.py.
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

    # ---- Pattern_Indicator ----------------------------------------------------------------
    pattern_indicator = -1 * distance_from_peak_series * hc_ratio

    # ---- RSI (14) -------------------------------------------------------------------------
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # ---- RSI / HC composite ---------------------------------------------------------------
    rsi_hc_composite = hc_ratio * (100 - rsi) / 50
    rsi_hc_composite_norm = _norm(rsi_hc_composite)

    # ---- Lyapunov_Scaled (single-ticker body of the source loop) --------------------------
    close_volatility = df["Close"].pct_change().rolling(50, min_periods=10).std()
    rolling_mean = close_volatility.shift(1).rolling(50, min_periods=10).mean()
    rolling_std = close_volatility.shift(1).rolling(50, min_periods=10).std()
    lyapunov_scaled = (close_volatility - rolling_mean) / rolling_std.replace(0, 1)
    lyapunov_scaled = lyapunov_scaled.clip(-3, 3) / 3

    # ---- HC_Predict_Regime (lagged-Lyapunov gated hc_ratio) -------------------------------
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

    # ---- the remaining *_norm columns (source norm loop) ----------------------------------
    pattern_indicator_norm = _norm(pattern_indicator)
    hc_ratio_norm = _norm(hc_ratio)

    # ---- HC_GP_Composite: needs two genetic features (computed inline, verbatim formulas) -
    # High_Lag2 / Low_Lag2 are shift(2) + epsilon (calculate_genetic_indicators L2532-2533).
    high_lag2 = df["High"].shift(2) + _EPSILON
    low_lag2 = df["Low"].shift(2) + _EPSILON
    g_momentum_confluence = _safe_divide(high_lag2, high_lag2 * df["Open"])              # L2537
    g_lagged_pv_convergence = _safe_divide(                                              # L2579
        low_lag2,
        (high_lag2 + _safe_divide(0.791, df["Low"] * df["High"])),
    )
    # _safe_divide returns ndarrays here; wrap so the multiply aligns on df.index.
    g_momentum_confluence = pd.Series(np.asarray(g_momentum_confluence), index=df.index)
    g_lagged_pv_convergence = pd.Series(np.asarray(g_lagged_pv_convergence), index=df.index)
    hc_gp_composite = hc_ratio * g_momentum_confluence * (1 + g_lagged_pv_convergence)

    # ---- Significant_Indicators_Ensemble = mean of the 4 norms -----------------------------
    # Source norm_columns from [Pattern_Indicator, RSI_HC_Composite, ratio_column, HC_Predict_Regime].
    # ratio_column is High_Close_Ratio (== HC_Ratio); its norm == hc_ratio_norm.
    ensemble = pd.concat(
        [pattern_indicator_norm, rsi_hc_composite_norm, hc_ratio_norm, hc_predict_regime_norm],
        axis=1,
    ).mean(axis=1)

    new_cols = {
        "HC_Ratio":                        hc_ratio,
        "HC_Ratio_norm":                   hc_ratio_norm,
        "High_Close_Ratio":                hc_ratio,
        "High_Close_Ratio_norm":           hc_ratio_norm,
        "Pattern_Indicator":               pattern_indicator,
        "Pattern_Indicator_norm":          pattern_indicator_norm,
        "RSI":                             rsi,
        "RSI_HC_Composite":                rsi_hc_composite,
        "RSI_HC_Composite_norm":           rsi_hc_composite_norm,
        "Lyapunov_Scaled":                 lyapunov_scaled,
        "HC_Predict_Regime":               hc_predict_regime,
        "HC_Predict_Regime_norm":          hc_predict_regime_norm,
        "HC_GP_Composite":                 hc_gp_composite,
        "Significant_Indicators_Ensemble": ensemble,
    }

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
