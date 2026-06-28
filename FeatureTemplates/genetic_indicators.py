import pandas as pd
import numpy as np

METADATA = {
    "name":        "genetic_indicators",
    "description": "Genetic-programming-derived price/volume interaction features (20 g_* columns)",
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "g_momentum_confluence_indicator",
        "g_price_gap_analyzer",
        "g_triple_high_trend_indicator",
        "g_cyclical_price_oscillator",
        "g_volume_adjusted_price_indicator",
        "g_volume_weighted_high_ratio",
        "g_high_price_momentum_indicator",
        "g_advanced_trend_synthesizer",
        "g_price_volatility_gauge",
        "g_multi_point_price_analyzer",
        "g_logarithmic_trend_detector",
        "g_complex_price_pattern_indicator",
        "g_log_scaled_price_ratio",
        "g_volume_price_impact_indicator",
        "g_volume_trend_analyzer",
        "g_price_open_ratio_indicator",
        "g_price_differential_analyzer",
        "g_price_volatility_trend_measure",
        "g_lagged_price_volume_convergence",
        "g_price_volume_disparity_index",
    ],
    "tags": ["experimental", "gp"],
    "version": "1.0",
    "author": "migration from monolith (calculate_genetic_indicators)",
}


# ---------------------------------------------------------------------------
# Local helpers — inlined from 3__AlphaSensitivity.py (lines 199-216)
# ---------------------------------------------------------------------------

def _safe_divide(a, b, fill_value=0):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, fill_value=fill_value)
    elif isinstance(a, pd.Series):
        b = pd.Series(b, index=a.index)
    elif isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, pd.Series):
            result = result.where((b != 0) & (b.notna()), fill_value)
        else:
            result = np.where((b != 0) & (~np.isnan(b)), result, fill_value)
    return result


def _safe_log(x, epsilon=1e-14):
    return np.log(np.maximum(x, epsilon))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-10

    # Build lagged series as local dicts — never written to df
    high_lag   = {i: df['High'].shift(i)   + epsilon for i in range(1, 8)}
    low_lag    = {i: df['Low'].shift(i)    + epsilon for i in range(1, 8)}
    volume_lag = {i: df['Volume'].shift(i) + epsilon for i in range(1, 8)}
    open_lag   = {i: df['Open'].shift(i)   + epsilon for i in range(1, 8)}

    new_cols = {}

    # g_momentum_confluence_indicator  (LOCKED name)
    new_cols['g_momentum_confluence_indicator'] = _safe_divide(
        high_lag[2], high_lag[2] * df['Open']
    )

    # g_price_gap_analyzer
    new_cols['g_price_gap_analyzer'] = _safe_divide(
        _safe_log(open_lag[2]), high_lag[1]
    )

    # g_triple_high_trend_indicator
    new_cols['g_triple_high_trend_indicator'] = _safe_divide(
        high_lag[2], high_lag[1] * df['High']
    )

    # g_cyclical_price_oscillator
    new_cols['g_cyclical_price_oscillator'] = _safe_divide(
        _safe_divide(
            np.cos(_safe_divide(high_lag[2], df['High'])),
            high_lag[1]
        ),
        np.sqrt(open_lag[1] + epsilon)
    )

    # g_volume_adjusted_price_indicator
    new_cols['g_volume_adjusted_price_indicator'] = _safe_divide(
        high_lag[2], _safe_divide(df['Volume'], low_lag[1])
    )

    # g_volume_weighted_high_ratio
    new_cols['g_volume_weighted_high_ratio'] = _safe_divide(
        _safe_divide(df['High'], high_lag[1]),
        _safe_log(df['Volume'] + 1)
    )

    # g_high_price_momentum_indicator
    new_cols['g_high_price_momentum_indicator'] = _safe_divide(
        df['High'], (high_lag[1] + high_lag[2]) / 2
    )

    # g_advanced_trend_synthesizer
    new_cols['g_advanced_trend_synthesizer'] = (
        _safe_log(_safe_divide(high_lag[1] + high_lag[5], high_lag[1])) *
        np.abs(_safe_log(_safe_divide(high_lag[2], high_lag[2])) - high_lag[7]) *
        _safe_divide(open_lag[2], df['Close'])
    )

    # g_price_volatility_gauge
    new_cols['g_price_volatility_gauge'] = _safe_divide(
        np.abs(high_lag[2] - df['High']), df['Open']
    )

    # g_multi_point_price_analyzer
    new_cols['g_multi_point_price_analyzer'] = np.abs(
        _safe_divide(
            _safe_divide(
                _safe_log(_safe_divide(df['High'], high_lag[2])),
                _safe_divide(df['Close'], high_lag[2])
            ),
            _safe_divide(df['Close'], high_lag[2]) * _safe_divide(df['Close'], high_lag[1])
        )
    )

    # g_logarithmic_trend_detector
    new_cols['g_logarithmic_trend_detector'] = -_safe_log(
        _safe_divide(high_lag[2], high_lag[4])
    )

    # g_complex_price_pattern_indicator
    new_cols['g_complex_price_pattern_indicator'] = _safe_log(
        _safe_divide(
            np.sqrt(np.sqrt(np.sqrt(np.sqrt(high_lag[7] * high_lag[4] + epsilon)))),
            df['Close']
        )
    )

    # g_log_scaled_price_ratio
    new_cols['g_log_scaled_price_ratio'] = _safe_log(
        _safe_divide(df['High'], (high_lag[1] + high_lag[2]) / 2)
    )

    # g_volume_price_impact_indicator
    new_cols['g_volume_price_impact_indicator'] = _safe_divide(
        -high_lag[1] + _safe_divide(high_lag[3], df['Close']),
        df['Volume']
    )

    # g_volume_trend_analyzer
    new_cols['g_volume_trend_analyzer'] = _safe_log(
        _safe_divide(df['Volume'], volume_lag[1])
    )

    # g_price_open_ratio_indicator
    new_cols['g_price_open_ratio_indicator'] = _safe_divide(
        _safe_divide(
            _safe_log(_safe_divide(high_lag[2], df['High'])),
            _safe_divide(df['High'], open_lag[2])
        ),
        _safe_divide(df['High'], open_lag[2])
    )

    # g_price_differential_analyzer
    new_cols['g_price_differential_analyzer'] = (
        (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    )

    # g_price_volatility_trend_measure
    new_cols['g_price_volatility_trend_measure'] = (
        0.278 - np.abs(
            _safe_divide(df['Low'], df['High']) /
            _safe_divide(high_lag[5], low_lag[5])
        )
    )

    # g_lagged_price_volume_convergence  (LOCKED name)
    new_cols['g_lagged_price_volume_convergence'] = _safe_divide(
        low_lag[2],
        high_lag[2] + _safe_divide(0.791, df['Low'] * df['High'])
    )

    # g_price_volume_disparity_index
    new_cols['g_price_volume_disparity_index'] = (
        np.abs(_safe_divide(low_lag[2], high_lag[2])) /
        (_safe_divide(high_lag[5], low_lag[5]) / -0.2831)
    )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['g_log_scaled_price_ratio', 'g_price_differential_analyzer']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
