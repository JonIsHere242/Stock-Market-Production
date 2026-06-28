"""
vix_features.py  --  VIX market-regime / volatility feature suite.

Ported verbatim (math-preserving) from 3__AlphaSensitivity.calculate_vix_features
(+ the inlined _dynamic_rsi_vectorized helper).

External VIX index data is loaded through the shared _indexes.py helper. The block
merges the daily-resampled VIX close onto the per-ticker frame with a backward
merge_asof (past-only, look-ahead safe) WITHOUT reordering df — df is already
ascending by Date.

NOTE on the monolith cleanup that is deliberately NOT reproduced here: the source
ended with a whole-frame `for col in numeric_cols: ffill()`, a `fillna(0)`, and a
`reset_index(drop=True)`. Per the framework contract those are STRIPPED — NaNs are
left as-is and the frame's order/index is never touched.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the underscore-prefixed shared index helper by path (auto-discovery
# skips it, so we import it explicitly).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


METADATA = {
    "name":        "vix_features",
    "description": "VIX market-regime, mean-reversion, vol-of-vol and VIX-adjusted price feature suite",
    "requires":    ["Date", "Close", "High", "Low"],
    "produces":    [
        "vix_close",
        "vix_regime_numeric",
        "vix_percentile_30d", "vix_percentile_60d", "vix_percentile_252d",
        "vix_change_1d", "vix_change_5d", "vix_change_10d", "vix_change_20d",
        "vix_momentum", "vix_acceleration",
        "vix_ma10", "vix_ma20", "vix_ma50",
        "vix_cross_10_50", "vix_cross_20_50",
        "vix_mean_20d", "vix_std_20d", "vix_z_20d",
        "vix_mean_50d", "vix_std_50d", "vix_z_50d",
        "vix_mean_100d", "vix_std_100d", "vix_z_100d",
        "vix_extreme_high", "vix_extreme_low",
        "vix_of_vix", "vix_of_vix_z",
        "log_return", "realized_vol_10d", "realized_vol_21d", "vix_vs_realized_21d",
        "vix_adjusted_atr", "vix_factor", "dynamic_rsi", "vix_risk_multiplier",
        "sma50", "trend_direction", "market_regime",
        "vix_spike", "vix_bottom",
    ],
    "tags":        ["market_regime", "volatility"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.calculate_vix_features",
}


def _dynamic_rsi_one_ticker(close: np.ndarray, vix_factor: np.ndarray) -> np.ndarray:
    """Inlined, single-ticker version of _dynamic_rsi_vectorized.

    Precompute RSI for windows 5-30 then select the VIX-driven window per row.
    The original per-ticker `if n < 50: continue` guard becomes: only fill
    positions >= 50 (everything else stays NaN).
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < 50:
        return out

    delta  = np.diff(close, prepend=np.nan)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Build 26-column RSI matrix (one column per window 5..30)
    rsi_matrix = np.full((n, 26), np.nan)
    for col, w in enumerate(range(5, 31)):
        avg_g = pd.Series(gains ).rolling(w, min_periods=w).mean().to_numpy()
        avg_l = pd.Series(losses).rolling(w, min_periods=w).mean().to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            rsi = np.where(
                avg_l == 0,
                np.where(avg_g > 0, 100.0, 50.0),
                100.0 - 100.0 / (1.0 + avg_g / avg_l)
            )
        rsi_matrix[:, col] = rsi

    # Map VIX_Factor → integer window, fallback 14 on NaN
    dyn_w = np.where(
        np.isfinite(vix_factor),
        np.clip(np.floor(14.0 * vix_factor).astype(int), 5, 30),
        14
    )
    selected = rsi_matrix[np.arange(n), dyn_w - 5]
    selected = np.where(np.arange(n) >= 50, selected, np.nan)
    return selected


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # Merge VIX close onto df WITHOUT reordering df (df is already ascending
    # by Date). merge_asof backward = last known VIX on/before each row's Date.
    # -----------------------------------------------------------------------
    vix_daily = _indexes.vix_daily_close()  # ['Date','vix_close'], ascending

    dates  = pd.to_datetime(df["Date"])
    tmp    = pd.DataFrame({"Date": dates.values})
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    vix_close = merged["vix_close"].ffill().to_numpy()   # aligned 1:1 to df rows

    # 'VIX_Close' stand-in used everywhere the source referenced result_df['VIX_Close'].
    vc = pd.Series(vix_close, index=df.index)

    new = {}
    new['vix_close'] = vc

    # 1. VIX REGIME DETECTION ------------------------------------------------
    new['vix_regime_numeric'] = pd.Series(
        np.select(
            [
                vc < 15,
                (vc >= 15) & (vc < 20),
                (vc >= 20) & (vc < 30),
                vc >= 30,
            ],
            [0, 1, 2, 3],
            default=1,
        ),
        index=df.index,
    )

    # 2. VIX RELATIVE LEVELS AND CHANGES ------------------------------------
    for window in [30, 60, 252]:
        new[f'vix_percentile_{window}d'] = (
            vc
            .rolling(window + 1, min_periods=window + 1)
            .apply(lambda x: (x[:-1] <= x[-1]).mean(), raw=True)
        )

    for period in [1, 5, 10, 20]:
        if len(df) > period:
            new[f'vix_change_{period}d'] = vc.pct_change(period, fill_method=None)

    # VIX momentum: today vs mean of past 10 days (excludes today via shift(1))
    _past_10d_mean = vc.shift(1).rolling(10, min_periods=10).mean()
    new['vix_momentum'] = pd.Series(
        np.where(_past_10d_mean > 0, vc / _past_10d_mean - 1, np.nan),
        index=df.index,
    )

    # VIX Acceleration: diff(5) of the 5d-change series (LOCKED — consumed elsewhere)
    if 'vix_change_5d' in new:
        new['vix_acceleration'] = new['vix_change_5d'].diff(5)
    else:
        new['vix_acceleration'] = pd.Series(np.nan, index=df.index)

    # 3. VIX MOVING AVERAGES AND CROSSOVERS ---------------------------------
    for window in [10, 20, 50]:
        new[f'vix_ma{window}'] = vc.shift(1).rolling(window, min_periods=window).mean()

    if 'vix_ma10' in new and 'vix_ma50' in new:
        new['vix_cross_10_50'] = pd.Series(
            np.where(
                (new['vix_ma10'].notna()) & (new['vix_ma50'].notna()),
                np.where(new['vix_ma10'] > new['vix_ma50'], 1, -1),
                np.nan,
            ),
            index=df.index,
        )

    if 'vix_ma20' in new and 'vix_ma50' in new:
        new['vix_cross_20_50'] = pd.Series(
            np.where(
                (new['vix_ma20'].notna()) & (new['vix_ma50'].notna()),
                np.where(new['vix_ma20'] > new['vix_ma50'], 1, -1),
                np.nan,
            ),
            index=df.index,
        )

    # 4. VIX MEAN REVERSION SIGNALS -----------------------------------------
    for window in [20, 50, 100]:
        if len(df) >= window:
            mean_col = f'vix_mean_{window}d'
            std_col  = f'vix_std_{window}d'
            z_col    = f'vix_z_{window}d'

            _mean = vc.rolling(window, min_periods=window).mean()
            _std  = vc.rolling(window, min_periods=window).std()
            new[mean_col] = _mean
            new[std_col]  = _std

            # Add epsilon to avoid division by zero
            new[z_col] = pd.Series(
                np.where(
                    (_mean.notna()) & (_std.notna()) & (_std > 0),
                    (vc - _mean) / (_std + 1e-10),
                    np.nan,
                ),
                index=df.index,
            )

    if 'vix_z_50d' in new:
        new['vix_extreme_high'] = pd.Series(np.where(new['vix_z_50d'] > 2, 1, 0), index=df.index)
        new['vix_extreme_low']  = pd.Series(np.where(new['vix_z_50d'] < -1, 1, 0), index=df.index)

    # 5. VOLATILITY OF VOLATILITY -------------------------------------------
    _vix_pct = vc.pct_change(fill_method=None)
    new['vix_of_vix'] = _vix_pct.shift(1).rolling(20, min_periods=5).std() * np.sqrt(252)

    _vov      = new['vix_of_vix']
    _vov_mean = _vov.shift(1).rolling(100, min_periods=30).mean()
    _vov_std  = _vov.shift(1).rolling(100, min_periods=30).std()
    new['vix_of_vix_z'] = pd.Series(
        np.where(_vov_std > 0, (_vov - _vov_mean) / (_vov_std + 1e-10), np.nan),
        index=df.index,
    )

    # 6. VIX-ADJUSTED PRICE INDICATORS --------------------------------------
    close_series = df['Close']
    new['log_return'] = np.log(close_series / close_series.shift(1))

    for window in [10, 21]:
        new[f'realized_vol_{window}d'] = new['log_return'].rolling(
            window, min_periods=window
        ).std() * np.sqrt(252)

    if 'realized_vol_21d' in new:
        new['vix_vs_realized_21d'] = pd.Series(
            np.where(
                new['realized_vol_21d'] > 0,
                vc / (100 * new['realized_vol_21d']),
                np.nan,
            ),
            index=df.index,
        )

    # VIX-adjusted ATR calculations (ATR is a local intermediate — NOT emitted)
    high_low   = df['High'] - df['Low']
    high_close = np.abs(df['High'] - close_series.shift(1))
    low_close  = np.abs(df['Low'] - close_series.shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    ATR = true_range.rolling(14, min_periods=14).mean()

    new['vix_adjusted_atr'] = pd.Series(
        np.where(
            ATR.notna() & (vc > 0),
            ATR * np.sqrt(vc / 20),
            np.nan,
        ),
        index=df.index,
    )

    # 7. DYNAMIC LOOKBACK PERIODS -------------------------------------------
    new['vix_factor'] = pd.Series(
        np.where(vc > 0, np.clip(20 / vc, 0.5, 2), 1.0),
        index=df.index,
    )

    # Dynamic RSI: inlined single-ticker version (needs vix_factor first).
    _close_arr  = close_series.to_numpy(dtype=np.float64)
    _vfactor_arr = new['vix_factor'].to_numpy(dtype=np.float64)
    new['dynamic_rsi'] = pd.Series(
        _dynamic_rsi_one_ticker(_close_arr, _vfactor_arr),
        index=df.index,
    )

    # 8. RISK ADJUSTMENT FEATURES -------------------------------------------
    new['vix_risk_multiplier'] = vc / 20

    # percent_change_Close is a local intermediate only — NOT emitted (another
    # block owns percent_change_close). Computed here purely to mirror source order.
    percent_change_Close = close_series.pct_change(fill_method=None)  # noqa: F841

    # 9. MARKET REGIME CLASSIFICATION ---------------------------------------
    new['sma50'] = close_series.rolling(50, min_periods=50).mean()

    new['trend_direction'] = pd.Series(
        np.where(
            new['sma50'].notna(),
            np.where(close_series > new['sma50'], 1, -1),
            np.nan,
        ),
        index=df.index,
    )

    new['market_regime'] = pd.Series(
        np.where(
            new['trend_direction'].notna(),
            np.where(
                vc < 20,
                np.where(new['trend_direction'] > 0, 0, 1),  # Low vol regimes
                np.where(new['trend_direction'] > 0, 2, 3),  # High vol regimes
            ),
            np.nan,
        ),
        index=df.index,
    )

    # 10. VIX PATTERN DETECTION ---------------------------------------------
    _v5    = vc.shift(5)                                       # VIX five days ago
    _v1    = vc.shift(1)                                       # VIX yesterday
    _vmean = vc.shift(1).rolling(20, min_periods=20).mean()    # past-20-day mean

    _has_history = _vmean.notna()

    _spike_cond = (
        _has_history &
        (_v5 < _v1) &             # was rising
        (_v1 > vc) &              # now falling
        (_v1 > _vmean * 1.5)      # was elevated vs recent mean
    )
    _bottom_cond = (
        _has_history &
        (_v5 > _v1) &             # was falling
        (_v1 < vc) &              # now rising
        (_v1 < _vmean * 0.8)      # was depressed vs recent mean
    )

    new['vix_spike']  = pd.Series(
        np.where(_has_history, _spike_cond.astype(float), np.nan), index=df.index
    )
    new['vix_bottom'] = pd.Series(
        np.where(_has_history, _bottom_cond.astype(float), np.nan), index=df.index
    )

    # -----------------------------------------------------------------------
    # STRIPPED: the source's whole-frame ffill loop, fillna(0), and
    # reset_index(drop=True) are intentionally NOT applied. NaNs stay as-is;
    # df order/index untouched. Only ADD the produced columns.
    # -----------------------------------------------------------------------
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['vix_factor', 'vix_risk_multiplier']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
