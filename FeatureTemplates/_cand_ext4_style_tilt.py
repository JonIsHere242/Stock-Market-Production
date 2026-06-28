"""
ext4_style_tilt — Growth-vs-small style tilt
Rolling 90-day correlation of a stock with QQQ minus its correlation with IWM.
Positive values indicate growth-tilt; negative values indicate small-cap tilt.
The 60-day change captures style rotation dynamics.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path — not a package import)
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_style_tilt",
    "description": (
        "Growth-vs-small style tilt per ticker. "
        "Computes rolling 90-day return correlation of the stock with QQQ (growth proxy) "
        "and IWM (small-cap proxy), then takes QQQ_corr − IWM_corr as the tilt signal. "
        "Positive = growth-tilted, negative = small-cap-tilted. "
        "A 60-day change of this tilt captures style rotation momentum. "
        "Per-ticker proxy — correlation is computed vs index series, not cross-sectionally. "
        "Source: Round-5 expansion (NEW: multi-index factor)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_style_tilt_level",   # 90d (QQQ corr - IWM corr)
        "ext4_style_tilt_rotation", # 60d change in tilt level
    ],
    "tags": ["style", "multi-index", "correlation", "growth", "small-cap", "rotation"],
    "version": "1.0.0",
    "author": "Round-5 expansion (NEW: multi-index factor)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOW_TILT = 90    # days for rolling correlation
_WINDOW_ROT  = 60    # days for style-rotation change


def _rolling_corr_series(
    stock_ret: pd.Series, index_ret: pd.Series, window: int
) -> pd.Series:
    """
    Rolling Pearson correlation between two aligned return series.
    Requires at least 20 observations in the window before emitting a value
    (min_periods=20) to avoid noise from tiny samples. Returns NaN otherwise.
    Guards against zero-variance windows (constant returns) → NaN.
    """
    combined = pd.concat(
        [stock_ret.rename("s"), index_ret.rename("i")], axis=1
    )
    # Rolling correlation uses the built-in pandas rolling().corr()
    corr = (
        combined["s"]
        .rolling(window, min_periods=max(20, window // 3))
        .corr(combined["i"])
    )
    return corr


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # We need at least a few rows
    if len(df) < 22:
        df["ext4_style_tilt_level"]    = np.nan
        df["ext4_style_tilt_rotation"] = np.nan
        return df

    # ---- Align index close series to df dates (merge_asof, lookahead-safe) ----
    dates_df = df[["Date"]].copy()
    dates_df["Date"] = pd.to_datetime(dates_df["Date"])

    def _get_index_close(symbol: str) -> pd.Series | None:
        try:
            s = _indexes.index_close(symbol)  # DatetimeIndex → Close values
            if s is None or len(s) == 0:
                return None
            idx_df = s.reset_index()
            idx_df.columns = ["Date", f"close_{symbol}"]
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            merged = pd.merge_asof(
                dates_df.sort_values("Date"),
                idx_df.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            # Restore original row order by re-aligning on the original df index
            merged.index = dates_df.sort_values("Date").index
            merged = merged.reindex(df.index)
            return merged[f"close_{symbol}"]
        except Exception:
            return None

    qqq_close = _get_index_close("QQQ")
    iwm_close = _get_index_close("IWM")

    # ---- Compute 1-day returns ----
    stock_close = df["Close"].values.astype(np.float64)
    stock_ret = pd.Series(
        np.concatenate([[np.nan], np.diff(np.log(np.where(stock_close > 0, stock_close, np.nan)))]),
        index=df.index,
    )

    def _index_ret(close_series: pd.Series | None) -> pd.Series:
        if close_series is None:
            return pd.Series(np.nan, index=df.index)
        arr = close_series.values.astype(np.float64)
        log_ret = np.concatenate(
            [[np.nan], np.diff(np.log(np.where(arr > 0, arr, np.nan)))]
        )
        return pd.Series(log_ret, index=df.index)

    qqq_ret = _index_ret(qqq_close)
    iwm_ret = _index_ret(iwm_close)

    # ---- Rolling correlation vs QQQ and IWM ----
    corr_qqq = _rolling_corr_series(stock_ret, qqq_ret, _WINDOW_TILT)
    corr_iwm = _rolling_corr_series(stock_ret, iwm_ret, _WINDOW_TILT)

    # ---- Style tilt = QQQ corr - IWM corr ----
    tilt = corr_qqq - corr_iwm

    # ---- 60d change in tilt (style rotation) ----
    rotation = tilt.diff(_WINDOW_ROT)

    # ---- Guard: replace inf/-inf with NaN ----
    tilt    = tilt.replace([np.inf, -np.inf], np.nan)
    rotation = rotation.replace([np.inf, -np.inf], np.nan)

    df["ext4_style_tilt_level"]    = tilt.values
    df["ext4_style_tilt_rotation"] = rotation.values

    return df
