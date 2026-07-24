from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# Load index/VIX helper
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282340e_regime_regime_volume_response_diff",
    "description": (
        "Regime-dependent volume-volatility coupling. VIX terciles (trailing 252d) define "
        "risk-off vs risk-on regimes. For each bar: compute rolling-120d correlation of "
        "absolute return vs volume over risk-off days minus same over risk-on days. "
        "Positive = volume-volatility link is tighter in stressed markets (typical for "
        "liquid stocks). Guards against zero variance. Per-ticker proxy: regime labels "
        "come from VIX, correlations are per-ticker. Level + z-score variant produced."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282340e_vol_resp_diff",
        "ff06282340e_vol_resp_diff_zscore",
        "ff06282340e_riskoff_frac",
    ],
    "tags": ["regime", "volume", "volatility", "correlation", "vix"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}


def _rolling_regime_corr(absret: np.ndarray, vol: np.ndarray, regime: np.ndarray,
                          window: int, target_regime: int) -> np.ndarray:
    """
    For each bar t, compute correlation of absret and vol over bars in
    [t-window+1, t] that belong to target_regime.
    Returns array of same length, NaN where fewer than 5 regime bars exist.
    """
    n = len(absret)
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        start = t - window + 1
        mask = regime[start: t + 1] == target_regime
        x = absret[start: t + 1][mask]
        y = vol[start: t + 1][mask]
        m = x.shape[0]
        if m < 5:
            continue
        mx, my = x.mean(), y.mean()
        dx, dy = x - mx, y - my
        denom = np.sqrt((dx * dx).sum() * (dy * dy).sum())
        if denom < 1e-30:
            continue
        result[t] = (dx * dy).sum() / denom
    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN up front (covers all early-return paths)
    df["ff06282340e_vol_resp_diff"] = np.nan
    df["ff06282340e_vol_resp_diff_zscore"] = np.nan
    df["ff06282340e_riskoff_frac"] = np.nan

    if len(df) < 30:
        return df

    # --- Fetch VIX and merge backward-safe ---
    try:
        vix_df = _indexes.vix_daily_close()
    except Exception:
        return df

    if vix_df is None or vix_df.empty:
        return df

    work = df[["Date"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df = vix_df.sort_values("Date").reset_index(drop=True)
    work = work.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(work, vix_df, on="Date", direction="backward")
    vix_vals = merged["vix_close"].values.astype(float)

    n = len(df)
    if n < 30:
        return df

    # --- Rolling 252d VIX tercile labels ---
    # Tercile thresholds computed rolling; label 0=risk-on, 1=mid, 2=risk-off
    vix_s = pd.Series(vix_vals)
    q33 = vix_s.rolling(252, min_periods=30).quantile(0.333)
    q67 = vix_s.rolling(252, min_periods=30).quantile(0.667)

    regime = np.full(n, np.nan)
    for i in range(n):
        v = vix_vals[i]
        lo = q33.iloc[i]
        hi = q67.iloc[i]
        if np.isnan(v) or np.isnan(lo) or np.isnan(hi):
            continue
        if v <= lo:
            regime[i] = 0   # risk-on
        elif v >= hi:
            regime[i] = 2   # risk-off
        else:
            regime[i] = 1   # mid

    # --- Per-bar signals ---
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # Absolute log return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_close = np.where(close > 0, np.log(close), np.nan)
    absret = np.abs(np.diff(log_close, prepend=np.nan))

    # Normalise volume (log) to reduce scale dominance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_vol = np.where(volume > 0, np.log(volume), np.nan)

    WINDOW = 120

    # Compute rolling regime correlations
    valid_mask = ~(np.isnan(absret) | np.isnan(log_vol) | np.isnan(regime))
    absret_c = np.where(valid_mask, absret, np.nan)
    log_vol_c = np.where(valid_mask, log_vol, np.nan)
    regime_c = np.where(valid_mask, regime, -1).astype(int)

    corr_riskoff = _rolling_regime_corr(absret_c, log_vol_c, regime_c, WINDOW, target_regime=2)
    corr_riskon = _rolling_regime_corr(absret_c, log_vol_c, regime_c, WINDOW, target_regime=0)

    diff = corr_riskoff - corr_riskon

    # Rolling z-score of the diff (126d lookback)
    diff_s = pd.Series(diff)
    roll_mean = diff_s.rolling(126, min_periods=20).mean()
    roll_std = diff_s.rolling(126, min_periods=20).std()
    zscore = (diff_s - roll_mean) / roll_std.replace(0, np.nan)

    # Fraction of risk-off bars in trailing 120d window
    regime_s = pd.Series(np.where(regime == 2, 1.0, np.where(np.isnan(regime), np.nan, 0.0)))
    riskoff_frac = regime_s.rolling(WINDOW, min_periods=20).mean()

    df["ff06282340e_vol_resp_diff"] = diff
    df["ff06282340e_vol_resp_diff_zscore"] = zscore.values
    df["ff06282340e_riskoff_frac"] = riskoff_frac.values

    return df
