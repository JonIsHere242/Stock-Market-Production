"""
ext4_variance_risk_proxy — Range-vs-close variance wedge (VRP-like proxy)

Parkinson intraday-range vol minus close-to-close vol over rolling 20d, plus the
60d trend of that wedge. A positive wedge means intraday range is wide relative to
overnight-inclusive moves, suggesting elevated intraday variance risk premium.
Per-ticker, pure OHLCV, causal/no-lookahead.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_variance_risk_proxy",
    "description": (
        "Parkinson range vol (from ln(H/L)) minus close-to-close vol over rolling 20d "
        "(variance-risk-premium-flavoured wedge). Positive = intraday range wide relative "
        "to overnight-inclusive variance. Also produces 60d rolling trend (slope) of the wedge "
        "as a dynamic signal. Pure OHLC, per-ticker, no lookahead."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ext4_variance_risk_proxy_wedge",   # Parkinson vol - c2c vol (20d)
        "ext4_variance_risk_proxy_ratio",   # Parkinson vol / c2c vol (20d)
        "ext4_variance_risk_proxy_trend",   # 60d OLS slope of the wedge
    ],
    "tags": ["volatility", "variance_risk_premium", "ohlc", "intraday", "rolling"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom_allan_variance); spec by project pipeline",
}

_WIN_SHORT = 20   # rolling window for vol estimation
_WIN_LONG  = 60   # rolling window for trend


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- Parkinson range volatility (annualised daily vol estimate) ---
    # Parkinson: sigma_park = sqrt(1/(4*ln2) * mean(ln(H/L)^2))
    # Guard against H==L (e.g. halted stocks) with np.where
    h = df["High"].values.astype(np.float64)
    l = df["Low"].values.astype(np.float64)
    c = df["Close"].values.astype(np.float64)

    hl_safe = np.where((l > 0) & (h >= l), h / l, np.nan)
    log_hl = np.log(hl_safe)            # ln(H/L), always >= 0 when valid
    log_hl_sq = log_hl ** 2             # (ln(H/L))^2

    park_factor = 1.0 / (4.0 * np.log(2.0))  # ~0.3607

    # --- Close-to-close log returns ---
    log_ret = np.where(
        (c[:-1] > 0) & np.isfinite(c[:-1]) & np.isfinite(c[1:]),
        np.log(c[1:] / c[:-1]),
        np.nan,
    )
    log_ret = np.concatenate([[np.nan], log_ret])

    n = len(df)
    park_vol   = np.full(n, np.nan)
    c2c_vol    = np.full(n, np.nan)

    # Rolling means over _WIN_SHORT using a simple loop-free sliding sum
    # (vectorised via cumsum trick)
    def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        """Causal rolling mean; result[i] = mean(arr[i-w+1 : i+1])."""
        out = np.full(len(arr), np.nan)
        valid = np.where(np.isfinite(arr), arr, np.nan)
        # cumsum of finite values + count of finite values
        cum   = np.nancumsum(np.where(np.isfinite(arr), arr, 0.0))
        cnt   = np.cumsum(np.isfinite(arr).astype(np.float64))
        # rolling sum = cum[i] - cum[i-w]  (require full w finite obs)
        for i in range(w - 1, len(arr)):
            s = cum[i] - (cum[i - w] if i >= w else 0.0)
            c_cnt = cnt[i] - (cnt[i - w] if i >= w else 0.0)
            out[i] = s / c_cnt if c_cnt >= w * 0.8 else np.nan  # allow 20% NaN
        return out

    mean_log_hl_sq = _rolling_mean(log_hl_sq, _WIN_SHORT)
    mean_log_ret_sq = _rolling_mean(log_ret ** 2, _WIN_SHORT)

    # Parkinson vol (daily, not annualised — keeps units comparable to c2c)
    park_vol = np.sqrt(park_factor * mean_log_hl_sq)
    c2c_vol  = np.sqrt(mean_log_ret_sq)

    # --- Wedge: Parkinson minus c2c ---
    wedge = park_vol - c2c_vol

    # --- Ratio: Parkinson / c2c (guard zero) ---
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(c2c_vol > 0, park_vol / c2c_vol, np.nan)
        ratio = np.where(np.isinf(ratio), np.nan, ratio)

    # --- 60d OLS slope of the wedge ---
    # Use vectorised rolling OLS via cross-products of (x, wedge) over _WIN_LONG
    trend = np.full(n, np.nan)
    x_idx = np.arange(n, dtype=np.float64)

    def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, w: int) -> np.ndarray:
        """Causal rolling OLS slope; requires ≥ w*0.8 finite y obs."""
        out = np.full(len(y), np.nan)
        for i in range(w - 1, len(y)):
            yi = y[i - w + 1 : i + 1]
            xi = x[i - w + 1 : i + 1]
            mask = np.isfinite(yi)
            if mask.sum() < w * 0.8:
                continue
            yi_m = yi[mask]
            xi_m = xi[mask]
            xi_m = xi_m - xi_m.mean()  # de-mean x for numerical stability
            denom = (xi_m * xi_m).sum()
            if denom == 0.0:
                continue
            out[i] = (xi_m * (yi_m - yi_m.mean())).sum() / denom
        return out

    trend = _rolling_ols_slope(wedge, x_idx, _WIN_LONG)

    df["ext4_variance_risk_proxy_wedge"] = wedge
    df["ext4_variance_risk_proxy_ratio"] = ratio
    df["ext4_variance_risk_proxy_trend"] = trend

    return df
