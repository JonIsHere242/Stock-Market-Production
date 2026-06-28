"""
osap_investppeinv — Change in PPE + Inventory / Lagged Assets
Source: Lyandres, Sun and Zhang (2008); OpenSourceAP (Chen-Zimmermann).
Economic signal: Investment growth (PPE + inventory) scaled by prior-year assets.
Predicted sign: -1 (high investment predicts lower future returns).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load PIT fundamentals helper
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_investppeinv",
    "description": (
        "One-year change in net PPE plus one-year change in inventory, scaled by "
        "one-year lagged total assets (Lyandres, Sun & Zhang 2008 investment factor). "
        "Predicted sign: -1 (higher capital/inventory build-up predicts lower returns). "
        "Uses point-in-time filed fundamentals (ppe_net, inventory, assets) merged "
        "backward-safe via _fundamentals.as_of(). Also emits a 4-quarter rolling "
        "slope variant and a signed-log level for robustness."
    ),
    "requires": [],
    "produces": [
        "osap_investppeinv_lvl",    # (ΔPPE + ΔInv) / lag_assets — core signal
        "osap_investppeinv_slope",  # rolling 4-quarter slope of the level series
        "osap_investppeinv_logabs", # sign(lvl) * log1p(|lvl|) — outlier-robust version
    ],
    "tags": ["fundamental", "investment", "accounting", "osap", "pit"],
    "version": "1.0",
    "author": "Lyandres, Sun and Zhang (2008); OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Attach PIT fundamentals (backward-safe filed_date merge)
    df = _fundamentals.as_of(df, fields=["ppe_net", "inventory", "assets"])

    ppe = df["fund_ppe_net"].copy()
    inv = df["fund_inventory"].copy()
    ast = df["fund_assets"].copy()

    # Approximate one-year lag: shift 252 trading days (per-ticker series)
    LAG = 252
    ppe_lag = ppe.shift(LAG)
    inv_lag = inv.shift(LAG)
    ast_lag = ast.shift(LAG)

    # Core signal: (ΔPPE + ΔInv) / prior-year assets
    delta_ppe = ppe - ppe_lag
    delta_inv = inv - inv_lag

    # Guard: zero or missing lagged assets → NaN
    denom = ast_lag.where(ast_lag != 0, other=np.nan)
    lvl = (delta_ppe + delta_inv) / denom

    # Clip extreme outliers (>|5| standard deviations are likely data errors)
    # Use rolling to stay forward-leakage-free
    roll_std = lvl.rolling(window=252, min_periods=60).std()
    roll_mean = lvl.rolling(window=252, min_periods=60).mean()
    lower = roll_mean - 5 * roll_std
    upper = roll_mean + 5 * roll_std
    lvl = lvl.clip(lower=lower, upper=upper)

    # Replace inf that may have slipped through
    lvl = lvl.replace([np.inf, -np.inf], np.nan)

    df["osap_investppeinv_lvl"] = lvl

    # Slope variant: rolling 63-bar (≈1 quarter) linear slope of lvl
    # Use a simple OLS-in-numpy via rolling apply on small windows
    SLOPE_WIN = 63
    if len(lvl.dropna()) >= SLOPE_WIN:
        x = np.arange(SLOPE_WIN, dtype=float)
        x_dm = x - x.mean()
        ss_x = float((x_dm ** 2).sum())

        def _slope(w: np.ndarray) -> float:
            if np.isnan(w).any():
                return np.nan
            yd = w - w.mean()
            return float(np.dot(x_dm, yd) / ss_x) if ss_x > 0 else np.nan

        slope = lvl.rolling(window=SLOPE_WIN, min_periods=SLOPE_WIN).apply(
            _slope, raw=True
        )
    else:
        slope = pd.Series(np.nan, index=df.index)

    slope = slope.replace([np.inf, -np.inf], np.nan)
    df["osap_investppeinv_slope"] = slope

    # Outlier-robust signed-log level
    logabs = np.sign(lvl) * np.log1p(np.abs(lvl))
    logabs = logabs.replace([np.inf, -np.inf], np.nan)
    df["osap_investppeinv_logabs"] = logabs

    # Drop scratch fund_* columns not in produces
    df = df.drop(columns=["fund_ppe_net", "fund_inventory", "fund_assets"], errors="ignore")

    return df
