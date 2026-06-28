"""
Abnormal Accruals (per-ticker proxy for Xie 2001 / OpenSourceAP Chen-Zimmermann).

TRUE METHOD: For each year t and 2-digit SIC code, regress
  Accruals_t = b0*(1/avg_assets) + b1*(delta_revenue/assets_t-1) + b2*(PPE/assets_t-1)
and take the residual as AbnormalAccrual. This is inherently cross-sectional.

PER-TICKER PROXY: We estimate the "expected" accrual from the Jones (1991) model
using only the firm's own time-series. Specifically:
  - Total accruals = (net_income_ttm - operating_cash_flow_ttm) / assets
  - Expected accrual = fitted value from rolling OLS on two regressors:
      X1 = 1 / assets  (scale factor, Jones model intercept analogue)
      X2 = delta_revenue / lagged_assets  (revenue change scaled by assets)
  - AbnormalAccrual = actual_accrual - expected_accrual (residual)
  - PPE/assets is also produced as a control/companion signal.

Sign: -1 (high accruals => lower future returns, consistent with accrual anomaly).
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
    "name": "osap_abnormalaccruals",
    "description": (
        "Abnormal accruals proxy based on Xie (2001) / OpenSourceAP (Chen-Zimmermann). "
        "True method: cross-sectional Jones-model regression within 2-digit SIC groups per year; "
        "residual = abnormal accrual. Per-ticker proxy: total accruals = (net_income_ttm - "
        "operating_cash_flow_ttm) / assets; 'expected' accrual estimated via rolling 8-quarter "
        "OLS of accruals on (1/assets, delta_revenue/lagged_assets); abnormal = residual. "
        "Predicted sign -1: high abnormal accruals => lower future returns. "
        "osap_abnormalaccruals_ppe_ratio: PPE/assets is the third Jones-model regressor, "
        "provided as a companion fundamental ratio."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_abnormalaccruals_total",        # raw accrual = (NI - OCF) / assets
        "osap_abnormalaccruals_abnormal",     # residual after removing expected portion
        "osap_abnormalaccruals_ppe_ratio",    # PPE / assets (Jones model 3rd regressor)
    ],
    "tags": ["accruals", "fundamentals", "accounting", "jones_model", "earnings_quality"],
    "version": "1.0",
    "author": "Xie 2001 / OpenSourceAP Chen-Zimmermann. Per-ticker proxy by Claude.",
}


def _rolling_ols_residuals(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    window: int = 8,
    min_periods: int = 5,
) -> np.ndarray:
    """
    Rolling OLS: y ~ b0*x1 + b1*x2 (no intercept; x1 already encodes 1/assets as intercept).
    Returns residuals (y - yhat). NaN when insufficient data.
    """
    n = len(y)
    residuals = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        # Gather valid observations in window [start..i]
        yw = y[start : i + 1]
        x1w = x1[start : i + 1]
        x2w = x2[start : i + 1]

        # Build design matrix
        mask = np.isfinite(yw) & np.isfinite(x1w) & np.isfinite(x2w)
        if mask.sum() < min_periods:
            continue

        Y = yw[mask]
        X = np.column_stack([x1w[mask], x2w[mask]])

        # OLS via normal equations: beta = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            XtY = X.T @ Y
            # Use lstsq for numerical stability
            beta, _, rank, _ = np.linalg.lstsq(XtX, XtY, rcond=None)
            if rank < 2:
                continue
            # Residual at current point i (only predict the last point)
            xi = np.array([x1[i], x2[i]])
            if np.isfinite(xi).all() and np.isfinite(y[i]):
                yhat = xi @ beta
                residuals[i] = y[i] - yhat
        except (np.linalg.LinAlgError, ValueError):
            continue

    return residuals


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: net_income_ttm, operating_cash_flow_ttm, assets, revenue_ttm, ppe_net
    df = _fundamentals.as_of(
        df,
        fields=[
            "net_income_ttm",
            "operating_cash_flow_ttm",
            "assets",
            "revenue_ttm",
            "ppe_net",
        ],
    )

    ni = df["fund_net_income_ttm"].values.astype(float)
    ocf = df["fund_operating_cash_flow_ttm"].values.astype(float)
    assets = df["fund_assets"].values.astype(float)
    revenue = df["fund_revenue_ttm"].values.astype(float)
    ppe = df["fund_ppe_net"].values.astype(float)

    n = len(df)

    # --- Total accruals = (NI - OCF) / assets ---
    assets_safe = np.where(assets == 0, np.nan, assets)
    total_accrual = np.where(
        np.isfinite(ni) & np.isfinite(ocf) & np.isfinite(assets_safe),
        (ni - ocf) / assets_safe,
        np.nan,
    )

    # --- Jones model regressors ---
    # X1 = 1 / assets  (Jones model intercept)
    x1 = np.where(np.isfinite(assets_safe), 1.0 / assets_safe, np.nan)

    # X2 = delta_revenue / lagged_assets  (revenue change scaled)
    rev_lag = np.empty(n)
    rev_lag[0] = np.nan
    rev_lag[1:] = revenue[:-1]

    assets_lag = np.empty(n)
    assets_lag[0] = np.nan
    assets_lag[1:] = assets_safe[:-1]

    delta_rev = revenue - rev_lag
    assets_lag_safe = np.where(assets_lag == 0, np.nan, assets_lag)
    x2 = np.where(
        np.isfinite(delta_rev) & np.isfinite(assets_lag_safe),
        delta_rev / assets_lag_safe,
        np.nan,
    )

    # --- Rolling OLS residual = abnormal accrual ---
    abnormal = _rolling_ols_residuals(total_accrual, x1, x2, window=8, min_periods=5)

    # --- PPE / assets ratio (3rd Jones-model regressor, useful standalone) ---
    ppe_ratio = np.where(
        np.isfinite(ppe) & np.isfinite(assets_safe),
        ppe / assets_safe,
        np.nan,
    )

    # Guard: replace inf/-inf with NaN
    total_accrual = np.where(np.isfinite(total_accrual), total_accrual, np.nan)
    abnormal = np.where(np.isfinite(abnormal), abnormal, np.nan)
    ppe_ratio = np.where(np.isfinite(ppe_ratio), ppe_ratio, np.nan)

    df["osap_abnormalaccruals_total"] = total_accrual
    df["osap_abnormalaccruals_abnormal"] = abnormal
    df["osap_abnormalaccruals_ppe_ratio"] = ppe_ratio

    # Drop scratch fundamentals columns
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
