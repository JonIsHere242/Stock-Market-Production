"""
_cand_osap_idiovol3f.py — Idiosyncratic Volatility (3-Factor Model)

Idiosyncratic volatility is defined as the standard deviation of daily residuals
from a Fama-French 3-factor regression over a trailing window. Higher idiosyncratic
volatility predicts LOWER subsequent returns (Ang et al. 2006, JF).

Per-ticker proxy:
  - Market factor (MKT-RF): SPY daily return (we omit the risk-free rate as it is
    a near-constant offset that doesn't affect residual std).
  - SMB proxy: IWM daily return minus SPY daily return (small-cap minus large-cap).
  - HML proxy: DIA daily return minus QQQ daily return (value-tilt minus growth-tilt).

We regress stock daily returns on [MKT, SMB_proxy, HML_proxy] over the trailing
window (default 21 trading days), then compute the std of the residuals.

Produces:
  osap_idiovol3f_21d  : trailing 21-day idiosyncratic vol (std of residuals, annualised)
  osap_idiovol3f_63d  : trailing 63-day idiosyncratic vol (std of residuals, annualised)
  osap_idiovol3f_chg  : 21d minus 63d (rising = increasing idio risk signal)
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path – required pattern)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_idiovol3f",
    "description": (
        "Idiosyncratic volatility: annualised std of residuals from a per-ticker "
        "OLS regression of daily returns on 3 factors (market=SPY, SMB proxy=IWM-SPY, "
        "HML proxy=DIA-QQQ) over trailing 21-day and 63-day windows. "
        "Higher idio-vol predicts lower returns (Ang et al. 2006 'The Cross-Section of "
        "Volatility and Expected Returns', JF). Cross-sectional ranking is NOT possible "
        "per-ticker; level values are used directly. SMB/HML are index-spread proxies, "
        "not the original Fama-French factor portfolios."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_idiovol3f_21d",
        "osap_idiovol3f_63d",
        "osap_idiovol3f_chg",
    ],
    "tags": ["volatility", "idiosyncratic", "fama-french", "risk", "osap"],
    "version": "1.0.0",
    "author": "Ang, Hodrick, Xing, Zhang (2006) 'The Cross-Section of Volatility and Expected Returns', JF 61(1):259-299. Per-ticker proxy implementation.",
}

# ---------------------------------------------------------------------------
# Annualisation constant
_SQRT252 = np.sqrt(252)


def _build_factor_matrix(df_stock: pd.DataFrame) -> pd.DataFrame | None:
    """
    Merge stock returns with factor returns (SPY, SMB_proxy, HML_proxy).
    Returns a DataFrame with columns [ret, mkt, smb, hml] indexed like df_stock,
    or None if index data is unavailable.
    """
    try:
        spy_close = _indexes.index_close("SPY")
        iwm_close = _indexes.index_close("IWM")
        dia_close = _indexes.index_close("DIA")
        qqq_close = _indexes.index_close("QQQ")
    except Exception:
        return None

    if spy_close is None or len(spy_close) == 0:
        return None

    # Build a combined index frame
    idx_df = pd.DataFrame({
        "spy": spy_close,
        "iwm": iwm_close if (iwm_close is not None and len(iwm_close) > 0) else np.nan,
        "dia": dia_close if (dia_close is not None and len(dia_close) > 0) else np.nan,
        "qqq": qqq_close if (qqq_close is not None and len(qqq_close) > 0) else np.nan,
    }).sort_index()

    # Daily returns for each index
    idx_df["mkt"] = idx_df["spy"].pct_change()
    idx_df["smb"] = (
        idx_df["iwm"].pct_change() - idx_df["spy"].pct_change()
        if "iwm" in idx_df.columns else np.nan
    )
    idx_df["hml"] = (
        idx_df["dia"].pct_change() - idx_df["qqq"].pct_change()
        if ("dia" in idx_df.columns and "qqq" in idx_df.columns) else np.nan
    )

    # Stock returns
    stock_ret = df_stock["Close"].pct_change()
    stock_ret.index = pd.to_datetime(df_stock["Date"].values)

    # Align on date
    combined = idx_df[["mkt", "smb", "hml"]].copy()
    combined["ret"] = stock_ret

    return combined.dropna(subset=["ret", "mkt"])


def _rolling_idiovol(
    combined: pd.DataFrame,
    window: int,
    min_obs: int,
) -> pd.Series:
    """
    Compute trailing rolling idiosyncratic vol (annualised std of OLS residuals).
    Returns a Series indexed by combined.index (DatetimeIndex).
    """
    n = len(combined)
    result = np.full(n, np.nan)

    ret = combined["ret"].values
    mkt = combined["mkt"].values

    # Build factor matrix X: [1, mkt, smb, hml] -- handle missing smb/hml gracefully
    has_smb = combined["smb"].notna().any()
    has_hml = combined["hml"].notna().any()

    smb = combined["smb"].values if has_smb else np.zeros(n)
    hml = combined["hml"].values if has_hml else np.zeros(n)

    for t in range(window - 1, n):
        start = t - window + 1
        r = ret[start : t + 1]
        m = mkt[start : t + 1]
        s = smb[start : t + 1]
        h = hml[start : t + 1]

        # Mask rows where any factor or return is NaN
        valid = np.isfinite(r) & np.isfinite(m) & np.isfinite(s) & np.isfinite(h)
        if valid.sum() < min_obs:
            continue

        r_v = r[valid]
        m_v = m[valid]
        s_v = s[valid]
        h_v = h[valid]

        # Build design matrix [intercept, mkt, smb, hml]
        X = np.column_stack([np.ones(valid.sum()), m_v, s_v, h_v])

        # OLS via pseudo-inverse (safe, no sklearn)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta, _, _, _ = np.linalg.lstsq(X, r_v, rcond=None)
            resid = r_v - X @ beta
            result[t] = resid.std(ddof=1) * _SQRT252
        except Exception:
            pass

    return pd.Series(result, index=combined.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN
    df["osap_idiovol3f_21d"] = np.nan
    df["osap_idiovol3f_63d"] = np.nan
    df["osap_idiovol3f_chg"] = np.nan

    if len(df) < 21:
        return df

    combined = _build_factor_matrix(df)
    if combined is None or len(combined) < 21:
        return df

    # Ensure Date column is datetime for merge
    df_dates = pd.to_datetime(df["Date"].values)

    # Compute rolling idiosyncratic vol at each window
    vol_21 = _rolling_idiovol(combined, window=21, min_obs=15)
    vol_63 = _rolling_idiovol(combined, window=63, min_obs=42)

    # Map back to df by date (merge_asof: backward lookup)
    factor_dates = combined.index  # DatetimeIndex

    vol_21_df = pd.DataFrame({"_date": factor_dates, "_v21": vol_21.values})
    vol_63_df = pd.DataFrame({"_date": factor_dates, "_v63": vol_63.values})

    stock_date_df = pd.DataFrame({"Date": df_dates})

    merged = pd.merge_asof(
        stock_date_df.sort_values("Date"),
        vol_21_df.sort_values("_date"),
        left_on="Date",
        right_on="_date",
        direction="backward",
    )
    merged63 = pd.merge_asof(
        stock_date_df.sort_values("Date"),
        vol_63_df.sort_values("_date"),
        left_on="Date",
        right_on="_date",
        direction="backward",
    )

    # Restore original order
    orig_order = np.argsort(np.argsort(df_dates.argsort()))  # identity permutation helper
    # Use the original df index order
    sorted_idx = np.argsort(df_dates)
    restore_idx = np.argsort(sorted_idx)

    v21_vals = merged["_v21"].values[restore_idx]
    v63_vals = merged63["_v63"].values[restore_idx]

    df["osap_idiovol3f_21d"] = v21_vals
    df["osap_idiovol3f_63d"] = v63_vals

    # Change: rising 21d vs 63d baseline = increasing idio risk
    with np.errstate(invalid="ignore"):
        chg = v21_vals - v63_vals
        chg = np.where(np.isfinite(chg), chg, np.nan)
    df["osap_idiovol3f_chg"] = chg

    return df
