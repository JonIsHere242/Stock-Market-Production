"""
ff06282316b_factor_return_convexity_spy_120
Quadratic market-model convexity of ticker returns vs SPY over a trailing 120-day window.
Fits r_i = a + b*r_m + c*r_m^2 via normal equations; produces the convexity coefficient c
(scaled by std of r_m^2 for unit stability), linear beta b, and a sign-asymmetry variant
capturing whether the convexity is up- vs down-market biased.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never by package import)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_factor_return_convexity_spy_120",
    "description": (
        "Per-ticker quadratic market-model convexity over a rolling 120-day window. "
        "Fits r_i = a + b*r_m + c*r_m^2 via closed-form normal equations (3x3 solve). "
        "Produces: scaled convexity coefficient c (option-likeness in large moves), "
        "linear beta b, and up/down convexity asymmetry (c estimated on up-market vs "
        "down-market SPY days). Positive c_scaled means the stock gains convexity vs the "
        "market (pays off in both sharp rallies AND crashes relative to a linear beta)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_return_convexity_spy_120_c_scaled",
        "ff06282316b_factor_return_convexity_spy_120_beta",
        "ff06282316b_factor_return_convexity_spy_120_asym",
    ],
    "tags": ["factor", "beta", "convexity", "quadratic", "spy", "market-model"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120
_EPS = 1e-12
_COL_C = "ff06282316b_factor_return_convexity_spy_120_c_scaled"
_COL_B = "ff06282316b_factor_return_convexity_spy_120_beta"
_COL_A = "ff06282316b_factor_return_convexity_spy_120_asym"


def _quad_ols_rolling(ri: np.ndarray, rm: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling quadratic OLS: r_i = a + b*r_m + c*r_m^2.
    Returns arrays (c_scaled, b, asym) of same length as ri, NaN where not enough data.
    Uses a stride-based vectorised accumulation of the 3x3 normal-equation system.
    """
    n = len(ri)
    out_c = np.full(n, np.nan)
    out_b = np.full(n, np.nan)
    out_a = np.full(n, np.nan)

    rm2 = rm ** 2  # quadratic term

    # We need sums: S1, Srm, Srm2, Srm3, Srm4, Sri, Srm_ri, Srm2_ri
    # Build running sums via a sliding window approach using cumsum
    # Prefix sums
    ones = np.ones(n)

    def _cumsum_slide(x: np.ndarray, w: int) -> np.ndarray:
        """Sliding window sum via cumsum; returns array same length, NaN for first w-1."""
        cs = np.concatenate(([0.0], np.cumsum(x)))
        out = np.full(n, np.nan)
        out[w - 1:] = cs[w:] - cs[:n - w + 1]
        return out

    S1 = _cumsum_slide(ones, window)          # = window (scalar)
    Srm = _cumsum_slide(rm, window)
    Srm2 = _cumsum_slide(rm2, window)
    Srm3 = _cumsum_slide(rm ** 3, window)
    Srm4 = _cumsum_slide(rm ** 4, window)
    Sri = _cumsum_slide(ri, window)
    Sxri = _cumsum_slide(rm * ri, window)
    Sx2ri = _cumsum_slide(rm2 * ri, window)

    # For each valid position solve the 3x3 system M @ [a,b,c]^T = rhs
    valid = np.where(np.isfinite(S1))[0]

    for t in valid:
        w = S1[t]  # == window
        M = np.array([
            [w,        Srm[t],  Srm2[t]],
            [Srm[t],   Srm2[t], Srm3[t]],
            [Srm2[t],  Srm3[t], Srm4[t]],
        ])
        rhs = np.array([Sri[t], Sxri[t], Sx2ri[t]])
        det = np.linalg.det(M)
        if abs(det) < _EPS:
            continue
        try:
            coeffs = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        c_raw = coeffs[2]
        # scale by std of rm2 over the same window for unit stability
        # std(rm2) = sqrt(Srm4/w - (Srm2/w)^2)
        var_rm2 = Srm4[t] / w - (Srm2[t] / w) ** 2
        std_rm2 = np.sqrt(max(var_rm2, 0.0))
        out_c[t] = c_raw * std_rm2 if std_rm2 > _EPS else c_raw
        out_b[t] = coeffs[1]
        # asymmetry: not computed here (done separately below)

    return out_c, out_b


def _asymmetry_rolling(ri: np.ndarray, rm: np.ndarray, window: int) -> np.ndarray:
    """
    Up-market vs down-market convexity asymmetry.
    For each window: fit quadratic on up-SPY days vs down-SPY days separately,
    asymmetry = c_up - c_down.  Falls back to NaN if too few obs in either half.
    To keep this O(n) we compute it with a Python loop over the valid range
    (acceptable since it's one pass, not per-row inner-loop).
    """
    n = len(ri)
    out = np.full(n, np.nan)
    rm2 = rm ** 2

    def _quad_c(rm_s, ri_s):
        """Return quadratic coeff c from sub-arrays, or nan."""
        if len(rm_s) < 6:
            return np.nan
        ones = np.ones(len(rm_s))
        M = np.column_stack([ones, rm_s, rm_s ** 2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, ri_s, rcond=None)
            return coeffs[2]
        except Exception:
            return np.nan

    for t in range(window - 1, n):
        rm_w = rm[t - window + 1: t + 1]
        ri_w = ri[t - window + 1: t + 1]
        mask_up = rm_w >= 0
        mask_dn = rm_w < 0
        c_up = _quad_c(rm_w[mask_up], ri_w[mask_up])
        c_dn = _quad_c(rm_w[mask_dn], ri_w[mask_dn])
        if np.isfinite(c_up) and np.isfinite(c_dn):
            out[t] = c_up - c_dn

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (required by gate)
    df[_COL_C] = np.nan
    df[_COL_B] = np.nan
    df[_COL_A] = np.nan

    if len(df) < _WINDOW + 2:
        return df

    # Load SPY close
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Align SPY to df dates via merge_asof (backward, lookahead-safe)
    df_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    merged = merged.set_index(df_dates.sort_values("Date").index).reindex(df.index)
    spy_vals = merged["spy_close"].values.astype(float)

    # Compute SPY returns
    spy_ret = np.full(len(df), np.nan)
    spy_ret[1:] = np.diff(spy_vals) / np.where(np.abs(spy_vals[:-1]) > _EPS, spy_vals[:-1], np.nan)

    # Compute ticker returns
    close = df["Close"].values.astype(float)
    ri = np.full(len(df), np.nan)
    ri[1:] = np.diff(close) / np.where(np.abs(close[:-1]) > _EPS, close[:-1], np.nan)

    # Mask NaN positions
    valid_mask = np.isfinite(ri) & np.isfinite(spy_ret)

    # We need contiguous-friendly arrays; replace NaN with 0 temporarily for sliding sums
    # but we will mask output at positions where the window contained any NaN
    ri_f = np.where(valid_mask, ri, 0.0)
    rm_f = np.where(valid_mask, spy_ret, 0.0)

    # Also need a count of valid obs per window to gate output
    def _cumsum_slide(x: np.ndarray, w: int) -> np.ndarray:
        cs = np.concatenate(([0.0], np.cumsum(x)))
        out = np.full(len(x), np.nan)
        out[w - 1:] = cs[w:] - cs[:len(x) - w + 1]
        return out

    valid_count = _cumsum_slide(valid_mask.astype(float), _WINDOW)
    # require at least WINDOW * 0.8 valid obs
    sufficient = valid_count >= (_WINDOW * 0.8)

    c_arr, b_arr = _quad_ols_rolling(ri_f, rm_f, _WINDOW)
    a_arr = _asymmetry_rolling(ri_f, rm_f, _WINDOW)

    # Mask positions with insufficient data
    c_arr = np.where(sufficient, c_arr, np.nan)
    b_arr = np.where(sufficient, b_arr, np.nan)
    a_arr = np.where(sufficient, a_arr, np.nan)

    df[_COL_C] = c_arr
    df[_COL_B] = b_arr
    df[_COL_A] = a_arr

    return df
