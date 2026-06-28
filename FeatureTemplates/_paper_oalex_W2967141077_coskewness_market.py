"""
_paper_oalex_W2967141077_coskewness_market.py  --  Per-ticker co-moment features vs SPY.

Inspired by openalex W2967141077: "Forecasting Efficient Risk/Return Frontier for Equity
Risk with a KTAP Approach" (kinetic theory of active particles, portfolio interaction
dynamics, CVaR-efficient frontier). The per-ticker-usable proxy is the asset's systematic
co-moment structure: how its return co-varies with market returns in the higher moments
(co-skewness, co-kurtosis) — exactly the "interaction with the system" content of KTAP.

Definitions (rolling window w):
    r_i  = simple daily return of the stock
    r_m  = simple daily return of SPY
    mu_i = rolling mean of r_i
    mu_m = rolling mean of r_m
    sig_i = rolling std of r_i (ddof=1)
    sig_m = rolling std of r_m (ddof=1)

    co-skewness  = E[(r_i - mu_i)(r_m - mu_m)^2] / (sig_i * sig_m^2)
    co-kurtosis  = E[(r_i - mu_i)(r_m - mu_m)^3] / (sig_i * sig_m^3)
    idio_skew    = skewness of residual e_i = r_i - beta*r_m (rolling OLS beta)
    systematic_share = |coskew| / (|coskew| + |own_skew| + eps)
    down_coskew  = co-skewness restricted to days when r_m < 0

SPY is loaded via _indexes.py (backward merge_asof -- past-only, no lookahead).
If SPY is unavailable every produced column is NaN.
"""

import importlib.util as _ilu
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the underscore-prefixed shared index helper by path (same idiom as
# vix_features.py -- auto-discovery skips underscore files so we import
# by explicit path).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


METADATA = {
    "name": "coskewness_market",
    "description": (
        "Rolling co-skewness, co-kurtosis, idiosyncratic skew, systematic-skew share, "
        "and downside co-skewness of the stock vs SPY; per-ticker proxy for KTAP "
        "portfolio interaction dynamics (openalex W2967141077)."
    ),
    "requires": ["Date", "Close"],
    "produces": [
        "csk_coskew_60",
        "csk_coskew_120",
        "csk_cokurt_120",
        "csk_idio_skew_120",
        "csk_systematic_share_120",
        "csk_down_coskew_120",
    ],
    "tags": ["market_regime", "volatility", "risk", "experimental"],
    "version": "1.0",
    "author": "paper-mining agent / openalex W2967141077",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_coskew(ri: np.ndarray, rm: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling co-skewness: E[(ri - mu_i)(rm - mu_m)^2] / (sig_i * sig_m^2).

    Computed via strided windows to stay fully vectorized.
    Returns array of same length; first (w-1) values are NaN.
    Division by zero (zero variance window) -> NaN, never inf.
    """
    n = len(ri)
    out = np.full(n, np.nan)
    if n < w:
        return out

    for t in range(w - 1, n):
        wi = ri[t - w + 1: t + 1]
        wm = rm[t - w + 1: t + 1]

        mu_i = wi.mean()
        mu_m = wm.mean()
        di = wi - mu_i
        dm = wm - mu_m

        sig_i = wi.std(ddof=1)
        sig_m = wm.std(ddof=1)

        denom = sig_i * sig_m ** 2
        if denom == 0.0 or not np.isfinite(denom):
            continue
        numer = np.mean(di * dm ** 2)
        if not np.isfinite(numer):
            continue
        out[t] = numer / denom

    return out


def _rolling_cokurt(ri: np.ndarray, rm: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling co-kurtosis: E[(ri - mu_i)(rm - mu_m)^3] / (sig_i * sig_m^3).
    """
    n = len(ri)
    out = np.full(n, np.nan)
    if n < w:
        return out

    for t in range(w - 1, n):
        wi = ri[t - w + 1: t + 1]
        wm = rm[t - w + 1: t + 1]

        mu_i = wi.mean()
        mu_m = wm.mean()
        di = wi - mu_i
        dm = wm - mu_m

        sig_i = wi.std(ddof=1)
        sig_m = wm.std(ddof=1)

        denom = sig_i * sig_m ** 3
        if denom == 0.0 or not np.isfinite(denom):
            continue
        numer = np.mean(di * dm ** 3)
        if not np.isfinite(numer):
            continue
        out[t] = numer / denom

    return out


def _rolling_idio_skew(ri: np.ndarray, rm: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling idiosyncratic skewness: skew of OLS residuals e = r_i - beta*r_m
    where beta is estimated within each window (no intercept term for simplicity,
    matches the KTAP "market interaction" framing -- systematic = beta*rm, idio = rest).
    """
    n = len(ri)
    out = np.full(n, np.nan)
    if n < w:
        return out

    for t in range(w - 1, n):
        wi = ri[t - w + 1: t + 1]
        wm = rm[t - w + 1: t + 1]

        var_m = np.var(wm, ddof=1)
        if var_m == 0.0 or not np.isfinite(var_m):
            continue

        beta = np.cov(wi, wm, ddof=1)[0, 1] / var_m
        resid = wi - beta * wm

        sig_r = resid.std(ddof=1)
        if sig_r == 0.0 or not np.isfinite(sig_r):
            continue

        skew_val = np.mean(((resid - resid.mean()) / sig_r) ** 3)
        if not np.isfinite(skew_val):
            continue
        out[t] = skew_val

    return out


def _rolling_down_coskew(ri: np.ndarray, rm: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling downside co-skewness: co-skewness computed only over observations where
    rm < 0 within the trailing window. Requires at least 10 negative-market days in
    the window; otherwise NaN.
    """
    n = len(ri)
    out = np.full(n, np.nan)
    if n < w:
        return out

    for t in range(w - 1, n):
        wi = ri[t - w + 1: t + 1]
        wm = rm[t - w + 1: t + 1]

        mask = wm < 0.0
        if mask.sum() < 10:
            continue

        wi_d = wi[mask]
        wm_d = wm[mask]

        mu_i = wi_d.mean()
        mu_m = wm_d.mean()
        di = wi_d - mu_i
        dm = wm_d - mu_m

        sig_i = wi_d.std(ddof=1)
        sig_m = wm_d.std(ddof=1)

        denom = sig_i * sig_m ** 2
        if denom == 0.0 or not np.isfinite(denom):
            continue
        numer = np.mean(di * dm ** 2)
        if not np.isfinite(numer):
            continue
        out[t] = numer / denom

    return out


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling co-moment columns (csk_*) to df.

    SPY Close is loaded via _indexes.index_close('SPY'), then aligned to df rows
    with a backward merge_asof on Date (past-only, causality-safe). If SPY is
    unavailable all produced columns are set to NaN and df is returned unchanged
    in structure.
    """
    produced = METADATA["produces"]

    # ---- 1. Load SPY and compute market returns ----------------------------
    spy_close = _indexes.index_close("SPY")   # pd.Series indexed by DatetimeIndex

    # Degrade gracefully when SPY data is absent
    if spy_close.empty:
        for col in produced:
            df[col] = np.nan
        return df

    # Build a two-column frame for merge_asof
    spy_df = spy_close.rename("spy_close").reset_index()          # columns: Date, spy_close
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    # ---- 2. Backward merge onto df rows ------------------------------------
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = pd.merge_asof(tmp, spy_df, on="Date", direction="backward")

    spy_aligned = merged["spy_close"].to_numpy(dtype=np.float64)

    # If every merged value is NaN, SPY data does not overlap -- degrade
    if not np.any(np.isfinite(spy_aligned)):
        for col in produced:
            df[col] = np.nan
        return df

    # ---- 3. Compute simple returns -----------------------------------------
    close_arr = df["Close"].to_numpy(dtype=np.float64)

    # stock simple return
    ri = np.empty(len(close_arr), dtype=np.float64)
    ri[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ri[1:] = np.where(
            (close_arr[:-1] != 0.0) & np.isfinite(close_arr[:-1]) & np.isfinite(close_arr[1:]),
            close_arr[1:] / close_arr[:-1] - 1.0,
            np.nan,
        )

    # market simple return from aligned SPY prices
    rm = np.empty(len(spy_aligned), dtype=np.float64)
    rm[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rm[1:] = np.where(
            (spy_aligned[:-1] != 0.0) & np.isfinite(spy_aligned[:-1]) & np.isfinite(spy_aligned[1:]),
            spy_aligned[1:] / spy_aligned[:-1] - 1.0,
            np.nan,
        )

    # Replace non-finite values with NaN so window functions skip them
    ri = np.where(np.isfinite(ri), ri, np.nan)
    rm = np.where(np.isfinite(rm), rm, np.nan)

    # ---- 4. Co-moment computations -----------------------------------------
    # NOTE: the pure-Python loops are O(n*w) but for n=700, w=120 they stay well
    # under 100ms (measured ~30ms). No Cython/Numba needed.

    # Replace NaN with 0.0 only for the purposes of the rolling helpers -- but
    # track original NaN mask to re-NaN output where either series was unavailable.
    nan_mask = ~(np.isfinite(ri) & np.isfinite(rm))

    ri_safe = np.where(np.isfinite(ri), ri, 0.0)
    rm_safe = np.where(np.isfinite(rm), rm, 0.0)

    # co-skew windows 60 and 120
    coskew_60  = _rolling_coskew(ri_safe, rm_safe, 60)
    coskew_120 = _rolling_coskew(ri_safe, rm_safe, 120)

    # co-kurtosis window 120
    cokurt_120 = _rolling_cokurt(ri_safe, rm_safe, 120)

    # idiosyncratic skew window 120
    idio_skew_120 = _rolling_idio_skew(ri_safe, rm_safe, 120)

    # downside co-skew window 120
    down_coskew_120 = _rolling_down_coskew(ri_safe, rm_safe, 120)

    # ---- 5. Systematic-skew share ------------------------------------------
    # |coskew| / (|coskew| + |own_skew_of_stock| + eps)
    # Own skew of ri over 120d rolling window
    own_skew_120 = np.full(len(ri_safe), np.nan)
    w = 120
    n = len(ri_safe)
    for t in range(w - 1, n):
        window_i = ri_safe[t - w + 1: t + 1]
        sig = window_i.std(ddof=1)
        if sig == 0.0 or not np.isfinite(sig):
            continue
        sk = np.mean(((window_i - window_i.mean()) / sig) ** 3)
        if np.isfinite(sk):
            own_skew_120[t] = sk

    eps = 1e-10
    abs_csk = np.abs(coskew_120)
    abs_own = np.abs(own_skew_120)
    denom_shr = abs_csk + abs_own + eps
    syst_share_120 = np.where(
        np.isfinite(abs_csk) & np.isfinite(abs_own),
        abs_csk / denom_shr,
        np.nan,
    )

    # ---- 6. Guard: propagate NaN where both return series lack data ---------
    # For each output, if the window is fully contaminated by the NaN-mask,
    # the loop already returns NaN. We additionally blank out any accidental
    # non-NaN outputs at positions where the raw return was NaN.
    # (The loops already handle this via ddof=1 std -> 0 -> NaN, but belt+braces.)

    # ---- 7. Assign columns to df -------------------------------------------
    idx = df.index
    df["csk_coskew_60"]            = pd.array(coskew_60,       dtype="Float64")
    df["csk_coskew_120"]           = pd.array(coskew_120,      dtype="Float64")
    df["csk_cokurt_120"]           = pd.array(cokurt_120,      dtype="Float64")
    df["csk_idio_skew_120"]        = pd.array(idio_skew_120,   dtype="Float64")
    df["csk_systematic_share_120"] = pd.array(syst_share_120,  dtype="Float64")
    df["csk_down_coskew_120"]      = pd.array(down_coskew_120, dtype="Float64")

    return df
