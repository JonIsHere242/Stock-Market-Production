"""
_p625_expected_idio_skew.py -- Ex-ante (expected) idiosyncratic skewness forecast.

Per-ticker rolling OLS of FUTURE realized idiosyncratic skewness on lagged
characteristics. SPY-residual e = r - (alpha_W + beta_W*m), W=63. At each t an
expanding (min 252) multivariate OLS regresses realized idio-skew[k] (skew of e
over k-62..k) on lagged X = [idioskew_lag, idiovol_lag, mom_lag, disp_lag]
measured at k-63 (plus intercept). X'X and X'y are accumulated via expanding
sums; a 5x5 system is solved once count >= 252. The fitted forecast (not the
realized skew) is the priced object (Boyer-Mitton-Vorkink 2010 RFS;
Conrad-Dittmar-Ghysels 2013 JF). Strictly causal: pair (X_{k-63}, y_k) is fully
observable at time k, and the forecast at t uses only coefficients estimated
from k <= t and features X_t known at t.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared index helper (auto-skipped by the framework; import by file path).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "_p625_expected_idio_skew",
    "description": (
        "Ex-ante idiosyncratic skewness forecast via expanding OLS of realized "
        "idio-skew on lagged characteristics, plus skew surprise "
        "(Boyer-Mitton-Vorkink 2010 RFS; Conrad-Dittmar-Ghysels 2013 JF)."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "exsk_fcast_63",
        "exsk_fcast_126",
        "exsk_resid_skew_63",
        "exsk_idiovol_load_63",
    ],
    "tags":        ["skewness", "lottery", "cross_section", "experimental"],
    "version":     "1.0",
    "author":      "paper:Boyer-Mitton-Vorkink 2010 RFS; Conrad-Dittmar-Ghysels 2013 JF",
}

# --- parameters -------------------------------------------------------------
_W_BETA  = 63    # market-model window for alpha/beta -> residual
_W_SKEW  = 63    # realized idio-skew / idiovol / dispersion window
_W_MOM   = 126   # momentum window (sum of returns)
_MIN_FIT = 252   # minimum (X,y) pairs before a regression is solved
_LAG     = 63    # X is measured at k - _LAG, paired with y measured at k


def _rolling_skew(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling sample skewness (bias-corrected, pandas-compatible).

    Returns NaN for the warmup region (< win finite obs) and where dispersion
    is ~0. Uses sliding_window_view for an O(n*win) vectorized pass.
    """
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if n < win:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    w = _swv(x, win)                      # shape (n-win+1, win), rows end at idx win-1..n-1
    finite = np.isfinite(w).all(axis=1)   # require a full clean window (e has leading NaN)
    if not finite.any():
        return out
    wf = w[finite]
    m = wf.mean(axis=1, keepdims=True)
    d = wf - m
    s2 = (d * d).mean(axis=1)
    std = np.sqrt(s2)
    m3 = (d ** 3).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = m3 / (std ** 3)              # population skew
    # bias correction to match pandas .skew(): sqrt(N(N-1))/(N-2)
    corr = np.sqrt(win * (win - 1.0)) / (win - 2.0)
    g1 = g1 * corr
    g1[~np.isfinite(g1)] = np.nan
    g1[std <= 0] = np.nan
    res = np.full(w.shape[0], np.nan, dtype="float64")
    res[finite] = g1
    out[win - 1:] = res
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    nan_col = np.full(n, np.nan, dtype="float64")

    close = df["Close"].astype(float)
    dates = pd.to_datetime(df["Date"])

    # ---- stock log returns -------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close / close.shift(1))
    r = r.replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float64")

    # ---- align SPY (backward merge_asof on Date) ---------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = pd.Series(dtype="float64")

    if spy_close is None or spy_close.empty:
        # SPY unavailable for this universe -> emit NaN columns, stay contract-clean.
        for c in METADATA["produces"]:
            df[c] = nan_col
        return df

    spy = pd.DataFrame({"Date": pd.to_datetime(spy_close.index.values),
                        "_spy_close": spy_close.to_numpy(dtype="float64")})
    spy = spy.sort_values("Date").reset_index(drop=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        spy["_spy_ret"] = np.log(spy["_spy_close"] / spy["_spy_close"].shift(1))
    spy["_spy_ret"] = spy["_spy_ret"].replace([np.inf, -np.inf], np.nan)

    left = pd.DataFrame({"Date": dates}).reset_index(drop=True)
    merged = pd.merge_asof(
        left.sort_values("Date"),
        spy[["Date", "_spy_ret"]].sort_values("Date"),
        on="Date", direction="backward",
    )
    # merged follows date-sorted order; df is already ascending by Date, so it
    # aligns row-for-row, but reindex defensively onto the original order.
    m = merged["_spy_ret"].to_numpy(dtype="float64")
    if m.shape[0] != n:
        m = np.full(n, np.nan, dtype="float64")

    # ---- rolling market model (alpha_W, beta_W) over W_BETA ----------------
    rs = pd.Series(r)
    ms = pd.Series(m)
    win = _W_BETA
    cov = rs.rolling(win, min_periods=win).cov(ms)
    var = ms.rolling(win, min_periods=win).var()
    var_safe = var.where(var > 0, np.nan)
    beta = (cov / var_safe)
    mean_r = rs.rolling(win, min_periods=win).mean()
    mean_m = ms.rolling(win, min_periods=win).mean()
    alpha = mean_r - beta * mean_m

    e = rs - (alpha + beta * ms)                       # idiosyncratic residual return
    e = e.replace([np.inf, -np.inf], np.nan)
    e_np = e.to_numpy(dtype="float64")

    # ---- realized idio-skew over W_SKEW ------------------------------------
    realized_skew = _rolling_skew(e_np, _W_SKEW)       # y candidate at each k

    # ---- lagged characteristics (measured "now", lagged by _LAG when paired)
    idiovol = e.rolling(_W_SKEW, min_periods=_W_SKEW).std()
    idiovol_np = idiovol.replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float64")

    mom = rs.rolling(_W_MOM, min_periods=_W_MOM).sum()
    mom_np = mom.to_numpy(dtype="float64")

    disp = rs.abs().rolling(_W_SKEW, min_periods=_W_SKEW).mean()
    disp_np = disp.to_numpy(dtype="float64")

    idioskew_np = realized_skew.copy()                 # idioskew characteristic == realized skew

    # Feature matrix X_t (current features at row t), columns:
    #   [1, idioskew, idiovol, mom, disp]
    X = np.column_stack([
        np.ones(n, dtype="float64"),
        idioskew_np,
        idiovol_np,
        mom_np,
        disp_np,
    ])

    # ---- expanding multivariate OLS via accumulated X'X, X'y ---------------
    # Pair (X measured at k-_LAG) -> (y = realized_skew at k). Both known at time k.
    p = X.shape[1]
    XtX = np.zeros((p, p), dtype="float64")
    Xty = np.zeros(p, dtype="float64")
    count = 0

    fcast63  = nan_col.copy()
    fcast126 = nan_col.copy()
    resid63  = nan_col.copy()
    iv_load  = nan_col.copy()

    # Precompute which pair index is valid (finite X_{k-LAG} and finite y_k).
    for t in range(n):
        # 1) Add the pair that becomes fully observable AT time t:
        #    y_t = realized_skew[t], paired with x = X[t - _LAG].
        kx = t - _LAG
        if kx >= 0:
            xrow = X[kx]
            yval = realized_skew[t]
            if np.isfinite(yval) and np.isfinite(xrow).all():
                XtX += np.outer(xrow, xrow)
                Xty += xrow * yval
                count += 1

        # 2) Forecast at t using coefficients estimated from pairs with k <= t
        #    and the CURRENT features X[t] (known at t).
        if count >= _MIN_FIT:
            xt = X[t]
            if np.isfinite(xt).all():
                try:
                    # tiny ridge for numerical stability (does not introduce leakage)
                    b = np.linalg.solve(XtX + 1e-8 * np.eye(p), Xty)
                except np.linalg.LinAlgError:
                    b = None
                if b is not None and np.isfinite(b).all():
                    f63 = float(xt @ b)
                    if np.isfinite(f63):
                        fcast63[t] = f63
                        fcast126[t] = f63        # same fitted object; 126 names the mom_lag horizon
                        rsk = realized_skew[t]
                        if np.isfinite(rsk):
                            resid63[t] = rsk - f63
                        iv_load[t] = float(b[2])  # fitted coefficient on idiovol_lag

    df["exsk_fcast_63"]        = fcast63
    df["exsk_fcast_126"]       = fcast126
    df["exsk_resid_skew_63"]   = resid63
    df["exsk_idiovol_load_63"] = iv_load
    return df
