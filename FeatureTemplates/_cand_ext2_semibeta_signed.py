"""
Bollerslev-Li-Patton (2022) signed semibetas.

Decompose rolling beta into sign-conditioned quadrant covariances:
  - beta_N  : concordant downside  (stock down & SPY down) -- priced positively
  - beta_P  : concordant upside    (stock up   & SPY up)
  - beta_Mn : mixed / discordant   (stock down & SPY up) + (stock up & SPY down) -- priced negatively

Per-ticker proxy using SPY daily returns from _indexes.
Window = 120 trading days (matching the original paper's monthly frequency on daily returns).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ── load _indexes helper ────────────────────────────────────────────────────
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ── metadata ────────────────────────────────────────────────────────────────
METADATA = {
    "name": "ext2_semibeta_signed",
    "description": (
        "Bollerslev-Li-Patton (2022) signed semibetas over a 120-day rolling window. "
        "Daily returns of the stock and SPY are decomposed into four sign quadrants: "
        "beta_N = concordant downside cov (stock<0 & SPY<0), "
        "beta_P = concordant upside cov (stock>0 & SPY>0), "
        "beta_Mn = mixed/discordant cov (sign-disagree days). "
        "beta_N is priced positively; beta_Mn is priced negatively. "
        "Per-ticker proxy using SPY from _indexes; zero-return days allocated to both "
        "halves consistently with Bollerslev et al. (2022)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_semibeta_signed_N",
        "ext2_semibeta_signed_P",
        "ext2_semibeta_signed_Mn",
    ],
    "tags": ["beta", "semibeta", "downside_risk", "market", "signed"],
    "version": "1.0",
    "author": "Bollerslev, Li & Patton (2022) 'Realized Semibetas'; spec: Round-3 xdom2 extension",
}

# ── helpers ──────────────────────────────────────────────────────────────────

_WINDOW = 120


def _rolling_semibeta(
    r_stock: np.ndarray,
    r_spy: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return arrays (beta_N, beta_P, beta_Mn) computed with a causal rolling window.
    Semibeta quadrants (Bollerslev et al. 2022, eq 3-5):
      beta_N  = E[r_s * r_m | r_s<=0, r_m<=0] / var_m(r_m<=0)   (normalised by full mkt variance)
      In practice we compute as:
        cov_NN = mean(r_s * r_m, where both <= 0) * frac_NN
      then divide by var(r_m) full-sample to keep units consistent with OLS beta.
    """
    n = len(r_stock)
    beta_N = np.full(n, np.nan, dtype=np.float64)
    beta_P = np.full(n, np.nan, dtype=np.float64)
    beta_Mn = np.full(n, np.nan, dtype=np.float64)

    for t in range(window - 1, n):
        rs = r_stock[t - window + 1 : t + 1]   # shape (window,)
        rm = r_spy[t - window + 1 : t + 1]

        var_m = np.var(rm, ddof=1)
        if var_m == 0 or np.isnan(var_m):
            continue

        # Quadrant masks (strict halves; zero treated as negative, consistent with BLP)
        mask_NN = (rs <= 0) & (rm <= 0)
        mask_PP = (rs > 0)  & (rm > 0)
        mask_NP = (rs <= 0) & (rm > 0)   # discordant A
        mask_PN = (rs > 0)  & (rm <= 0)  # discordant B

        frac_NN = mask_NN.sum() / window
        frac_PP = mask_PP.sum() / window
        frac_NP = mask_NP.sum() / window
        frac_PN = mask_PN.sum() / window

        # Quadrant semi-covariances (mean product within quadrant * quadrant frequency)
        def _qcov(mask: np.ndarray, frac: float) -> float:
            if frac == 0:
                return 0.0
            return float(np.mean(rs[mask] * rm[mask])) * frac

        cov_NN = _qcov(mask_NN, frac_NN)
        cov_PP = _qcov(mask_PP, frac_PP)
        cov_NP = _qcov(mask_NP, frac_NP)
        cov_PN = _qcov(mask_PN, frac_PN)

        beta_N[t]  = cov_NN / var_m
        beta_P[t]  = cov_PP / var_m
        beta_Mn[t] = (cov_NP + cov_PN) / var_m   # discordant total

    return beta_N, beta_P, beta_Mn


# ── main compute ─────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # initialise output columns to NaN
    df["ext2_semibeta_signed_N"]  = np.nan
    df["ext2_semibeta_signed_P"]  = np.nan
    df["ext2_semibeta_signed_Mn"] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # --- SPY returns via _indexes ---
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # align SPY to this stock's dates (backward merge = no lookahead)
    dates = pd.to_datetime(df["Date"])
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        pd.DataFrame({"Date": dates}).sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # restore original order
    merged = merged.set_index(dates.sort_values().index).reindex(df.index)

    spy_vals = merged["spy_close"].values.astype(float)

    # stock close
    stock_close = df["Close"].values.astype(float)

    # daily log returns (shift-safe: only backward references)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_stock = np.diff(np.log(np.where(stock_close > 0, stock_close, np.nan)))
        r_spy   = np.diff(np.log(np.where(spy_vals   > 0, spy_vals,   np.nan)))

    # r_* have length n-1; index offset = 1 (day 0 has no return)
    n_ret = len(r_stock)
    if n_ret < _WINDOW:
        return df

    beta_N, beta_P, beta_Mn = _rolling_semibeta(r_stock, r_spy, _WINDOW)

    # beta_N/P/Mn are indexed over returns (length n_ret).
    # Return at position t (0-based in r_*) corresponds to df row t+1.
    # Assign via iloc slice so we stay vectorised.
    n_df = len(df)
    # rows 1..n_ret (or n_df-1, whichever is smaller)
    out_len = min(n_ret, n_df - 1)
    if out_len > 0:
        iloc_idx = df.index[1 : out_len + 1]
        df.loc[iloc_idx, "ext2_semibeta_signed_N"]  = beta_N[:out_len]
        df.loc[iloc_idx, "ext2_semibeta_signed_P"]  = beta_P[:out_len]
        df.loc[iloc_idx, "ext2_semibeta_signed_Mn"] = beta_Mn[:out_len]

    return df
