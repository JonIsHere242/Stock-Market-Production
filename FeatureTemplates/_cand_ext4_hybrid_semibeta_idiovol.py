"""
_cand_ext4_hybrid_semibeta_idiovol.py
Candidate feature block: Downside semibeta scaled by idiosyncratic volatility.

Rolling 120-day downside semibeta vs SPY, divided by idiosyncratic vol (= residual vol
after removing the market component). This gives a risk-adjusted downside loading.
Also produces the product (semibeta * idiovol = raw downside risk).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never via package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_hybrid_semibeta_idiovol",
    "description": (
        "Rolling 120-day downside semibeta of the stock vs SPY (only days where SPY "
        "return < 0 enter the regression numerator/denominator), divided by the "
        "stock's idiosyncratic volatility (residual std from a 120-day rolling "
        "OLS market model). The ratio (ext4_hybrid_semibeta_idiovol_ratio) isolates "
        "how much downside market loading a unit of idio-risk carries — a higher "
        "value means disproportionate downside exposure relative to idio-noise. "
        "The product (ext4_hybrid_semibeta_idiovol_product) captures total downside "
        "risk. Per-ticker proxy faithful to the xdom2_downside_beta extension."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_hybrid_semibeta_idiovol_semibeta",   # raw 120d downside semibeta
        "ext4_hybrid_semibeta_idiovol_idiovol",    # 120d idiosyncratic vol (annualised)
        "ext4_hybrid_semibeta_idiovol_ratio",      # semibeta / idiovol  (main signal)
        "ext4_hybrid_semibeta_idiovol_product",    # semibeta * idiovol  (total downside)
    ],
    "tags": ["risk", "beta", "downside", "idiovol", "market-model", "SPY"],
    "version": "1.0",
    "author": "Round-5 expansion (xdom2_downside_beta) — Jonathan Bellmont / Claude",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120          # rolling window in trading days
_MIN_OBS = 40          # minimum observations inside window before we emit a value
_MIN_DOWN = 10         # minimum down-market days inside window for semibeta to be valid
_ANNUALISE = np.sqrt(252)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute downside semibeta / idiovol features.

    Steps per rolling window:
      1. Align SPY daily close to df dates (merge_asof backward).
      2. Compute daily log-returns for stock and SPY.
      3. Rolling 120-day:
         a. Downside semibeta = Cov(r_stock, r_spy | r_spy<0) / Var(r_spy | r_spy<0)
            implemented as a rolling sum of products / rolling sum of sq on down-days.
         b. Full-beta OLS (via covariance) + residual std = idiovol.
      4. ratio = semibeta / idiovol; product = semibeta * idiovol.
    """
    # ------------------------------------------------------------------ #
    # 1. Fetch SPY index series and align
    # ------------------------------------------------------------------ #
    spy_close: pd.Series | None = None
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    # Build a helper dataframe of dates + stock returns
    dates = pd.to_datetime(df["Date"])

    stock_ret = np.log(df["Close"].replace(0, np.nan)).diff()

    spy_ret: pd.Series
    if spy_close is not None and len(spy_close) > 0:
        spy_idx = pd.DataFrame({"Date": spy_close.index, "_spy_close": spy_close.values})
        spy_idx["Date"] = pd.to_datetime(spy_idx["Date"])

        tmp = pd.DataFrame({"Date": dates.values, "_stock_ret": stock_ret.values})
        tmp = pd.merge_asof(
            tmp.sort_values("Date"),
            spy_idx.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original row order
        tmp = tmp.set_index(dates.sort_values().index)
        tmp = tmp.reindex(df.index)

        spy_ret = np.log(tmp["_spy_close"].replace(0, np.nan)).diff()
        # spy_ret is per-date not per-spy-bar; since we merged asof, the spy_close
        # might repeat on non-trading SPY days.  Recompute SPY return on aligned series.
        # Actually, we need SPY daily return aligned to stock dates.
        # Safer: compute SPY ret from the merged aligned SPY closes directly.
        spy_ret_aligned = pd.Series(
            np.log(tmp["_spy_close"].replace(0, np.nan).values),
            index=df.index,
        ).diff()
    else:
        spy_ret_aligned = pd.Series(np.nan, index=df.index)

    sr = stock_ret.values.astype(float)       # shape (n,)
    mr = spy_ret_aligned.values.astype(float) # shape (n,)

    n = len(df)
    W = _WINDOW

    semibeta_arr  = np.full(n, np.nan)
    idiovol_arr   = np.full(n, np.nan)
    ratio_arr     = np.full(n, np.nan)
    product_arr   = np.full(n, np.nan)

    for i in range(W - 1, n):
        s_win = sr[i - W + 1 : i + 1]   # shape (W,)
        m_win = mr[i - W + 1 : i + 1]

        # Drop NaN pairs
        mask_valid = np.isfinite(s_win) & np.isfinite(m_win)
        sv = s_win[mask_valid]
        mv = m_win[mask_valid]

        if len(sv) < _MIN_OBS:
            continue

        # ---- Full-market model (OLS): stock_ret = alpha + beta * mkt_ret + resid ----
        m_mean = mv.mean()
        s_mean = sv.mean()
        mkt_var = np.sum((mv - m_mean) ** 2)
        if mkt_var < 1e-14:
            continue
        beta_full = np.sum((sv - s_mean) * (mv - m_mean)) / mkt_var
        resid = sv - (s_mean + beta_full * (mv - m_mean))
        idiovol = resid.std(ddof=1) * _ANNUALISE   # annualised idio vol
        if not np.isfinite(idiovol) or idiovol < 1e-10:
            idiovol = np.nan

        # ---- Downside semibeta (only on down-market days) ----
        down_mask = mv < 0
        n_down = down_mask.sum()
        if n_down < _MIN_DOWN:
            semibeta = np.nan
        else:
            sv_d = sv[down_mask]
            mv_d = mv[down_mask]
            mv_d_mean = mv_d.mean()
            sv_d_mean = sv_d.mean()
            var_d = np.sum((mv_d - mv_d_mean) ** 2)
            if var_d < 1e-14:
                semibeta = np.nan
            else:
                semibeta = np.sum((sv_d - sv_d_mean) * (mv_d - mv_d_mean)) / var_d

        semibeta_arr[i] = semibeta

        if np.isfinite(idiovol):
            idiovol_arr[i] = idiovol

        if np.isfinite(semibeta) and np.isfinite(idiovol) and idiovol > 0:
            ratio_arr[i]   = semibeta / idiovol
            product_arr[i] = semibeta * idiovol

    df["ext4_hybrid_semibeta_idiovol_semibeta"] = semibeta_arr
    df["ext4_hybrid_semibeta_idiovol_idiovol"]  = idiovol_arr
    df["ext4_hybrid_semibeta_idiovol_ratio"]    = ratio_arr
    df["ext4_hybrid_semibeta_idiovol_product"]  = product_arr

    return df
