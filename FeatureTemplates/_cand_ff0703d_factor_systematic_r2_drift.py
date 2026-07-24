"""
_cand_ff0703d_factor_systematic_r2_drift.py -- systematic (factor) R^2 level + drift.

METHOD
------
Build three daily factor return series aligned by Date from the market indexes:
    m = SPY.pct_change()                     (market)
    s = IWM.pct_change() - m                  (size, small-minus-broad-market proxy)
    g = QQQ.pct_change() - m                  (growth, tech/growth-minus-broad-market proxy)

On a FIXED-FROM-START stride (i % 5 == 0, forward-filled between grid points) run a
trailing 60-day OLS of the stock's daily return r on [1, m, s, g] and record
    R^2 = 1 - SSres / SStot
i.e. the share of the stock's return variance explained by the three common factors
over the trailing 60 sessions ("systematic" vs idiosyncratic variance split). SStot<=0
(constant returns) guards to NaN. R^2 is clipped to [0, 1].

LEVEL   = current (forward-filled) systematic R^2.
DYNAMIC = R^2_t - R^2_{t-60}: positive means the stock has been "de-idiosyncratizing" --
          loading up on common/systematic factors over the last quarter; negative means
          it has been decoupling into idiosyncratic-return territory.

This is a per-ticker proxy for a genuinely cross-sectional factor model (no true
multi-factor cross-section, no fitted factor loadings shared across stocks) -- SPY/IWM/QQQ
excess returns stand in for market/size/growth factors, which is the closest faithful
implementation reachable from OHLCV + the shared index helper alone.

Causality: the stride grid is anchored to the START of the series (i % 5 == 0) and each
regression uses only the trailing 60 rows ending at the grid point -- both survive
truncation-based lookahead checks. Index factor data is joined with a backward merge_asof
on Date (never anchors to a future bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import importlib.util as _ilu
from pathlib import Path as _P

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

_SPEC = "ff0703d_factor_systematic_r2_drift"
_COL_LEVEL = f"{_SPEC}_level"
_COL_DYN = f"{_SPEC}_dynamic"

_STRIDE = 5
_WINDOW = 60
_MIN_VALID = 20  # minimum valid (non-NaN-factor) rows within the 60-row window to fit

METADATA = {
    "name": _SPEC,
    "description": (
        "Trailing 60-day OLS R^2 of stock returns on [1, SPY_mkt, IWM-SPY_size, "
        "QQQ-SPY_growth] factor returns, computed on a from-start stride-5 grid and "
        "forward-filled ('systematic' variance share). Level = current R^2 (clipped "
        "[0,1]); dynamic = 60-day change in R^2 (rising => de-idiosyncratizing / "
        "loading up on common factors). Per-ticker proxy for a cross-sectional "
        "multi-factor model using SPY/IWM/QQQ as market/size/growth stand-ins."
    ),
    "requires": ["Close"],
    "produces": [_COL_LEVEL, _COL_DYN],
    "tags": ["factor", "beta", "systematic-risk", "regime"],
    "version": "1.0",
    "author": "feature-factory (ff0703d, honest per-ticker index-factor proxy)",
}


def _causal_ffill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = np.nan
    for i in range(out.shape[0]):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    level = np.full(n, np.nan, dtype="float64")

    if n == 0:
        df[_COL_LEVEL] = level
        df[_COL_DYN] = level.copy()
        return df

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype="float64")
    r = np.full(n, np.nan, dtype="float64")
    if n > 1:
        prev = close[:-1]
        cur = close[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(prev != 0, (cur - prev) / prev, np.nan)
        r[1:] = pct

    # --- factor series from shared index helper, merged backward onto df's Date ---
    m_col = np.full(n, np.nan, dtype="float64")
    s_col = np.full(n, np.nan, dtype="float64")
    g_col = np.full(n, np.nan, dtype="float64")

    try:
        spy = _indexes.index_close("SPY")
        iwm = _indexes.index_close("IWM")
        qqq = _indexes.index_close("QQQ")
    except Exception:
        spy = pd.Series(dtype="float64")
        iwm = pd.Series(dtype="float64")
        qqq = pd.Series(dtype="float64")

    if len(spy) > 1 and "Date" in df.columns:
        spy = spy.sort_index()
        m = spy.pct_change()
        factors = pd.DataFrame({"Date": m.index, "mkt_m": m.to_numpy(dtype="float64")})

        if len(iwm) > 1:
            iwm = iwm.sort_index()
            m_iwm = iwm.pct_change()
            iwm_df = pd.DataFrame({"Date": m_iwm.index, "_iwm_ret": m_iwm.to_numpy(dtype="float64")})
            factors = factors.merge(iwm_df, on="Date", how="outer")
        else:
            factors["_iwm_ret"] = np.nan

        if len(qqq) > 1:
            qqq = qqq.sort_index()
            m_qqq = qqq.pct_change()
            qqq_df = pd.DataFrame({"Date": m_qqq.index, "_qqq_ret": m_qqq.to_numpy(dtype="float64")})
            factors = factors.merge(qqq_df, on="Date", how="outer")
        else:
            factors["_qqq_ret"] = np.nan

        factors = factors.sort_values("Date").drop_duplicates(subset="Date", keep="last")
        factors["mkt_s"] = factors["_iwm_ret"] - factors["mkt_m"]
        factors["mkt_g"] = factors["_qqq_ret"] - factors["mkt_m"]
        factors = factors[["Date", "mkt_m", "mkt_s", "mkt_g"]]

        # df is guaranteed ascending-by-Date per the framework contract, so a direct
        # merge_asof (no re-sort/re-index gymnastics needed) preserves row order.
        left = pd.DataFrame({"Date": pd.to_datetime(df["Date"]).to_numpy()})
        factors["Date"] = pd.to_datetime(factors["Date"])
        merged = pd.merge_asof(left, factors.sort_values("Date"), on="Date", direction="backward")

        m_col = merged["mkt_m"].to_numpy(dtype="float64")
        s_col = merged["mkt_s"].to_numpy(dtype="float64")
        g_col = merged["mkt_g"].to_numpy(dtype="float64")

    # --- trailing-60 OLS R^2 on a from-start stride-5 grid ---
    if n >= _WINDOW:
        for idx in range(_WINDOW - 1, n, _STRIDE):
            lo = idx - _WINDOW + 1
            y_win = r[lo:idx + 1]
            x1 = m_col[lo:idx + 1]
            x2 = s_col[lo:idx + 1]
            x3 = g_col[lo:idx + 1]

            valid = (
                ~np.isnan(y_win) & ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(x3)
            )
            n_valid = int(valid.sum())
            if n_valid < _MIN_VALID:
                continue

            y_v = y_win[valid]
            sstot = float(np.sum((y_v - y_v.mean()) ** 2))
            if not np.isfinite(sstot) or sstot <= 0:
                continue

            X = np.column_stack([
                np.ones(n_valid, dtype="float64"),
                x1[valid], x2[valid], x3[valid],
            ])
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
            except Exception:
                continue

            resid = y_v - X @ coef
            ssres = float(np.sum(resid ** 2))
            if not np.isfinite(ssres):
                continue

            r2 = 1.0 - ssres / sstot
            if not np.isfinite(r2):
                continue
            r2 = min(1.0, max(0.0, r2))
            level[idx] = r2

    level = _causal_ffill(level)

    dyn = np.full(n, np.nan, dtype="float64")
    if n > _WINDOW:
        dyn[_WINDOW:] = level[_WINDOW:] - level[:-_WINDOW]

    df[_COL_LEVEL] = level
    df[_COL_DYN] = dyn
    return df
