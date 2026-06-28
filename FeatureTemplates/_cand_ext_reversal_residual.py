"""
_cand_ext_reversal_residual.py
Candidate block — market-residual short-term reversal.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# _indexes helper (load by file path — never via package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_reversal_residual",
    "description": (
        "Market-residual short-term reversal.  "
        "Computes the 21-day cumulative idiosyncratic return = stock 21d return "
        "minus (trailing-120d OLS beta vs SPY) * SPY 21d return.  "
        "The reversal signal is the NEGATIVE of this residual (so positive = "
        "idiosyncratically weak stock → expected mean-reversion upward).  "
        "A volume-weighted variant scales by relative volume to emphasise "
        "conviction of the move being reversed.  "
        "This is orthogonal to raw short-term reversal (osap_streversal) because "
        "it removes the market component; the pure idiosyncratic component is the "
        "residual after subtracting beta*market.  "
        "Cross-sectional ranking is NOT applied here (per-ticker proxy only); "
        "the cross-sectional ranking step is deferred to the downstream pipeline."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_reversal_residual_level",   # -1 * 21d idiosyncratic return
        "ext_reversal_residual_vw",      # volume-weighted variant
    ],
    "tags": ["reversal", "residual", "market-neutral", "momentum", "medium-term"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner osap_streversal. "
        "Method: trailing OLS-beta market-residual reversal (classic cross-sectional "
        "literature, e.g. Blitz, Huij & Martens 2011; Gutierrez & Pirinsky 2007)."
    ),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BETA_WINDOW = 120   # trading days for rolling OLS beta estimation
_REV_WINDOW  = 21    # 1-month cumulative return for the reversal signal
_VOL_WINDOW  = 21    # same window for volume-weighting


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext_reversal_residual_level and ext_reversal_residual_vw to df."""
    # Need at least BETA_WINDOW + REV_WINDOW rows for a valid observation
    n = len(df)
    if n < _BETA_WINDOW + _REV_WINDOW:
        df["ext_reversal_residual_level"] = np.nan
        df["ext_reversal_residual_vw"]    = np.nan
        return df

    # ------------------------------------------------------------------
    # 1. Stock log returns (daily)
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    stock_ret = np.empty(n, dtype=np.float64)
    stock_ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # ------------------------------------------------------------------
    # 2. SPY log returns aligned to this ticker's dates
    # ------------------------------------------------------------------
    dates = pd.to_datetime(df["Date"])

    try:
        spy_close = _indexes.index_close("SPY")
        # merge_asof requires sorted Series; spy_close is indexed by DatetimeIndex
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        ticker_dates = pd.DataFrame({"Date": dates})
        merged = pd.merge_asof(
            ticker_dates.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original order
        merged = merged.set_index(ticker_dates.sort_values("Date").index).reindex(df.index)
        spy_c = merged["spy_close"].values.astype(np.float64)
    except Exception:
        spy_c = np.full(n, np.nan)

    spy_ret = np.empty(n, dtype=np.float64)
    spy_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        spy_ret[1:] = np.log(np.where(spy_c[:-1] > 0, spy_c[1:] / spy_c[:-1], np.nan))

    # ------------------------------------------------------------------
    # 3. Rolling 120-day OLS beta (causal, no lookahead)
    #    beta_t = cov(stock_ret, spy_ret) / var(spy_ret) over [t-119, t]
    # ------------------------------------------------------------------
    # Use a sliding view approach; fall back to pd.Series.rolling for clarity
    s_ret = pd.Series(stock_ret)
    m_ret = pd.Series(spy_ret)

    # rolling covariance and variance (ddof=1, but ratio is same for ddof=0)
    roll_cov = s_ret.rolling(_BETA_WINDOW, min_periods=60).cov(m_ret)
    roll_var = m_ret.rolling(_BETA_WINDOW, min_periods=60).var()

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(
            (roll_var.values > 0),
            roll_cov.values / roll_var.values,
            np.nan,
        )

    # ------------------------------------------------------------------
    # 4. 21-day cumulative returns (stock and SPY)
    #    cum_ret[t] = Close[t] / Close[t - 21] - 1  (simple return)
    # ------------------------------------------------------------------
    # Use log for additivity then convert; but simple return is standard in
    # short-term reversal literature.  We use simple returns here.
    with np.errstate(divide="ignore", invalid="ignore"):
        past_close = np.empty(n, dtype=np.float64)
        past_close[:] = np.nan
        past_close[_REV_WINDOW:] = close[: n - _REV_WINDOW]

        past_spy = np.empty(n, dtype=np.float64)
        past_spy[:] = np.nan
        past_spy[_REV_WINDOW:] = spy_c[: n - _REV_WINDOW]

        stock_cum = np.where(
            (past_close > 0) & np.isfinite(past_close),
            close / past_close - 1.0,
            np.nan,
        )
        spy_cum = np.where(
            (past_spy > 0) & np.isfinite(past_spy),
            spy_c / past_spy - 1.0,
            np.nan,
        )

    # ------------------------------------------------------------------
    # 5. Idiosyncratic 21d return = stock_cum - beta * spy_cum
    # ------------------------------------------------------------------
    idio = np.where(
        np.isfinite(beta) & np.isfinite(stock_cum) & np.isfinite(spy_cum),
        stock_cum - beta * spy_cum,
        np.nan,
    )

    # Reversal signal: NEGATIVE of idiosyncratic return
    reversal_level = -idio

    # ------------------------------------------------------------------
    # 6. Volume-weighted variant
    #    Scale by relative volume: vol_t / avg_vol over same 21d window
    #    High relative volume ⟹ larger position for the reversal bet
    # ------------------------------------------------------------------
    vol = df["Volume"].values.astype(np.float64)
    avg_vol = (
        pd.Series(vol)
        .rolling(_VOL_WINDOW, min_periods=max(1, _VOL_WINDOW // 2))
        .mean()
        .values
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_vol = np.where(avg_vol > 0, vol / avg_vol, np.nan)

    reversal_vw = np.where(
        np.isfinite(reversal_level) & np.isfinite(rel_vol),
        reversal_level * rel_vol,
        np.nan,
    )

    # Guard: replace any inf/-inf with NaN
    reversal_level = np.where(np.isfinite(reversal_level), reversal_level, np.nan)
    reversal_vw    = np.where(np.isfinite(reversal_vw),    reversal_vw,    np.nan)

    # ------------------------------------------------------------------
    # 7. Assign back (preserve index alignment)
    # ------------------------------------------------------------------
    df["ext_reversal_residual_level"] = reversal_level
    df["ext_reversal_residual_vw"]    = reversal_vw

    return df
