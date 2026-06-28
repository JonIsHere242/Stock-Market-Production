"""
Intangible Return using EP (IntanEP) — per-ticker proxy.

SOURCE: OpenSourceAP (Chen-Zimmermann); Daniel and Titman 2006.
SPEC ID: osap_intanep

Original method: monthly cross-sectional regression of the firm's 5-year stock
return on (a) lagged EP (net_income / market_cap) and (b) the change in EP over
5 years plus the 5-year stock return.  The residual is IntanEP.

Per-ticker proxy rationale:
  Cross-sectional regression requires the full universe on the same day, which is
  unavailable inside a per-stock compute().  Instead we capture the *same economic
  signal* — the portion of the 5-year return NOT explained by EP change — as:

      proxy_return   = 5-yr log return  (rolling 1260 trading days)
      proxy_ep_chg   = EP_now - EP_5yr_ago   (lagged 1260 bars of EP)
      intanep_proxy  = proxy_return - proxy_ep_chg  (EP contribution subtracted)
      intanep_slope  = 1-yr change in intanep_proxy  (trend / momentum of intangible)

  EP = fund_eps_basic / Close  (inverse of P/E).  Using PIT earnings so no
  look-ahead.  Predicted sign: -1 (long LOW intanEP = tangible outperformers).
  Coverage ~84% of universe; rows without fundamentals emit NaN (expected).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_intanep",
    "description": (
        "Per-ticker proxy for Intangible Return using EP (Daniel & Titman 2006 / "
        "OpenSourceAP Chen-Zimmermann). Original: monthly XS regression of 5-yr "
        "return on lagged EP and delta-EP + 5yr-return; residual = IntanEP. "
        "Proxy: 5-yr log return minus change in EP (earnings/price) over the same "
        "window, capturing the return component unexplained by EP improvement. "
        "Predicted sign: -1 (overperformance driven by intangibles predicts reversal). "
        "Cross-sectional regression replaced by per-ticker EP-adjusted return."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_intanep_proxy",   # 5-yr log return minus 5-yr EP change (intangible proxy)
        "osap_intanep_ep_now",  # current EP level (net_income / market_cap ~ eps/close)
        "osap_intanep_slope",   # 1-yr change in intanep_proxy (trend of intangible component)
    ],
    "tags": ["fundamental", "long_term_reversal", "intangible", "ep", "valuation"],
    "version": "1.0",
    "author": "Daniel and Titman 2006 (OpenSourceAP Chen-Zimmermann); per-ticker proxy by Claude",
}

# Trading-day window constants
_WINDOW_5YR = 1260   # ~5 trading years
_WINDOW_1YR = 252    # ~1 trading year


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_intanep_* columns to a single-ticker OHLCV DataFrame."""

    # -----------------------------------------------------------------------
    # Pull PIT fundamentals: eps_basic for EP calculation
    # -----------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=["eps_basic"])

    close = df["Close"].to_numpy(dtype=float)
    eps = df["fund_eps_basic"].to_numpy(dtype=float)
    n = len(close)

    # -----------------------------------------------------------------------
    # EP = earnings / price  (using trailing eps_basic from PIT filings)
    # Guard: Close==0 -> NaN
    # -----------------------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        ep = np.where(close > 0, eps / close, np.nan)

    # -----------------------------------------------------------------------
    # 5-yr log return:  log(Close_t / Close_{t-1260})
    # -----------------------------------------------------------------------
    ret_5yr = np.full(n, np.nan)
    for t in range(_WINDOW_5YR, n):
        c_now = close[t]
        c_lag = close[t - _WINDOW_5YR]
        if c_now > 0 and c_lag > 0:
            ret_5yr[t] = np.log(c_now / c_lag)

    # -----------------------------------------------------------------------
    # 5-yr EP change:  EP_t - EP_{t-1260}
    # -----------------------------------------------------------------------
    ep_chg_5yr = np.full(n, np.nan)
    for t in range(_WINDOW_5YR, n):
        ep_now = ep[t]
        ep_lag = ep[t - _WINDOW_5YR]
        if np.isfinite(ep_now) and np.isfinite(ep_lag):
            ep_chg_5yr[t] = ep_now - ep_lag

    # -----------------------------------------------------------------------
    # IntanEP proxy:  5-yr return  minus  EP change over 5 years
    # (return component unexplained by fundamentals improvement)
    # -----------------------------------------------------------------------
    intanep = np.where(
        np.isfinite(ret_5yr) & np.isfinite(ep_chg_5yr),
        ret_5yr - ep_chg_5yr,
        np.nan,
    )

    # -----------------------------------------------------------------------
    # 1-yr slope of intanep proxy:  intanep_t - intanep_{t-252}
    # -----------------------------------------------------------------------
    slope = np.full(n, np.nan)
    for t in range(_WINDOW_1YR, n):
        v_now = intanep[t]
        v_lag = intanep[t - _WINDOW_1YR]
        if np.isfinite(v_now) and np.isfinite(v_lag):
            slope[t] = v_now - v_lag

    # -----------------------------------------------------------------------
    # Write produced columns; drop scratch fund_* not in produces
    # -----------------------------------------------------------------------
    df["osap_intanep_proxy"] = intanep
    df["osap_intanep_ep_now"] = ep          # current EP (useful as standalone signal too)
    df["osap_intanep_slope"] = slope

    df.drop(columns=["fund_eps_basic"], inplace=True, errors="ignore")

    return df
