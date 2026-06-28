"""
osap_iomom_cust — Customers Momentum (per-ticker proxy)
========================================================
SOURCE: OpenSourceAP (Chen-Zimmermann); Menzly & Ozbas (2010)
"Market Segmentation and Cross-predictability of Returns", JF 2010.

TRUE METHOD (cross-sectional, not implementable here):
  Download BEA Make-Use Tables, match firms to NAICS industries, compute
  weighted average returns of *customer* industries (rows of Make table →
  cols of Use table), sort into deciles, long deciles 8-10 / short decile 1.
  Captures the lead-lag effect where customer industry returns predict
  supplier stock returns.

PER-TICKER PROXY (this file):
  Without cross-sectional IO table data we approximate the economic signal
  with two components that share the same lead-lag logic:

  1. osap_iomom_cust_lead  — broad-market (SPY) lagged 1-month return as a
     proxy for the "customer industry" momentum that should predict this
     stock. Positive sign expected (strong customer demand → supplier
     outperforms).

  2. osap_iomom_cust_spread — stock's own 1-month return minus the lagged
     SPY 1-month return. A high spread means the stock has already moved
     *with* the demand signal (momentum absorbed); a low/negative spread
     means the stock has *not yet* responded (the IO lead-lag gap). The raw
     signal is the market lead, the spread captures whether the stock has
     already repriced.

  3. osap_iomom_cust_z  — 12-month rolling z-score of the lead signal,
     giving a stationary, scale-free version for model consumption.

  All windows are computed entirely from past data (no negative shifts).
  Expected sign on osap_iomom_cust_lead = +1 (matches paper's predicted sign).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY as broad-market / customer-industry proxy)
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
    "name": "osap_iomom_cust",
    "description": (
        "Per-ticker proxy for customer-industry momentum (Menzly & Ozbas 2010). "
        "True method uses BEA IO tables to weight customer-industry returns and "
        "predict supplier returns (cross-sectional, requires NAICS mapping). "
        "Proxy: lagged SPY 1-month return as the demand-chain signal, stock-vs-market "
        "spread to measure remaining lead-lag gap, and a 12m rolling z-score. "
        "Predicted sign on lead feature = +1 (high customer momentum → buy)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_iomom_cust_lead",    # lagged SPY 21-day return (customer proxy)
        "osap_iomom_cust_spread",  # stock 21d ret minus lagged SPY 21d ret
        "osap_iomom_cust_z",       # 252-day rolling z-score of lead signal
    ],
    "tags": ["momentum", "lead_lag", "cross_predictability", "io_tables", "proxy"],
    "version": "1.0.0",
    "author": "Menzly & Ozbas (2010) JF; Chen & Zimmermann OpenSourceAP; proxy impl by Claude",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MOM_WINDOW = 21      # ~1 trading month for return computation
_LAG = _MOM_WINDOW    # lag the market signal by 1 month (the lead-lag gap)
_Z_WINDOW = 252       # 12-month rolling window for z-score normalisation
_MIN_PERIODS_MOM = 10
_MIN_PERIODS_Z = 63   # at least 3 months before emitting z-score


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker customer-momentum proxy features.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    n = len(df)

    # Initialise output columns to NaN
    df["osap_iomom_cust_lead"] = np.nan
    df["osap_iomom_cust_spread"] = np.nan
    df["osap_iomom_cust_z"] = np.nan

    if n < _MOM_WINDOW + _LAG + 2:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close aligned to this ticker's dates (backward merge)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # DatetimeIndex → float Series
    except Exception:
        return df  # helper unavailable; degrade gracefully

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_dates = pd.to_datetime(df["Date"])
    left = pd.DataFrame({"Date": ticker_dates})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = pd.merge_asof(
            left.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
    # Re-align to original df order
    merged = merged.set_index("Date").reindex(ticker_dates.values)
    spy_vals = merged["spy_close"].values.astype(float)

    # ------------------------------------------------------------------
    # 2. SPY 21-day log return  (past-only)
    #    r_spy[t] = log(spy[t] / spy[t - 21])
    # ------------------------------------------------------------------
    spy_series = pd.Series(spy_vals)
    spy_ret21 = np.log(
        spy_series / spy_series.shift(_MOM_WINDOW)
    )  # length n, leading NaNs expected

    # ------------------------------------------------------------------
    # 3. Lagged SPY return = shift forward by _LAG bars
    #    spy_lead[t] = spy_ret21[t - LAG]
    #    This is the "customer industry" signal that should predict t's stock.
    #    shift(+k) shifts values *down* (uses past data) — no lookahead.
    # ------------------------------------------------------------------
    spy_lead = spy_ret21.shift(_LAG)  # entirely past-looking

    # ------------------------------------------------------------------
    # 4. Stock 21-day log return
    # ------------------------------------------------------------------
    close = pd.Series(df["Close"].values, dtype=float)
    stock_ret21 = np.log(close / close.shift(_MOM_WINDOW))

    # ------------------------------------------------------------------
    # 5. Spread: stock - lagged SPY (measures how much of lead is absorbed)
    # ------------------------------------------------------------------
    spread = stock_ret21 - spy_lead

    # ------------------------------------------------------------------
    # 6. Rolling z-score of the lead signal (normalisation)
    #    z[t] = (spy_lead[t] - mean(spy_lead[t-252:t])) / std(spy_lead[t-252:t])
    # ------------------------------------------------------------------
    lead_mean = spy_lead.rolling(_Z_WINDOW, min_periods=_MIN_PERIODS_Z).mean()
    lead_std = spy_lead.rolling(_Z_WINDOW, min_periods=_MIN_PERIODS_Z).std(ddof=1)
    lead_std = lead_std.replace(0, np.nan)
    lead_z = (spy_lead - lead_mean) / lead_std

    # ------------------------------------------------------------------
    # 7. Write back (replace inf/-inf with NaN for safety)
    # ------------------------------------------------------------------
    def _safe(s: pd.Series) -> np.ndarray:
        arr = s.values.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    df["osap_iomom_cust_lead"] = _safe(spy_lead)
    df["osap_iomom_cust_spread"] = _safe(spread)
    df["osap_iomom_cust_z"] = _safe(lead_z)

    return df
