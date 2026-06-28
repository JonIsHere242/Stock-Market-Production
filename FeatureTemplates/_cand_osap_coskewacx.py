"""
Coskewness (Ang, Chen & Xing 2006) — per-ticker proxy using SPY as market.

Signal = E[r̃_i * r̃_m²] / (SD[r̃_i] * SD[r̃_m]²)

where r̃_i is the de-meaned daily log-return of the stock and r̃_m is the
de-meaned daily log-return of SPY (proxy for market).  Computed over a rolling
252-day lookback (roughly one trading year).  The original paper uses the CRSP
NYSE VW index; SPY is the closest available per-ticker proxy.

Predicted cross-sectional sign: -1 (stocks with lower coskewness earn higher
returns as compensation for downside co-movement with the market).

CROSS-SECTIONAL NOTE: The original signal ranks stocks against each other.
Here each ticker is computed standalone, so the absolute level is comparable
across tickers at the same point in time but cannot use cross-sectional
standardisation.  The slope variant (_chg) measures momentum in coskewness
(26-week change) which captures deteriorating/improving co-skew dynamics.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_coskewacx",
    "description": (
        "Coskewness (Ang, Chen & Xing 2006): rolling 252-day sample coskewness "
        "E[r̃_i * r̃_m²] / (SD[r̃_i] * SD[r̃_m]²) using SPY log-returns as the "
        "market proxy.  Predicted sign -1 (low coskewness = downside co-movement "
        "risk premium).  Also emits a 26-week change to capture dynamic shifts. "
        "Per-ticker proxy; cross-sectional ranking not applied."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_coskewacx_252d",   # rolling 252-day coskewness
        "osap_coskewacx_126d",   # rolling 126-day coskewness (shorter window)
        "osap_coskewacx_chg",    # 26-week (63 trading-day) change in 252d coskewness
    ],
    "tags": ["risk", "coskewness", "market", "ang-chen-xing", "rolling"],
    "version": "1.0",
    "author": "Ang, Chen and Xing (2006) — OpenSourceAP (Chen-Zimmermann); per-ticker proxy by Claude",
}


# ---------------------------------------------------------------------------
# Load _indexes helper (by file path — do NOT use a package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes",
    _P(__file__).resolve().parent / "_indexes.py",
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_coskewacx_* columns to df (one stock at a time, sorted ascending by Date)."""

    # ------------------------------------------------------------------ #
    # 1. Stock log-returns (no lookahead — shift(1) uses yesterday's close)
    # ------------------------------------------------------------------ #
    close = df["Close"].values.astype(np.float64)
    log_ret_i = np.empty(len(close), dtype=np.float64)
    log_ret_i[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret_i[1:] = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )

    # ------------------------------------------------------------------ #
    # 2. Market (SPY) log-returns aligned to this stock's dates
    # ------------------------------------------------------------------ #
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
        spy_close = spy_close.dropna()

        # Build a two-column frame for merge_asof
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "_spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        stock_dates = pd.to_datetime(df["Date"]).sort_values()
        left_df = pd.DataFrame({"Date": stock_dates})

        merged = pd.merge_asof(
            left_df.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        spy_vals = merged["_spy_close"].values.astype(np.float64)

        log_ret_m = np.empty(len(spy_vals), dtype=np.float64)
        log_ret_m[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret_m[1:] = np.where(
                (spy_vals[:-1] > 0) & (spy_vals[1:] > 0),
                np.log(spy_vals[1:] / spy_vals[:-1]),
                np.nan,
            )

        # If merged result is not in original df order, re-index
        # (merge_asof can reorder rows if dates were not aligned)
        # We must map back to df's original row order.
        orig_dates = pd.to_datetime(df["Date"]).values
        merged_dates = merged["Date"].values
        if not np.array_equal(orig_dates, merged_dates):
            # Build a mapping via sorted positions
            date_to_mret = dict(zip(merged["Date"].values, log_ret_m))
            log_ret_m = np.array(
                [date_to_mret.get(d, np.nan) for d in orig_dates], dtype=np.float64
            )
    except Exception:
        log_ret_m = np.full(len(df), np.nan)

    # ------------------------------------------------------------------ #
    # 3. Vectorised rolling coskewness helper
    # ------------------------------------------------------------------ #
    def _rolling_coskew(ri: np.ndarray, rm: np.ndarray, window: int) -> np.ndarray:
        """
        For each position t compute over [t-window+1 .. t]:
            num   = mean( r̃_i * r̃_m² )
            denom = std(r̃_i) * std(r̃_m)²
        Both r̃ are de-meaned WITHIN the window (zero-mean residuals).
        Returns array of same length, NaN where window not yet full.
        """
        n = len(ri)
        out = np.full(n, np.nan, dtype=np.float64)

        for t in range(window - 1, n):
            wi = ri[t - window + 1 : t + 1]
            wm = rm[t - window + 1 : t + 1]

            # Drop any NaN pairs
            mask = np.isfinite(wi) & np.isfinite(wm)
            if mask.sum() < max(20, window // 5):
                continue

            wi_c = wi[mask]
            wm_c = wm[mask]

            # De-mean within window
            wi_dm = wi_c - wi_c.mean()
            wm_dm = wm_c - wm_c.mean()

            numerator = np.mean(wi_dm * wm_dm ** 2)

            std_i = wi_dm.std(ddof=1)
            std_m = wm_dm.std(ddof=1)

            denom = std_i * std_m ** 2
            if denom == 0 or not np.isfinite(denom):
                continue

            out[t] = numerator / denom

        return out

    # ------------------------------------------------------------------ #
    # 4. Compute features
    # ------------------------------------------------------------------ #
    coskew_252 = _rolling_coskew(log_ret_i, log_ret_m, 252)
    coskew_126 = _rolling_coskew(log_ret_i, log_ret_m, 126)

    # 26-week change (63 trading days) in the 252-day coskewness
    coskew_chg = np.empty(len(coskew_252), dtype=np.float64)
    coskew_chg[:] = np.nan
    coskew_chg[63:] = coskew_252[63:] - coskew_252[:-63]

    # Replace any inf that slipped through
    def _clean(arr: np.ndarray) -> np.ndarray:
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    df["osap_coskewacx_252d"] = _clean(coskew_252)
    df["osap_coskewacx_126d"] = _clean(coskew_126)
    df["osap_coskewacx_chg"] = _clean(coskew_chg)

    return df
