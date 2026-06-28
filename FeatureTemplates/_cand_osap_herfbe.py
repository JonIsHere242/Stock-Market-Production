"""
Per-ticker proxy for Hou & Robinson (2006) industry concentration (Herfindahl-HHI
based on book equity).

The true measure is cross-sectional: HHI = sum_i(s_i^2) where s_i = firm_i book
equity / total industry book equity, averaged over 3 years.  We cannot compute
that per-ticker in isolation, so we build three PIT-fundamental proxies that
capture the same economic signal:

  osap_herfbe_pb_inv  : inverse price-to-book (book_value_per_share / Close).
      High book/price = low-growth / stable firm → concentrated industry proxy.
      Predicted sign in Hou-Robinson is −1 (high concentration → low returns),
      so high pb_inv stocks are expected to UNDER-perform.

  osap_herfbe_eq_stab : stability of book equity (1 − rolling CV over 3yr
      annual windows resampled from quarterly filings).  Concentrated industries
      show stable, slowly growing equity; high stability ≈ high concentration.

  osap_herfbe_eq_growth : 3-year log-growth of book equity per share (from PIT
      fundamentals).  Slow equity growth = mature concentrated industry.
      Again predicted negative for returns.

All values are backward-merged on filing date (point-in-time safe).
ETFs / foreign firms with no fundamentals will emit NaN — expected.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (by file path — required pattern)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_herfbe",
    "description": (
        "Per-ticker proxy for Hou & Robinson (2006) industry Herfindahl concentration "
        "based on book equity (OpenSourceAP / Chen-Zimmermann). True measure requires "
        "cross-sectional HHI across all firms in a 3-digit SIC code; per-ticker proxy "
        "uses: (1) inverse price-to-book as a concentration/maturity signal, "
        "(2) rolling book-equity stability (low CV = concentrated/stable), and "
        "(3) 3-year book-equity growth (slow growth = concentrated industry). "
        "Predicted sign is −1: high concentration predicts lower subsequent returns."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_herfbe_pb_inv",
        "osap_herfbe_eq_stab",
        "osap_herfbe_eq_growth",
    ],
    "tags": ["fundamentals", "concentration", "herfindahl", "book_equity", "cross_sectional_proxy"],
    "version": "1.0.0",
    "author": "Hou and Robinson 2006 (OpenSourceAP / Chen-Zimmermann); per-ticker proxy implementation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker Herfindahl-concentration proxies from PIT fundamentals.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending Date, columns include at minimum
        Date, Ticker, Close.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    # Pull PIT fundamentals: book_value_per_share and equity (total book equity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["book_value_per_share", "equity"])

    bvps = df["fund_book_value_per_share"]
    eq = df["fund_equity"]
    close = df["Close"]

    # ------------------------------------------------------------------
    # 1. Inverse price-to-book  (book / price)
    #    High = low-growth / mature firm → proxy for concentrated industry
    # ------------------------------------------------------------------
    pb = close.where(close > 0, other=np.nan)
    bvps_pos = bvps.where(bvps > 0, other=np.nan)
    pb_inv = bvps_pos / pb                # book / price
    df["osap_herfbe_pb_inv"] = pb_inv

    # ------------------------------------------------------------------
    # 2. Book-equity stability: 1 − rolling_std / rolling_mean over ~3yr
    #    window on equity.  We use 756 trading days (~3yr) with min 252.
    #    High stability → concentrated/mature industry proxy.
    # ------------------------------------------------------------------
    window = 756
    min_p = 252
    eq_pos = eq.where(eq > 0, other=np.nan)

    roll_mean = eq_pos.rolling(window, min_periods=min_p).mean()
    roll_std = eq_pos.rolling(window, min_periods=min_p).std(ddof=1)

    # CV (coefficient of variation); guard zero mean
    cv = roll_std / roll_mean.where(roll_mean != 0, other=np.nan)
    eq_stab = 1.0 - cv.clip(lower=0.0)   # stability in (−∞, 1]; high = stable
    df["osap_herfbe_eq_stab"] = eq_stab

    # ------------------------------------------------------------------
    # 3. 3-year book-equity growth (log scale, backward looking)
    #    Slow / negative growth = mature / concentrated industry proxy.
    #    We shift by 756 trading days to get the value ~3yr ago.
    # ------------------------------------------------------------------
    eq_lag = eq_pos.shift(window)
    # log growth; guard divide by zero (already guarded by positivity mask)
    log_growth = np.log(
        eq_pos / eq_lag.where(eq_lag > 0, other=np.nan)
    )
    df["osap_herfbe_eq_growth"] = log_growth

    # ------------------------------------------------------------------
    # Drop scratch fund_* columns that are NOT in produces
    # ------------------------------------------------------------------
    df = df.drop(columns=["fund_book_value_per_share", "fund_equity"], errors="ignore")

    return df
