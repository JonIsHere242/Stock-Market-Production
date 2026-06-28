"""
osap_bmdec — Book-to-Market using December Market Equity
Fama and French (1992), via OpenSourceAP (Chen-Zimmermann).

Per-ticker proxy: BM = book_equity / december_market_equity
  - book_equity  = fund_equity (PIT, backward merge_asof on filed_date)
  - december_ME  = Close[Dec] × fund_shares_outstanding[Dec], using the most
                   recent December month-end close and PIT shares as of that date.
                   Updated once per year (calendar year of the most recent December).

Cross-sectional note: the original factor sorts ALL stocks by BM in June using
December ME; here we compute the raw BM ratio per ticker. Sorting/ranking across
the universe is left to the downstream cross-sectional feature layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# load fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_bmdec",
    "description": (
        "Book-to-Market ratio using December market equity (Fama-French 1992). "
        "book_equity from PIT SEC fundamentals (fund_equity); market equity anchored "
        "to the most recent December month-end (Close × shares_outstanding as of that "
        "date). Produces the raw BM level and a 12-month rolling z-score of BM "
        "(per-ticker valuation cheapness trend). Cross-sectional sorting is NOT done "
        "here; downstream XS layer handles ranking. Coverage ~84% (ETFs/foreign = NaN)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_bmdec_ratio",   # BM = book_equity / december_ME
        "osap_bmdec_log",     # log(BM) -- more normal distribution
        "osap_bmdec_zscore",  # 12-month rolling z-score of log(BM)
    ],
    "tags": ["valuation", "fundamental", "fama-french", "book-to-market"],
    "version": "1.0",
    "author": "Fama and French 1992 / OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BM using December ME per Fama-French 1992.
    df is a single ticker, ascending Date, columns: Date, Ticker, Open, High, Low, Close, Volume.
    """
    # -- pull PIT fundamentals: equity + shares_outstanding
    df = _fundamentals.as_of(df, fields=["equity", "shares_outstanding"])

    # Ensure Date is datetime for month/year extraction
    dates = pd.to_datetime(df["Date"])

    # -- identify December month-end rows (month == 12)
    is_december = dates.dt.month == 12

    # Build a per-row "most recent December ME" via forward-fill from December values.
    # December ME = Close × shares_outstanding on that December bar.
    dec_me = pd.Series(np.nan, index=df.index)

    shares = df["fund_shares_outstanding"].values
    close = df["Close"].values

    # compute ME only on December rows; shares may be PIT-lagged (safe)
    me_vals = np.where(is_december, close * shares, np.nan)
    dec_me_series = pd.Series(me_vals, index=df.index)

    # forward-fill so every row carries the most recent December ME
    dec_me_ffill = dec_me_series.ffill()

    # -- book equity (PIT from fundamentals)
    book_eq = df["fund_equity"]  # already NaN where no data

    # -- BM ratio: guard against zero / negative ME
    with np.errstate(divide="ignore", invalid="ignore"):
        bm_ratio = np.where(
            (dec_me_ffill.values > 0) & (~np.isnan(book_eq.values)) & (~np.isnan(dec_me_ffill.values)),
            book_eq.values / dec_me_ffill.values,
            np.nan,
        )

    bm_ratio = pd.Series(bm_ratio, index=df.index)

    # -- log BM (more normally distributed; standard in FF literature)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_bm = np.where(bm_ratio.values > 0, np.log(bm_ratio.values), np.nan)
    log_bm = pd.Series(log_bm, index=df.index)

    # -- 12-month (≈252 trading day) rolling z-score of log BM (per-ticker trend)
    window = 252
    roll_mean = log_bm.rolling(window, min_periods=63).mean()
    roll_std = log_bm.rolling(window, min_periods=63).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        bm_zscore = np.where(
            roll_std.values > 0,
            (log_bm.values - roll_mean.values) / roll_std.values,
            np.nan,
        )

    df["osap_bmdec_ratio"] = bm_ratio.values
    df["osap_bmdec_log"] = log_bm.values
    df["osap_bmdec_zscore"] = bm_zscore

    # drop scratch fund_ columns not in produces
    for col in ["fund_equity", "fund_shares_outstanding"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
