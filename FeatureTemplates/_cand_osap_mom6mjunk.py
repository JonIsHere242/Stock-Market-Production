"""
Junk Stock Momentum (Avramov et al. 2007) — per-ticker proxy.

Cross-sectional method: original paper computes 6-month momentum restricted to
stocks with a credit rating of BBB or lower (junk/high-yield universe).
Since splticrm (S&P credit rating) is not available, this block proxies
"junk" status using PIT fundamental signals: debt-to-equity, interest coverage
(EBIT / interest expense), and net margin. Stocks that look financially
distressed / speculative are flagged as "junk-like"; the momentum signal is
then conditioned on that flag.

Columns produced:
  osap_mom6mjunk_raw   : raw 6-month momentum (t-1 to t-126 business days),
                         available for all stocks regardless of junk status.
  osap_mom6mjunk_score : junk-proxy score (higher = more junk-like; 0-3 scale
                         summing three fundamental distress flags). NaN when no
                         fundamental data available.
  osap_mom6mjunk_sig   : the conditioned signal — mom6m value when the stock
                         looks junk-like (score >= 2), else NaN. This is the
                         closest faithful per-ticker proxy to the paper's
                         cross-sectional filter. Use osap_mom6mjunk_raw if you
                         want unconditional momentum.

Proxy compromise: credit ratings are not available; distress proxies (leverage,
interest coverage, net margin) are imperfect substitutes — coverage is ~84%
for stocks with SEC filings; ETFs/foreign companies produce NaN.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional: PIT fundamentals helper
# ---------------------------------------------------------------------------
try:
    _s2 = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _HAS_FUND = False

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_mom6mjunk",
    "description": (
        "Junk Stock Momentum (Avramov et al. 2007 via Chen-Zimmermann OpenSourceAP). "
        "6-month price momentum restricted to speculative/distressed stocks. "
        "Original cross-sectional filter uses S&P credit ratings (BBB or lower); "
        "proxied here with PIT fundamental distress signals: high debt-to-equity, "
        "low interest coverage, and negative/low net margin. "
        "osap_mom6mjunk_raw is unconditional; osap_mom6mjunk_sig fires only when "
        "the stock appears junk-like (score >= 2/3 distress flags triggered)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_mom6mjunk_raw",
        "osap_mom6mjunk_score",
        "osap_mom6mjunk_sig",
    ],
    "tags": ["momentum", "credit", "junk", "distress", "osap"],
    "version": "1.0",
    "author": "Avramov, Chordia, Jostova, Philipov (2007); OpenSourceAP Chen-Zimmermann; proxy impl.",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame for a single ticker, ascending by Date.
         Columns guaranteed: Date, Ticker, Open, High, Low, Close, Volume.

    Returns
    -------
    df with three new columns appended.
    """
    n = len(df)

    # ------------------------------------------------------------------
    # 1. 6-month momentum: return from t-126 to t-1 (skip last day to
    #    avoid microstructure reversal bias, matching the paper convention).
    #    126 ~ 6 months of trading days.
    # ------------------------------------------------------------------
    close = df["Close"].to_numpy(dtype=np.float64)

    # Price 126 days ago and 1 day ago (both strictly in the past)
    # shift(1)  = yesterday's close  (t-1)
    # shift(126)= ~6 months ago close (t-126)
    close_s = pd.Series(close, index=df.index)
    p_1   = close_s.shift(1)
    p_126 = close_s.shift(126)

    # raw momentum: (p_{t-1} / p_{t-126}) - 1
    denom = p_126.to_numpy(dtype=np.float64)
    denom_safe = np.where(denom == 0.0, np.nan, denom)
    mom6m_raw = (p_1.to_numpy(dtype=np.float64) / denom_safe) - 1.0
    # Replace inf/-inf
    mom6m_raw = np.where(np.isfinite(mom6m_raw), mom6m_raw, np.nan)

    df["osap_mom6mjunk_raw"] = mom6m_raw

    # ------------------------------------------------------------------
    # 2. Junk proxy score from PIT fundamentals (0–3 scale).
    #    Flag 1: high leverage — debt_to_equity > 2.0
    #    Flag 2: low interest coverage — operating_income / interest_expense < 3
    #            (interest_expense in fundamentals is typically signed positive)
    #    Flag 3: low profitability — net_margin < 0.02 (i.e. thin / negative)
    # ------------------------------------------------------------------
    junk_score = np.zeros(n, dtype=np.float64)
    has_fund_data = False

    if _HAS_FUND:
        try:
            df = _fundamentals.as_of(
                df,
                fields=[
                    "debt_to_equity",
                    "operating_income",
                    "interest_expense",
                    "net_margin",
                ],
            )
            has_fund_data = True

            dte   = df["fund_debt_to_equity"].to_numpy(dtype=np.float64)
            oi    = df["fund_operating_income"].to_numpy(dtype=np.float64)
            ie    = df["fund_interest_expense"].to_numpy(dtype=np.float64)
            nm    = df["fund_net_margin"].to_numpy(dtype=np.float64)

            # Flag 1: leverage
            flag_lev = np.where(np.isfinite(dte), (dte > 2.0).astype(np.float64), np.nan)

            # Flag 2: interest coverage < 3  (avoid div-by-zero on ie==0)
            ie_safe = np.where(ie == 0.0, np.nan, ie)
            icr = oi / ie_safe
            # If interest expense is zero/nan (no debt), company is NOT junk on this metric
            flag_icr = np.where(
                np.isfinite(icr),
                (icr < 3.0).astype(np.float64),
                np.nan,
            )

            # Flag 3: thin/negative profitability
            flag_nm = np.where(np.isfinite(nm), (nm < 0.02).astype(np.float64), np.nan)

            # Score: sum of flags (NaN if all three are NaN)
            stack = np.column_stack([flag_lev, flag_icr, flag_nm])
            any_valid = np.any(np.isfinite(stack), axis=1)
            junk_score_vals = np.nansum(stack, axis=1).astype(np.float64)
            junk_score_vals = np.where(any_valid, junk_score_vals, np.nan)

            junk_score = junk_score_vals

            # Drop scratch columns
            df = df.drop(
                columns=[
                    c for c in [
                        "fund_debt_to_equity",
                        "fund_operating_income",
                        "fund_interest_expense",
                        "fund_net_margin",
                    ]
                    if c in df.columns
                ]
            )
        except Exception:
            junk_score = np.full(n, np.nan)
            has_fund_data = False

    if not has_fund_data:
        junk_score = np.full(n, np.nan)

    df["osap_mom6mjunk_score"] = junk_score

    # ------------------------------------------------------------------
    # 3. Conditioned signal: momentum only when stock looks junk-like
    #    (score >= 2, i.e. at least 2 of 3 distress flags triggered).
    # ------------------------------------------------------------------
    score_arr = df["osap_mom6mjunk_score"].to_numpy(dtype=np.float64)
    sig = np.where(
        np.isfinite(score_arr) & (score_arr >= 2.0),
        mom6m_raw,
        np.nan,
    )
    df["osap_mom6mjunk_sig"] = sig

    return df
