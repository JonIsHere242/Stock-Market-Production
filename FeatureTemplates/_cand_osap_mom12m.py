"""
Candidate feature block: osap_mom12m
12-month price momentum (Jegadeesh-Titman / OpenSourceAP mom12m).

Per OpenSourceAP (Chen & Zimmermann) the signal is the cumulative return
from month t-12 to month t-2 (skipping the most recent month to avoid
the well-known 1-month reversal contamination).

On a DAILY basis we operationalise this as:
  - main  : cumulative return from 252 trading days ago to 21 trading days
             ago (approx 12-month window skipping ~1 month lag).
  - decay : ratio of the last-63-day sub-window return (within the mom12m
             window) divided by the first-63-day sub-window return, capped
             at [-3, 3].  Captures whether momentum is accelerating (>1)
             or decelerating (<1) -- a refinement used in later papers.
  - rank  : trailing percentile rank (252-day expanding from 252) of the
             main signal across the stock's own history.  NOT cross-
             sectional; this is a per-ticker time-series rank that captures
             how extreme the current momentum reading is for THIS stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_mom12m",
    "description": (
        "12-month price momentum (t-12 to t-2, skipping most-recent month). "
        "Per OpenSourceAP (Chen & Zimmermann), following Jegadeesh & Titman (1993). "
        "Predicted sign: +1 (high past-year return predicts high future return). "
        "Daily proxy: cumulative log-return from 252 to 21 bars ago. "
        "Also produces a momentum-decay ratio and a per-ticker TS rank of the main signal."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_mom12m_main",    # cumulative return t-252 to t-21
        "osap_mom12m_decay",   # ratio of recent 63d vs early 63d sub-window
        "osap_mom12m_tsrank",  # per-ticker percentile rank of main (252-bar rolling)
    ],
    "tags": ["momentum", "price", "osap", "jegadeesh_titman"],
    "version": "1.0",
    "author": (
        "Jegadeesh & Titman (1993); OpenSourceAP catalog: "
        "Chen & Zimmermann (2022) 'Open Source Cross-Sectional Asset Pricing'. "
        "Per-ticker daily proxy implemented for this pipeline."
    ),
}

# ---------------------------------------------------------------------------
# Constants (in trading days)
# ---------------------------------------------------------------------------
_SKIP   = 21    # ~1 calendar month skip (avoids short-term reversal)
_TOTAL  = 252   # ~12 calendar months
_SUB    = 63    # ~3 months sub-window for decay ratio


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    main   = np.full(n, np.nan)
    decay  = np.full(n, np.nan)

    # We need at least _TOTAL + 1 bars to have a valid observation
    for i in range(_TOTAL, n):
        # Price at start of 12m window (i - 252) and end (i - 21)
        p_start = close[i - _TOTAL]
        p_end   = close[i - _SKIP]

        if p_start <= 0 or p_end <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
            continue

        # Cumulative log return from t-252 to t-21
        main[i] = np.log(p_end / p_start)

        # Decay ratio: recent 63d (t-84 to t-21) vs early 63d (t-252 to t-189)
        p_early_start = close[i - _TOTAL]        # t-252
        p_early_end   = close[i - _TOTAL + _SUB] # t-189
        p_late_start  = close[i - _SKIP - _SUB]  # t-84
        p_late_end    = close[i - _SKIP]          # t-21

        if (p_early_start > 0 and p_early_end > 0 and
                p_late_start > 0 and p_late_end > 0 and
                np.isfinite(p_early_start) and np.isfinite(p_early_end) and
                np.isfinite(p_late_start) and np.isfinite(p_late_end)):

            ret_early = np.log(p_early_end / p_early_start)
            ret_late  = np.log(p_late_end  / p_late_start)

            # Avoid dividing by zero / near-zero; express as difference
            # (log scale: decay = late minus early momentum contribution)
            decay[i] = np.clip(ret_late - ret_early, -3.0, 3.0)

    # TS rank of main over trailing 252 bars (per-ticker rank in its own history)
    main_s  = pd.Series(main, index=df.index)
    tsrank  = main_s.rolling(window=_TOTAL, min_periods=_TOTAL).rank(pct=True)

    df["osap_mom12m_main"]   = main
    df["osap_mom12m_decay"]  = decay
    df["osap_mom12m_tsrank"] = tsrank.to_numpy()

    return df
