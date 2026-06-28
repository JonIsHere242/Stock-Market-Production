"""
Underwater duration & recovery factor (ext3_underwater_duration)

Tracks how long a stock has been trading below its rolling 120-day peak,
how deep into a current drawdown run it is, and how much it has recovered
from the trough relative to the maximum drawdown. Pure OHLCV; no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_underwater_duration",
    "description": (
        "Three per-ticker signals derived from rolling 120-day peak tracking. "
        "(1) ext3_underwater_frac: fraction of the last 120 calendar days spent "
        "below the running 120d closing peak -- a sustained-stress indicator. "
        "(2) ext3_underwater_runlen: number of consecutive days (up to today) "
        "since the price last touched or exceeded its 120d peak -- current run length. "
        "(3) ext3_recovery_factor: gain from the deepest trough back to today's close, "
        "divided by the maximum drawdown from peak to trough over the 120d window; "
        "captures whether the stock is bouncing from a completed drawdown. "
        "All three are causal (use only past Close), fully vectorised, and guard "
        "against divide-by-zero by returning NaN. Orthogonal to downside-beta / "
        "volatility features -- duration of stress is a distinct axis."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_underwater_frac",
        "ext3_underwater_runlen",
        "ext3_recovery_factor",
    ],
    "tags": ["drawdown", "underwater", "recovery", "duration", "ohlcv"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom2_downside_beta); spec: ext3_underwater_duration",
}

_WINDOW = 120  # look-back window in trading days


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    underwater_frac = np.full(n, np.nan)
    underwater_runlen = np.full(n, np.nan)
    recovery_factor = np.full(n, np.nan)

    # We need at least 2 rows to produce anything meaningful; wait for _WINDOW
    # rows before emitting (leading NaNs are expected and acceptable).
    if n < 2:
        df["ext3_underwater_frac"] = underwater_frac
        df["ext3_underwater_runlen"] = underwater_runlen
        df["ext3_recovery_factor"] = recovery_factor
        return df

    # Compute rolling 120d peak using cummax within a sliding window.
    # For each day t we look at close[max(0,t-_WINDOW+1) : t+1].
    # To vectorise efficiently, use pandas rolling on the close series.
    close_s = pd.Series(close)

    # Rolling peak (causal: includes current bar, no lookahead).
    rolling_peak = close_s.rolling(_WINDOW, min_periods=1).max().to_numpy()

    # --- (1) Underwater fraction ---
    # For each day t: fraction of the _WINDOW days in [t-_WINDOW+1..t] where
    # close < rolling peak AT THAT DAY (i.e., the stock was underwater).
    # "Underwater on day s" = close[s] < running_peak_up_to_s.
    underwater_bool = (close < rolling_peak).astype(np.float64)  # 1 if below peak
    uw_frac_s = pd.Series(underwater_bool).rolling(_WINDOW, min_periods=1).mean()
    underwater_frac = uw_frac_s.to_numpy()

    # --- (2) Current underwater run length ---
    # Number of consecutive trailing days (ending today) where close < rolling_peak.
    # Vectorise: for each t, find the most recent day s <= t where close[s] >= peak[s],
    # then runlen = t - s.  We walk forward; reset counter when not underwater.
    runlen = np.zeros(n, dtype=np.float64)
    for t in range(n):
        if close[t] < rolling_peak[t]:
            runlen[t] = (runlen[t - 1] + 1) if t > 0 else 1.0
        else:
            runlen[t] = 0.0
    underwater_runlen = runlen

    # --- (3) Recovery factor ---
    # Over the _WINDOW look-back:
    #   peak_price  = max close in window (= rolling_peak[t])
    #   trough_price = min close in window
    #   max_drawdown = peak_price - trough_price  (in price terms)
    #   recovery     = close[t] - trough_price
    #   recovery_factor = recovery / max_drawdown  (0 = still at trough, 1 = full recovery)
    rolling_trough = close_s.rolling(_WINDOW, min_periods=1).min().to_numpy()
    max_dd = rolling_peak - rolling_trough          # >= 0 always
    recovery = close - rolling_trough               # >= 0 always

    # Guard: when max_dd == 0 (flat price), return NaN (no drawdown to speak of).
    with np.errstate(invalid="ignore", divide="ignore"):
        rf = np.where(max_dd > 0.0, recovery / max_dd, np.nan)

    recovery_factor = rf

    # Mask leading rows where window is not yet full to keep semantics clean.
    # min_periods=1 gives values from row 0, which is fine per spec (leading NaNs
    # are only *expected*, not required); we leave them as computed.

    df["ext3_underwater_frac"] = underwater_frac
    df["ext3_underwater_runlen"] = underwater_runlen
    df["ext3_recovery_factor"] = recovery_factor

    return df
