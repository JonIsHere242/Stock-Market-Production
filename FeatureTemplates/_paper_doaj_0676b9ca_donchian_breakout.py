"""
_paper_doaj_0676b9ca_donchian_breakout.py

Donchian channel / Turtle breakout features, inspired by the channel-breakout
timing strategy described in:

  "An SVM improvement prediction in multifactor model for stocks selection"
  (DOAJ paper id 0676b9ca)

The paper builds a multifactor stock-selection model that uses Moving Average
(MA) and Channel Breakout (CB) timing strategies to reduce drawdown. This block
implements the classic Donchian / Turtle Trader interpretation of CB at two
horizons (20-day and 55-day).

CAUSALITY DESIGN
----------------
- POSITION columns (dch_pos_*): use the CURRENT-row-inclusive channel.
  The channel upper/lower at row t includes all highs/lows through row t,
  which is acceptable for a positional ratio (where does today's Close
  sit inside today's full observed range?). These are descriptive scalars
  with no lookahead, because they only reference prices already known at
  close-of-day t.

- BREAKOUT columns (dch_breakout_*): use the PRIOR-window channel via
  rolling(N).shift(1) so the current bar's High/Low do NOT contribute to
  the threshold that the current Close is tested against. This prevents a
  bar from being its own breakout trigger.

- dch_time_since_breakout_*: trailing bar count since the last up or down
  breakout event, computed from the prior-window breakout flags. Fully causal.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_doaj_0676b9ca_donchian_breakout",
    "description": (
        "Donchian channel position, width, and breakout signals at 20-day and "
        "55-day horizons; proxy for the channel-breakout timing strategy in the "
        "DOAJ SVM multifactor stock-selection paper (id 0676b9ca)."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "dch_pos_20",
        "dch_pos_55",
        "dch_width_20",
        "dch_width_55",
        "dch_breakout_up_20",
        "dch_breakout_dn_20",
        "dch_breakout_up_55",
        "dch_breakout_dn_55",
        "dch_time_since_breakout_20",
        "dch_time_since_breakout_55",
    ],
    "tags": ["trend", "momentum", "technical", "breakout"],
    "version": "1.0",
    "author": "paper proxy — DOAJ 0676b9ca (SVM multifactor model / channel breakout)",
}


# ---------------------------------------------------------------------------
# Helper: bars since any True in a boolean Series (vectorised, causal)
# ---------------------------------------------------------------------------
def _bars_since_event(event: pd.Series) -> pd.Series:
    """
    Return a Series of the number of bars elapsed since the most recent True
    in `event`. Returns NaN until the first event occurs.

    Fully causal: at position t the result only depends on event[0..t].
    """
    arr = event.to_numpy(dtype=bool)
    n = len(arr)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if arr[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return pd.Series(out, index=event.index)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Donchian channel features at 20-day and 55-day windows.

    Position features (dch_pos_*):
        Current Close relative to the current-bar-inclusive channel.
        = (Close - lower_N) / (upper_N - lower_N), guarded against zero width.

    Width features (dch_width_*):
        Normalised channel width relative to Close.
        = (upper_N - lower_N) / Close, guarded against zero Close.

    Breakout features (dch_breakout_up/dn_*):
        1.0 when the current Close breaches the PRIOR window's upper/lower band;
        otherwise 0.0.  NaN during the warmup period (< N rows of prior history).
        Prior-window band: rolling(N).max(High).shift(1) / rolling(N).min(Low).shift(1)

    Time-since-breakout (dch_time_since_breakout_*):
        Bars elapsed since the most recent up OR down breakout event (either
        direction). NaN until the first event.
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # Safe Close denominator (replace zero with NaN to avoid inf)
    close_safe = close.replace(0.0, np.nan)

    for N in (20, 55):
        # ------------------------------------------------------------------
        # POSITION: current-bar-inclusive channel
        # ------------------------------------------------------------------
        upper_incl = high.rolling(N, min_periods=N).max()
        lower_incl = low.rolling(N,  min_periods=N).min()

        width = upper_incl - lower_incl
        # Guard: if channel has zero width set position to NaN
        pos = np.where(
            width > 0,
            (close - lower_incl) / width,
            np.nan,
        )
        df[f"dch_pos_{N}"] = pd.Series(pos, index=df.index)

        # ------------------------------------------------------------------
        # WIDTH: normalised by Close
        # ------------------------------------------------------------------
        norm_width = np.where(
            close_safe.notna() & (width >= 0),
            width / close_safe,
            np.nan,
        )
        df[f"dch_width_{N}"] = pd.Series(norm_width, index=df.index)

        # ------------------------------------------------------------------
        # BREAKOUT: prior-window channel (shift so current bar excluded)
        # ------------------------------------------------------------------
        upper_prior = high.rolling(N, min_periods=N).max().shift(1)
        lower_prior = low.rolling(N,  min_periods=N).min().shift(1)

        # Valid only where the prior window was fully populated
        prior_valid = upper_prior.notna() & lower_prior.notna()

        breakout_up = np.where(
            prior_valid,
            (close > upper_prior).astype(float),
            np.nan,
        )
        breakout_dn = np.where(
            prior_valid,
            (close < lower_prior).astype(float),
            np.nan,
        )

        df[f"dch_breakout_up_{N}"] = pd.Series(breakout_up, index=df.index)
        df[f"dch_breakout_dn_{N}"] = pd.Series(breakout_dn, index=df.index)

        # ------------------------------------------------------------------
        # TIME SINCE BREAKOUT: bars since last up-OR-down breakout event
        # NaN rows in breakout flags are treated as non-events (False).
        # ------------------------------------------------------------------
        any_breakout = (
            pd.Series(breakout_up, index=df.index).fillna(0.0).astype(bool)
            | pd.Series(breakout_dn, index=df.index).fillna(0.0).astype(bool)
        )
        df[f"dch_time_since_breakout_{N}"] = _bars_since_event(any_breakout)

    return df
