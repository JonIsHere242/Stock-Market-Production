"""Book-rule targets: what the 3-slot trigger book actually pays for.

One place for the outcome columns the feature gate, the screen and the predictor lab all
need, so every rig scores the same event. Rule (matches 5__NightlyBackTester / pool_quality):
    depth  = k x daily_vol            (k 1.5; optional |beta| scaling and clamp)
    limit  = Close[t] x (1 - depth)   resting one bar
    touch  = Low[t+1] <= limit
    win    = from the fill, High >= fill x (1+target) strictly before Low <= fill x (1-stop)
             within `horizon` sessions (same-session tie counts as a stop)
    ret    = +target / -stop / Close[t+horizon]/fill - 1 (touching rows only)

Targets (name -> Series aligned to df, NaN where undefined):
    logret     next-day log return (the old gate target)
    ret5       5-session close-to-close return
    hit8       ret5 >= +8%
    touch      would the limit have filled next session (all rows)
    touch_win  touch AND win (all rows)               = the book_5d_all label
    win_touch  win given touch (touching rows only)   = P(win | touch)
    book_ret   realised fixed-rule return (touching rows only)
    book_ret_all  same with non-touch rows = 0
All are forward-looking by construction and must never be used as features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGETS = ("logret", "ret5", "hit8", "touch", "touch_win", "win_touch", "book_ret", "book_ret_all")


def daily_vol(df: pd.DataFrame, vol_col: str = "Realized_Vol_21d", window: int = 21) -> np.ndarray:
    """Daily vol per row. Uses vol_col when present (annualised values are divided by sqrt(252)),
    else a rolling std of log close returns."""
    if vol_col in df.columns:
        v = pd.to_numeric(df[vol_col], errors="coerce").astype(float)
        if np.nanmedian(v.values) > 0.1:
            v = v / np.sqrt(252.0)
        return v.values
    c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    return np.log(c / c.shift(1)).rolling(window, min_periods=window // 2).std().values


def book_outcomes(df: pd.DataFrame, k: float = 1.5, stop: float = 0.03, target: float = 0.08,
                  horizon: int = 5, vol_col: str = "Realized_Vol_21d", beta_col: str | None = None,
                  min_depth: float = 0.0, max_depth: float = 1.0) -> pd.DataFrame:
    """df: one ticker, ascending Date, with Open/High/Low/Close. Returns the outcome frame."""
    c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    lo = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    hi = pd.to_numeric(df["High"], errors="coerce").astype(float)
    v = daily_vol(df, vol_col)
    if beta_col and beta_col in df.columns:
        v = v * pd.to_numeric(df[beta_col], errors="coerce").abs().fillna(1.0).values
    depth = np.clip(k * v, min_depth, max_depth)
    limit = c.values * (1.0 - depth)
    lows = np.column_stack([lo.shift(-j).values for j in range(1, horizon + 1)])
    highs = np.column_stack([hi.shift(-j).values for j in range(1, horizon + 1)])
    touch = lows[:, 0] <= limit
    stop_hit = lows <= limit[:, None] * (1.0 - stop)
    tgt_hit = highs >= limit[:, None] * (1.0 + target)
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), 99)
    f_t, f_s = first(tgt_hit), first(stop_hit)
    win = f_t < f_s
    stopped = (f_s < 99) & ~win
    with np.errstate(divide="ignore", invalid="ignore"):
        last = c.shift(-horizon).values / limit - 1.0
    ret = np.where(win, target, np.where(stopped, -stop, last))
    ret5 = (c.shift(-horizon) / c - 1.0).replace([np.inf, -np.inf], np.nan).clip(-1.0, 2.0).values
    valid = np.isfinite(limit) & np.isfinite(lows).all(axis=1)
    out = pd.DataFrame(index=df.index)
    out["logret"] = np.log(c.shift(-1) / c).values
    out["ret5"] = ret5
    out["hit8"] = np.where(np.isfinite(ret5), (ret5 >= 0.08).astype(float), np.nan)
    out["touch"] = np.where(valid, touch.astype(float), np.nan)
    out["touch_win"] = np.where(valid, (touch & win).astype(float), np.nan)
    out["win_touch"] = np.where(valid & touch, win.astype(float), np.nan)
    out["book_ret"] = np.where(valid & touch, np.clip(ret, -0.5, 0.5), np.nan)
    out["book_ret_all"] = np.where(valid, np.where(touch, np.clip(ret, -0.5, 0.5), 0.0), np.nan)
    return out


def make_target(df: pd.DataFrame, name: str = "logret", **kw) -> pd.Series:
    """Single target Series for the gate. name in TARGETS."""
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; choose from {TARGETS}")
    if name == "logret":
        c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
        return np.log(c.shift(-1) / c)
    return book_outcomes(df, **kw)[name]
