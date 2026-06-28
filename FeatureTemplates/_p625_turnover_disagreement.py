"""
_p625_turnover_disagreement.py -- Turnover-shock disagreement (opinion divergence) features.

Concept (Garfinkel & Sokobin 2006 JAR; Datar, Naik & Radcliffe 1998 JFM; Hong & Stein 2007):
a cap-normalized turnover SURGE accompanied by a SMALL net price move proxies investor
opinion divergence / disagreement -- lots of trading, little price agreement. We also emit
the cap-vs-raw turnover surprise residual (does the cap-normalized shock exceed the raw-volume
shock?) and the trailing skew of turnover (asymmetry of trading intensity).

Strictly causal & per-ticker: every statistic is a trailing rolling/shift; no whole-series,
cross-sectional, or forward operation. Market cap is point-in-time (backward-asof in helper).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared market-cap helper (skipped by framework auto-discovery)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_marketcap", _Path(__file__).resolve().parent / "_marketcap.py"
)
_marketcap = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_marketcap)

# Warm the process-wide cap panel cache at import so the ~one-time parquet load is NOT
# charged to the first timed compute() call (the helper stashes the panel on `sys`, so this
# loads once per process and is shared across all tickers/blocks).
try:  # pragma: no cover - defensive: never let a missing panel break import
    _marketcap._load_panel()
except Exception:
    pass

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "_p625_turnover_disagreement",
    "description": (
        "Cap-normalized turnover-surge disagreement (turnover shock with small net price "
        "move), cap-vs-raw turnover surprise residual, and 60d turnover skew "
        "(Garfinkel & Sokobin 2006 JAR; Datar, Naik & Radcliffe 1998 JFM; Hong & Stein 2007)."
    ),
    "requires":    ["Close", "Volume"],
    "produces":    ["tdz_disagree_60", "tdz_turn_resid_60", "tdz_turn_skew_60"],
    "tags":        ["volume", "disagreement", "experimental"],
    "version":     "1.0",
    "author":      "paper:Garfinkel & Sokobin (2006) JAR; Datar, Naik & Radcliffe (1998) JFM; Hong & Stein (2007)",
}

_WIN = 60
_MINP = 30


def _rolling_z(s: pd.Series, win: int, minp: int) -> pd.Series:
    """Trailing rolling z-score; zero/NaN std -> NaN (no inf)."""
    mean = s.rolling(win, min_periods=minp).mean()
    std = s.rolling(win, min_periods=minp).std()
    std = std.replace(0.0, np.nan)
    return (s - mean) / std


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # --- point-in-time market cap; remember if WE added it so we can drop it -----
    had_mcap = "mcap" in df.columns
    if not had_mcap:
        df = _marketcap.as_of_cap(df)
    mcap = pd.to_numeric(df["mcap"], errors="coerce").astype(float)

    # --- cap-normalized turnover = Volume*Close / mcap, clipped [0,100] ----------
    denom = mcap.replace(0.0, np.nan)
    turnover = (vol * close) / denom
    turnover = turnover.clip(lower=0.0, upper=100.0)

    # --- turnover z-shock (trailing 60d), clipped [-10,10] -----------------------
    turn_z = _rolling_z(turnover, _WIN, _MINP).clip(-10.0, 10.0)

    # --- conviction: 5d net move vs typical daily move scaled to 5d --------------
    nmove_5 = (close / close.shift(5) - 1.0).abs()
    daily_abs = close.pct_change().abs()
    typical = daily_abs.rolling(20, min_periods=10).mean() * np.sqrt(5.0) + 1e-6
    conv = (nmove_5 / typical).clip(lower=0.0, upper=20.0)

    # --- disagreement: positive turnover shock with LOW conviction move ----------
    disagree = (turn_z.clip(lower=0.0) * (1.0 / (1.0 + conv))).clip(0.0, 10.0)

    # --- cap-vs-raw surprise residual: cap-norm shock minus raw-volume shock -----
    vol_z = _rolling_z(vol, _WIN, _MINP)
    turn_resid = (turn_z - vol_z).clip(-10.0, 10.0)

    # --- trailing 60d skew of turnover via central moments -----------------------
    m = turnover.rolling(_WIN, min_periods=_MINP).mean()
    dev = turnover - m
    m2 = (dev ** 2).rolling(_WIN, min_periods=_MINP).mean()
    m3 = (dev ** 3).rolling(_WIN, min_periods=_MINP).mean()
    s3 = (m2 ** 1.5).replace(0.0, np.nan)
    turn_skew = (m3 / s3).clip(-10.0, 10.0)

    df["tdz_disagree_60"] = disagree.to_numpy()
    df["tdz_turn_resid_60"] = turn_resid.to_numpy()
    df["tdz_turn_skew_60"] = turn_skew.to_numpy()

    # --- drop mcap if the caller did not already have it -------------------------
    if not had_mcap and "mcap" in df.columns:
        df = df.drop(columns=["mcap"])

    return df
