"""
_paper_doaj_42a0cc34_candle_pattern_rates.py

Per-ticker candlestick shadow asymmetry and rolling pattern-occurrence rates.

Inspired by: "Forecasting stock returns: the role of VIX-based upper and lower
shadow of Japanese candlestick" (DOAJ 42a0cc34).  The paper's key predictor is
the Upper-minus-Lower-shadow Difference (ULD): higher ULD precedes lower
subsequent returns.  The paper applied this to VIX; here we compute the
per-ticker analogue from each stock's own OHLC geometry so that the signal
varies cross-sectionally.

Shadow geometry (per bar, rng = High - Low):
    upper_shadow  = High  - max(Open, Close)
    lower_shadow  = min(Open, Close) - Low
    body          = abs(Close - Open)
    cdl_uld       = (upper_shadow - lower_shadow) / rng   [paper's core signal]

All ratio columns guard rng == 0 → NaN for that bar.
Rolling columns use a trailing window (rows <= t); first (window-1) rows are NaN.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_doaj_42a0cc34_candle_pattern_rates",
    "description": (
        "Per-ticker upper/lower shadow asymmetry (ULD) and rolling candlestick "
        "pattern-occurrence rates derived from the paper's VIX-shadow predictor "
        "ported to per-stock OHLC geometry."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "cdl_upper_shadow_ratio",
        "cdl_lower_shadow_ratio",
        "cdl_uld",
        "cdl_shadow_asym_20",
        "cdl_doji_rate_20",
        "cdl_hammer_rate_20",
        "cdl_body_ratio_mean_20",
    ],
    "tags": ["candlestick", "price_structure", "pattern", "experimental"],
    "version": "1.0",
    "author": "paper doaj 42a0cc34 — per-ticker port",
}

# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
_WINDOW = 20  # trailing window for all rolling statistics


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shadow-asymmetry and rolling pattern-occurrence features.

    Per-bar geometry is computed from that bar's own OHLC only (no lookahead).
    Rolling statistics use a trailing window of _WINDOW bars (min_periods=_WINDOW
    so that the first _WINDOW-1 rows stay NaN rather than emit partial estimates).

    Parameters
    ----------
    df : pd.DataFrame
        Per-ticker DataFrame sorted ascending by Date.
        Must contain Open, High, Low, Close columns.

    Returns
    -------
    pd.DataFrame
        Original df with seven new cdl_* columns appended.
    """
    o = df["Open"].to_numpy(dtype=np.float64)
    h = df["High"].to_numpy(dtype=np.float64)
    l = df["Low"].to_numpy(dtype=np.float64)
    c = df["Close"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Per-bar geometry (vectorised, no lookahead — uses only that bar's row)
    # ------------------------------------------------------------------
    rng          = h - l                                       # High - Low
    upper_shadow = h - np.maximum(o, c)                       # above body
    lower_shadow = np.minimum(o, c) - l                       # below body
    body         = np.abs(c - o)

    # Guard division by rng == 0 → NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        upper_shadow_ratio = np.where(rng > 0, upper_shadow / rng, np.nan)
        lower_shadow_ratio = np.where(rng > 0, lower_shadow / rng, np.nan)
        uld                = np.where(rng > 0, (upper_shadow - lower_shadow) / rng, np.nan)
        body_ratio         = np.where(rng > 0, body / rng, np.nan)

    idx = df.index

    df["cdl_upper_shadow_ratio"] = pd.array(upper_shadow_ratio, dtype="float64")
    df["cdl_lower_shadow_ratio"] = pd.array(lower_shadow_ratio, dtype="float64")
    df["cdl_uld"]                = pd.array(uld,                dtype="float64")

    # ------------------------------------------------------------------
    # Rolling statistics — trailing _WINDOW bars (no lookahead)
    # ------------------------------------------------------------------
    uld_series        = pd.Series(uld,        index=idx)
    body_ratio_series = pd.Series(body_ratio, index=idx)

    # cdl_shadow_asym_20: trailing mean of ULD
    df["cdl_shadow_asym_20"] = (
        uld_series.rolling(_WINDOW, min_periods=_WINDOW).mean()
    )

    # cdl_body_ratio_mean_20: trailing mean of body/rng
    df["cdl_body_ratio_mean_20"] = (
        body_ratio_series.rolling(_WINDOW, min_periods=_WINDOW).mean()
    )

    # ------------------------------------------------------------------
    # Pattern flags — binary per bar, then rolling fraction over window
    # ------------------------------------------------------------------

    # Doji: body is a tiny fraction of range (< 10 % of range)
    # Guard: if rng == 0, body_ratio is NaN → doji_flag stays NaN
    with np.errstate(invalid="ignore"):
        doji_flag   = np.where(np.isfinite(body_ratio), (body_ratio < 0.1).astype(np.float64), np.nan)
        # Hammer / inverted-hammer: lower_shadow > 2 * body AND upper_shadow < body
        hammer_flag = np.where(
            np.isfinite(body_ratio),
            ((lower_shadow > 2.0 * body) & (upper_shadow < body)).astype(np.float64),
            np.nan,
        )

    df["cdl_doji_rate_20"] = (
        pd.Series(doji_flag, index=idx)
        .rolling(_WINDOW, min_periods=_WINDOW)
        .mean()
    )

    df["cdl_hammer_rate_20"] = (
        pd.Series(hammer_flag, index=idx)
        .rolling(_WINDOW, min_periods=_WINDOW)
        .mean()
    )

    return df
