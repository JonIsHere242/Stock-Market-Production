"""
_paper_2604_26063_volume_price_macd.py  --  Volume-Price-Adjusted MACD feature pack.

Based on: "A Volume-Price-Adjusted MACD Trading Strategy with Sensitivity
Calibration for U.S. Equity Indices", arXiv 2604.26063.

CONCEPT
-------
Classic MACD uses only raw-price EMAs (12/26) and a 9-bar signal-line EMA.
The paper's key insight: the oscillator should reflect *conviction* — i.e.
how much volume stands behind each price move.  We achieve this by replacing
the raw Close EMA with an EMA of a volume-weighted price proxy:

    vp_basis[t] = Close[t] * (1 + ln(Volume[t] / EMA_26(Volume)[t]))

The additive log-ratio term is positive (negative) when volume is above
(below) its long-run trend, scaling the effective price up or down by the
degree of participation.  This makes fast moves on thin volume look smaller
and slow grinds on heavy volume look larger — precisely the "sensitivity
calibration" the paper describes.

All arithmetic uses only data up to and including row t (causal).

NOTE (honesty)
--------------
The paper focuses on index-level (SPY/QQQ) application; applying per-ticker
is an extrapolation.  The volume normalisation window (26-bar) is our
reasonable stand-in for the paper's rolling market-volume reference.  The
produced columns are marked experimental / unproven (leading underscore on
the file name).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2604_26063_volume_price_macd",
    "description": (
        "Volume-price-adjusted MACD pack (arXiv 2604.26063): replaces raw-price "
        "EMAs with a volume-conviction-weighted price basis to produce a MACD "
        "oscillator, signal line, histogram, histogram slope/momentum, "
        "ATR-normalised MACD, zero/signal-crossover state, and volume-confirmation flag."
    ),
    "requires": ["Close", "Volume", "High", "Low"],
    "produces": [
        "vpm_macd",            # VP-adjusted MACD line  (fast EMA − slow EMA of vp_basis)
        "vpm_signal",          # 9-bar EMA of vpm_macd  (signal line)
        "vpm_hist",            # histogram = vpm_macd − vpm_signal
        "vpm_hist_slope",      # 1-bar diff of vpm_hist  (momentum of histogram)
        "vpm_hist_mom5",       # 5-bar diff of vpm_hist  (medium-term histogram momentum)
        "vpm_macd_norm",       # vpm_macd normalised by 14-bar ATR (price-scale-free)
        "vpm_cross_state",     # +1 MACD > signal, −1 MACD < signal  (crossover state)
        "vpm_vol_confirm",     # 1 if |vpm_hist| expanding AND volume > 20-bar avg, else 0
    ],
    "tags": ["momentum", "volume", "technical", "experimental"],
    "version": "1.0",
    "author": "arXiv 2604.26063 — volume-price-adjusted MACD",
}

# ---------------------------------------------------------------------------
# Internal helpers (module-level, no IO, no randomness)
# ---------------------------------------------------------------------------
_FAST   = 12   # fast EMA period
_SLOW   = 26   # slow EMA period (also volume-normalisation window)
_SIG    = 9    # signal-line EMA period
_ATR_W  = 14   # ATR window for normalisation
_VOL_W  = 20   # lookback for volume-confirmation average


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EWM EMA, min_periods = span so early values are NaN."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """True-range ATR using shift(1) for prior close (causal)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the volume-price-adjusted MACD feature pack.

    Volume-weighted price basis:
        vol_ema26[t]  = EMA_26(Volume)[t]
        vp_basis[t]   = Close[t] * (1 + log(Volume[t] / vol_ema26[t]))

    The log-ratio is capped to ±2 (i.e. volume can shift the effective price
    by at most e^2 ≈ 7.4×) to prevent outlier volume spikes from dominating.

    All produced columns are prefixed "vpm_".
    """
    close  = df["Close"]
    volume = df["Volume"].astype(float)
    high   = df["High"]
    low    = df["Low"]

    # ------------------------------------------------------------------
    # 1. Volume-weighted price basis
    # ------------------------------------------------------------------
    vol_ema26 = _ema(volume, _SLOW)  # long-run volume trend (causal)

    # log-ratio: how much today's volume deviates from trend (signed)
    # Guard against zero volume (use prior EMA as fallback)
    safe_vol     = volume.where(volume > 0, other=np.nan)
    safe_ema26   = vol_ema26.where(vol_ema26 > 0, other=np.nan)
    log_vol_ratio = np.log(safe_vol / safe_ema26).clip(-2.0, 2.0)

    # When either term is NaN (first ~26 bars or zero-vol bars), vp_basis
    # falls back to raw Close so the EMA can still warm up gracefully.
    vp_basis = close * (1.0 + log_vol_ratio.fillna(0.0))

    # ------------------------------------------------------------------
    # 2. VP-adjusted MACD = fast EMA − slow EMA of vp_basis
    # ------------------------------------------------------------------
    vp_fast = _ema(vp_basis, _FAST)
    vp_slow = _ema(vp_basis, _SLOW)

    vpm_macd   = vp_fast - vp_slow            # NaN until slow EMA warms up
    # EWM propagates NaN correctly when min_periods is set, so signal is also
    # NaN during warmup — that is correct.
    vpm_signal = _ema(vpm_macd, _SIG)        # signal line EMA

    vpm_hist   = vpm_macd - vpm_signal        # histogram

    # ------------------------------------------------------------------
    # 3. Histogram slope / momentum
    # ------------------------------------------------------------------
    vpm_hist_slope = vpm_hist.diff(1)   # 1-bar difference
    vpm_hist_mom5  = vpm_hist.diff(5)   # 5-bar difference

    # ------------------------------------------------------------------
    # 4. Normalised MACD  (price-scale-free, comparable across stocks)
    # ------------------------------------------------------------------
    atr14 = _atr(high, low, close, _ATR_W)
    # Divide by ATR; if ATR is NaN or zero use NaN
    vpm_macd_norm = np.where(
        atr14.notna() & (atr14 > 0),
        vpm_macd / atr14,
        np.nan,
    )
    vpm_macd_norm = pd.Series(vpm_macd_norm, index=df.index)

    # ------------------------------------------------------------------
    # 5. Crossover state  (+1 bullish, −1 bearish, NaN during warmup)
    # ------------------------------------------------------------------
    both_valid = vpm_macd.notna() & vpm_signal.notna()
    vpm_cross_state = pd.Series(
        np.where(both_valid, np.where(vpm_macd > vpm_signal, 1.0, -1.0), np.nan),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # 6. Volume-confirmation flag
    # ------------------------------------------------------------------
    # Condition A: histogram is expanding in absolute terms  (|hist| > |hist[t-1]|)
    hist_expanding = vpm_hist.abs() > vpm_hist.abs().shift(1)

    # Condition B: today's volume exceeds its 20-bar simple moving average
    vol_avg20    = volume.rolling(_VOL_W, min_periods=_VOL_W).mean()
    vol_above_avg = volume > vol_avg20

    # Both conditions must be met; emit NaN during warmup (where vol_avg is NaN)
    vol_confirm_raw = (hist_expanding & vol_above_avg).astype(float)
    vpm_vol_confirm = pd.Series(
        np.where(vol_avg20.notna(), vol_confirm_raw, np.nan),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # Assemble and return — only ADD produced columns, never modify existing
    # ------------------------------------------------------------------
    new = pd.DataFrame(
        {
            "vpm_macd":        vpm_macd,
            "vpm_signal":      vpm_signal,
            "vpm_hist":        vpm_hist,
            "vpm_hist_slope":  vpm_hist_slope,
            "vpm_hist_mom5":   vpm_hist_mom5,
            "vpm_macd_norm":   vpm_macd_norm,
            "vpm_cross_state": vpm_cross_state,
            "vpm_vol_confirm": vpm_vol_confirm,
        },
        index=df.index,
    )

    return pd.concat([df, new], axis=1)
