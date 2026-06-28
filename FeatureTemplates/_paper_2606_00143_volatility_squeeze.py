"""
Volatility Squeeze / Band-Compression feature block.

Inspired by arxiv 2606.00143 "Regime-Adaptive Continual Learning for Portfolio
Management" — markets exhibit non-stationary regime shifts; this block provides
a per-ticker, lookahead-safe proxy for regime-transition readiness via the
classic Bollinger-inside-Keltner squeeze and MA-ribbon compression.

No external data required — pure OHLCV arithmetic only.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "volatility_squeeze",
    "description": (
        "Bollinger-inside-Keltner squeeze + MA-ribbon compression features "
        "as a per-ticker proxy for imminent volatility-regime transitions."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "sqz_bb_bandwidth_20",
        "sqz_bb_pct_252",
        "sqz_keltner_width_20",
        "sqz_squeeze_on",
        "sqz_ribbon_dispersion",
        "sqz_bandwidth_velocity",
        "sqz_pct_b_20",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper proxy — arxiv 2606.00143",
}


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-squeeze features on a single-ticker ascending OHLCV frame.

    All rolling windows use only past rows (min_periods enforced); no forward-
    looking operations. Leading NaN rows are expected and left intact.

    Columns produced
    ----------------
    sqz_bb_bandwidth_20
        (BB_upper - BB_lower) / BB_mid  — proportional Bollinger width.
        Narrow = low-vol squeeze; wide = expansion.
    sqz_bb_pct_252
        Trailing-252-bar percentile rank of sqz_bb_bandwidth_20.
        Low value (near 0) signals a historically tight squeeze.
    sqz_keltner_width_20
        (KC_upper - KC_lower) / KC_mid  — proportional Keltner channel width.
    sqz_squeeze_on
        1.0 when BOTH Bollinger bands sit entirely inside the Keltner channel,
        else 0.0.  NaN where either channel is undefined.
    sqz_ribbon_dispersion
        Std-dev of {SMA10, SMA20, SMA50, SMA100} divided by Close — MA-fan
        compression; near zero when moving averages converge.
    sqz_bandwidth_velocity
        sqz_bb_bandwidth_20 minus its value 5 bars ago  (positive = expanding,
        negative = contracting).
    sqz_pct_b_20
        %B location: (Close - BB_lower) / (BB_upper - BB_lower).
        0 = at lower band, 1 = at upper band.
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # -----------------------------------------------------------------------
    # True Range  (for ATR used in Keltner)
    # TR = max(H-L, |H-prev_C|, |L-prev_C|)
    # -----------------------------------------------------------------------
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # -----------------------------------------------------------------------
    # Bollinger Bands (20-day SMA ± 2 rolling-std)
    # -----------------------------------------------------------------------
    _WINDOW = 20

    bb_mid   = close.rolling(_WINDOW, min_periods=_WINDOW).mean()
    bb_std   = close.rolling(_WINDOW, min_periods=_WINDOW).std(ddof=1)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    # sqz_bb_bandwidth_20  — guard zero mid
    bb_width_raw = bb_upper - bb_lower
    sqz_bb_bandwidth_20 = np.where(
        bb_mid.notna() & (bb_mid.abs() > 0),
        bb_width_raw / bb_mid,
        np.nan,
    )
    sqz_bb_bandwidth_20 = pd.Series(sqz_bb_bandwidth_20, index=df.index)

    # sqz_pct_b_20 — guard zero bandwidth
    sqz_pct_b_20 = np.where(
        bb_width_raw.notna() & (bb_width_raw.abs() > 0),
        (close - bb_lower) / bb_width_raw,
        np.nan,
    )
    sqz_pct_b_20 = pd.Series(sqz_pct_b_20, index=df.index)

    # -----------------------------------------------------------------------
    # Keltner Channel (20-day EMA of Close ± 1.5 * ATR20)
    # EWM span=20 corresponds to alpha = 2/(20+1) — standard exponential MA.
    # -----------------------------------------------------------------------
    kc_mid   = close.ewm(span=_WINDOW, min_periods=_WINDOW, adjust=False).mean()
    atr20    = tr.rolling(_WINDOW, min_periods=_WINDOW).mean()
    kc_upper = kc_mid + 1.5 * atr20
    kc_lower = kc_mid - 1.5 * atr20

    # sqz_keltner_width_20 — guard zero mid
    kc_width_raw = kc_upper - kc_lower
    sqz_keltner_width_20 = np.where(
        kc_mid.notna() & (kc_mid.abs() > 0),
        kc_width_raw / kc_mid,
        np.nan,
    )
    sqz_keltner_width_20 = pd.Series(sqz_keltner_width_20, index=df.index)

    # -----------------------------------------------------------------------
    # sqz_squeeze_on
    # 1.0 when both BB bands sit strictly inside the KC channel, else 0.0.
    # NaN when either channel is undefined.
    # -----------------------------------------------------------------------
    both_defined = (
        bb_upper.notna() & bb_lower.notna() &
        kc_upper.notna() & kc_lower.notna()
    )
    squeeze_on_mask = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    sqz_squeeze_on = np.where(
        both_defined,
        squeeze_on_mask.astype(float),
        np.nan,
    )
    sqz_squeeze_on = pd.Series(sqz_squeeze_on, index=df.index)

    # -----------------------------------------------------------------------
    # sqz_bb_pct_252
    # Trailing-252-bar PERCENTILE RANK of sqz_bb_bandwidth_20 (past-only).
    # rank of today relative to the prior 251 observations (window covers 252,
    # [:-1] is previous rows, [-1] is today's value).
    # -----------------------------------------------------------------------
    sqz_bb_pct_252 = (
        sqz_bb_bandwidth_20
        .rolling(252, min_periods=252)
        .apply(lambda x: float((x[:-1] <= x[-1]).mean()), raw=True)
    )

    # -----------------------------------------------------------------------
    # sqz_ribbon_dispersion
    # Std-dev of {SMA10, SMA20, SMA50, SMA100} / Close
    # -----------------------------------------------------------------------
    sma10  = close.rolling(10,  min_periods=10 ).mean()
    sma20  = close.rolling(20,  min_periods=20 ).mean()
    sma50  = close.rolling(50,  min_periods=50 ).mean()
    sma100 = close.rolling(100, min_periods=100).mean()

    ribbon = pd.concat([sma10, sma20, sma50, sma100], axis=1)
    ribbon.columns = ["s10", "s20", "s50", "s100"]

    # All four SMAs must be valid to form a meaningful ribbon
    ribbon_std = ribbon.std(axis=1, ddof=1)  # NaN where any MA is NaN (dropna default)

    sqz_ribbon_dispersion = np.where(
        ribbon.notna().all(axis=1) & (close.abs() > 0),
        ribbon_std / close,
        np.nan,
    )
    sqz_ribbon_dispersion = pd.Series(sqz_ribbon_dispersion, index=df.index)

    # -----------------------------------------------------------------------
    # sqz_bandwidth_velocity
    # sqz_bb_bandwidth_20 minus its value 5 bars ago (positive diff → expansion)
    # -----------------------------------------------------------------------
    sqz_bandwidth_velocity = sqz_bb_bandwidth_20 - sqz_bb_bandwidth_20.shift(5)

    # -----------------------------------------------------------------------
    # Assign all produced columns at once
    # -----------------------------------------------------------------------
    df["sqz_bb_bandwidth_20"]    = sqz_bb_bandwidth_20
    df["sqz_bb_pct_252"]         = sqz_bb_pct_252
    df["sqz_keltner_width_20"]   = sqz_keltner_width_20
    df["sqz_squeeze_on"]         = sqz_squeeze_on
    df["sqz_ribbon_dispersion"]  = sqz_ribbon_dispersion
    df["sqz_bandwidth_velocity"] = sqz_bandwidth_velocity
    df["sqz_pct_b_20"]           = sqz_pct_b_20

    return df
