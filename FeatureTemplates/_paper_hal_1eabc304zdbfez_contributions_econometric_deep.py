"""
MS-GARCH Regime Proxy Features  —  HAL:1eabc304zdbfez
"Contributions to Econometric and Deep Learning Methods for Time Series Forecasting"
(2024 PhD thesis: Temporal KAN / TKAT / MS-GARCH / path-signature portfolios)

The thesis proposes Markov-Switching GARCH models (MS-GARCH) for detecting market
regimes and estimating transition probabilities between volatility states.  We
cannot train a neural network or MS-GARCH model per ticker inside compute(), so
we implement a OHLCV proxy:

PROXY APPROACH (per-ticker, causal):
  1. GARCH-style conditional variance: EWMA variance with two decay rates
     (fast=12d, slow=60d) captures short- vs long-run volatility clustering.
  2. REGIME STATE: label bar as HIGH-vol regime when EWMA-fast variance exceeds
     a rolling 75th-percentile threshold (analogous to one Markov state).
  3. TRANSITION INDICATOR: 1-bar change in regime label → transition count over
     a short window approximates the Markov transition probability mass.
  4. REGIME PERSISTENCE: rolling fraction of HIGH-vol bars (persistence measure).
  5. VOL RATIO (fast/slow EWMA var): captures relative vol acceleration, which
     drives regime-switch timing in MS-GARCH.
  6. REALISED GARCH INNOVATION: standardised residual (ret / sqrt(EWMA-var))
     measures deviation from the current conditional-vol estimate; a key input
     to GARCH updating equations.

DROPPED: TKAN/TKAT neural architectures, simulation-based estimation, path
signatures, AF-LSTM, anomaly-detection autoencoder — none are implementable
as a stateless OHLCV feature without model training.

Proxy is described as such in METADATA['author'].
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_hal_1eabc304zdbfez_contributions_econometric_deep",
    "description": (
        "OHLCV proxy for MS-GARCH regime detection from HAL thesis (2024): EWMA "
        "conditional variance at two speeds, binary high-vol regime label, regime "
        "transition count, persistence fraction, vol-acceleration ratio, and "
        "standardised GARCH innovation.  Neural network / MS-GARCH training dropped."
    ),
    "requires": ["Close"],
    "produces": [
        "msgarch_ewma_var_fast_12",
        "msgarch_ewma_var_slow_60",
        "msgarch_regime_hi_vol",
        "msgarch_transition_cnt_20",
        "msgarch_regime_persist_60",
        "msgarch_vol_ratio_fast_slow",
        "msgarch_garch_innov_12",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": (
        "proxy:hal-1eabc304zdbfez — MS-GARCH regime state approximated via "
        "EWMA conditional variance thresholding; neural-net components omitted."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MS-GARCH regime proxy features from Close prices.

    All operations are causal (no look-ahead). Leading NaNs expected during warm-up.
    """
    close = df["Close"]

    # ---- Log returns (more GARCH-appropriate than arithmetic) ---------------
    log_ret = np.log(close / close.shift(1))

    # ---- EWMA conditional variance (GARCH-lite: span-based) -----------------
    fast_span = 12   # ~2-week half-life
    slow_span = 60   # ~3-month half-life

    # ewm variance on squared returns (proxy for conditional variance)
    sq_ret = log_ret ** 2
    ewma_var_fast = sq_ret.ewm(span=fast_span, min_periods=fast_span).mean()
    ewma_var_slow = sq_ret.ewm(span=slow_span, min_periods=slow_span).mean()

    df["msgarch_ewma_var_fast_12"] = ewma_var_fast
    df["msgarch_ewma_var_slow_60"] = ewma_var_slow

    # ---- Regime label: HIGH vol when fast-EWMA var exceeds rolling 75th pct -
    # Rolling 75th percentile of fast EWMA var over past 252 trading days
    roll_q75 = ewma_var_fast.rolling(window=252, min_periods=60).quantile(0.75)
    regime_hi = (ewma_var_fast > roll_q75).astype(float)
    # NaN where not enough history
    regime_hi[ewma_var_fast.isna() | roll_q75.isna()] = np.nan

    df["msgarch_regime_hi_vol"] = regime_hi

    # ---- Regime transition count over trailing 20 bars ----------------------
    # A transition is a change in regime label (0→1 or 1→0)
    regime_change = regime_hi.diff().abs()  # 1 at transitions, 0 otherwise, NaN at first
    transition_cnt = regime_change.rolling(window=20, min_periods=10).sum()
    df["msgarch_transition_cnt_20"] = transition_cnt

    # ---- Regime persistence: fraction of HIGH-vol bars in trailing 60d ------
    persist = regime_hi.rolling(window=60, min_periods=30).mean()
    df["msgarch_regime_persist_60"] = persist

    # ---- Vol-acceleration ratio: fast / slow EWMA variance ------------------
    vol_ratio = ewma_var_fast / ewma_var_slow.replace(0, np.nan)
    df["msgarch_vol_ratio_fast_slow"] = vol_ratio

    # ---- Standardised GARCH innovation: ret / sqrt(conditional var) ---------
    # This is the key residual that drives GARCH parameter updating
    cond_std_fast = np.sqrt(ewma_var_fast.replace(0, np.nan))
    garch_innov = log_ret / cond_std_fast
    df["msgarch_garch_innov_12"] = garch_innov

    return df
