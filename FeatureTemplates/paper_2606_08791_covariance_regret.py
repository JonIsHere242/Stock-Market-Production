"""
Covariance Regret Features  —  arxiv:2606.08791
"Evaluating AI Investment Strategies"

The paper proves an exact decomposition: cumulative regret of a rolling-window
policy equals the sum of per-period covariances between the cost vector and the
policy's decisions. Applied to OHLCV: the "policy" is a simple threshold signal
derived from price/volume primitives; the "cost" is realized next-bar return
(used causally — we compute rolling covariance of LAGGED signal vs return).

Key idea: rolling covariance between a momentum/reversal signal and realized
return is a *live* measure of whether that signal is in a regime of positive
or negative predictive power. High positive cov_regret → signal has been
predictive recently; near-zero → regime breakdown.

We compute:
  1. signal_vec: a Z-scored OHLCV primitive (momentum, volume surprise, body ratio)
  2. cost_vec:   realized log-return (known — no lookahead)
  3. rolling_cov(signal[t-lag], ret[t]) over windows [10, 21, 42, 63]
  4. Standardised (z-score) of each covariance series = "covreg_z_*"
  5. A composite "covreg_composite" = sign-weighted average of z-scores

Produces 13 columns prefixed "covreg_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_08791_covariance_regret",
    "description": (
        "Rolling policy-covariance regret features from arxiv:2606.08791 — "
        "rolling covariance between OHLCV momentum/volume signals and realized "
        "returns, capturing live regime quality of each signal family."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "covreg_mom_10d",
        "covreg_mom_21d",
        "covreg_mom_42d",
        "covreg_vol_10d",
        "covreg_vol_21d",
        "covreg_vol_42d",
        "covreg_body_10d",
        "covreg_body_21d",
        "covreg_body_42d",
        "covreg_mom_z21",
        "covreg_vol_z21",
        "covreg_body_z21",
        "covreg_composite",
    ],
    "tags": ["experimental", "market_regime", "momentum"],
    "version": "1.0",
    "author": "paper:2606.08791",
}


def _rolling_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling covariance of x and y using a centered dot-product approach."""
    xm = x - x.rolling(window, min_periods=window).mean()
    ym = y - y.rolling(window, min_periods=window).mean()
    return (xm * ym).rolling(window, min_periods=window).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(np.float64)
    V = df["Volume"].astype(np.float64)
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)

    # Realized log return (cost vector — known, no lookahead)
    ret = np.log(C / C.shift(1))

    # --- Signal 1: Momentum (5-day log return, z-scored over 63d) ---
    mom5 = np.log(C / C.shift(5))
    mom5_z = (mom5 - mom5.rolling(63, min_periods=21).mean()) / (
        mom5.rolling(63, min_periods=21).std() + 1e-9
    )

    # --- Signal 2: Volume surprise (vol vs 21d MA, z-scored) ---
    vol_log = np.log(V.replace(0, np.nan))
    vol_z = (vol_log - vol_log.rolling(21, min_periods=10).mean()) / (
        vol_log.rolling(21, min_periods=10).std() + 1e-9
    )

    # --- Signal 3: Candle body ratio (close position within H-L range) ---
    hl = (H - L).replace(0, np.nan)
    body = (C - L) / hl  # [0, 1]; high = bullish close
    body_z = (body - body.rolling(21, min_periods=10).mean()) / (
        body.rolling(21, min_periods=10).std() + 1e-9
    )

    # Rolling covariance: signal[t-1] vs ret[t] — lag=1 ensures no lookahead
    for sig, prefix in [(mom5_z, "mom"), (vol_z, "vol"), (body_z, "body")]:
        sig_lag = sig.shift(1)  # use yesterday's signal vs today's return
        for w in [10, 21, 42]:
            col = f"covreg_{prefix}_{w}d"
            df[col] = _rolling_cov(sig_lag, ret, w)

    # Z-score of 21d covariance series over 126d window (regime standardisation)
    for prefix in ["mom", "vol", "body"]:
        cov21 = df[f"covreg_{prefix}_21d"]
        z = (cov21 - cov21.rolling(126, min_periods=42).mean()) / (
            cov21.rolling(126, min_periods=42).std() + 1e-9
        )
        df[f"covreg_{prefix}_z21"] = z

    # Composite: average of the three z-scored regime signals
    df["covreg_composite"] = (
        df["covreg_mom_z21"] + df["covreg_vol_z21"] + df["covreg_body_z21"]
    ) / 3.0

    return df
