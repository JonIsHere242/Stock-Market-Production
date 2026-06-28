"""
Volatility-decay (compounding drag) features derived from:
  "A Levered ETF Anomaly Explained" (arXiv 2604.27287).

The paper proves that ~2/3 of the levered-ETF return gap vs the index is
due to compounding + volatility drag, with the rest from covariance between
leverage deviation and index return. For a leverage factor L, the daily
compounding drag is approximately:

    drag_daily ≈ (L^2 - L) / 2 * sigma_t^2

where sigma_t^2 is daily variance. Over a window, this accumulates as:

    cumulative_drag ≈ (L^2 - L) / 2 * sum(r_t^2)   [sum of squared daily returns]

For L=2 (2x): drag ≈ (4-2)/2 * realized_var = realized_var
For L=3 (3x): drag ≈ (9-3)/2 * realized_var = 3 * realized_var

Per-ticker signal: estimate the prospective compounding drag on the stock
itself (if it were a levered product), and the covariance term between
leverage deviation and return. High drag → the stock's daily volatility is
compounding AGAINST a buy-and-hold position.

Features:
  - voldrag_drag_20d:     rolling 20-day compounding drag estimate (L=2 proxy)
  - voldrag_drag_60d:     rolling 60-day compounding drag estimate
  - voldrag_cov_term_20d: covariance between |r_t| (leverage deviation proxy) and r_t sign
  - voldrag_bias_ratio:   realized cum return / (expected-no-drag cum return) rolling 20d
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2604_27287_vol_decay_drag",
    "description": (
        "Compounding volatility drag features: rolling accumulated (L^2-L)/2 * sigma^2 "
        "drag estimate and return-vs-compounding bias ratio; per-ticker proxy for "
        "arXiv 2604.27287 levered ETF anomaly (compounding + vol drag mechanism)."
    ),
    "requires":    ["Close"],
    "produces": [
        "voldrag_drag_20d",
        "voldrag_drag_60d",
        "voldrag_cov_term_20d",
        "voldrag_bias_ratio_20d",
    ],
    "tags":        ["volatility", "momentum", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2604.27287",
}

_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n     = len(df)
    close = df["Close"].values.astype(np.float64)

    # Daily log returns
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(np.maximum(close[1:], _EPS) / np.maximum(close[:-1], _EPS))

    # For L=2: drag_daily = (L^2 - L)/2 * r^2 = 1 * r^2 = r^2
    # (log-return squared approximates daily variance for compounding drag)
    r2 = log_ret ** 2  # NaN where log_ret is NaN

    drag_20 = np.full(n, np.nan)
    drag_60 = np.full(n, np.nan)
    cov_20  = np.full(n, np.nan)
    bias_20 = np.full(n, np.nan)

    for window in [20, 60]:
        for i in range(window, n):
            sl = r2[i - window + 1: i + 1]
            valid = ~np.isnan(sl)
            if valid.sum() < window // 2:
                continue
            # Accumulated drag = sum of r^2 (for L=2, factor = 1)
            drag = np.nansum(sl)
            if window == 20:
                drag_20[i] = drag
            else:
                drag_60[i] = drag

    # Covariance term: cov(|r_t|, sign(r_t)) over 20d rolling
    # (leverage deviation proxy ~ |r_t|, index return proxy ~ r_t)
    for i in range(20, n):
        sl_r = log_ret[i - 20 + 1: i + 1]
        valid = ~np.isnan(sl_r)
        if valid.sum() < 10:
            continue
        sl_r_v = sl_r[valid]
        abs_r  = np.abs(sl_r_v)
        sign_r = np.sign(sl_r_v)
        # Demean
        abs_r_dm  = abs_r  - abs_r.mean()
        sign_r_dm = sign_r - sign_r.mean()
        n_v = len(abs_r_dm)
        if n_v < 4:
            continue
        cov_20[i] = (abs_r_dm * sign_r_dm).mean()

    # Bias ratio: actual geometric cum return vs. arithmetic (no-compounding) approximation.
    # actual_geo  = exp(sum(log_ret)) - 1  (compounding)
    # actual_arith = sum(simple_ret)       (no compounding, pure arithmetic)
    # Bias = actual_geo / actual_arith when both nonzero, else NaN.
    # < 1 → compounding drag is hurting; > 1 → convexity benefit (trending market).
    simple_ret = np.full(n, np.nan)
    simple_ret[1:] = close[1:] / np.maximum(close[:-1], _EPS) - 1.0
    simple_ret[0] = np.nan

    for i in range(20, n):
        sl_log = log_ret[i - 20 + 1: i + 1]
        sl_sim = simple_ret[i - 20 + 1: i + 1]
        valid = ~(np.isnan(sl_log) | np.isnan(sl_sim))
        if valid.sum() < 10:
            continue
        sl_log_v = sl_log[valid]
        sl_sim_v = sl_sim[valid]
        cum_geo   = np.exp(sl_log_v.sum()) - 1.0   # geometric cum return
        cum_arith = sl_sim_v.sum()                  # arithmetic (no compounding)
        if abs(cum_arith) > _EPS:
            bias_20[i] = cum_geo / cum_arith
        # else leave as NaN (mean-zero period, undefined ratio)

    df["voldrag_drag_20d"]       = drag_20
    df["voldrag_drag_60d"]       = drag_60
    df["voldrag_cov_term_20d"]   = cov_20
    df["voldrag_bias_ratio_20d"] = bias_20

    return df
