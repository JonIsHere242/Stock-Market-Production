"""
Heavy-Tail & ARCH-Effect Dynamics Features  —  arXiv:2605.26890
"Nonlinear and Heavy-Tailed Predictability in Transition-Energy Financial Markets"

The paper documents that transition-related financial returns depart from
Gaussian-linear behavior, exhibiting excess kurtosis, volatility clustering
(ARCH effects), and remaining nonlinear dependence after VAR filtering.

Per-ticker OHLCV proxies (faithfully capturing the paper's measured phenomena):

  1. EXCESS KURTOSIS (rolling): Fisher kurtosis of returns over trailing windows.
     High positive values signal fat-tailed / leptokurtic distribution.

  2. ARCH EFFECT STRENGTH: rolling autocorrelation of squared returns (lag-1).
     Positive autocorr of r^2 = volatility clustering = ARCH signature.

  3. VOL-OF-VOL (vol clustering intensity): rolling std of rolling 10d realized vol.
     Captures how much volatility itself fluctuates (a regime-sensitivity signal).

  4. TAIL ASYMMETRY: ratio of negative tail mass (days below -2 std) to positive
     tail mass (days above +2 std) in a rolling window. > 1 means left-heavy.

  5. NONLINEARITY SCORE: rolling R² of squared-return on lagged squared-return
     (i.e., regression R² of |r_t|^2 ~ |r_{t-1}|^2). Measures how much of
     vol-clustering is captured by simple ARCH(1) — leftover = nonlinear.

All computations are purely causal (rolling backward), vectorised, no sklearn.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2605_26890_heavy_tail_dynamics",
    "description": (
        "Rolling excess kurtosis, ARCH-effect (squared-return autocorr), vol-of-vol, "
        "tail asymmetry, and nonlinearity score; proxy for the heavy-tailed/nonlinear "
        "dynamics documented in arXiv:2605.26890 for per-ticker OHLCV."
    ),
    "requires": ["Close"],
    "produces": [
        "htd_excess_kurtosis_60",
        "htd_excess_kurtosis_120",
        "htd_arch_strength_60",
        "htd_vol_of_vol_60",
        "htd_tail_asymmetry_60",
        "htd_nonlinearity_r2_60",
    ],
    "tags": ["volatility", "statistical", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2605.26890",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute heavy-tail and ARCH-effect dynamics features.

    Uses rolling windows only. Leading NaNs are expected.
    """
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Log returns (more stable for kurtosis / tail measures than arithmetic)
    log_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    sq_ret = log_ret ** 2  # squared returns for ARCH

    # -------------------------------------------------------------------------
    # 1. Rolling excess kurtosis — 60d and 120d
    # -------------------------------------------------------------------------
    for window in (60, 120):
        kurt_arr = np.full(n, np.nan)
        min_obs = window // 2
        for i in range(min_obs, n):
            start = max(0, i - window + 1)
            r = log_ret[start: i + 1]
            r = r[~np.isnan(r)]
            k = len(r)
            if k < 8:
                continue
            mu = r.mean()
            sig = r.std()
            if sig < 1e-12:
                continue
            # Fisher kurtosis (excess kurtosis, 0 for normal)
            m4 = np.mean((r - mu) ** 4)
            kurt_arr[i] = m4 / (sig ** 4) - 3.0
        df[f"htd_excess_kurtosis_{window}"] = kurt_arr

    # -------------------------------------------------------------------------
    # 2. ARCH effect: rolling lag-1 autocorrelation of squared returns (60d)
    # -------------------------------------------------------------------------
    arch_arr = np.full(n, np.nan)
    window = 60
    min_obs = 20
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        sq = sq_ret[start: i + 1]
        mask = ~np.isnan(sq)
        sq = sq[mask]
        k = len(sq)
        if k < 6:
            continue
        y = sq[1:]
        x = sq[:-1]
        xm = x.mean()
        ym = y.mean()
        num = np.sum((x - xm) * (y - ym))
        denom = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
        if denom < 1e-14:
            continue
        arch_arr[i] = num / denom
    df["htd_arch_strength_60"] = arch_arr

    # -------------------------------------------------------------------------
    # 3. Vol-of-vol: rolling std of 10d realized vol (std of log_ret), over 60d
    # -------------------------------------------------------------------------
    # First compute 10d rolling realised vol as a series
    rv10 = pd.Series(log_ret).rolling(window=10, min_periods=5).std().values
    vov_arr = np.full(n, np.nan)
    window = 60
    min_obs = 20
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        seg = rv10[start: i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 5:
            continue
        vov_arr[i] = np.std(seg)
    df["htd_vol_of_vol_60"] = vov_arr

    # -------------------------------------------------------------------------
    # 4. Tail asymmetry: negative tail fraction / positive tail fraction (60d)
    #    Threshold: 2 rolling-std below/above zero
    # -------------------------------------------------------------------------
    tail_arr = np.full(n, np.nan)
    window = 60
    min_obs = 20
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        r = log_ret[start: i + 1]
        r = r[~np.isnan(r)]
        k = len(r)
        if k < 8:
            continue
        sig = r.std()
        if sig < 1e-12:
            continue
        thresh = 2.0 * sig
        neg_mass = np.sum(r < -thresh)
        pos_mass = np.sum(r > thresh)
        if pos_mass == 0:
            tail_arr[i] = np.nan
        else:
            tail_arr[i] = neg_mass / pos_mass
    df["htd_tail_asymmetry_60"] = tail_arr

    # -------------------------------------------------------------------------
    # 5. Nonlinearity R²: R² from OLS of sq_ret ~ lagged_sq_ret (rolling 60d)
    #    Measures how much of vol clustering a simple ARCH(1) explains.
    #    High R² = strong linear ARCH; low R² with high ARCH strength = nonlinear.
    # -------------------------------------------------------------------------
    nl_arr = np.full(n, np.nan)
    window = 60
    min_obs = 20
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        sq = sq_ret[start: i + 1]
        mask = ~np.isnan(sq)
        sq = sq[mask]
        k = len(sq)
        if k < 6:
            continue
        y = sq[1:]
        x = sq[:-1]
        xm = x.mean()
        ym = y.mean()
        ss_xx = np.sum((x - xm) ** 2)
        ss_yy = np.sum((y - ym) ** 2)
        if ss_xx < 1e-14 or ss_yy < 1e-14:
            continue
        beta = np.sum((x - xm) * (y - ym)) / ss_xx
        resid = y - (ym + beta * (x - xm))
        ss_res = np.sum(resid ** 2)
        nl_arr[i] = 1.0 - ss_res / ss_yy
    df["htd_nonlinearity_r2_60"] = nl_arr

    return df
