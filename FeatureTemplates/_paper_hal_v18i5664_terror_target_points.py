"""
OHLCV Proxy for:
  "Terror of Target Points and Loss Limits Modeling in the New York Trading Market
   Based on Deep Learning"  —  doi:10.9734/ajrcos/2025/v18i5664

The paper predicts implied volatility of option contracts using a probabilistic
neural network (PNN) + Brownian-curve (geometric Brownian motion) feature
extraction for dimensionality reduction, to model target-point and stop-loss
placement in the New York market.

PROXY IMPLEMENTATION
--------------------
Implied volatility, option contract data, intraday Brownian feature extraction,
and the PNN/GBM classification pipeline cannot be replicated from daily OHLCV.
The paper's core signal is REALIZED VOLATILITY — the primary input to any IV
model.  We implement three complementary realized-volatility proxies and one
Brownian-motion-inspired diffusion metric:

  1. entloss_rv_cc_22    — Close-to-Close realized variance (22-day sum of
                           squared log-returns, annualized).  Foundation of the
                           Brownian-motion IV formula (sigma^2 * T).

  2. entloss_rv_rs_22    — Garman-Klass range-based realized variance (22-day
                           rolling); more efficient estimator of true variance
                           using intraday H/L range without intraday data.

  3. entloss_rv_ratio_22 — Ratio of GK-RV to CC-RV: >1 means intraday vol
                           exceeds close-to-close vol (gap risk vs. intraday
                           moves), which informs target/stop placement.

  4. entloss_bm_drift_22 — GBM drift estimate: mu = mean(log-ret) / annualized
                           over 22 days.  The Sharpe-like ratio drift/sigma
                           governs how far a Brownian bridge target is likely to
                           be reached.

  5. entloss_bm_target_22 — Brownian first-passage approximation: expected
                            fraction of the daily 1-sigma band that the price
                            is likely to exceed over a 22-day horizon.
                            = Phi(|mu|*sqrt(T) / sigma), where T=22/252.

  6. entloss_stop_ratio_22 — Stop-loss asymmetry: downside-vol / upside-vol
                             (semi-deviations), analogous to the paper's
                             loss-limit vs target-point asymmetry.

Dropped: option/IV data, PNN, Brownian-curve dimension reduction, NY market
cross-sectional panel, classification into clusters.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

METADATA = {
    "name": "paper_hal_v18i5664_terror_target_points",
    "description": (
        "Realized-volatility and GBM-based target/stop proxies for implied-volatility "
        "prediction; proxies doi:10.9734/ajrcos/2025/v18i5664 (PNN+Brownian IV model). "
        "Dropped: options/IV data, PNN, Brownian-curve dimensionality reduction. "
        "Implements CC-RV, Garman-Klass RV, drift/target estimates from GBM."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "entloss_rv_cc_22",
        "entloss_rv_rs_22",
        "entloss_rv_ratio_22",
        "entloss_bm_drift_22",
        "entloss_bm_target_22",
        "entloss_stop_ratio_22",
    ],
    "tags": ["volatility", "risk", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.9734/ajrcos/2025/v18i5664 (Sadeghi et al., 2025). "
        "Implied-vol/options dropped; realized-vol (CC + Garman-Klass) + GBM "
        "drift metrics are the implementable per-ticker OHLCV core."
    ),
}

_ANNFACT = 252.0


def compute(df: pd.DataFrame) -> pd.DataFrame:
    w = 22  # ~1 trading month

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    close = df["Close"]

    # ------------------------------------------------------------------
    # 1. Close-to-Close realized variance (annualized)
    # ------------------------------------------------------------------
    rv_cc = log_ret.pow(2).rolling(w, min_periods=w // 2).sum() * (_ANNFACT / w)
    df["entloss_rv_cc_22"] = rv_cc

    # ------------------------------------------------------------------
    # 2. Garman-Klass range-based realized variance (annualized)
    # ------------------------------------------------------------------
    # GK estimator per bar: 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    gk_daily = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    rv_rs = gk_daily.rolling(w, min_periods=w // 2).mean() * _ANNFACT
    df["entloss_rv_rs_22"] = rv_rs

    # ------------------------------------------------------------------
    # 3. RV ratio: GK / CC (intraday vs close-to-close vol)
    # ------------------------------------------------------------------
    df["entloss_rv_ratio_22"] = rv_rs / rv_cc.replace(0, np.nan)

    # ------------------------------------------------------------------
    # 4. GBM drift estimate (annualized mean log-return)
    # ------------------------------------------------------------------
    mu_ann = log_ret.rolling(w, min_periods=w // 2).mean() * _ANNFACT
    df["entloss_bm_drift_22"] = mu_ann

    # ------------------------------------------------------------------
    # 5. Brownian first-passage: Phi(|mu| * sqrt(T) / sigma)
    #    T = w/252,  sigma = sqrt(rv_cc)
    #    Approximates probability a GBM path hits a 1-sigma target in T days
    # ------------------------------------------------------------------
    sigma_ann = rv_cc.pow(0.5).replace(0, np.nan)
    T = w / _ANNFACT
    z_score = mu_ann.abs() * np.sqrt(T) / sigma_ann
    df["entloss_bm_target_22"] = z_score.apply(
        lambda z: norm.cdf(z) if not np.isnan(z) else np.nan
    )

    # ------------------------------------------------------------------
    # 6. Stop-loss asymmetry: downside semi-dev / upside semi-dev
    # ------------------------------------------------------------------
    def _stop_ratio(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 4:
            return np.nan
        up = valid[valid > 0]
        dn = valid[valid < 0]
        up_std = np.std(up) if len(up) > 1 else np.nan
        dn_std = np.std(np.abs(dn)) if len(dn) > 1 else np.nan
        if up_std is None or np.isnan(up_std) or up_std < 1e-12:
            return np.nan
        return float(dn_std / up_std)

    df["entloss_stop_ratio_22"] = (
        log_ret.rolling(w, min_periods=w // 2)
        .apply(_stop_ratio, raw=True)
    )

    return df
