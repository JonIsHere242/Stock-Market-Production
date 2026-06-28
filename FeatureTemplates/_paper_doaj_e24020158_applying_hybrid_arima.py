"""
Applying Hybrid ARIMA-SGARCH in Algorithmic Investment Strategies on S&P500 Index
DOI: 10.3390/e24020158  (Entropy, 2022)

The paper fits rolling ARIMA models to predict the sign of next-day log returns,
then fits GARCH/SGARCH/EGARCH on the ARIMA residuals to capture heteroskedasticity.
The combined model's sign forecast drives a long/short strategy on S&P500, beating
buy-and-hold over 2000–2019.

PROXY IMPLEMENTED:
  1. ARIMA residual proxy:   actual log-return minus its rolling mean (AR(1)-like
     de-meaned residual).  The sign of the rolling mean gives the ARIMA
     directional forecast.
  2. GARCH conditional-vol proxy: EWMA of squared residuals (GARCH(1,1)-like
     conditional variance), normalised to allow cross-section comparison.
  3. Composite signal: ARIMA direction * vol-scaled confidence (high conditional-vol
     periods inflate uncertainty → attenuate the directional signal).

DROPPED: Actual ARIMA/GARCH fitting (requires statsmodels, iterative MLE),
rolling window refit, S&P500 index-level context.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_e24020158_hybrid_arima_garch",
    "description": (
        "GARCH-style conditional-variance proxy + ARIMA-like de-meaned direction "
        "signal from rolling log-return moments; implements the information content "
        "of the ARIMA-SGARCH strategy of Sakowski et al. (2022) using EWMA variance "
        "instead of parametric GARCH fitting; statsmodels/ARIMA dropped."
    ),
    "requires": ["Close"],
    "produces": [
        "ag_log_ret",
        "ag_arima_resid_proxy_20",
        "ag_arima_dir_proxy_20",
        "ag_garch_condvar_proxy_20",
        "ag_garch_condvar_proxy_60",
        "ag_garch_condvol_proxy_20",
        "ag_composite_signal_20",
        "ag_composite_signal_60",
        "ag_egarch_leverage_60",
    ],
    "tags": ["volatility", "momentum", "trend", "experimental"],
    "version": "1.0",
    "author": "proxy: paper DOI:10.3390/e24020158 (Sakowski et al., 2022); ARIMA/GARCH MLE dropped",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ARIMA-residual proxy + GARCH conditional-variance proxy.

    Log return:         r_t = log(Close_t / Close_{t-1})
    AR(1) mean proxy:   rolling mean of r over [window] days
    Residual proxy:     r_t - rolling_mean (de-meaned, like ARIMA(1,0,0) residual)
    GARCH cond-var:     EWMA(residual^2) with span=window  [GARCH(1,1)-like]
    EGARCH leverage:    skewness of residuals in rolling window (captures sign-vol
                        asymmetry, the core of EGARCH/SGARCH vs plain GARCH)
    Composite signal:   sign(rolling_mean) / sqrt(cond_var)  (directional / uncertainty)
    """
    close = df["Close"].astype(float)

    # Log returns (causal)
    log_ret = np.log(close / close.shift(1))
    df["ag_log_ret"] = log_ret

    for window in (20, 60):
        # ARIMA direction proxy: rolling mean of log_ret (AR drift / trend direction)
        roll_mean = log_ret.rolling(window, min_periods=window // 2).mean()

        # Residual proxy: deviation from rolling mean
        resid = log_ret - roll_mean

        # GARCH(1,1)-style conditional variance: EWMA of squared residuals
        sq_resid = resid ** 2
        cond_var  = sq_resid.ewm(span=window, min_periods=window // 2, adjust=False).mean()
        cond_vol  = np.sqrt(cond_var.clip(lower=0.0))

        # Composite: direction * inverse-uncertainty weighting
        # sign(roll_mean) ∈ {-1, 0, +1}, attenuated by cond_vol
        safe_vol  = cond_vol.replace(0.0, float("nan"))
        composite = np.sign(roll_mean) / safe_vol

        if window == 20:
            df["ag_arima_resid_proxy_20"]  = resid
            df["ag_arima_dir_proxy_20"]    = roll_mean
            df["ag_garch_condvar_proxy_20"] = cond_var
            df["ag_garch_condvol_proxy_20"] = cond_vol
            df["ag_composite_signal_20"]   = composite
        else:
            df["ag_garch_condvar_proxy_60"] = cond_var
            df["ag_composite_signal_60"]   = composite

    # EGARCH leverage effect proxy: rolling skewness of 60-day residuals
    # Negative skew (left tail) = leverage effect (vol spikes on down moves)
    resid_60_roll = log_ret.rolling(60, min_periods=30)
    # Rolling skewness via (mean of cubed deviations) / std^3
    roll_mean60 = resid_60_roll.mean()
    roll_std60  = resid_60_roll.std(ddof=1)
    roll_skew60 = (
        ((log_ret - roll_mean60) ** 3).rolling(60, min_periods=30).mean()
        / (roll_std60 ** 3).replace(0.0, float("nan"))
    )
    df["ag_egarch_leverage_60"] = roll_skew60

    return df
