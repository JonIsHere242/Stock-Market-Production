"""
OHLCV Proxy for:
  "Enhancing Prediction by Incorporating Entropy Loss in Volatility Forecasting"
  —  doi:10.3390/e27080806

The paper fits Heterogeneous Autoregressive (HAR) models using five estimation
techniques (OLS, WLS, robust regression, entropy-loss, QLIKE) and four forecast
horizons (1-day, 5-day, 10-day, 22-day) on 5-min intraday SPX/VIX realized
volatility.  Key finding: the Entropy Loss / QLIKE objective yields the best
QLIKE forecast error across all horizons, especially weekly.

PROXY IMPLEMENTATION
--------------------
5-min intraday data and VIX are not available per-ticker.  We implement a
faithful per-ticker OHLCV HAR-RV proxy:

  1. RANGE-BASED REALIZED VOLATILITY (Parkinson, daily proxy for RV):
     rv_daily = (ln(H) - ln(L))^2 / (4 * ln2)

  2. HAR-RV COMPONENTS (the autoregressive structure of the paper):
     - RV_d  = rv_daily[t]           (daily lag)
     - RV_w  = mean(rv[t-4:t])       (weekly average, 5-day)
     - RV_m  = mean(rv[t-21:t])      (monthly average, 22-day)

  3. OLS HAR-RV FITTED VALUE: rolling OLS regressing rv[t] on
     rv_d[t-1], rv_w[t-1], rv_m[t-1]; the fitted value serves as the
     1-day-ahead realized-vol forecast (the paper's primary output).

  4. HAR-RV QLIKE ERROR: QLIKE = RV/RV_hat - ln(RV/RV_hat) - 1, computed
     on a rolling in-sample window; lower = better-calibrated model.

  5. MULTI-HORIZON FORECASTS: the paper studies 1/5/10/22-day horizons.
     We produce horizon-averaged RV targets by rolling mean-forward on the
     range-based RV series (still causal — these are LAG features of RV,
     not future values):
     - harv_rv_5d  : rolling 5-day mean of past rv (acts as 5d forecast)
     - harv_rv_22d : rolling 22-day mean of past rv (acts as 22d forecast)

  6. ENTROPY SIGNAL: the paper's key innovation is the entropy objective.
     We compute a rolling information-theoretic RV entropy: the Shannon
     entropy of the rolling distribution of daily RV values (normalized
     histogram over 60-day window), as a proxy for the model's uncertainty
     regime.  Low entropy = concentrated / predictable vol; high = diffuse.

Dropped: 5-min realized variance, VIX exogenous variable, HARQ quarticity
extension, WLS/robust-LM coefficient estimation, formal Mincer-Zarnowitz tests.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_doaj_e27080806_enhancing_incorporating_entropy",
    "description": (
        "HAR-RV volatility forecast features with QLIKE error and rolling entropy; "
        "proxies doi:10.3390/e27080806 (entropy-loss HAR models). "
        "Uses Parkinson range-based RV in place of 5-min intraday RV. "
        "Dropped: VIX, HARQ quarticity, intraday data, WLS/robust estimation."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "harv_rv_daily",
        "harv_rv_5d",
        "harv_rv_22d",
        "harv_har_forecast_1d",
        "harv_qlike_60",
        "harv_entropy_60",
    ],
    "tags": ["volatility", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.3390/e27080806 (Beran et al., 2025). "
        "5-min RV/VIX dropped; Parkinson range-based daily RV used for HAR-RV "
        "components; QLIKE rolling error + entropy regime signal retained."
    ),
}

_ANNFACT = 252.0
_LOG2 = np.log(2.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Parkinson range-based realized variance (daily, annualized)
    # ------------------------------------------------------------------
    log_hl = np.log(df["High"] / df["Low"].replace(0, np.nan))
    rv_daily = (log_hl ** 2) / (4.0 * _LOG2) * _ANNFACT
    df["harv_rv_daily"] = rv_daily

    # ------------------------------------------------------------------
    # 2. HAR-RV components: weekly and monthly rolling means
    # ------------------------------------------------------------------
    rv_5 = rv_daily.rolling(5, min_periods=3).mean()
    rv_22 = rv_daily.rolling(22, min_periods=11).mean()
    df["harv_rv_5d"] = rv_5
    df["harv_rv_22d"] = rv_22

    # ------------------------------------------------------------------
    # 3. Rolling OLS HAR-RV forecast (1-day ahead)
    #    Regress rv[t] on rv_d[t-1], rv_w[t-1], rv_m[t-1]
    #    Use a 60-day estimation window; produce fitted value as forecast
    # ------------------------------------------------------------------
    rv_arr = rv_daily.values.astype(np.float64)
    rv5_arr = rv_5.values.astype(np.float64)
    rv22_arr = rv_22.values.astype(np.float64)

    har_forecast = np.full(n, np.nan)
    min_w = 40  # minimum window for OLS stability

    for i in range(min_w + 1, n):
        # estimation window: use up to 60 past obs (excluding i)
        w_start = max(0, i - 60)
        # y = rv at times w_start+1 .. i (shifted: outcome at t+1)
        # x = [1, rv_d[t], rv_5[t], rv_22[t]] at times w_start .. i-1
        y = rv_arr[w_start + 1: i + 1]
        xd = rv_arr[w_start:i]
        xw = rv5_arr[w_start:i]
        xm = rv22_arr[w_start:i]

        # keep only rows with no NaN
        valid = ~(np.isnan(y) | np.isnan(xd) | np.isnan(xw) | np.isnan(xm))
        y_v = y[valid]
        if y_v.shape[0] < 20:
            continue
        X = np.column_stack([
            np.ones(y_v.shape[0]),
            xd[valid],
            xw[valid],
            xm[valid],
        ])
        # OLS via lstsq
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Forecast at i: use rv_d[i-1], rv_5[i-1], rv_22[i-1]
        if not (np.isnan(rv_arr[i - 1]) or
                np.isnan(rv5_arr[i - 1]) or
                np.isnan(rv22_arr[i - 1])):
            hat = coef[0] + coef[1] * rv_arr[i - 1] + coef[2] * rv5_arr[i - 1] + coef[3] * rv22_arr[i - 1]
            har_forecast[i] = max(hat, 1e-8)   # RV must be non-negative

    df["harv_har_forecast_1d"] = har_forecast

    # ------------------------------------------------------------------
    # 4. Rolling QLIKE error over 60-day window
    #    QLIKE(rv, rv_hat) = rv/rv_hat - ln(rv/rv_hat) - 1  (per bar)
    #    Then rolling mean over 60 bars.  Lower = better calibration.
    # ------------------------------------------------------------------
    rv_hat = pd.Series(har_forecast, index=df.index)
    ratio = rv_daily / rv_hat.replace(0, np.nan)
    qlike_bar = ratio - np.log(ratio.clip(lower=1e-8)) - 1.0
    df["harv_qlike_60"] = qlike_bar.rolling(60, min_periods=30).mean()

    # ------------------------------------------------------------------
    # 5. Rolling Shannon entropy of daily RV distribution (60-day)
    #    Proxy for the paper's entropy-loss calibration uncertainty
    #    Discretise RV into 10 equal-frequency bins per window
    # ------------------------------------------------------------------
    def _rv_entropy(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 10:
            return np.nan
        n_bins = min(10, len(valid) // 3)
        if n_bins < 2:
            return np.nan
        counts, _ = np.histogram(valid, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    df["harv_entropy_60"] = (
        rv_daily.rolling(60, min_periods=30)
        .apply(_rv_entropy, raw=True)
    )

    return df
