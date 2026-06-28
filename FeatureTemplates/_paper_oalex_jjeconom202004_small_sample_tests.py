"""
Stock Return Predictability Regressors  —  OpenAlex:jjeconom202004
"Small-sample tests for stock return predictability with possibly
non-stationary regressors and GARCH-type effects"
Journal of Econometrics, 2020.

The paper TESTS whether financial variables predict excess stock returns using a
simulation-based procedure robust to non-stationary regressors and GARCH effects.
It is fundamentally a STATISTICAL TEST, not a feature.  However, it identifies
the EMPIRICAL REGRESSORS that have predictive power and characterises the GARCH-
type effects that confound naive tests.

PROXY APPROACH:
  We implement the underlying predictive signals the paper studies:

  1. TERM-SPREAD ANALOGUE (the paper finds this the main predictor):
     The paper identifies the term spread as the single variable with evidence
     of predictability (1948-2014).  At the per-ticker level we proxy this via:
     - Momentum mean-reversion z-score: rolling EMA_short / EMA_long spread
       normalised by its own rolling std.  This captures the persistent, near-
       integrated trend component the term-spread reflects.
     - Cross-over ratio: (EMA_12 - EMA_26) / EMA_26 — MACD-like persistent
       spread between two exponentials.

  2. GARCH-EFFECT CONDITIONAL VARIANCE REGRESSORS:
     The paper explicitly accounts for GARCH effects in return residuals as a
     confound.  We implement the GARCH-effect covariates the procedure controls
     for:
     - Conditional variance (EWMA, lambda=0.94): a standard GARCH(1,1) proxy.
     - Standardised return: r_t / sqrt(h_t) — pivoting by cond-vol as the paper
       does in its test statistic.
     - Squared-return autocorrelation: rolling mean of r^2 * r_{t-k}^2 products
       (ARCH-effect summary used in GARCH test diagnostics).

  3. AUTOCORRELATION REGRESSOR:
     The paper's simulation procedure uses the autocorrelation structure of
     regressors (persistence parameter rho).  We approximate with:
     - Rolling 1-lag return autocorrelation over 60 bars.
     - Rolling variance-of-variance (vol-of-vol) which the GARCH test relies on.

  CAVEAT: This is a pure OHLCV proxy.  The actual paper uses macro predictors
  (dividend yield, earnings yield, short rate, term spread) that require external
  data.  Our "term-spread analogue" is a within-ticker persistence measure only.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_oalex_jjeconom202004_small_sample_tests",
    "description": (
        "Per-ticker OHLCV proxies for the stock-return predictability regressors "
        "studied in Gungor & Luger (2020 J.Econometrics): term-spread analogue "
        "(EMA-spread z-score + MACD ratio), GARCH-effect covariates (EWMA cond-var, "
        "standardised return, ARCH-effect autocorrelation), and regressor persistence "
        "(1-lag autocorr, vol-of-vol).  Paper is a statistical TEST; these are the "
        "implied feature proxies. External macro data (term spread, div yield) unavailable."
    ),
    "requires": ["Close"],
    "produces": [
        "sstest_ema_spread_z_12_26",
        "sstest_macd_ratio_12_26",
        "sstest_ewma_cond_var_94",
        "sstest_std_return_94",
        "sstest_arch_autocorr_5",
        "sstest_ret_autocorr_60",
        "sstest_vol_of_vol_60",
    ],
    "tags": ["momentum", "volatility", "mean_reversion", "experimental"],
    "version": "1.0",
    "author": (
        "proxy:openalex-jjeconom202004 — per-ticker OHLCV approximation of the "
        "predictability regressors identified in Gungor & Luger (2020); paper is "
        "a small-sample test procedure, macro variables unavailable, implemented "
        "closest-feasible within-ticker analogue."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stock-return predictability regressor proxies.

    All operations are causal. Leading NaNs expected during warm-up.
    """
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    # ========================================================================
    # 1. TERM-SPREAD ANALOGUE
    # ========================================================================
    # EMA spread (analogous to yield curve spread: short minus long-run mean)
    ema_fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, min_periods=26, adjust=False).mean()

    ema_spread = ema_fast - ema_slow

    # Normalise by rolling std (z-score of the spread)
    spread_std = ema_spread.rolling(window=60, min_periods=30).std()
    sstest_ema_spread_z = ema_spread / spread_std.replace(0, np.nan)
    df["sstest_ema_spread_z_12_26"] = sstest_ema_spread_z

    # MACD-style ratio: (EMA12 - EMA26) / EMA26
    macd_ratio = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    df["sstest_macd_ratio_12_26"] = macd_ratio

    # ========================================================================
    # 2. GARCH-EFFECT COVARIATES
    # ========================================================================
    # EWMA conditional variance (RiskMetrics lambda=0.94, i.e. decay = 1-0.94)
    # h_t = 0.94 * h_{t-1} + 0.06 * r_{t-1}^2
    sq_ret = log_ret ** 2
    # span = 2/(1-lambda) - 1 = 2/0.06 - 1 ≈ 32.3
    ewma_span = 32
    ewma_cond_var = sq_ret.ewm(span=ewma_span, min_periods=ewma_span, adjust=False).mean()
    df["sstest_ewma_cond_var_94"] = ewma_cond_var

    # Standardised return: r_t / sqrt(h_t) — pivoted test statistic
    cond_std = np.sqrt(ewma_cond_var.replace(0, np.nan))
    std_ret = log_ret / cond_std
    df["sstest_std_return_94"] = std_ret

    # ARCH-effect autocorrelation: rolling mean of r^2_t * r^2_{t-5}
    # Measures squared-return autocorrelation (ARCH effect diagnostic)
    sq_ret_lag5 = sq_ret.shift(5)
    arch_prod = sq_ret * sq_ret_lag5
    arch_autocorr = arch_prod.rolling(window=60, min_periods=30).mean()
    # Normalise by product of rolling variances for a correlation-like measure
    sq_ret_var = sq_ret.rolling(window=60, min_periods=30).var()
    sq_ret_var_lag5 = sq_ret_lag5.rolling(window=60, min_periods=30).var()
    denom = np.sqrt(sq_ret_var * sq_ret_var_lag5).replace(0, np.nan)
    df["sstest_arch_autocorr_5"] = arch_autocorr / denom

    # ========================================================================
    # 3. REGRESSOR PERSISTENCE MEASURES
    # ========================================================================
    # Rolling 1-lag return autocorrelation over 60 bars
    # Corr(r_t, r_{t-1}) over rolling window
    ret_lag1 = log_ret.shift(1)

    def _roll_corr(x: pd.Series, y: pd.Series, w: int, min_p: int) -> pd.Series:
        """Rolling Pearson correlation between two series."""
        xw = x.rolling(window=w, min_periods=min_p)
        yw = y.rolling(window=w, min_periods=min_p)
        x_mean = xw.mean()
        y_mean = yw.mean()
        cov = (x - x_mean).rolling(window=w, min_periods=min_p).mean()
        # Use vectorised rolling covariance
        cov_xy = (x * y).rolling(window=w, min_periods=min_p).mean() - x_mean * y_mean
        std_x = xw.std(ddof=0)
        std_y = yw.std(ddof=0)
        corr = cov_xy / (std_x * std_y).replace(0, np.nan)
        return corr

    ret_autocorr_60 = _roll_corr(log_ret, ret_lag1, w=60, min_p=30)
    df["sstest_ret_autocorr_60"] = ret_autocorr_60

    # Vol-of-vol: rolling std of the rolling 20-bar realised variance
    # Key input for the GARCH-test variance-of-variance check
    realised_var_20 = sq_ret.rolling(window=20, min_periods=10).mean()
    vol_of_vol_60 = realised_var_20.rolling(window=60, min_periods=30).std()
    df["sstest_vol_of_vol_60"] = vol_of_vol_60

    return df
