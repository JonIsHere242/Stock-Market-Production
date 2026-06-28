import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_range_vol_of_vol",
    "description": "Per-ticker vol-of-vol on range-based daily variance proxies: coefficient of "
                   "variation of the Parkinson/Garman-Klass variance plus its lag-1 autocorrelation "
                   "(Baltussen, van Bekkum & van der Grient 2018 RFS; Parkinson 1980; Garman-Klass 1980).",
    "requires":    [],
    "produces":    ["rvvv_park_cv_21", "rvvv_park_cv_63", "rvvv_gk_cv_63", "rvvv_park_acf1_63"],
    "tags":        ["volatility", "vol_of_vol", "range", "experimental"],
    "version":     "1.0",
    "author":      "paper:Baltussen,vanBekkum,vanderGrient(2018)RFS;Parkinson(1980);Garman-Klass(1980)",
}

_LN2 = np.log(2.0)


def _cv(x: pd.Series, w: int) -> pd.Series:
    """Trailing coefficient of variation: std(ddof=1) / (mean + 1e-8), clipped to [0, 10]."""
    mp = w // 2
    mean = x.rolling(w, min_periods=mp).mean()
    std = x.rolling(w, min_periods=mp).std(ddof=1)
    cv = std / (mean + 1e-8)
    return cv.clip(0.0, 10.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)

    # log(H/L) and log(C/O); guard non-positive denominators -> NaN (never inf)
    hl_ratio = high / low.replace(0.0, np.nan)
    hl_ratio = hl_ratio.where(hl_ratio > 0.0, np.nan)
    ln_hl = np.log(hl_ratio)

    co_ratio = close / open_.replace(0.0, np.nan)
    co_ratio = co_ratio.where(co_ratio > 0.0, np.nan)
    ln_co = np.log(co_ratio)

    ln_hl_sq = ln_hl ** 2
    ln_co_sq = ln_co ** 2

    # Parkinson daily variance proxy: ln(H/L)^2 / (4 ln2)
    park_d = ln_hl_sq / (4.0 * _LN2)
    # Garman-Klass daily variance proxy: 0.5*ln(H/L)^2 - (2ln2 - 1)*ln(C/O)^2
    gk_d = 0.5 * ln_hl_sq - (2.0 * _LN2 - 1.0) * ln_co_sq

    df["rvvv_park_cv_21"] = _cv(park_d, 21)
    df["rvvv_park_cv_63"] = _cv(park_d, 63)
    df["rvvv_gk_cv_63"] = _cv(gk_d, 63)

    # lag-1 autocorrelation of park_d over a trailing 63d window via rolling cov/var.
    # corr(x_t, x_{t-1}) = cov(x_t, x_{t-1}) / sqrt(var(x_t) * var(x_{t-1})).
    w = 63
    mp = w // 2
    x = park_d
    x_lag = park_d.shift(1)
    mean_x = x.rolling(w, min_periods=mp).mean()
    mean_l = x_lag.rolling(w, min_periods=mp).mean()
    mean_xl = (x * x_lag).rolling(w, min_periods=mp).mean()
    var_x = (x * x).rolling(w, min_periods=mp).mean() - mean_x ** 2
    var_l = (x_lag * x_lag).rolling(w, min_periods=mp).mean() - mean_l ** 2
    cov_xl = mean_xl - mean_x * mean_l
    denom = np.sqrt(var_x.clip(lower=0.0) * var_l.clip(lower=0.0))
    acf1 = cov_xl / denom.replace(0.0, np.nan)
    acf1 = acf1.where(denom > 0.0, np.nan)
    df["rvvv_park_acf1_63"] = acf1.clip(-1.0, 1.0)

    return df
