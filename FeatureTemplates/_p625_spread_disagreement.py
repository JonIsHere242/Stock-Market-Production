import warnings

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_spread_disagreement",
    "description": "Cross-estimator spread disagreement: dispersion across Corwin-Schultz, Roll, "
                   "Abdi-Ranaldo daily spread estimators as a regime-instability signal "
                   "(Corwin&Schultz 2012 JF 67(2); Roll 1984 JF 39(4); Abdi&Ranaldo 2017 RFS 30(12); "
                   "Goyenko,Holden&Trzcinka 2009 JFE 92(2)).",
    "requires":    [],
    "produces":    ["spdis_std_20", "spdis_range_20", "spdis_cv_20", "spdis_roll_zero_frac_60"],
    "tags":        ["liquidity", "microstructure", "spread", "experimental"],
    "version":     "1.0",
    "author":      "paper:Corwin&Schultz 2012; Roll 1984; Abdi&Ranaldo 2017; Goyenko et al 2009",
}

_SMOOTH = 20
_ROLLW = 20
_ZW = 60
_CLIP_HI = 0.25


def _safe_log(x: pd.Series) -> pd.Series:
    """log of a positive price; non-positive -> NaN (no inf/-inf)."""
    x = x.astype(float)
    return np.log(x.where(x > 0.0))


def _corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Corwin-Schultz (2012) 2-day high-low spread estimator, computed per day t
    using days {t-1, t} only (strictly causal).
    """
    lh = _safe_log(high)
    ll = _safe_log(low)

    # single-day log(high/low)^2
    beta_single = (lh - ll) ** 2  # >= 0 where defined

    # beta_t uses days t-1 and t: sum of two consecutive single-day terms
    beta = beta_single + beta_single.shift(1)

    # gamma_t: log( max(H_t,H_{t-1}) / min(L_t,L_{t-1}) )^2  over the 2-day window
    h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = (_safe_log(h2) - _safe_log(l2)) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    # alpha = (sqrt(2*beta)-sqrt(beta))/denom - sqrt(gamma/denom)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.where(np.isfinite(alpha))
    # spread S = 2*(exp(alpha)-1)/(1+exp(alpha)); negative alpha -> floor at 0
    ea = np.exp(alpha.clip(upper=50.0))  # guard overflow; alpha rarely large
    spread = 2.0 * (ea - 1.0) / (1.0 + ea)
    spread = spread.where(np.isfinite(spread))
    return spread.clip(lower=0.0, upper=_CLIP_HI)


def _roll(close: pd.Series, window: int) -> pd.Series:
    """
    Roll (1984) effective spread = 2*sqrt(max(-cov(ret_t, ret_{t-1}), 0))
    over a trailing `window`-day rolling covariance. Causal.
    """
    logc = _safe_log(close)
    ret = logc.diff()
    ret_lag = ret.shift(1)
    # rolling sample covariance of (ret, ret_lag); pandas rolling.cov is trailing
    cov = ret.rolling(window, min_periods=window).cov(ret_lag)
    neg_cov = (-cov).clip(lower=0.0)
    spread = 2.0 * np.sqrt(neg_cov)
    spread = spread.where(np.isfinite(spread))
    return spread.clip(lower=0.0, upper=_CLIP_HI)


def _abdi_ranaldo(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """
    Abdi-Ranaldo (2017) CHL estimator. eta_t = (log H_t + log L_t)/2 (mid).
    S^2_t = 4 * (log C_t - eta_t) * (log C_t - eta_{t+1}).
    Causality: at day t we only know info up to t, so we align so the
    *value reported at row t* uses days <= t. We compute the per-pair product
    using (eta_{t-1}, c_{t-1}, c_t) i.e. p_t = 4*(c_{t-1}-eta_{t-1})*(c_{t-1}-eta_t),
    then average p over a trailing window. clip negatives to 0, sqrt.
    """
    lc = _safe_log(close)
    lh = _safe_log(high)
    ll = _safe_log(low)
    eta = (lh + ll) / 2.0

    # p_t built only from rows {t-1, t}: (c_{t-1}-eta_{t-1})*(c_{t-1}-eta_t)
    c_lag = lc.shift(1)
    eta_lag = eta.shift(1)
    p = 4.0 * (c_lag - eta_lag) * (c_lag - eta)
    s2 = p.rolling(window, min_periods=window).mean()
    s2 = s2.clip(lower=0.0)
    spread = np.sqrt(s2)
    spread = spread.where(np.isfinite(spread))
    return spread.clip(lower=0.0, upper=_CLIP_HI)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # ---- three inline daily spread estimators ----
    cs = _corwin_schultz(high, low)                       # per-day 2-day estimator
    roll = _roll(close, _ROLLW)                           # 20d rolling Roll
    ar = _abdi_ranaldo(high, low, close, _ROLLW)          # 20d Abdi-Ranaldo CHL

    # ---- smooth each over 20d (trailing) ----
    cs20 = cs.rolling(_SMOOTH, min_periods=_SMOOTH).mean()
    roll20 = roll.rolling(_SMOOTH, min_periods=_SMOOTH).mean()
    ar20 = ar.rolling(_SMOOTH, min_periods=_SMOOTH).mean()

    est = np.column_stack([
        cs20.to_numpy(dtype="float64"),
        roll20.to_numpy(dtype="float64"),
        ar20.to_numpy(dtype="float64"),
    ])

    # rowwise dispersion across the three estimators (ddof=0).
    # Warmup rows are all-NaN -> nan* funcs emit benign RuntimeWarnings; we mask
    # those rows back to NaN below, so silence the noise here.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_std = np.nanstd(est, axis=1, ddof=0)
        row_max = np.nanmax(est, axis=1)
        row_min = np.nanmin(est, axis=1)
        row_mean = np.nanmean(est, axis=1)

    # rows where all three are NaN -> nan* funcs warn & give nan; keep as NaN
    all_nan = np.all(~np.isfinite(est), axis=1)
    row_std = np.where(all_nan, np.nan, row_std)
    row_max = np.where(all_nan, np.nan, row_max)
    row_min = np.where(all_nan, np.nan, row_min)
    row_mean = np.where(all_nan, np.nan, row_mean)

    df["spdis_std_20"] = row_std
    df["spdis_range_20"] = row_max - row_min

    cv = row_std / (row_mean + 1e-9)
    cv = np.where(np.isfinite(cv), cv, np.nan)
    df["spdis_cv_20"] = np.clip(cv, 0.0, 5.0)

    # ---- roll-zero fraction over 60d, conditional on cs20 above its trailing 60d median ----
    roll20_s = pd.Series(roll20.to_numpy(dtype="float64"), index=df.index)
    cs20_s = pd.Series(cs20.to_numpy(dtype="float64"), index=df.index)

    cs_med60 = cs20_s.rolling(_ZW, min_periods=_ZW).median()
    # event: roll20 == 0 (illiquidity/zero-cov) while cs20 indicates a wide spread regime
    roll_is_zero = (roll20_s <= 0.0) & np.isfinite(roll20_s)
    cs_wide = cs20_s > cs_med60
    event = (roll_is_zero & cs_wide).astype(float)
    # only count rows where both inputs are defined (else NaN, excluded from mean)
    valid = (np.isfinite(roll20_s) & np.isfinite(cs_med60) & np.isfinite(cs20_s))
    event = event.where(valid)
    zero_frac = event.rolling(_ZW, min_periods=_ZW).mean()
    df["spdis_roll_zero_frac_60"] = np.clip(zero_frac.to_numpy(dtype="float64"), 0.0, 1.0)

    return df
