"""
Network diffusion factor coupling — per-ticker proxy derived from:
  "Emergence of Statistical Financial Factors by a Diffusion Process"
  (arxiv:2604.12197).

The paper models factor emergence via a Laplacian-coupled iterated-map network
where assets' returns depend on their own past returns AND those of structurally
related neighbors through a coupling matrix derived from a graph Laplacian.
Factors emerge as stable co-movement patterns when coupling λ is above a threshold.

Per-ticker OHLCV proxy using SPY as the single "market neighbor":
  - Multi-lag cross-correlation between stock return and SPY return at lags 1..L,
    summed with exponential (heat-kernel) weights: captures how strongly the stock's
    dynamics are "absorbed" into the market diffusion over multiple time steps.
  - Idiosyncratic residual variance: the fraction of rolling return variance NOT
    explained by the lagged market diffusion term.
  - Diffusion absorption rate: the decay rate of the lagged cross-correlation
    profile (estimated as the slope of log|xcorr| vs lag k), measuring how
    quickly the stock re-couples to the market factor after a shock.

All rolling, causal, no lookahead. Requires SPY index data via _indexes.
"""

import numpy as np
import pandas as pd

try:
    from _indexes import index_close
    _HAS_INDEXES = True
except ImportError:
    _HAS_INDEXES = False

METADATA = {
    "name": "network_diffusion_coupling",
    "description": (
        "Multi-lag heat-kernel diffusion coupling to market (SPY): captures how strongly "
        "a ticker's return dynamics get 'absorbed' into the market factor over multiple "
        "lags (from arxiv:2604.12197 Laplacian-coupled diffusion factor emergence model)."
    ),
    "requires": ["Close"],
    "produces": [
        "ndf_coupling_strength_20",    # rolling 20d: sum of exp-weighted lagged xcorr with market
        "ndf_coupling_strength_60",    # rolling 60d: same
        "ndf_idio_var_ratio_20",       # rolling 20d: idiosyncratic var / total var (1 - R^2)
        "ndf_idio_var_ratio_60",       # rolling 60d: same
        "ndf_xcorr_decay_rate_60",     # rolling 60d: decay rate of lagged xcorr profile (slope)
    ],
    "tags": ["experimental", "market_coupling", "network_diffusion"],
    "version": "1.0",
    "author": "paper-mining slate 5",
}

_LAGS = 3          # max lag L for diffusion coupling (3 days captures short-term market absorption)
_TAU = 2.0         # heat-kernel decay timescale (exponential weights ~ exp(-k/tau))
_WIN_SHORT = 20
_WIN_LONG = 60
_MIN_OBS = 15


def _exp_weights(lags: int, tau: float) -> np.ndarray:
    """Exponential heat-kernel weights for lags 1..lags."""
    k = np.arange(1, lags + 1, dtype=np.float64)
    w = np.exp(-k / tau)
    return w / w.sum()


_EXP_W = _exp_weights(_LAGS, _TAU)


def _compute_lag_xcorrs(s_win: np.ndarray, m_win: np.ndarray, win: int) -> np.ndarray:
    """
    Vectorized computation of lagged cross-correlations for lags 1.._LAGS.
    Returns array of length _LAGS (NaN where insufficient data).
    """
    xcorrs = np.full(_LAGS, np.nan)
    for lag_idx, lag in enumerate(range(1, _LAGS + 1)):
        if win - lag < _MIN_OBS:
            continue
        s_lag = s_win[lag:]
        m_lag = m_win[: win - lag]
        mask = np.isfinite(s_lag) & np.isfinite(m_lag)
        n_valid = mask.sum()
        if n_valid < _MIN_OBS:
            continue
        s_v = s_lag[mask]
        m_v = m_lag[mask]
        s_std = s_v.std()
        m_std = m_v.std()
        if s_std < 1e-12 or m_std < 1e-12:
            continue
        xcorrs[lag_idx] = np.dot(s_v - s_v.mean(), m_v - m_v.mean()) / (n_valid * s_std * m_std)
    return xcorrs


def _rolling_xcorr_coupling(stock_ret: np.ndarray, mkt_ret: np.ndarray,
                             t: int, win: int) -> float:
    """
    Compute the heat-kernel-weighted sum of lagged cross-correlations between
    stock_ret and mkt_ret inside window [t-win+1, t].

    For lag k: corr(stock_ret[s], mkt_ret[s-k]) over the window (removing
    the first k observations so both series are aligned).

    Returns NaN if insufficient data.
    """
    start = t - win + 1
    if start < _LAGS:
        return np.nan

    s_win = stock_ret[start: t + 1]
    m_win = mkt_ret[start: t + 1]

    xcorrs = _compute_lag_xcorrs(s_win, m_win, win)
    valid_mask = np.isfinite(xcorrs)
    if not valid_mask.any():
        return np.nan

    # Weighted sum with exp weights; use only valid lags
    w = _EXP_W.copy()
    w[~valid_mask] = 0.0
    w_sum = w.sum()
    if w_sum < 1e-12:
        return np.nan
    return float(np.dot(w / w_sum, np.where(valid_mask, xcorrs, 0.0)))


def _precompute_idio_var(stock_ret: np.ndarray, mkt_ret: np.ndarray,
                         win: int, min_obs: int) -> np.ndarray:
    """
    Vectorized precomputation of rolling idiosyncratic variance ratio (1 - R²).
    Uses the formula: R² = (cov(s,m))² / (var(s) * var(m)).
    All done via pandas rolling to avoid per-row Python loops.
    Returns array of length n.
    """
    s = pd.Series(stock_ret, dtype=np.float64)
    m = pd.Series(mkt_ret, dtype=np.float64)

    roll_cov = s.rolling(win, min_periods=min_obs).cov(m)
    roll_var_s = s.rolling(win, min_periods=min_obs).var()
    roll_var_m = m.rolling(win, min_periods=min_obs).var()

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (roll_cov ** 2) / (roll_var_s * roll_var_m)

    idio = 1.0 - r2.clip(0.0, 1.0)
    return idio.to_numpy(dtype=np.float64)


def _rolling_xcorr_decay(stock_ret: np.ndarray, mkt_ret: np.ndarray,
                         t: int, win: int) -> float:
    """
    Estimate the decay rate of the lagged cross-correlation profile over lags 1.._LAGS.
    Fit a linear regression of log|xcorr[lag]| ~ lag (slope = -decay_rate).
    Positive slope = growing coupling (stock absorbing market shock over time).
    Negative slope = decaying coupling (fast absorption, quick convergence).
    Reuses _compute_lag_xcorrs to avoid redundant inner loops.
    """
    start = t - win + 1
    if start < _LAGS:
        return np.nan

    s_win = stock_ret[start: t + 1]
    m_win = mkt_ret[start: t + 1]

    xcorrs = _compute_lag_xcorrs(s_win, m_win, win)
    abs_xcorrs = np.abs(xcorrs)
    valid = np.isfinite(abs_xcorrs) & (abs_xcorrs > 1e-10)
    if valid.sum() < 3:
        return np.nan

    lag_arr = np.arange(1, _LAGS + 1, dtype=np.float64)[valid]
    xc_arr = np.log(abs_xcorrs[valid])

    lag_mean = lag_arr.mean()
    xc_mean = xc_arr.mean()
    lag_dem = lag_arr - lag_mean
    denom = np.dot(lag_dem, lag_dem)
    if denom < 1e-14:
        return np.nan

    slope = np.dot(lag_dem, xc_arr - xc_mean) / denom
    return float(slope)   # negative = decaying xcorr (fast market absorption)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network diffusion coupling features for a single ticker.
    Aligns SPY closes to ticker dates via backward merge_asof.
    All features are rolling/causal — no lookahead.
    """
    n = len(df)

    # Initialize outputs
    ndf_coupling_s20  = np.full(n, np.nan)
    ndf_coupling_s60  = np.full(n, np.nan)
    ndf_idio_v20      = np.full(n, np.nan)
    ndf_idio_v60      = np.full(n, np.nan)
    ndf_xcorr_decay60 = np.full(n, np.nan)

    # ---- Load SPY close and compute market returns ----------------------------
    if not _HAS_INDEXES:
        df["ndf_coupling_strength_20"] = np.nan
        df["ndf_coupling_strength_60"] = np.nan
        df["ndf_idio_var_ratio_20"]    = np.nan
        df["ndf_idio_var_ratio_60"]    = np.nan
        df["ndf_xcorr_decay_rate_60"]  = np.nan
        return df

    spy_close = index_close("SPY")  # pd.Series indexed by Date (DatetimeIndex)
    if spy_close.empty:
        df["ndf_coupling_strength_20"] = np.nan
        df["ndf_coupling_strength_60"] = np.nan
        df["ndf_idio_var_ratio_20"]    = np.nan
        df["ndf_idio_var_ratio_60"]    = np.nan
        df["ndf_xcorr_decay_rate_60"]  = np.nan
        return df

    # Build a date -> SPY close lookup for backward merge
    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    # Align to ticker dates
    ticker_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(
        ticker_dates.sort_values("Date"),
        spy_df,
        on="Date",
        direction="backward",
    )
    # Reorder merged to match original df order (merge_asof sorts, but df may already be sorted)
    spy_aligned = merged["spy_close"].values  # already in ascending date order

    # Stock log-returns and market log-returns
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    stock_ret = np.empty(n, dtype=np.float64)
    stock_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret[1:] = np.log(close_arr[1:] / close_arr[:-1])

    mkt_ret = np.empty(n, dtype=np.float64)
    mkt_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mkt_ret[1:] = np.log(
            np.where(spy_aligned[:-1] > 0, spy_aligned[1:] / spy_aligned[:-1], np.nan)
        )

    # Precompute rolling idiosyncratic variance ratio (vectorized via pandas)
    idio20 = _precompute_idio_var(stock_ret, mkt_ret, _WIN_SHORT, _MIN_OBS)
    idio60 = _precompute_idio_var(stock_ret, mkt_ret, _WIN_LONG, _MIN_OBS)

    # Pre-compute constants
    _lag_arr_full = np.arange(1, _LAGS + 1, dtype=np.float64)

    # ---- Compute rolling xcorr coupling features ----------------------------
    for t in range(_LAGS, n):
        # Coupling strength - 20d window
        if t >= _WIN_SHORT - 1:
            start20 = t - _WIN_SHORT + 1
            s20 = stock_ret[start20: t + 1]
            m20 = mkt_ret[start20: t + 1]
            xc20 = _compute_lag_xcorrs(s20, m20, _WIN_SHORT)
            valid20 = np.isfinite(xc20)
            if valid20.any():
                w20 = _EXP_W.copy(); w20[~valid20] = 0.0
                ws20 = w20.sum()
                if ws20 > 1e-12:
                    ndf_coupling_s20[t] = float(np.dot(w20 / ws20, np.where(valid20, xc20, 0.0)))
            # Idio var: precomputed
            ndf_idio_v20[t] = idio20[t]

        # Coupling strength + decay - 60d window
        if t >= _WIN_LONG - 1:
            start60 = t - _WIN_LONG + 1
            s60 = stock_ret[start60: t + 1]
            m60 = mkt_ret[start60: t + 1]
            xc60 = _compute_lag_xcorrs(s60, m60, _WIN_LONG)
            valid60 = np.isfinite(xc60)
            if valid60.any():
                w60 = _EXP_W.copy(); w60[~valid60] = 0.0
                ws60 = w60.sum()
                if ws60 > 1e-12:
                    ndf_coupling_s60[t] = float(np.dot(w60 / ws60, np.where(valid60, xc60, 0.0)))
                # Decay rate from the same xcorr profile
                abs_xc = np.abs(xc60)
                decay_valid = valid60 & (abs_xc > 1e-10)
                if decay_valid.sum() >= 3:
                    lags_v = _lag_arr_full[decay_valid]
                    log_xc = np.log(abs_xc[decay_valid])
                    lag_m = lags_v.mean(); xc_m = log_xc.mean()
                    lag_d = lags_v - lag_m
                    denom = np.dot(lag_d, lag_d)
                    if denom > 1e-14:
                        ndf_xcorr_decay60[t] = float(np.dot(lag_d, log_xc - xc_m) / denom)
            # Idio var: precomputed
            ndf_idio_v60[t] = idio60[t]

    # Attach to df
    idx = df.index
    new_cols = pd.DataFrame(
        {
            "ndf_coupling_strength_20": ndf_coupling_s20,
            "ndf_coupling_strength_60": ndf_coupling_s60,
            "ndf_idio_var_ratio_20":    ndf_idio_v20,
            "ndf_idio_var_ratio_60":    ndf_idio_v60,
            "ndf_xcorr_decay_rate_60":  ndf_xcorr_decay60,
        },
        index=idx,
    )
    return pd.concat([df, new_cols], axis=1)
