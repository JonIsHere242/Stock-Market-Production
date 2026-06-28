"""
Idiosyncratic return kurtosis (tail thickness) -- candidate block.
Spec: ext_idio_kurtosis
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_idio_kurtosis",
    "description": (
        "Idiosyncratic return kurtosis (tail thickness). "
        "Residuals are computed from a per-ticker rolling OLS beta on SPY daily returns "
        "(trailing 120-day window). Two features are produced: "
        "(1) ext_idio_kurtosis_60d -- rolling 60-day excess kurtosis of residuals "
        "(captures fat-tailedness of idiosyncratic moves); "
        "(2) ext_idio_kurtosis_jump -- rolling 60-day abs-max-to-std ratio of residuals "
        "(single-name jumpiness proxy). "
        "These are orthogonal to the parent osap_idiovolaht (which measures volatility level) "
        "because kurtosis and max/std capture distribution shape, not scale. "
        "Cross-sectional ranking is not possible per-ticker; the raw statistics are "
        "computed in-series (time-series rolling) which is a faithful per-ticker proxy."
    ),
    "requires": ["Close"],
    "produces": ["ext_idio_kurtosis_60d", "ext_idio_kurtosis_jump"],
    "tags": ["idiosyncratic", "kurtosis", "tail", "risk", "higher_moment"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner osap_idiovolaht; spec: ext_idio_kurtosis",
}

# ---------------------------------------------------------------------------
_BETA_WIN = 120     # window for rolling OLS beta estimation
_KURT_WIN = 60      # window for kurtosis / jump stat

# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute idiosyncratic kurtosis and jumpiness from market-model residuals."""
    n = len(df)

    # Initialise output columns to NaN
    df["ext_idio_kurtosis_60d"] = np.nan
    df["ext_idio_kurtosis_jump"] = np.nan

    if n < _BETA_WIN + 2:
        return df

    # ------------------------------------------------------------------
    # 1.  Get SPY daily returns aligned to the stock's dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        return df  # degrade gracefully if index data unavailable

    if spy_close is None or len(spy_close) == 0:
        return df

    dates = pd.to_datetime(df["Date"])

    # SPY daily log-returns (shift(1) = previous day, no lookahead)
    spy_ret = spy_close.pct_change()  # daily simple return

    # Align SPY returns to stock dates via backward merge_asof
    spy_df = spy_ret.rename("spy_ret").reset_index()
    spy_df.columns = ["Date", "spy_ret"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    stock_dates = dates.reset_index(drop=True).to_frame("Date")
    aligned = pd.merge_asof(
        stock_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    aligned = aligned.set_index(stock_dates.sort_values("Date").index).reindex(stock_dates.index)
    spy_r = aligned["spy_ret"].values  # shape (n,)

    # Stock daily returns (simple)
    stock_ret = df["Close"].pct_change().values  # shape (n,)

    # ------------------------------------------------------------------
    # 2.  Rolling beta (OLS slope) via rolling covariance / variance
    #     beta_t = cov(r_stock, r_spy) / var(r_spy) over trailing BETA_WIN
    # ------------------------------------------------------------------
    sr = pd.Series(stock_ret)
    mr = pd.Series(spy_r)

    roll_cov = sr.rolling(_BETA_WIN, min_periods=_BETA_WIN // 2).cov(mr)
    roll_var = mr.rolling(_BETA_WIN, min_periods=_BETA_WIN // 2).var()

    beta = np.where(np.abs(roll_var.values) > 1e-12, roll_cov.values / roll_var.values, np.nan)

    # Rolling mean of SPY and stock (for alpha in OLS: alpha = mean(r_s) - beta*mean(r_spy))
    roll_mean_s = sr.rolling(_BETA_WIN, min_periods=_BETA_WIN // 2).mean().values
    roll_mean_m = mr.rolling(_BETA_WIN, min_periods=_BETA_WIN // 2).mean().values
    alpha = roll_mean_s - beta * roll_mean_m

    # Idiosyncratic residuals: e_t = r_stock_t - (alpha_t-1 + beta_t-1 * r_spy_t)
    # Use shift(1) of fitted params so they are known before day t (no lookahead)
    beta_lag = np.roll(beta, 1)
    beta_lag[0] = np.nan
    alpha_lag = np.roll(alpha, 1)
    alpha_lag[0] = np.nan

    resid = stock_ret - (alpha_lag + beta_lag * spy_r)
    resid_s = pd.Series(resid)

    # ------------------------------------------------------------------
    # 3.  Rolling 60-day excess kurtosis of residuals
    # ------------------------------------------------------------------
    def _excess_kurtosis(x: np.ndarray) -> float:
        """Fisher excess kurtosis (kurtosis - 3) with bias correction."""
        x = x[~np.isnan(x)]
        k = len(x)
        if k < 4:
            return np.nan
        mu = x.mean()
        m2 = np.mean((x - mu) ** 2)
        m4 = np.mean((x - mu) ** 4)
        if m2 < 1e-20:
            return np.nan
        kurt = m4 / (m2 ** 2)
        # bias-corrected excess kurtosis (Fisher G2)
        n_f = float(k)
        kurt_bc = (n_f - 1) / ((n_f - 2) * (n_f - 3)) * ((n_f + 1) * kurt - 3 * (n_f - 1))
        return kurt_bc

    kurt_vals = resid_s.rolling(_KURT_WIN, min_periods=_KURT_WIN // 2).apply(
        _excess_kurtosis, raw=True
    )

    # ------------------------------------------------------------------
    # 4.  Rolling 60-day max-abs-residual / std  (jumpiness)
    # ------------------------------------------------------------------
    roll_std = resid_s.rolling(_KURT_WIN, min_periods=_KURT_WIN // 2).std()
    roll_maxabs = resid_s.abs().rolling(_KURT_WIN, min_periods=_KURT_WIN // 2).max()
    jump = np.where(roll_std.values > 1e-12, roll_maxabs.values / roll_std.values, np.nan)

    df["ext_idio_kurtosis_60d"] = kurt_vals.values
    df["ext_idio_kurtosis_jump"] = jump

    return df
