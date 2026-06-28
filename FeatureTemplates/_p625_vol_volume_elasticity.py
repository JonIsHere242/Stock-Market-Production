import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_vol_volume_elasticity",
    "description": "Volume-volatility elasticity (MDH): rolling log-log OLS slope of |return| on volume, its R^2, and recent standardized residual. Tauchen & Pitts (1983) Econometrica; Karpoff (1987) JFQA; Gallant, Rossi & Tauchen (1992) RFS.",
    "requires":    [],
    "produces":    ["vve_elasticity_60", "vve_elast_r2_60", "vve_elast_resid_5"],
    "tags":        ["volume", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "paper:Tauchen&Pitts(1983)Econometrica; Karpoff(1987)JFQA; Gallant,Rossi&Tauchen(1992)RFS",
}

W = 60   # regression window
MP = 36  # min periods


def _roll_sum(s: pd.Series, w: int, mp: int) -> pd.Series:
    return s.rolling(window=w, min_periods=mp).sum()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # x = log volume, y = log |return| ; clip to keep logs finite
    r = close.pct_change()
    x = np.log(vol.clip(lower=1.0))
    y = np.log(r.abs().clip(lower=1e-6))

    # rows where either x or y is non-finite must not enter the rolling sums
    valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    xv = x.where(valid, 0.0)
    yv = y.where(valid, 0.0)
    n_ind = pd.Series(np.where(valid, 1.0, 0.0), index=df.index)

    # rolling sufficient statistics over W (closed-form OLS), trailing only
    n = _roll_sum(n_ind, W, MP)
    Sx = _roll_sum(xv, W, MP)
    Sy = _roll_sum(yv, W, MP)
    Sxx = _roll_sum(xv * xv, W, MP)
    Syy = _roll_sum(yv * yv, W, MP)
    Sxy = _roll_sum(xv * yv, W, MP)

    # need a real OLS (>=2 valid points and non-degenerate x); guard all denoms
    n_safe = n.where(n >= 2.0, np.nan)
    Sxx_c = Sxx - Sx * Sx / n_safe          # n * var(x)
    Syy_c = Syy - Sy * Sy / n_safe          # n * var(y)
    Sxy_c = Sxy - Sx * Sy / n_safe          # n * cov(x, y)

    denom_x = Sxx_c.replace(0.0, np.nan)
    beta = Sxy_c / denom_x
    beta = beta.replace([np.inf, -np.inf], np.nan)
    df["vve_elasticity_60"] = beta.clip(-5.0, 5.0)

    # R^2 = cov^2 / (var_x * var_y)
    denom_r2 = (Sxx_c * Syy_c).replace(0.0, np.nan)
    r2 = (Sxy_c * Sxy_c) / denom_r2
    r2 = r2.replace([np.inf, -np.inf], np.nan)
    df["vve_elast_r2_60"] = r2.clip(0.0, 1.0)

    # intercept a = ybar - beta * xbar  (trailing window means)
    xbar = Sx / n_safe
    ybar = Sy / n_safe
    a = ybar - beta * xbar

    # per-row residual using the trailing-window OLS line at that row
    resid = (y - (a + beta * x))
    # only meaningful where the current point's x,y are finite
    resid = resid.where(pd.Series(valid, index=df.index), np.nan)
    resid = resid.replace([np.inf, -np.inf], np.nan)

    # standardize by trailing rolling std of the residual, then mean of last 5d
    resid_std = resid.rolling(window=W, min_periods=MP).std()
    resid_std = resid_std.replace(0.0, np.nan)
    resid_z = resid / resid_std
    resid_z = resid_z.replace([np.inf, -np.inf], np.nan)
    resid5 = resid_z.rolling(window=5, min_periods=5).mean()
    df["vve_elast_resid_5"] = resid5.clip(-5.0, 5.0)

    return df
