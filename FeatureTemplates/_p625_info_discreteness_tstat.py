import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_info_discreteness_tstat",
    "description": "DGW continuous-information via t-stat of cumulative log-return regressed on time, plus t-gated continuation momentum (Da, Gurun & Warachka 2014 RFS 'Frog in the Pan').",
    "requires":    [],
    "produces":    ["idc_tstat_63", "idc_tstat_126", "idc_contmom_63", "idc_contmom_126", "idc_signrel_126"],
    "tags":        ["momentum", "information", "experimental"],
    "version":     "1.0",
    "author":      "paper:Da, Gurun & Warachka (2014) RFS 'Frog in the Pan'",
}


def _rolling_tstat(logp: pd.Series, W: int, mp: int) -> pd.Series:
    """
    Rolling OLS t-statistic of logp regressed on time index t = 0..W-1 within each window.

    slope b = cov(t, logp) / var(t),  var(t) = (W^2 - 1) / 12  (constant within a window).
    R^2 = corr(t, logp)^2.
    resid_var = var(logp) * (1 - R^2).
    se = sqrt(resid_var / (var(t) * (W - 2))).
    tstat = b / se.

    All rolling moments use pandas rolling sums (causal: window ends at row t, uses rows <= t).
    Divisions guarded -> np.nan.
    """
    n = len(logp)
    if n < W:
        return pd.Series(np.full(n, np.nan), index=logp.index)

    # Time vector within a window is the same every window: t = 0, 1, ..., W-1
    var_t = (W * W - 1.0) / 12.0  # population variance of 0..W-1, constant

    # Rolling means / variances of logp (population, ddof=0) via rolling sums.
    s_y = logp.rolling(W, min_periods=mp).sum()
    s_yy = (logp * logp).rolling(W, min_periods=mp).sum()

    cnt = logp.rolling(W, min_periods=mp).count()  # actual finite count in window
    # Use the nominal window size for the closed-form t with the centered-time vector;
    # var_t and the t-mean assume a full W-length window, so require full windows only.
    mean_y = s_y / W
    var_y = s_yy / W - mean_y * mean_y  # population variance of logp over the window

    # cov(t, logp) = E[t*y] - E[t]*E[y].  E[t] = (W-1)/2, mean_y as above.
    # E[t*y] = sum(t_i * y_i) / W; sum(t_i * y_i) is a weighted rolling sum.
    # Build t-weights 0..W-1 and convolve via a dot-window using sliding_window_view.
    y = logp.to_numpy(dtype="float64")
    tw = np.arange(W, dtype="float64")  # 0..W-1
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    if n >= W:
        win = _swv(y, W)  # shape (n-W+1, W); window k covers rows k..k+W-1, ends at k+W-1
        # sum(t_i * y_i) per window
        s_ty_vals = win @ tw  # (n-W+1,)
        s_ty = np.full(n, np.nan)
        s_ty[W - 1:] = s_ty_vals
    else:
        s_ty = np.full(n, np.nan)
    s_ty = pd.Series(s_ty, index=logp.index)

    mean_t = (W - 1.0) / 2.0
    e_ty = s_ty / W
    cov_ty = e_ty - mean_t * mean_y

    b = cov_ty / var_t  # var_t is constant, nonzero for W>=2

    # R^2 = cov^2 / (var_t * var_y)
    denom_r2 = var_t * var_y
    r2 = (cov_ty * cov_ty) / denom_r2.replace(0.0, np.nan)
    # numerical guard: clip R^2 into [0,1]
    r2 = r2.clip(lower=0.0, upper=1.0)

    resid_var = var_y * (1.0 - r2)
    se_sq = resid_var / (var_t * (W - 2.0))
    se = np.sqrt(se_sq.clip(lower=0.0))
    tstat = b / se.replace(0.0, np.nan)

    # Only keep values where the window is fully populated (count == W); else NaN.
    tstat = tstat.where(cnt >= W, np.nan)
    return tstat


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    logp = np.log(close.where(close > 0.0, np.nan))

    tstats = {}
    for W in (63, 126):
        mp = int(0.7 * W)
        ts = _rolling_tstat(logp, W, mp).clip(lower=-15.0, upper=15.0)
        tstats[W] = ts
        df[f"idc_tstat_{W}"] = ts

        # continuation momentum: (logp - logp.shift(W)) * tanh(|tstat|/4)
        ret_W = logp - logp.shift(W)
        df[f"idc_contmom_{W}"] = ret_W * np.tanh(np.abs(ts) / 4.0)

    # sign-relative: tstat_126 / W (W = 126)
    df["idc_signrel_126"] = tstats[126] / 126.0

    # Final inf guard (should not trigger, but enforce contract)
    for c in METADATA["produces"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df
