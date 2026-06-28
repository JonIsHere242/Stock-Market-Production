from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_trended_momentum",
    "description": "Trended Momentum: past return signed/scaled by trend clarity (R^2 of log-price on time), plus clarity-conditioned reversal of erratic movers (Cai-Li-Keasey 2024, SSRN 4740445; Da-Gurun-Warachka 2014 RFS; Han-Zhou-Zhu 2016).",
    "requires":    [],
    "produces":    [
        "trmo_clarity_126",
        "trmo_trended_126",
        "trmo_erratic_mom_126",
        "trmo_clarity_x_id_126",
    ],
    "tags":        ["momentum", "trend", "experimental"],
    "version":     "1.0",
    "author":      "paper:Cai-Li-Keasey 2024 SSRN 4740445; Da-Gurun-Warachka 2014 RFS; Han-Zhou-Zhu 2016",
}

W = 126   # formation / clarity window (causal, ends at row t)
SKIP = 5  # skip the most recent days for the momentum proxy
MP = 90   # momentum-proxy span (documented param; PRET uses shift(SKIP)..shift(W))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    n = len(close)

    # --- log price; trailing log-returns for the erratic/clarity-interaction sign ---
    logp = np.log(close.where(close > 0.0))
    logp_vals = logp.to_numpy(dtype=float)            # may contain NaN where price<=0

    # =========================================================================
    # Rolling trend clarity R^2 = cov(t, logp)^2 / (var(t) * var(logp))
    # over a trailing window of length W ending at (and including) row t.
    # var(t) is constant for a fixed window length; cov(t, logp) and var(logp)
    # are computed per-window. Uses ONLY rows <= t (sliding_window_view is causal
    # because each window's last element is row t).
    # =========================================================================
    r2 = np.full(n, np.nan, dtype=float)
    if n >= W:
        windows = np.lib.stride_tricks.sliding_window_view(logp_vals, W)  # shape (n-W+1, W)
        # time index within each window: 0..W-1 (centered for numerical stability)
        t = np.arange(W, dtype=float)
        t_c = t - t.mean()
        var_t = np.dot(t_c, t_c)  # scalar, > 0 for W >= 2

        # mask windows that contain any NaN (price<=0) -> leave R^2 = NaN there
        valid = ~np.isnan(windows).any(axis=1)

        wv = windows[valid]
        if wv.shape[0] > 0:
            mean_y = wv.mean(axis=1, keepdims=True)
            y_c = wv - mean_y
            cov_ty = y_c @ t_c                     # (m,)  = sum(t_c * y_c)
            var_y = (y_c * y_c).sum(axis=1)        # (m,)
            denom = var_t * var_y
            with np.errstate(divide="ignore", invalid="ignore"):
                r2_valid = np.where(denom > 0.0, (cov_ty * cov_ty) / denom, np.nan)
            # clip tiny FP overshoot into [0,1]
            r2_valid = np.clip(r2_valid, 0.0, 1.0)
            out = np.full(windows.shape[0], np.nan, dtype=float)
            out[valid] = r2_valid
            r2[W - 1:] = out

    r2_s = pd.Series(r2, index=close.index)

    # =========================================================================
    # PRET = log(Close.shift(SKIP) / Close.shift(W))  -- strictly past returns.
    # Guard non-positive prices -> NaN.
    # =========================================================================
    c_skip = close.shift(SKIP).where(lambda s: s > 0.0)
    c_form = close.shift(W).where(lambda s: s > 0.0)
    ratio = c_skip / c_form.replace(0.0, np.nan)
    pret = np.log(ratio.where(ratio > 0.0))

    # =========================================================================
    # neg_frac - pos_frac of daily log-returns over the trailing window W.
    # =========================================================================
    dlog = logp.diff()
    pos = (dlog > 0.0).astype(float)
    neg = (dlog < 0.0).astype(float)
    cnt = dlog.notna().astype(float)
    win_cnt = cnt.rolling(W, min_periods=W).sum()
    pos_sum = pos.rolling(W, min_periods=W).sum()
    neg_sum = neg.rolling(W, min_periods=W).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_frac = pos_sum / win_cnt.replace(0.0, np.nan)
        neg_frac = neg_sum / win_cnt.replace(0.0, np.nan)
    sign_skew = neg_frac - pos_frac   # positive => more down-days (erratic/reversal lean)

    # =========================================================================
    # Emit features.
    # =========================================================================
    df["trmo_clarity_126"]      = r2_s
    df["trmo_trended_126"]      = pret * r2_s
    df["trmo_erratic_mom_126"]  = pret * (1.0 - r2_s)
    df["trmo_clarity_x_id_126"] = r2_s * sign_skew

    return df
