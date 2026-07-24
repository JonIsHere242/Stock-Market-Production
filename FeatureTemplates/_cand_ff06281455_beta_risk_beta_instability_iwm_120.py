from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load index helper by path ------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06281455_beta_risk_beta_instability_iwm_120",
    "description": (
        "Rolling beta instability vs IWM. Computes a 40-day rolling OLS beta "
        "of the stock's daily returns vs IWM daily returns, then over the last "
        "120 bars measures: (1) std of that rolling-beta series, (2) range "
        "(max-min) of that rolling-beta series, and (3) the difference between "
        "the beta instability in down-IWM days vs up-IWM days. "
        "Captures whether a stock's market sensitivity is stable or erratic, "
        "a signal axis distinct from raw beta level."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06281455_beta_risk_beta_instability_iwm_120_std",
        "ff06281455_beta_risk_beta_instability_iwm_120_range",
        "ff06281455_beta_risk_beta_instability_iwm_120_asymm",
    ],
    "tags": ["beta", "risk", "instability", "regime", "iwm"],
    "version": "1.0.0",
    "author": "feature-factory ff06281455",
}

# Window parameters
_BETA_WIN = 40   # rolling window for each beta estimate
_STAB_WIN = 120  # window over which beta-series stability is measured


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    out_std   = np.full(len(df), np.nan)
    out_range = np.full(len(df), np.nan)
    out_asymm = np.full(len(df), np.nan)

    cols = [
        "ff06281455_beta_risk_beta_instability_iwm_120_std",
        "ff06281455_beta_risk_beta_instability_iwm_120_range",
        "ff06281455_beta_risk_beta_instability_iwm_120_asymm",
    ]

    if len(df) < _BETA_WIN + 2:
        df[cols[0]] = out_std
        df[cols[1]] = out_range
        df[cols[2]] = out_asymm
        return df

    # -- fetch IWM close series -----------------------------------------------
    try:
        iwm_series = _indexes.index_close("IWM")  # DatetimeIndex → Close
    except Exception:
        df[cols[0]] = out_std
        df[cols[1]] = out_range
        df[cols[2]] = out_asymm
        return df

    if iwm_series is None or len(iwm_series) == 0:
        df[cols[0]] = out_std
        df[cols[1]] = out_range
        df[cols[2]] = out_asymm
        return df

    # -- align IWM to our dates via merge_asof --------------------------------
    iwm_df = (
        iwm_series
        .rename("iwm_close")
        .reset_index()
        .rename(columns={"index": "Date", "Date": "Date"})
    )
    # Ensure Date column is datetime in both frames
    iwm_df["Date"] = pd.to_datetime(iwm_df["Date"])
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        iwm_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    work = work.set_index(df.index)

    stock_close = work["Close"].values.astype(float)
    iwm_close   = work["iwm_close"].values.astype(float)

    # Daily log returns (no lookahead: return[t] uses close[t] and close[t-1])
    stock_ret = np.empty(len(df), dtype=float)
    iwm_ret   = np.empty(len(df), dtype=float)
    stock_ret[0] = np.nan
    iwm_ret[0]   = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret[1:] = np.where(
            stock_close[:-1] > 0,
            np.log(stock_close[1:] / stock_close[:-1]),
            np.nan,
        )
        iwm_ret[1:] = np.where(
            iwm_close[:-1] > 0,
            np.log(iwm_close[1:] / iwm_close[:-1]),
            np.nan,
        )

    n = len(df)

    # -- compute rolling 40-day beta at each bar ------------------------------
    # Beta[t] = cov(stock[t-W+1..t], iwm[t-W+1..t]) / var(iwm[t-W+1..t])
    # Vectorised with numpy stride tricks for correctness and speed.
    # For each t >= W-1: use indices [t-W+1 .. t]
    W = _BETA_WIN
    rolling_beta = np.full(n, np.nan)

    # Build a padded matrix of shape (n, W): row t = [ret[t-W+1..t]]
    # Use numpy cumsum trick for rolling cov/var to stay vectorised.
    # cov(x,y) = mean(x*y) - mean(x)*mean(y); var(y) = mean(y^2) - mean(y)^2
    # We compute rolling sums via cumsum.

    sr = stock_ret
    ir = iwm_ret

    # Replace NaN with 0 temporarily for cumsum, but mask positions that had NaN
    valid = np.isfinite(sr) & np.isfinite(ir)
    sr_v = np.where(valid, sr, 0.0)
    ir_v = np.where(valid, ir, 0.0)
    xy_v = np.where(valid, sr * ir, 0.0)
    ii_v = np.where(valid, ir * ir, 0.0)
    cnt_v = valid.astype(float)

    # Cumulative sums (pad with a leading 0 for prefix-sum indexing)
    cs_sr  = np.concatenate([[0.0], np.cumsum(sr_v)])
    cs_ir  = np.concatenate([[0.0], np.cumsum(ir_v)])
    cs_xy  = np.concatenate([[0.0], np.cumsum(xy_v)])
    cs_ii  = np.concatenate([[0.0], np.cumsum(ii_v)])
    cs_cnt = np.concatenate([[0.0], np.cumsum(cnt_v)])

    for t in range(W - 1, n):
        start = t - W + 1  # inclusive
        end   = t + 1      # exclusive (prefix-sum index)
        c = cs_cnt[end] - cs_cnt[start]
        if c < 5:          # need at least 5 valid pairs
            continue
        mean_s  = (cs_sr[end] - cs_sr[start]) / c
        mean_i  = (cs_ir[end] - cs_ir[start]) / c
        mean_xy = (cs_xy[end] - cs_xy[start]) / c
        mean_ii = (cs_ii[end] - cs_ii[start]) / c
        cov_si  = mean_xy - mean_s * mean_i
        var_i   = mean_ii - mean_i * mean_i
        if var_i <= 0:
            continue
        rolling_beta[t] = cov_si / var_i

    # -- compute 120-day stability metrics over rolling_beta ------------------
    S = _STAB_WIN

    # Also need iwm direction for asymmetry: up-day vs down-day
    # up_day[t] = iwm_ret[t] > 0
    iwm_up = (iwm_ret > 0).astype(float)   # 1 = up, 0 = down/flat/nan

    for t in range(S - 1, n):
        window_beta = rolling_beta[t - S + 1 : t + 1]
        wv = window_beta[np.isfinite(window_beta)]
        if len(wv) < 10:
            continue
        out_std[t]   = np.std(wv, ddof=1)
        out_range[t] = np.max(wv) - np.min(wv)

        # Asymmetry: std of beta in up-IWM windows vs down-IWM windows
        # iwm_up is per-bar; we mark each beta[i] by direction of iwm_ret[i]
        window_up = iwm_up[t - S + 1 : t + 1]
        wb_full   = window_beta  # same slice
        mask_up   = np.isfinite(wb_full) & (window_up == 1)
        mask_dn   = np.isfinite(wb_full) & (window_up == 0)
        beta_up   = wb_full[mask_up]
        beta_dn   = wb_full[mask_dn]
        if len(beta_up) >= 5 and len(beta_dn) >= 5:
            std_up = np.std(beta_up, ddof=1)
            std_dn = np.std(beta_dn, ddof=1)
            out_asymm[t] = std_dn - std_up  # positive = more erratic in down markets

    df[cols[0]] = out_std
    df[cols[1]] = out_range
    df[cols[2]] = out_asymm
    return df
