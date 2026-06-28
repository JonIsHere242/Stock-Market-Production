"""
AR rollout divergence features derived from:
  "Exposure Bias as Epistemic Underidentification in Recursive Forecasting"
  (arXiv 2606.13302).

The paper studies how k-step recursive AR rollout diverges from 1-step forecasts
due to "induced states" (self-generated predictions used as inputs). The gap between
1-step and k-step rollout is an epistemic uncertainty signal.

Key signals for OHLCV:
  1. ar_phi: rolling AR(1) coefficient on log-returns (momentum vs reversion regime)
  2. ar_rollout_div: divergence between 1-step*k and k-step cumulative prediction
     — measures how unstable the AR dynamics are locally
  3. ar_resid_autocorr: autocorrelation of AR residuals (residual structure = missed signal)
  4. ar_induced_state_dev: distance of current state from the distribution of training states
     (provenance gap: is today in the "rollout territory" = unusual past context?)
  5. ar_forecast_reversal: sign disagreement between 1-step and 5-step prediction
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13302_ar_rollout_divergence",
    "description": (
        "AR(1) rollout divergence, induced-state deviation and residual structure "
        "as epistemic instability signals; based on arXiv 2606.13302 exposure bias."
    ),
    "requires":    ["Close"],
    "produces": [
        "ar_phi_20d",
        "ar_phi_60d",
        "ar_rollout_div_5d",
        "ar_rollout_div_10d",
        "ar_resid_autocorr_20d",
        "ar_induced_state_dev_20d",
        "ar_forecast_reversal_5d",
        "ar_resid_signed_20d",
        "ar_provenance_pct_40d",
    ],
    "tags":        ["momentum", "mean_reversion", "statistical", "experimental"],
    "version":     "1.1",
    "author":      "paper:2606.13302",
}


def _rollout_k(phi: float, c: float, r0: float, k: int) -> float:
    """k-step recursive AR(1) rollout: cumulative predicted return."""
    total = 0.0
    r = r0
    for _ in range(k):
        r_next = c + phi * r
        total += r_next
        r = r_next
    return total


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values.astype(np.float64)
    n = len(df)

    phi20_arr   = np.full(n, np.nan)
    phi60_arr   = np.full(n, np.nan)
    div5_arr    = np.full(n, np.nan)
    div10_arr   = np.full(n, np.nan)
    racorr_arr  = np.full(n, np.nan)
    istate_arr  = np.full(n, np.nan)
    frev_arr    = np.full(n, np.nan)
    resid_arr   = np.full(n, np.nan)

    prev_resid_20 = [np.nan] * 5  # ring buffer for residuals

    for i in range(10, n):
        # --- 20-day window ---
        s20 = max(0, i - 20 + 1)
        r20 = log_ret[s20: i + 1]
        valid_mask = ~np.isnan(r20)
        rv = r20[valid_mask]
        k20 = len(rv)
        if k20 >= 8:
            y = rv[1:]; x = rv[:-1]
            xm = x.mean(); ym = y.mean()
            ss_xx = ((x - xm) ** 2).sum()
            if ss_xx > 1e-12:
                phi = ((x - xm) * (y - ym)).sum() / ss_xx
                c   = ym - phi * xm
                phi20_arr[i] = phi

                r_last = rv[-1]
                one_step = c + phi * r_last

                # Rollout divergence
                k5_pred  = _rollout_k(phi, c, r_last, 5)
                k10_pred = _rollout_k(phi, c, r_last, 10)
                # 1-step linear extrapolation vs recursive rollout
                div5_arr[i]  = abs(k5_pred  - 5  * one_step)
                div10_arr[i] = abs(k10_pred - 10 * one_step)

                # Forecast reversal: sign flip between 1-step and 5-step
                frev_arr[i] = float(np.sign(one_step) != np.sign(k5_pred))

                # Current AR residual
                if i > 0 and not np.isnan(log_ret[i]) and not np.isnan(log_ret[i-1]):
                    cur_resid = log_ret[i] - (c + phi * log_ret[i-1])
                    resid_arr[i] = cur_resid
                    # Track residuals for autocorrelation
                    prev_resid_20.append(cur_resid)
                    prev_resid_20 = prev_resid_20[-10:]
                    if len(prev_resid_20) >= 5:
                        pr = np.array(prev_resid_20)
                        pm = ~np.isnan(pr)
                        if pm.sum() >= 4:
                            ra = np.corrcoef(pr[pm][:-1], pr[pm][1:])[0, 1]
                            racorr_arr[i] = ra

                # Induced-state deviation: is current return in the tail of training returns?
                r_std = rv.std()
                if r_std > 1e-10:
                    istate_arr[i] = abs(r_last - rv.mean()) / r_std

        # --- 60-day window ---
        if i >= 20:
            s60 = max(0, i - 60 + 1)
            r60 = log_ret[s60: i + 1]
            vm = ~np.isnan(r60)
            rv60 = r60[vm]
            if len(rv60) >= 15:
                y = rv60[1:]; x = rv60[:-1]
                xm = x.mean(); ym = y.mean()
                ss = ((x - xm) ** 2).sum()
                if ss > 1e-12:
                    phi = ((x - xm) * (y - ym)).sum() / ss
                    phi60_arr[i] = phi

    df["ar_phi_20d"]              = phi20_arr
    df["ar_phi_60d"]              = phi60_arr
    df["ar_rollout_div_5d"]       = div5_arr
    df["ar_rollout_div_10d"]      = div10_arr
    df["ar_resid_autocorr_20d"]   = racorr_arr
    df["ar_induced_state_dev_20d"] = istate_arr
    df["ar_forecast_reversal_5d"] = frev_arr
    df["ar_resid_signed_20d"]     = resid_arr

    # Provenance: rolling percentile rank of induced-state deviation
    is_s = pd.Series(istate_arr, index=df.index)
    df["ar_provenance_pct_40d"] = is_s.rolling(40, min_periods=10).rank(pct=True)

    return df
