"""
Causal Shock Propagation Features
Paper: "NetCause: Counterfactual Learning for Root Cause Analysis in Large-Scale Networks"
       (arXiv 2606.13543)

The paper's computable method:
  - Model network incidents as GRAPH-TEMPORAL PROCESSES
  - Use COUNTERFACTUAL SIMULATION to rank candidate root causes:
      "What would the system state be if this node had NOT faulted?"
  - Root cause attribution by measuring counterfactual impact on downstream nodes

OHLCV adaptation — "price shock root cause propagation":
  1. **Shock detection**: identify abnormal price moves (root cause events) via
     z-score threshold on rolling return distribution
  2. **Counterfactual baseline**: "what would price be if no shock occurred" =
     rolling pre-shock trend extrapolation
  3. **Propagation measurement**: measure how the price evolves AFTER a shock
     relative to the counterfactual baseline (continuation vs reversal)
  4. **Attribution score**: cumulative deviation from counterfactual over 1d/3d/5d
     after each detected shock → indicates "impact magnitude" of past shocks

Key insight from the paper: the COUNTERFACTUAL DELTA is the signal, not the raw value.
We measure: actual[t+k] - counterfactual[t+k] = shock propagation depth.
Then we turn this into a predictive feature by looking at recent shock history:
  - How many shocks occurred in the past W days?
  - What is the cumulative propagation of those shocks (mean-reversion vs continuation)?
  - Is the market currently in "post-shock counterfactual recovery" territory?
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13543_causal_propagation",
    "description": (
        "Price shock detection + counterfactual propagation attribution features "
        "inspired by NetCause graph-temporal counterfactual root cause analysis "
        "(arXiv 2606.13543)"
    ),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces":    [
        # Rolling z-score of daily return (shock severity)
        "csp_return_zscore_20d",
        # Binary shock flag: |z| > 2.0 within rolling distribution
        "csp_shock_flag",
        # Shock density: fraction of bars in past W days that were shocks
        "csp_shock_density_10d",
        "csp_shock_density_20d",
        # Post-shock counterfactual deviation: price vs pre-shock trend extrapolation
        # Measured at current bar: how far are we from the "no-shock" trajectory?
        "csp_counterfactual_dev_5d",
        "csp_counterfactual_dev_10d",
        # Shock propagation direction: rolling sign of (actual - counterfactual) after shocks
        # +1 = continuation (shock propagated), -1 = reversal (counterfactual restored)
        "csp_propagation_sign_10d",
        # Volume spike co-occurrence: fraction of shocks accompanied by volume spike (>2x avg)
        "csp_volume_shock_cooccurrence_20d",
        # Post-shock mean return: average 1d return following shock events in past 60d
        "csp_post_shock_mean_ret_60d",
    ],
    "tags":        ["volatility", "market_regime", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13543",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    volume = df["Volume"]
    n      = len(df)

    # ── 1. Return z-score (shock severity measure) ────────────────────────────
    ret = np.log(close / close.shift(1).clip(lower=1e-8))
    roll_mean = ret.rolling(20, min_periods=10).mean()
    roll_std  = ret.rolling(20, min_periods=10).std()
    z = (ret - roll_mean) / roll_std.clip(lower=1e-8)
    df["csp_return_zscore_20d"] = z

    # ── 2. Shock flag: |z| > 2.0 ─────────────────────────────────────────────
    shock = (z.abs() > 2.0).astype(float)
    df["csp_shock_flag"] = shock

    # ── 3. Shock density ──────────────────────────────────────────────────────
    df["csp_shock_density_10d"] = shock.rolling(10, min_periods=5).mean()
    df["csp_shock_density_20d"] = shock.rolling(20, min_periods=10).mean()

    # ── 4. Counterfactual deviation ───────────────────────────────────────────
    # "Counterfactual" = what price WOULD be if the last shock hadn't happened:
    # extrapolate the pre-shock linear trend over the window following each shock.
    # Implemented as: rolling regression slope over W days BEFORE current bar,
    # applied W/2 days forward → expected "undisturbed" log-price.
    # Deviation = log(close) - expected_log_close.

    log_close = np.log(close.clip(lower=1e-8))

    for w in [5, 10]:
        # Slope of log-close over past w bars (pre-shock linear trend)
        def slope_and_project(arr, steps_fwd=None):
            """Return slope of linear fit over arr."""
            if len(arr) < max(3, len(arr) // 2):
                return np.nan
            x = np.arange(len(arr), dtype=float)
            # Least-squares slope
            xm = x.mean()
            ym = arr.mean()
            denom = ((x - xm) ** 2).sum()
            if denom < 1e-10:
                return np.nan
            slope = ((x - xm) * (arr - ym)).sum() / denom
            return slope

        slopes = log_close.rolling(w, min_periods=w // 2).apply(
            slope_and_project, raw=True
        )
        # Expected log-close at current bar if trend had continued from w/2 bars ago:
        # expected = log_close[t - w//2] + slope * (w//2)
        half_w = w // 2
        expected = log_close.shift(half_w) + slopes.shift(half_w) * half_w
        df[f"csp_counterfactual_dev_{w}d"] = log_close - expected

    # ── 5. Propagation sign: sign of (actual - counterfactual) after shocks ──
    # We use the 10d counterfactual deviation and smooth its sign
    dev10 = df["csp_counterfactual_dev_10d"]
    df["csp_propagation_sign_10d"] = (
        np.sign(dev10).rolling(10, min_periods=5).mean()
    )

    # ── 6. Volume-shock co-occurrence ─────────────────────────────────────────
    vol_avg = volume.rolling(20, min_periods=10).mean()
    vol_spike = (volume > 2.0 * vol_avg).astype(float)
    shock_and_vol = (shock * vol_spike)
    # Among shock days, fraction also having volume spike
    shock_sum = shock.rolling(20, min_periods=10).sum().clip(lower=1e-8)
    df["csp_volume_shock_cooccurrence_20d"] = (
        shock_and_vol.rolling(20, min_periods=10).sum() / shock_sum
    )

    # ── 7. Post-shock mean return (average next-day return after shock events) ─
    # For each bar t, look back 60 days; for each shock at s in [t-60, t-1],
    # record ret[s+1]. Post-shock mean ret = mean over those events.
    ret_next = ret.shift(-1)   # next-day return (lookahead for building the stat
                               # BUT: we shift the LABEL, not the features below)
    # Actually: post_shock_mean_ret is a LOOKBACK stat only:
    # "in the past 60d, after each shock, what was the SUBSEQUENT 1-day return?"
    # ret_next[t] = return AT t+1 — but we're at bar t looking back at shocks in [t-60,t-2]
    # and their subsequent day's return in [t-59,t-1]. No lookahead.
    # ret_after_shock[s] = ret[s+1] for shock[s]=1, s < t
    # Use shift(1) on ret to get "return the day after" (already realised as of yesterday)
    ret_lag1 = ret.shift(-1)   # this is FUTURE — CANNOT use directly
    # Correct approach: for bar t, "post-shock return" at shock time s means
    # the return on day s+1 which is historical as of day t if s+1 <= t-1.
    # So use: ret_following = ret (not shifted), aligned to shock[t-1] → ret[t]
    # i.e., pair shock[s] with ret[s+1] = ret.shift(-1)[s] BUT store as of s+1.
    # At bar t, available: {(shock[s], ret[s+1]) for s <= t-2}
    # → product series: shock_at_s * ret_at_s+1 = shock.shift(1) * ret
    shock_contrib = shock.shift(1) * ret   # ret[t] is return of today, shock.shift(1) = shock yesterday
    shock_count   = shock.shift(1).rolling(60, min_periods=10).sum().clip(lower=1e-8)
    df["csp_post_shock_mean_ret_60d"] = (
        shock_contrib.rolling(60, min_periods=10).sum() / shock_count
    )

    return df
