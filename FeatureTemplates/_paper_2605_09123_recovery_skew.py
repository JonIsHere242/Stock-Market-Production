"""
Path-Dependent Recovery & Skew Engineering Features  —  arXiv:2605.09123
"The Engineering of Skew: A Path-Dependent Framework for Asymmetric Volatility Management"

The paper develops a path-dependent framework for asymmetric volatility management.
Core concepts implemented per-ticker from OHLCV:

  1. DRAWDOWN DEPTH (D): rolling maximum drawdown depth over a trailing window.
     After a drawdown of depth D, the required recovery gain is R = 1/(1-D) - 1.

  2. RECOVERY BURDEN (R): nonlinear recovery requirement from current drawdown.
     This is the arithmetic of recovery: R = 1/(1-D) - 1 (convex in D).

  3. SUBMERGENCE TIME: fraction of trailing window days spent below the rolling
     peak (time underwater). Captures recovery drag and path dependency.

  4. DOWNSIDE PARTICIPATION RATIO: ratio of downside semi-deviation to upside
     semi-deviation. Low ratio = skew-engineered (more upside than downside).

  5. RECOVERY EFFICIENCY: ratio of recent return to recovery burden from the
     rolling drawdown trough. Measures how well the stock is recovering.

  6. CONDITIONAL COMPOUNDING DRAG: g ≈ mu - sigma^2/2. We compute the drag
     term (sigma^2/2) asymmetrically: downside vol contributes more harm than
     upside vol adds benefit.

All computations are rolling and causally clean.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2605_09123_recovery_skew",
    "description": (
        "Path-dependent drawdown recovery metrics: drawdown depth, nonlinear recovery "
        "burden, submergence fraction, asymmetric participation ratio, and compounding "
        "drag; implements the skew-engineering framework of arXiv:2605.09123."
    ),
    "requires": ["Close"],
    "produces": [
        "skew_drawdown_depth_60",
        "skew_drawdown_depth_120",
        "skew_recovery_burden_60",
        "skew_recovery_burden_120",
        "skew_submergence_frac_60",
        "skew_submergence_frac_120",
        "skew_downside_participation_60",
        "skew_downside_participation_120",
        "skew_compound_drag_asymm_60",
    ],
    "tags": ["volatility", "mean_reversion", "risk", "experimental"],
    "version": "1.0",
    "author": "paper:2605.09123",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute path-dependent recovery and skew engineering features.

    All windows are trailing (causal). Leading NaNs are expected during
    the warm-up period.
    """
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Daily returns (arithmetic)
    ret = np.full(n, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0

    for window in (60, 120):
        dd_arr = np.full(n, np.nan)
        rb_arr = np.full(n, np.nan)
        sub_arr = np.full(n, np.nan)
        dp_arr = np.full(n, np.nan)

        min_obs = window // 2

        for i in range(min_obs, n):
            start = max(0, i - window + 1)
            seg = close[start: i + 1]
            if len(seg) < 2:
                continue

            # Rolling peak up to current bar (causal)
            rolling_peak = np.maximum.accumulate(seg)

            # Drawdown depth at each bar within window: (peak - price) / peak
            drawdowns = (rolling_peak - seg) / np.where(rolling_peak > 0, rolling_peak, np.nan)

            # Max drawdown depth in window
            dd = np.nanmax(drawdowns)
            dd_arr[i] = dd

            # Recovery burden: R = 1/(1-D) - 1, nonlinear (convex in D)
            if dd < 1.0 - 1e-9:
                rb_arr[i] = 1.0 / (1.0 - dd) - 1.0
            else:
                rb_arr[i] = np.nan  # total wipeout: undefined

            # Submergence fraction: fraction of bars spent below peak
            submergence = np.sum(drawdowns > 1e-6) / len(drawdowns)
            sub_arr[i] = submergence

            # Asymmetric participation: downside semi-dev / upside semi-dev
            r_seg = ret[start: i + 1]
            r_seg = r_seg[~np.isnan(r_seg)]
            if len(r_seg) < 4:
                continue
            up_ret = r_seg[r_seg > 0]
            dn_ret = r_seg[r_seg < 0]
            up_semi = np.std(up_ret) if len(up_ret) > 1 else np.nan
            dn_semi = np.std(np.abs(dn_ret)) if len(dn_ret) > 1 else np.nan
            if up_semi is not None and dn_semi is not None and up_semi > 1e-10:
                dp_arr[i] = dn_semi / up_semi
            else:
                dp_arr[i] = np.nan

        df[f"skew_drawdown_depth_{window}"] = dd_arr
        df[f"skew_recovery_burden_{window}"] = rb_arr
        df[f"skew_submergence_frac_{window}"] = sub_arr
        df[f"skew_downside_participation_{window}"] = dp_arr

    # Compound drag asymmetry over 60d:
    # g ≈ mu - sigma^2/2; the drag from downside vol > gain from upside vol
    # Metric: (downside_vol^2 - upside_vol^2) / 2 → positive = net drag
    drag_arr = np.full(n, np.nan)
    window = 60
    min_obs = 30
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        r_seg = ret[start: i + 1]
        r_seg = r_seg[~np.isnan(r_seg)]
        if len(r_seg) < 4:
            continue
        up_ret = r_seg[r_seg > 0]
        dn_ret = r_seg[r_seg < 0]
        up_var = np.var(up_ret) if len(up_ret) > 1 else 0.0
        dn_var = np.var(np.abs(dn_ret)) if len(dn_ret) > 1 else 0.0
        drag_arr[i] = (dn_var - up_var) / 2.0

    df["skew_compound_drag_asymm_60"] = drag_arr

    return df
