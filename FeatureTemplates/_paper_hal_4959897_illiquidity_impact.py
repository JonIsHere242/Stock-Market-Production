"""
_paper_hal_4959897_illiquidity_impact.py
-----------------------------------------
Daily-OHLCV microstructure illiquidity & price-impact proxy feature pack.

Inspired by: "Trading volumes in stock markets: Forecasts, Trading Strategies,
Market Impact..." (hal:4959897), which links intra-daily volume patterns to
price impact and liquidity regimes.

All measures are DAILY-BAR approximations of classic microstructure quantities
that normally require tick data. The results are per-ticker proxies only; they
capture directional regimes but not true intraday market impact.

NOTE: This is an UNPROVEN candidate block (leading underscore). Auto-discovery
skips it; it must be manually promoted once validated.

Columns produced (prefix: ilq_):
    ilq_amihud_21d      -- 21-day rolling Amihud illiquidity ratio
    ilq_amihud_63d      -- 63-day rolling Amihud illiquidity ratio (normalised)
    ilq_kyle_lambda_21d -- Kyle's lambda proxy (21d rolling OLS slope of ret on
                           signed dollar volume)
    ilq_roll_spread_21d -- Roll's effective spread (from serial cov of Δprice)
    ilq_ret_vol_zscore  -- |return| / trailing 21d volume z-score
                           (high z = high impact per unit of abnormal volume)
    ilq_depth_score     -- high-volume / low-move "depth" indicator (positive = deep)
    ilq_fragility_score -- low-volume / high-move "fragility" indicator
    ilq_impact_regime   -- discrete regime: 0=liquid/deep, 1=normal, 2=illiquid/fragile
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_hal_4959897_illiquidity_impact",
    "description": (
        "Daily-OHLCV microstructure illiquidity & price-impact proxy pack: "
        "Amihud illiquidity (21d/63d), Kyle lambda proxy, Roll spread, "
        "volume-normalised return impact, depth/fragility indicators, and "
        "a discrete liquidity-regime label — all distinct from liquidity_features.py "
        "dollar-volume stats."
    ),
    "requires":    ["Open", "Close", "Volume"],
    "produces":    [
        "ilq_amihud_21d",
        "ilq_amihud_63d",
        "ilq_kyle_lambda_21d",
        "ilq_roll_spread_21d",
        "ilq_ret_vol_zscore",
        "ilq_depth_score",
        "ilq_fragility_score",
        "ilq_impact_regime",
    ],
    "tags":        ["liquidity", "volume", "microstructure", "experimental"],
    "version":     "1.0",
    "author":      "paper: hal:4959897  — daily-bar microstructure proxies",
}


# ---------------------------------------------------------------------------
# Helper: causal rolling OLS slope (price move per unit signed-vol)
# ---------------------------------------------------------------------------
def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised rolling OLS slope of y on x over `window` bars.
    At each position t the window covers rows [t-window+1 .. t] (causal).
    Returns NaN wherever the window has fewer than window valid pairs or
    where Var(x)==0.
    """
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        yw = y[t - window + 1 : t + 1]
        xw = x[t - window + 1 : t + 1]
        # Mask joint NaNs
        mask = np.isfinite(yw) & np.isfinite(xw)
        if mask.sum() < max(5, window // 2):
            continue
        yw_m = yw[mask]
        xw_m = xw[mask]
        xvar = np.var(xw_m)
        if xvar < 1e-30:
            continue
        out[t] = np.cov(xw_m, yw_m, ddof=1)[0, 1] / xvar
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute illiquidity and price-impact proxy features.

    All rolling windows are fully causal: at row t the computation uses only
    rows 0..t (no future data). Rolling operations use shift(1) only where
    the measure itself should exclude today (e.g. z-score denominators); the
    primary Amihud and Roll spread use the CURRENT bar's data (|ret|/dolvol
    at t uses Close_t and Volume_t, which are already realised at close).
    """
    close  = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    open_  = df["Open"].to_numpy(dtype=np.float64)
    n      = len(df)
    idx    = df.index

    # ------------------------------------------------------------------
    # 0. Guard: replace zero volume with NaN (avoids div-by-zero silently)
    # ------------------------------------------------------------------
    vol_safe = np.where(volume > 0, volume, np.nan)

    # ------------------------------------------------------------------
    # 1. Daily log-return and dollar-volume
    # ------------------------------------------------------------------
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])               # causal: row t uses close[t-1]

    dolvol = close * vol_safe                                    # close_t × vol_t (realised at close)
    dolvol_safe = np.where(dolvol > 0, dolvol, np.nan)

    # ------------------------------------------------------------------
    # 2. Amihud illiquidity ratio  =  |log_ret| / dollar_volume
    #    Roll-window mean — causal at every bar
    # ------------------------------------------------------------------
    amihud_daily = np.abs(log_ret) / dolvol_safe                # NaN where vol=0 or first row

    # Scale up for readability (×1e6): typical value ~ 1e-9 raw
    amihud_daily_scaled = amihud_daily * 1e6

    amihud_s = pd.Series(amihud_daily_scaled, index=idx)
    ilq_amihud_21d = amihud_s.rolling(21, min_periods=10).mean()
    ilq_amihud_63d = amihud_s.rolling(63, min_periods=21).mean()

    # ------------------------------------------------------------------
    # 3. Kyle's lambda proxy
    #    slope of log_ret ~ signed_dollar_volume  (rolling 21d OLS)
    #    signed_dolvol = sign(ret) × dolvol  (or sign(close-open) × dolvol)
    #    We use sign(log_ret) to be consistent with the return series.
    # ------------------------------------------------------------------
    sign_ret = np.sign(log_ret)                                  # -1, 0, +1
    signed_dolvol = sign_ret * dolvol_safe                       # NaN propagates

    # Normalise signed_dolvol by 1e6 so slope is in sensible units
    signed_dolvol_norm = signed_dolvol / 1e6

    kyle_arr = _rolling_ols_slope(log_ret, signed_dolvol_norm, window=21)
    ilq_kyle_lambda_21d = pd.Series(kyle_arr, index=idx)

    # ------------------------------------------------------------------
    # 4. Roll's effective spread
    #    From serial covariance of consecutive price changes:
    #    spread_t ≈ 2 × sqrt(−cov(Δp_t, Δp_{t-1}))   when cov < 0
    #    We use log-return changes (Δp = log_ret), rolling 21 bars.
    #    Normalised to close price to get relative spread.
    # ------------------------------------------------------------------
    delta_p = log_ret                                            # Δp_t = log_ret_t
    delta_p_lag = np.full(n, np.nan)
    delta_p_lag[1:] = delta_p[:-1]                              # Δp_{t-1}

    dp_s    = pd.Series(delta_p,     index=idx)
    dp_lag_s = pd.Series(delta_p_lag, index=idx)

    roll_spread_arr = np.full(n, np.nan)
    for t in range(21 - 1, n):
        dp_w   = dp_s.iloc[t - 21 + 1 : t + 1].to_numpy()
        lag_w  = dp_lag_s.iloc[t - 21 + 1 : t + 1].to_numpy()
        mask   = np.isfinite(dp_w) & np.isfinite(lag_w)
        if mask.sum() < 10:
            continue
        cov = np.cov(dp_w[mask], lag_w[mask], ddof=1)[0, 1]
        if cov < 0:
            roll_spread_arr[t] = 2.0 * np.sqrt(-cov)
        # If cov >= 0 (no bid-ask bounce signal) leave as NaN

    ilq_roll_spread_21d = pd.Series(roll_spread_arr, index=idx)

    # ------------------------------------------------------------------
    # 5. Return / volume-z-score  (impact per unit abnormal volume)
    #    vol_zscore_t = (vol_t - mean_vol_{t-1..t-22}) / std_vol_{t-1..t-22}
    #    Then ratio = |log_ret_t| / (|vol_zscore_t| + epsilon)
    # ------------------------------------------------------------------
    vol_s = pd.Series(vol_safe, index=idx)
    vol_mean_21 = vol_s.shift(1).rolling(21, min_periods=10).mean()
    vol_std_21  = vol_s.shift(1).rolling(21, min_periods=10).std()

    vol_z = (vol_s - vol_mean_21) / (vol_std_21 + 1e-8)         # causal: mean/std from past bars

    abs_ret_s = pd.Series(np.abs(log_ret), index=idx)
    ilq_ret_vol_zscore = abs_ret_s / (vol_z.abs() + 0.1)        # +0.1 to damp noise at vol_z≈0

    # Replace inf that could slip through
    ilq_ret_vol_zscore = ilq_ret_vol_zscore.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 6. Depth vs Fragility indicators
    #
    #    depth_score:     high volume, low absolute return (liquid / absorbed)
    #                     = vol_z_21 when abs_ret < 21d median abs_ret
    #    fragility_score: low volume, high absolute return (impact-sensitive)
    #                     = |abs_ret - median_ret| / (vol_z + ε) when vol_z < 0
    #
    #    Both are signed continuous scores; positivity = more depth/fragility.
    # ------------------------------------------------------------------
    # Rolling 21d median of |return| for threshold
    abs_ret_med21 = abs_ret_s.shift(1).rolling(21, min_periods=10).median()

    high_vol_mask = vol_z > 0
    low_ret_mask  = abs_ret_s <= abs_ret_med21

    # Depth: positive volume surprise × small return (absorbed)
    depth_arr = np.where(
        high_vol_mask & low_ret_mask,
        vol_z.to_numpy(),
        np.where(
            low_ret_mask,
            vol_z.to_numpy(),              # any low-return day, vol_z tells depth
            np.nan,
        ),
    )
    # Simpler: depth = vol_z × (1 - abs_ret / (abs_ret_med21 + 1e-8)) → positive = deep
    ret_norm = abs_ret_s / (abs_ret_med21 + 1e-8)
    depth_raw = vol_z * (1.0 - ret_norm)
    ilq_depth_score = depth_raw.replace([np.inf, -np.inf], np.nan)

    # Fragility: negative vol surprise × large return → fragility_score > 0
    fragility_raw = -vol_z * (ret_norm - 1.0)                   # high when vol_z<0 and ret>med
    ilq_fragility_score = fragility_raw.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 7. Discrete impact regime
    #    Uses percentile ranks of Amihud (21d) over a trailing 63-bar window
    #    to classify each day into 3 liquidity regimes:
    #      0 = liquid / deep   (Amihud in bottom tercile)
    #      1 = normal          (middle tercile)
    #      2 = illiquid / fragile (top tercile)
    # ------------------------------------------------------------------
    amihud_pctile = ilq_amihud_21d.rolling(63, min_periods=21).rank(pct=True)

    regime_arr = np.select(
        [
            amihud_pctile.isna(),
            amihud_pctile < 1/3,
            amihud_pctile >= 2/3,
        ],
        [
            np.nan,
            0.0,
            2.0,
        ],
        default=1.0,
    )
    ilq_impact_regime = pd.Series(regime_arr, index=idx)

    # ------------------------------------------------------------------
    # 8. Assemble — only ADD the produced columns, never modify originals
    # ------------------------------------------------------------------
    new_cols = {
        "ilq_amihud_21d":      ilq_amihud_21d,
        "ilq_amihud_63d":      ilq_amihud_63d,
        "ilq_kyle_lambda_21d": ilq_kyle_lambda_21d,
        "ilq_roll_spread_21d": ilq_roll_spread_21d,
        "ilq_ret_vol_zscore":  ilq_ret_vol_zscore,
        "ilq_depth_score":     ilq_depth_score,
        "ilq_fragility_score": ilq_fragility_score,
        "ilq_impact_regime":   ilq_impact_regime,
    }

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
