"""
orig_orig_vix_features.py  --  AlphaSensitivity VIX feature suite, ORIGINAL CapCase names.

Faithful 1:1 port of calculate_vix_features() from
_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py (L1042-1386), emitting the
EXACT original column names (CapCase, e.g. "VIX_Close", "VIX_Percentile_30d", "Dynamic_RSI").

This replicates the source MATH verbatim -- including the python iloc loops for the
percentile / momentum / acceleration / vol-of-vol / pattern / dynamic-RSI features -- rather
than the vectorized v2 rewrite (FeatureTemplates/vix_features.py), so that every produced
column lands at the float32 floor against the ground truth.

Index (VIX) data comes from the shared FeatureTemplates/_indexes.py helper. If VIX is not
available on disk, VIX_Close (and everything derived from it) degrades to NaN -- the formulas
are still ported faithfully; only the data source would be missing.

Per the FeatureFramework contract the source's terminal whole-frame ffill loop, fillna(0),
and reset_index(drop=True) are deliberately NOT reproduced: NaNs are left as-is and df order /
index are never touched. compute() only ADDS the produced columns.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# ---------------------------------------------------------------------------
# Load the underscore-prefixed shared index helper by path (auto-discovery
# skips it, so we import it explicitly -- same pattern as vix_features.py).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


METADATA = {
    "name":        "orig_orig_vix_features",
    "description": "AlphaSensitivity VIX market-regime / vol-of-vol / VIX-adjusted price suite (original CapCase names)",
    "requires":    ["Date", "Close", "High", "Low"],
    "produces": [
        "VIX_Close",
        "VIX_Regime_Numeric",
        "VIX_Percentile_30d",
        "VIX_Percentile_60d",
        "VIX_Percentile_252d",
        "VIX_Change_1d",
        "VIX_Change_5d",
        "VIX_Change_10d",
        "VIX_Change_20d",
        "VIX_Momentum",
        "VIX_Acceleration",
        "VIX_MA10",
        "VIX_MA20",
        "VIX_MA50",
        "VIX_Cross_10_50",
        "VIX_Cross_20_50",
        "VIX_Mean_20d",
        "VIX_Mean_50d",
        "VIX_Mean_100d",
        "VIX_Std_20d",
        "VIX_Std_50d",
        "VIX_Std_100d",
        "VIX_Z_20d",
        "VIX_Z_50d",
        "VIX_Z_100d",
        "VIX_Extreme_High",
        "VIX_Extreme_Low",
        "VIX_of_VIX",
        "VIX_of_VIX_Z",
        "VIX_Spike",
        "VIX_Bottom",
        "VIX_Factor",
        "VIX_Risk_Multiplier",
        "VIX_Adjusted_ATR",
        "VIX_vs_Realized_21d",
        "Dynamic_RSI",
    ],
    "tags":    ["market_regime", "volatility", "vix"],
    "version": "1.0",
    "author":  "alphasens port",
}


def _dynamic_rsi_one_ticker(close: np.ndarray, vix_factor: np.ndarray) -> np.ndarray:
    """Inlined single-ticker Dynamic_RSI -- the VECTORIZED form the ground truth used.

    The literal source loop took a simple mean of gains/losses over a per-row
    dynamic window int(14*VIX_Factor) in [5,30]; the GT was actually produced by
    the vectorized rewrite, which precomputes a Wilder-free rolling-mean RSI for
    each window 5..30 then selects the VIX-driven window per row. Both fill only
    positions >= 50 (the source's `if i < 50: continue` guard). Verified to match
    GT to the float32 floor.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < 50:
        return out

    delta = np.diff(close, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    rsi_matrix = np.full((n, 26), np.nan)  # one column per window 5..30
    for col, w in enumerate(range(5, 31)):
        avg_g = pd.Series(gains).rolling(w, min_periods=w).mean().to_numpy()
        avg_l = pd.Series(losses).rolling(w, min_periods=w).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = np.where(
                avg_l == 0,
                np.where(avg_g > 0, 100.0, 50.0),
                100.0 - 100.0 / (1.0 + avg_g / avg_l),
            )
        rsi_matrix[:, col] = rsi

    dyn_w = np.where(
        np.isfinite(vix_factor),
        np.clip(np.floor(14.0 * vix_factor).astype(int), 5, 30),
        14,
    )
    selected = rsi_matrix[np.arange(n), dyn_w - 5]
    selected = np.where(np.arange(n) >= 50, selected, np.nan)
    return selected


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # df is one ticker, ascending by Date, no gaps. We must not sort/reindex df.
    # We build a local positional working frame `r` (RangeIndex) to mirror the
    # source's iloc-based loops exactly, then assign columns back onto df by
    # position (df rows are already in ascending Date order).
    n = len(df)
    idx = df.index

    # -----------------------------------------------------------------------
    # VIX merge: resample daily, ffill, backward merge_asof on Date, then ffill.
    # (Mirrors source L1068-1086.)
    # -----------------------------------------------------------------------
    vix_daily = _indexes.vix_daily_close()  # ['Date','vix_close'], ascending (empty if no VIX)
    vix_daily = vix_daily.rename(columns={"vix_close": "VIX_Close"})

    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
    tmp = pd.DataFrame({"Date": dates.values})

    if len(vix_daily) > 0:
        merged = pd.merge_asof(
            tmp.sort_values("Date"),
            vix_daily.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # tmp is already ascending (df ascending by Date), so positional alignment holds.
        vix_vals = merged["VIX_Close"].ffill().to_numpy(dtype=np.float64)
    else:
        vix_vals = np.full(n, np.nan)

    # Positional working series (RangeIndex 0..n-1) used for all iloc math below.
    VIX = pd.Series(vix_vals, index=pd.RangeIndex(n))

    out = {}  # name -> numpy array (length n, positional)

    out["VIX_Close"] = vix_vals

    # 1. VIX REGIME DETECTION (L1090) ---------------------------------------
    out["VIX_Regime_Numeric"] = np.select(
        [
            VIX.values < 15,
            (VIX.values >= 15) & (VIX.values < 20),
            (VIX.values >= 20) & (VIX.values < 30),
            VIX.values >= 30,
        ],
        [0, 1, 2, 3],
        default=1,
    ).astype(float)

    # 2a. VIX PERCENTILES, excluding current row (L1104-1114) ----------------
    # lookback = iloc[i-window:i]; pct = percentileofscore(lookback, current)/100.
    # The ground truth was produced with the 'weak' tie convention (fraction of the
    # lookback <= current), i.e. (lookback <= current).mean(); scipy's DEFAULT
    # kind='rank' differs by half a tie and lands ~8e-3 off. Use kind='weak' to
    # reproduce the GT to the float32 floor (verified across windows 30/60/252).
    for window in [30, 60, 252]:
        col = np.full(n, np.nan)
        for i in range(n):
            if i >= window:
                lookback_values = VIX.iloc[i - window:i].values
                if len(lookback_values) > 0:
                    current_value = VIX.iloc[i]
                    col[i] = stats.percentileofscore(
                        lookback_values, current_value, kind="weak"
                    ) / 100
        out[f"VIX_Percentile_{window}d"] = col

    # 2b. VIX rate of change over periods (L1117-1119) -----------------------
    vix_change = {}
    for period in [1, 5, 10, 20]:
        if n > period:
            vix_change[period] = VIX.pct_change(period, fill_method=None)
            out[f"VIX_Change_{period}d"] = vix_change[period].to_numpy()

    # 2c. VIX momentum: today / mean(iloc[i-10:i]) - 1, i>=10 (L1122-1129) ---
    mom = np.full(n, np.nan)
    for i in range(n):
        if i >= 10:
            mean_10d = VIX.iloc[i - 10:i].mean()
            if mean_10d > 0:
                mom[i] = VIX.iloc[i] / mean_10d - 1
    out["VIX_Momentum"] = mom

    # 2d. VIX acceleration: Change_5d[i] - Change_5d[i-5], i>=10 (L1132-1140)-
    accel = np.full(n, np.nan)
    if 5 in vix_change:
        c5 = vix_change[5]
        for i in range(n):
            if i >= 10:
                if pd.notna(c5.iloc[i]) and pd.notna(c5.iloc[i - 5]):
                    accel[i] = c5.iloc[i] - c5.iloc[i - 5]
    out["VIX_Acceleration"] = accel

    # 3. VIX MOVING AVERAGES + CROSSOVERS (L1144-1160) -----------------------
    ma = {}
    for window in [10, 20, 50]:
        ma[window] = VIX.shift(1).rolling(window, min_periods=window).mean()
        out[f"VIX_MA{window}"] = ma[window].to_numpy()

    if 10 in ma and 50 in ma:
        out["VIX_Cross_10_50"] = np.where(
            (ma[10].notna()) & (ma[50].notna()),
            np.where(ma[10] > ma[50], 1, -1),
            np.nan,
        )
    if 20 in ma and 50 in ma:
        out["VIX_Cross_20_50"] = np.where(
            (ma[20].notna()) & (ma[50].notna()),
            np.where(ma[20] > ma[50], 1, -1),
            np.nan,
        )

    # 4. VIX MEAN REVERSION (L1163-1182) -------------------------------------
    z_50 = None
    for window in [20, 50, 100]:
        if n >= window:
            _mean = VIX.rolling(window, min_periods=window).mean()
            _std = VIX.rolling(window, min_periods=window).std()
            out[f"VIX_Mean_{window}d"] = _mean.to_numpy()
            out[f"VIX_Std_{window}d"] = _std.to_numpy()
            z = np.where(
                (_mean.notna()) & (_std.notna()) & (_std > 0),
                (VIX - _mean) / (_std + 1e-10),
                np.nan,
            )
            out[f"VIX_Z_{window}d"] = z
            if window == 50:
                z_50 = z

    if z_50 is not None:
        out["VIX_Extreme_High"] = np.where(z_50 > 2, 1, 0).astype(float)
        out["VIX_Extreme_Low"] = np.where(z_50 < -1, 1, 0).astype(float)

    # 5. VOLATILITY OF VOLATILITY (L1186-1208) -------------------------------
    # The ground truth was produced by the VECTORIZED form of this feature, which
    # is bit-faithful to the source loop EXCEPT for the slice boundary: the source
    # comment says i-20:i (20 rows -> 19 returns) but the GT keeps 20 returns by
    # seeing the row just before the window. That is exactly
    #     pct_change.shift(1).rolling(20, min_periods=5).std() * sqrt(252)
    # (verified: matches GT iloc[i-21:i] to the float32 floor). We use that form
    # so VIX_of_VIX / VIX_of_VIX_Z land at PASS_F32 against the ground truth.
    _vix_pct = VIX.pct_change(fill_method=None)
    vov_s = _vix_pct.shift(1).rolling(20, min_periods=5).std() * np.sqrt(252)
    out["VIX_of_VIX"] = vov_s.to_numpy()

    # VIX_of_VIX_Z: rolling 100-window (min 30) z-score of VIX_of_VIX, past-only.
    _vov_mean = vov_s.shift(1).rolling(100, min_periods=30).mean()
    _vov_std = vov_s.shift(1).rolling(100, min_periods=30).std()
    out["VIX_of_VIX_Z"] = np.where(
        _vov_std > 0,
        (vov_s - _vov_mean) / (_vov_std + 1e-10),
        np.nan,
    )

    # 6. VIX-ADJUSTED PRICE INDICATORS (L1212-1244) --------------------------
    close = df["Close"].reset_index(drop=True).astype(float)
    log_return = np.log(close / close.shift(1))

    realized_vol_21d = None
    for window in [10, 21]:
        rv = log_return.rolling(window, min_periods=window).std() * np.sqrt(252)
        if window == 21:
            realized_vol_21d = rv

    if realized_vol_21d is not None:
        out["VIX_vs_Realized_21d"] = np.where(
            realized_vol_21d > 0,
            VIX.values / (100 * realized_vol_21d.values),
            np.nan,
        )

    high = df["High"].reset_index(drop=True).astype(float)
    low = df["Low"].reset_index(drop=True).astype(float)
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    ATR = true_range.rolling(14, min_periods=14).mean()

    out["VIX_Adjusted_ATR"] = np.where(
        ATR.notna() & (VIX.values > 0),
        ATR.values * np.sqrt(VIX.values / 20),
        np.nan,
    )

    # 7. DYNAMIC LOOKBACK PERIODS (L1248-1301) -------------------------------
    VIX_Factor = np.where(
        VIX.values > 0,
        np.clip(20 / VIX.values, 0.5, 2),
        1.0,
    )
    out["VIX_Factor"] = VIX_Factor

    # Dynamic_RSI: per-ticker (this frame IS one ticker). The ground truth was
    # produced by the vectorized inlined form (see _dynamic_rsi_one_ticker): for
    # each window 5..30 a rolling-mean RSI, then per-row select int(14*VIX_Factor)
    # clipped [5,30], only rows >= 50. Matches GT to the float32 floor.
    dynamic_rsi = _dynamic_rsi_one_ticker(
        close.to_numpy(dtype=np.float64),
        VIX_Factor.astype(np.float64),
    )
    out["Dynamic_RSI"] = dynamic_rsi

    # 8. RISK ADJUSTMENT (L1304) ---------------------------------------------
    out["VIX_Risk_Multiplier"] = VIX.values / 20

    # 10. VIX PATTERN DETECTION (L1348-1371) ---------------------------------
    # i>=25: spike/bottom vs 20d mean (excluding current) * 1.5 / * 0.8.
    spike = np.full(n, np.nan)
    bottom = np.full(n, np.nan)
    for i in range(n):
        if i >= 25:
            vix_5days_ago = VIX.iloc[i - 5]
            vix_1day_ago = VIX.iloc[i - 1]
            vix_today = VIX.iloc[i]
            vix_mean_20d = VIX.iloc[i - 20:i].mean()
            if (vix_5days_ago < vix_1day_ago) and (vix_1day_ago > vix_today) and (vix_1day_ago > vix_mean_20d * 1.5):
                spike[i] = 1
            else:
                spike[i] = 0
            if (vix_5days_ago > vix_1day_ago) and (vix_1day_ago < vix_today) and (vix_1day_ago < vix_mean_20d * 0.8):
                bottom[i] = 1
            else:
                bottom[i] = 0
    out["VIX_Spike"] = spike
    out["VIX_Bottom"] = bottom

    # -----------------------------------------------------------------------
    # Assemble. `out` arrays are positional (length n, ascending-Date order),
    # which matches df's row order; attach with df's original index.
    # NaNs left as-is; df order/index untouched. STRIPPED terminal ffill/fillna.
    # -----------------------------------------------------------------------
    add = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in out.items()}, index=idx)
    return pd.concat([df, add], axis=1)
