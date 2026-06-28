"""
mst_range_liquidity.py — Range-based volatility & volume-stability liquidity proxies.

Daily-OHLCV liquidity/microstructure signals built from the high-low range and
from the STABILITY of trading activity (not its level), chosen to be orthogonal
to plain close-to-close momentum/volatility and to the dollar-volume block:

  1. Parkinson (1980) range volatility = sqrt( mean[ ln(H/L)^2 ] / (4 ln2) ),
     a high-low range estimator of volatility (20d).
  2. Garman-Klass (1980) volatility = sqrt of a weighted blend of squared
     log-range and squared log open-close move (20d) — uses OHLC efficiently.
  3. Parkinson / Garman-Klass ratio: the two estimators differ mainly through
     the open-close (overnight + drift) term, so their ratio isolates how much
     variance comes from the close-to-open jump vs the intraday range — a
     gappiness / overnight-risk microstructure tell.
  4. Relative-range trend: 5d-vs-20d ratio of (H-L)/Close — is the bar widening
     (deteriorating liquidity / rising uncertainty) or tightening?
  5. Volume coefficient of variation (20/60d): std/mean of raw volume — a
     trading-activity STABILITY measure independent of its average level.
  6. Turnover stability: 20d mean / 60d mean of the volume CV's inverse, i.e.
     how steady recent participation is vs the medium term (autocorrelation of
     log-volume as a persistence/stability proxy).

All ratios guarded against divide-by-zero and clipped to sane bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "mst_range_liquidity",
    "description": (
        "Range-based volatility-implied liquidity and volume-stability proxies: "
        "Parkinson & Garman-Klass range vol (20d), their gappiness ratio, "
        "relative-range 5d/20d trend, volume coefficient of variation (20/60d), "
        "and log-volume autocorrelation turnover stability (20d)."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces":    [
        "mst_parkinson_vol_20d",
        "mst_garman_klass_vol_20d",
        "mst_pk_gk_ratio_20d",
        "mst_rel_range_trend_5_20",
        "mst_volume_cv_20d",
        "mst_volume_cv_60d",
        "mst_turnover_stability_20d",
    ],
    "tags":        ["liquidity", "volatility", "volume", "microstructure"],
    "version":     "1.0",
    "author":      "feature-gen",
}

_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Positive-price guards for logs.
    high_s = high.where(high > 0)
    low_s = low.where(low > 0)
    open_s = open_.where(open_ > 0)
    close_s = close.where(close > 0)

    new_cols = {}

    # ------------------------------------------------------------------
    # 1. Parkinson range volatility (20d)
    #    sigma_P = sqrt( (1 / (4 ln2)) * mean[ ln(H/L)^2 ] )
    # ------------------------------------------------------------------
    log_hl_sq = np.log(high_s / low_s) ** 2
    pk_var = log_hl_sq.rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))
    pk_vol = np.sqrt(pk_var.clip(lower=0.0))
    pk_vol = pk_vol.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=5.0)
    new_cols["mst_parkinson_vol_20d"] = pk_vol

    # ------------------------------------------------------------------
    # 2. Garman-Klass volatility (20d)
    #    term = 0.5*ln(H/L)^2 - (2 ln2 - 1)*ln(C/O)^2
    #    sigma_GK = sqrt( mean[ term ] )
    # ------------------------------------------------------------------
    log_co_sq = np.log(close_s / open_s) ** 2
    gk_term = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    gk_var = gk_term.rolling(20, min_periods=10).mean()
    gk_vol = np.sqrt(gk_var.clip(lower=0.0))
    gk_vol = gk_vol.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=5.0)
    new_cols["mst_garman_klass_vol_20d"] = gk_vol

    # ------------------------------------------------------------------
    # 3. Parkinson / Garman-Klass ratio — isolates the open-close (gap+drift)
    #    contribution.  ~1 when intraday range dominates; deviates when the
    #    close-to-open jump carries variance.
    # ------------------------------------------------------------------
    pk_gk = pk_vol / (gk_vol + _EPS)
    pk_gk = pk_gk.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=10.0)
    new_cols["mst_pk_gk_ratio_20d"] = pk_gk

    # ------------------------------------------------------------------
    # 4. Relative-range trend: short (5d) vs medium (20d) average of the
    #    relative bar width (H-L)/Close.  >1 => widening bars.
    # ------------------------------------------------------------------
    rel_range = (high_s - low_s) / (close_s + _EPS)
    rr_5 = rel_range.rolling(5, min_periods=3).mean()
    rr_20 = rel_range.rolling(20, min_periods=10).mean()
    rr_trend = rr_5 / (rr_20 + _EPS)
    rr_trend = rr_trend.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=10.0)
    new_cols["mst_rel_range_trend_5_20"] = rr_trend

    # ------------------------------------------------------------------
    # 5. Volume coefficient of variation (20/60d): std/mean of raw volume.
    #    Measures how ERRATIC participation is, independent of its level.
    # ------------------------------------------------------------------
    vol_pos = volume.clip(lower=0.0)
    for w in (20, 60):
        v_win = vol_pos.rolling(w, min_periods=w // 2)
        v_mean = v_win.mean()
        v_std = v_win.std()
        cv = v_std / (v_mean + _EPS)
        cv = cv.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=20.0)
        new_cols[f"mst_volume_cv_{w}d"] = cv

    # ------------------------------------------------------------------
    # 6. Turnover stability (20d): lag-1 autocorrelation of log-volume.
    #    High autocorrelation => steady, persistent participation (stable
    #    turnover); low/negative => choppy, unstable trading.  Distinct from
    #    the CV above (dispersion) — this measures temporal persistence.
    # ------------------------------------------------------------------
    log_vol = np.log(vol_pos + 1.0)
    log_vol_lag = log_vol.shift(1)
    ac = log_vol.rolling(20, min_periods=10).corr(log_vol_lag)
    ac = ac.replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0)
    new_cols["mst_turnover_stability_20d"] = ac

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
