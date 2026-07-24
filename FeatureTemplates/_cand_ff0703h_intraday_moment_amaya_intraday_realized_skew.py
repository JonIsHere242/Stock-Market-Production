"""
_cand_ff0703h_intraday_moment_amaya_intraday_realized_skew.py

VEIN: intraday_moment
SPEC: ff0703h_intraday_moment_amaya_intraday_realized_skew

Amaya-Christoffersen-style realized skewness, adapted per-ticker from daily
open-to-close log returns (a faithful proxy: the original estimator uses
intraday/high-frequency returns within a single day; we do not have
intraday bars in this OHLCV-only pipeline, so we treat each day's
open-to-close log return as one "intraday" return observation and roll
the skewness estimator across a window of N trading days instead of across
intraday bars within one day). This preserves the economic signal (a
rolling measure of return-distribution asymmetry derived from open-close
moves) in a causal, per-ticker, OHLCV-only way.

RS_N = sqrt(N) * sum(r^3) / (sum(r^2))^1.5   over the trailing N days,
where r = log(Close/Open) for each day.

Columns produced:
  ff0703h_intraday_moment_amaya_intraday_realized_skew_rs40   -- level, N=40 (slow)
  ff0703h_intraday_moment_amaya_intraday_realized_skew_rs20   -- level, N=20 (fast)
  ff0703h_intraday_moment_amaya_intraday_realized_skew_dyn    -- RS40 minus RS40 20 rows earlier
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703h_intraday_moment_amaya_intraday_realized_skew",
    "description": (
        "Amaya-Christoffersen realized skewness of open-to-close log returns, rolled "
        "over trailing windows of N=40 (slow/level) and N=20 (fast) trading days: "
        "RS_N = sqrt(N) * sum(r^3) / (sum(r^2))^1.5. Faithful per-ticker proxy: the "
        "original estimator sums intraday (sub-daily) returns within one day; this "
        "OHLCV-only pipeline has no intraday bars, so each day's open-to-close log "
        "return is used as a single 'intraday' observation and the sum runs across "
        "days instead of across intraday bars within a day. Also emits a 20-day "
        "dynamic (change) of the RS40 level to capture skew regime shifts."
    ),
    "requires": ["Open", "Close"],
    "produces": [
        "ff0703h_intraday_moment_amaya_intraday_realized_skew_rs40",
        "ff0703h_intraday_moment_amaya_intraday_realized_skew_rs20",
        "ff0703h_intraday_moment_amaya_intraday_realized_skew_dyn",
    ],
    "tags": ["intraday_moment", "skewness", "amaya_christoffersen", "distribution", "candidate"],
    "version": "1.0",
    "author": (
        "feature-factory (auto-generated); proxy note: intraday realized skew "
        "estimator applied across daily open-to-close returns rolled over N days "
        "(no sub-daily bars available in OHLCV-only pipeline)."
    ),
}

_COL_RS40 = "ff0703h_intraday_moment_amaya_intraday_realized_skew_rs40"
_COL_RS20 = "ff0703h_intraday_moment_amaya_intraday_realized_skew_rs20"
_COL_DYN = "ff0703h_intraday_moment_amaya_intraday_realized_skew_dyn"


def _rolling_realized_skew(r: pd.Series, n: int) -> pd.Series:
    """Vectorised rolling Amaya-Christoffersen realized skewness over window n."""
    r2 = r * r
    r3 = r2 * r
    sum_r2 = r2.rolling(window=n, min_periods=n).sum()
    sum_r3 = r3.rolling(window=n, min_periods=n).sum()
    denom = sum_r2.pow(1.5)
    denom_safe = denom.where(denom != 0, np.nan)
    rs = np.sqrt(float(n)) * sum_r3 / denom_safe
    rs = rs.replace([np.inf, -np.inf], np.nan)
    return rs


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    df[_COL_RS40] = np.nan
    df[_COL_RS20] = np.nan
    df[_COL_DYN] = np.nan

    if n == 0:
        return df

    open_ = pd.to_numeric(df["Open"], errors="coerce")
    close_ = pd.to_numeric(df["Close"], errors="coerce")

    valid = (open_ > 0) & (close_ > 0)
    ratio = (close_ / open_.where(open_ != 0, np.nan)).where(valid, np.nan)
    r = np.log(ratio.where(ratio > 0, np.nan))

    rs40 = _rolling_realized_skew(r, 40)
    rs20 = _rolling_realized_skew(r, 20)

    df[_COL_RS40] = rs40.to_numpy()
    df[_COL_RS20] = rs20.to_numpy()

    dyn = rs40 - rs40.shift(20)
    df[_COL_DYN] = dyn.to_numpy()

    return df
