"""
osap_zerotrade1m — Days with zero trades (Liu 2006 liquidity proxy)

Per Chen-Zimmermann OpenSourceAP. Cross-sectional in origin; implemented here
as a per-ticker rolling-month time series.

Formula (1-month window ending at each bar):
  zerotrade = (zero_days + (sum_monthly_turnover / 4_800_000)) * 21 / trading_days

where:
  zero_days          = count of days in the month where Volume == 0
  sum_monthly_turnover = sum(daily_volume / shares_outstanding) over the month
  trading_days       = number of trading days (bars) in the month window

Per-ticker proxy note: shares_outstanding from PIT fundamentals (filed_date);
missing coverage (ETFs, foreign) degrades gracefully to NaN in the turnover term.
The zero-day count remains valid even without fundamentals data.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (optional — degrades if unavailable)
# ---------------------------------------------------------------------------
try:
    _s2 = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _fundamentals = None  # type: ignore
    _HAS_FUND = False

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_zerotrade1m",
    "description": (
        "Liu (2006) zero-trade liquidity measure. In a rolling ~21-day window "
        "counts the number of days with zero volume, adds the scaled inverse "
        "of total monthly turnover (vol/shrout / 4.8e6), and normalises by "
        "trading days. Higher values = less liquid. Per-ticker proxy of the "
        "cross-sectional measure from Chen-Zimmermann OpenSourceAP. "
        "shares_outstanding from PIT SEC fundamentals (degrades to NaN for "
        "ETFs/foreign). Also produces the raw zero-day count and a 3-month "
        "z-score of the composite (trend / mean-reversion signal)."
    ),
    "requires": ["Volume", "Close"],
    "produces": [
        "osap_zerotrade1m_score",    # composite Liu measure (higher = illiquid)
        "osap_zerotrade1m_zdays",    # raw 21-day zero-volume day count
        "osap_zerotrade1m_zscore",   # 63-day rolling z-score of the composite
    ],
    "tags": ["liquidity", "volume", "zero-trade", "liu2006", "openSourceAP"],
    "version": "1.0",
    "author": "Liu 2006; OpenSourceAP (Chen-Zimmermann). Block by claude-sonnet-4-6.",
}

# rolling window (trading days in 1 month)
_WIN = 21
# z-score window (~3 months)
_ZWIN = 63
# Liu 2006 scaling constant
_SCALE = 48 * 1e5  # 4,800,000


def compute(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["Volume"].to_numpy(dtype=np.float64)
    n = len(vol)

    # ------------------------------------------------------------------
    # PIT shares outstanding (fund_shares_outstanding)
    # ------------------------------------------------------------------
    shrout = np.full(n, np.nan)
    if _HAS_FUND and _fundamentals is not None:
        try:
            df_aug = _fundamentals.as_of(df, fields=["shares_outstanding"])
            so_col = "fund_shares_outstanding"
            if so_col in df_aug.columns:
                raw = df_aug[so_col].to_numpy(dtype=np.float64)
                # Replace zero/negative with NaN to avoid bad divisions
                raw = np.where(raw > 0, raw, np.nan)
                shrout = raw
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Daily turnover = volume / shares_outstanding (NaN if shrout missing)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_turnover = np.where(shrout > 0, vol / shrout, np.nan)

    # zero-trade indicator (1 if no trades that day)
    is_zero = (vol == 0).astype(np.float64)

    # ------------------------------------------------------------------
    # Rolling _WIN-day statistics using stride tricks (fast, no O(n^2) loops)
    # ------------------------------------------------------------------
    from numpy.lib.stride_tricks import sliding_window_view

    if n < _WIN:
        # Not enough data — return NaN columns
        df["osap_zerotrade1m_score"] = np.nan
        df["osap_zerotrade1m_zdays"] = np.nan
        df["osap_zerotrade1m_zscore"] = np.nan
        return df

    # sliding_window_view shape: (n - WIN + 1, WIN)
    win_zeros = sliding_window_view(is_zero, _WIN)
    win_turn = sliding_window_view(daily_turnover, _WIN)

    # count of zero-trade days in the window
    zero_count = win_zeros.sum(axis=1)  # shape (n - WIN + 1,)

    # sum of turnover over the window (nansum: missing shrout -> skip those days)
    sum_turnover = np.nansum(win_turn, axis=1)  # shape same
    # number of actual trading-day bars in window
    trading_days = np.float64(_WIN)

    # Liu (2006) composite
    with np.errstate(divide="ignore", invalid="ignore"):
        composite = (zero_count + (sum_turnover / _SCALE)) * (21.0 / trading_days)

    # Pad leading NaNs so output aligns with df index
    pad = n - len(composite)
    score_arr = np.concatenate([np.full(pad, np.nan), composite])
    zdays_arr = np.concatenate([np.full(pad, np.nan), zero_count])

    # ------------------------------------------------------------------
    # 63-day rolling z-score of the composite (trend detector)
    # ------------------------------------------------------------------
    if n < _ZWIN:
        zscore_arr = np.full(n, np.nan)
    else:
        win_score = sliding_window_view(score_arr[pad:], _ZWIN - pad)
        # shape: (n - WIN + 1 - (ZWIN - WIN), ZWIN - WIN) -- recompute cleanly
        # Simpler: work on score_arr directly with pandas rolling
        s_series = pd.Series(score_arr)
        roll_mean = s_series.rolling(_ZWIN, min_periods=_ZWIN).mean()
        roll_std = s_series.rolling(_ZWIN, min_periods=_ZWIN).std(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore_arr = np.where(
                roll_std.to_numpy() > 0,
                (score_arr - roll_mean.to_numpy()) / roll_std.to_numpy(),
                np.nan,
            )

    df["osap_zerotrade1m_score"] = score_arr
    df["osap_zerotrade1m_zdays"] = zdays_arr
    df["osap_zerotrade1m_zscore"] = zscore_arr

    return df
