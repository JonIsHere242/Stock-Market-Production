"""
_paper_hal_2573075_vol_adjusted_trend.py
=========================================
Volatility-adjusted multi-horizon trend-following (CTA/TSMOM) signal pack.

Inspired by:
  "Trends everywhere? The case of hedge fund styles"
  Joenvaara, Karppinen, Lundqvist, Vansteenkiste (HAL id: hal-02573075)

DESIGN INTENT
-------------
This block implements the CANONICAL time-series momentum / CTA signal:
scale each stock's trailing return by its own realised volatility over the
SAME horizon, then combine across horizons.  It answers "how strongly and
consistently is this stock TRENDING *relative to how volatile it is*?" — which
is exactly the sizing rule used by systematic trend-following funds.

HOW IT DIFFERS FROM momentum_score.py
  momentum_score_63d is a raw-magnitude composite of log-returns z-scored over
  a fixed 63-day window.  It has NO volatility normalisation and does NOT produce
  per-horizon signals or inverse-vol leverage estimates.  This block's signals
  are dimensionless (ret / vol) and directly interpretable as Sharpe-per-horizon.

HOW IT DIFFERS FROM trend_revert_state.py (tvr_*)
  tvr_* is a REGIME DIAGNOSTIC: it measures whether returns exhibit statistical
  persistence (variance-ratio, Hurst, autocorrelation).  It does NOT produce a
  directional trade signal or a vol-target leverage factor.  This block produces
  actionable, signed CTA signals and an explicit inverse-vol leverage multiplier.

Columns produced (all prefixed `vtr_`):
  vtr_ret21, vtr_ret63, vtr_ret126, vtr_ret252
      Trailing log-returns over 21 / 63 / 126 / 252 calendar days (approx
      1 / 3 / 6 / 12 months).  These are raw inputs kept for diagnostics.

  vtr_vol21, vtr_vol63, vtr_vol126, vtr_vol252
      Realised annualised volatility (rolling std of 1d log-returns × √252) over
      each horizon, computed on the SAME window — so the vol estimate and the
      return estimate share the same look-back.

  vtr_ts21, vtr_ts63, vtr_ts126, vtr_ts252
      Vol-scaled TSMOM signal per horizon = trailing_log_return / (realized_vol × √h/252).
      Values are dimensionless (units: "Sharpe-equivalent over horizon h").
      These are the canonical CTA entry signals from the paper.

  vtr_combined
      Equal-weight average of the four per-horizon vol-scaled signals.  Positive
      ⇒ multi-horizon up-trend; negative ⇒ multi-horizon down-trend.

  vtr_horizon_agree
      Fraction of the four horizons whose vtr_ts* signal is positive, mapped to
      [-1, +1]:  +1 = all four agree up; -1 = all four agree down; 0 = split.
      (Equivalent to mean of sign(vtr_ts*) across horizons.)

  vtr_cta_pos
      tanh-capped CTA position signal: tanh(vtr_combined / 2.0).  Bounded to
      (-1, +1) by construction; mirrors the standard trend-follower sizing rule
      that clips extreme signals without a hard cutoff.

  vtr_inv_vol_lev
      Inverse-vol leverage multiplier based on the 63-day realised vol:
      target_vol / realised_vol, where target_vol = 0.15 (15% annualised),
      capped to [0.1, 5.0].  This is the vol-targeting factor that scales
      position size in a live CTA strategy.

Honest note: all features are per-ticker and purely price-based.  Cross-sectional
predictive power relative to the incumbent feature set requires in-model marginal-
contribution testing and multi-seed backtests before any promotion to the live set.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "vol_adjusted_trend",
    "description": (
        "Per-ticker volatility-adjusted multi-horizon CTA/TSMOM signal pack: "
        "vol-scaled trailing returns at 21/63/126/252-day horizons, combined score, "
        "horizon agreement, tanh-capped position signal, and inverse-vol leverage — "
        "distinct from momentum_score (no vol-normalisation) and trend_revert_state "
        "(regime diagnostic, not a directional trade signal)."
    ),
    "requires": ["Close"],
    "produces": [
        # raw building blocks (diagnostic)
        "vtr_ret21", "vtr_ret63", "vtr_ret126", "vtr_ret252",
        "vtr_vol21", "vtr_vol63", "vtr_vol126", "vtr_vol252",
        # per-horizon vol-scaled TSMOM signals
        "vtr_ts21", "vtr_ts63", "vtr_ts126", "vtr_ts252",
        # composite / sizing columns
        "vtr_combined",
        "vtr_horizon_agree",
        "vtr_cta_pos",
        "vtr_inv_vol_lev",
    ],
    "tags": ["trend", "momentum", "volatility", "experimental"],
    "version": "1.0",
    "author": "paper: hal-02573075 (Trends everywhere? Hedge fund CTA styles)",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_ANN_FACTOR = 252.0          # trading days per year for annualisation
_ANN_SQRT = np.sqrt(_ANN_FACTOR)

# Horizons in *trading days*.  The paper uses 1/3/6/12-month look-backs.
# Approximated as 21 / 63 / 126 / 252 daily bars.
_HORIZONS: Tuple[int, ...] = (21, 63, 126, 252)

# Vol-target for the inverse-vol leverage column (15 % annualised)
_TARGET_VOL = 0.15

# Leverage cap / floor (avoid absurd positions in near-zero-vol periods)
_LEV_MIN = 0.1
_LEV_MAX = 5.0

# Minimum 1d-returns required inside a window to emit a value
_MIN_OBS = 10

# tanh dampening factor: tanh(combined / _TANH_SCALE)
# scale=2 ⇒ signal must be ~2× "Sharpe" to push to ±0.96; moderate.
_TANH_SCALE = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a/b, NaN where b==0 or either is non-finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((b != 0.0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan)
    return out


def _rolling_log_ret(close: np.ndarray, h: int) -> np.ndarray:
    """
    Trailing log-return over h bars ending at each row t.

    ret[t] = log(close[t] / close[t-h])   (causal: uses close[t-h..t])
    First h rows are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= h:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prev = close[:n - h]
        curr = close[h:]
        # avoid log of non-positive or NaN
        valid = (prev > 0.0) & (curr > 0.0) & np.isfinite(prev) & np.isfinite(curr)
        result = np.where(valid, np.log(curr / prev), np.nan)
        out[h:] = result
    return out


def _rolling_realised_vol(close: np.ndarray, h: int) -> np.ndarray:
    """
    Annualised realised volatility = std(1d log-returns over h bars) × √252.

    Window of h 1d-log-returns ending at t  =>  uses close[t-h..t].
    Needs at least _MIN_OBS valid 1d returns inside the window; otherwise NaN.
    First h rows are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out

    # 1d log-returns (length n, first element NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r1d = np.empty(n, dtype=np.float64)
        r1d[0] = np.nan
        pc, cc = close[:-1], close[1:]
        valid1 = (pc > 0.0) & (cc > 0.0) & np.isfinite(pc) & np.isfinite(cc)
        r1d[1:] = np.where(valid1, np.log(cc / pc), np.nan)

    # Rolling std over h 1d-returns ending at t; t-th position uses r1d[t-h+1..t]
    # We use pandas rolling for correctness (ddof=1 by default).
    s = pd.Series(r1d)
    # min_periods=_MIN_OBS ensures we only emit when enough obs exist
    rolling_std = s.rolling(h, min_periods=_MIN_OBS).std()
    out[:] = rolling_std.to_numpy(dtype=np.float64) * _ANN_SQRT
    return out


def _tsmom_signal(log_ret_h: np.ndarray, real_vol_h: np.ndarray, h: int) -> np.ndarray:
    """
    Per-horizon TSMOM signal = log_ret_h / (real_vol_h * sqrt(h / ANN_FACTOR)).

    The denominator converts annualised vol to the vol expected over h days,
    making the ratio a Sharpe-equivalent (dimensionless) for horizon h.
    inf/nan inputs produce NaN output.
    """
    # expected vol over horizon h days = annualised_vol * sqrt(h / 252)
    expected_vol = real_vol_h * np.sqrt(h / _ANN_FACTOR)
    return _safe_div(log_ret_h, expected_vol)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vol-adjusted trend-following (CTA/TSMOM) features.

    All operations are causal (row t uses only data at rows <= t).
    NaN is emitted at leading rows where insufficient history exists.
    No existing column is modified.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    idx = df.index

    # ---- 1. Trailing log-returns and realised vols per horizon ----------------
    ret_arrays: dict[int, np.ndarray] = {}
    vol_arrays: dict[int, np.ndarray] = {}

    for h in _HORIZONS:
        ret_arrays[h] = _rolling_log_ret(close, h)
        vol_arrays[h] = _rolling_realised_vol(close, h)

    # ---- 2. Per-horizon TSMOM signals ----------------------------------------
    ts_arrays: dict[int, np.ndarray] = {}
    for h in _HORIZONS:
        ts_arrays[h] = _tsmom_signal(ret_arrays[h], vol_arrays[h], h)

    # ---- 3. Combined multi-horizon score (equal-weight mean, NaN-aware) -------
    ts_stack = np.stack([ts_arrays[h] for h in _HORIZONS], axis=1)  # (n, 4)
    # nanmean: if ALL four are NaN -> NaN; if some are valid -> use valid ones
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combined = np.where(
            np.all(np.isnan(ts_stack), axis=1),
            np.nan,
            np.nanmean(ts_stack, axis=1),
        )

    # ---- 4. Horizon agreement: mean of sign(ts_h), NaN-aware -----------------
    sign_stack = np.where(
        np.isnan(ts_stack),
        np.nan,
        np.sign(ts_stack),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        horizon_agree = np.where(
            np.all(np.isnan(sign_stack), axis=1),
            np.nan,
            np.nanmean(sign_stack, axis=1),
        )
    # horizon_agree is already in [-1, +1] because sign(x) ∈ {-1, 0, +1}

    # ---- 5. tanh-capped CTA position signal ----------------------------------
    cta_pos = np.where(
        np.isnan(combined),
        np.nan,
        np.tanh(combined / _TANH_SCALE),
    )
    # tanh output is strictly in (-1, +1); no further clip needed.
    # Guard against any spurious inf that might appear in combined:
    cta_pos = np.where(np.isfinite(cta_pos) | np.isnan(cta_pos), cta_pos, np.nan)

    # ---- 6. Inverse-vol leverage factor (based on 63-day vol) ----------------
    vol63 = vol_arrays[63]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_vol_lev = np.where(
            (vol63 > 0.0) & np.isfinite(vol63),
            np.clip(_TARGET_VOL / vol63, _LEV_MIN, _LEV_MAX),
            np.nan,
        )

    # ---- 7. Assign to df (ONLY produced columns, no existing columns touched) -
    df["vtr_ret21"]  = pd.Series(ret_arrays[21],  index=idx)
    df["vtr_ret63"]  = pd.Series(ret_arrays[63],  index=idx)
    df["vtr_ret126"] = pd.Series(ret_arrays[126], index=idx)
    df["vtr_ret252"] = pd.Series(ret_arrays[252], index=idx)

    df["vtr_vol21"]  = pd.Series(vol_arrays[21],  index=idx)
    df["vtr_vol63"]  = pd.Series(vol_arrays[63],  index=idx)
    df["vtr_vol126"] = pd.Series(vol_arrays[126], index=idx)
    df["vtr_vol252"] = pd.Series(vol_arrays[252], index=idx)

    df["vtr_ts21"]  = pd.Series(ts_arrays[21],  index=idx)
    df["vtr_ts63"]  = pd.Series(ts_arrays[63],  index=idx)
    df["vtr_ts126"] = pd.Series(ts_arrays[126], index=idx)
    df["vtr_ts252"] = pd.Series(ts_arrays[252], index=idx)

    df["vtr_combined"]      = pd.Series(combined,      index=idx)
    df["vtr_horizon_agree"] = pd.Series(horizon_agree, index=idx)
    df["vtr_cta_pos"]       = pd.Series(cta_pos,       index=idx)
    df["vtr_inv_vol_lev"]   = pd.Series(inv_vol_lev,   index=idx)

    return df
