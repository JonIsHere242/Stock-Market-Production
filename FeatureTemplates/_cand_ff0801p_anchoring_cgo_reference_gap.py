"""
Capital Gains Overhang (CGO) -- price vs. the turnover-weighted reference price
==============================================================================
Spec: ff0801p_anchoring_cgo_reference_gap
Vein: anchoring
Papers: Grinblatt & Han (2005, JFE) "Prospect theory, mental accounting, and
        momentum"; An (2016, JF) "Asset pricing when traders sell extreme
        winners and losers" (V-shaped selling propensity).

Idea
----
Holders anchor on what they paid. The market's aggregate cost basis is
approximated by a reference price RP that decays old purchases at the rate
shares turn over:

    turnover_t = Volume_t / shares_outstanding_t          (clipped to [0, 1])
    RP_t       = turnover_t * Close_t + (1 - turnover_t) * RP_{t-1}

seeded with the first valid Close. The capital-gains overhang is the gap
between today's price and the cost basis the market carried into today:

    cgo_t = (Close_t - RP_{t-1}) / Close_t

RP_{t-1} (the PRIOR reference price) is used deliberately so the current bar
does not price its own basis. A positive cgo means the average holder sits on
an unrealised gain (disposition-effect selling pressure -> price under-reacts
to good news); negative means an unrealised loss.

Columns
-------
1. _cgo      -- the raw overhang level.
2. _z120     -- cgo z-scored against its own trailing 120-bar mean/std, so the
                signal is a *deviation from this name's usual overhang* rather
                than a level that mostly encodes long-run drift.
3. _vshape   -- An (2016) V-shaped selling propensity:
                max(cgo, 0) - 0.79 * min(cgo, 0). Both legs are non-negative
                contributions; the 0.79 is the paper's loss-leg slope relative
                to the gain leg. High values = extreme winners OR extreme
                losers, i.e. names with the most latent selling pressure.

Honesty notes / proxies
-----------------------
* shares_outstanding comes from the point-in-time SEC helper (backward
  merge_asof on filed_date). Coverage is ~84% of the universe; ETFs and
  foreign issuers have none.
* Where shares are missing or non-positive on a given bar, turnover falls back
  to a purely price-panel proxy: 0.01 * Volume_t / rolling-250d-mean(Volume).
  That rolling ratio has a median near 1 by construction, so the fallback
  turnover has a median near 0.01, matching the typical daily turnover of a
  liquid US name. This is a scale assumption, not a measurement.
* True CGO in the literature uses the full daily turnover history of the free
  float; this is the standard recursive approximation, computed per ticker.
* Bars before 60 recursion steps have elapsed since the seed are NaN (burn-in),
  as are bars with a non-positive or missing Close.
* cgo is clipped to [-3, 1] purely to keep pathological penny-stock reversals
  from dominating; the upper bound of 1 is structural (RP > 0).

Leakage
-------
The recursion is strictly forward (RP_t depends only on bars <= t), the rolling
statistics are trailing, and the fundamentals join is backward-only, so
truncating future bars cannot change any past value.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by file path to satisfy sandbox rules)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)


_PREFIX = "ff0801p_anchoring_cgo_reference_gap"
_COL_CGO = f"{_PREFIX}_cgo"
_COL_Z = f"{_PREFIX}_z120"
_COL_V = f"{_PREFIX}_vshape"

_BURN_IN = 60          # bars of recursion before anything is emitted
_Z_WINDOW = 120        # trailing window for the z-score
_Z_MINP = 60
_VOL_WINDOW = 250      # rolling mean volume for the turnover fallback
_VOL_MINP = 20
_FALLBACK_MEDIAN_TURNOVER = 0.01
_LOSS_LEG_SLOPE = 0.79  # An (2016) loss-leg slope relative to the gain leg
_CGO_FLOOR = -3.0
_CGO_CAP = 1.0


METADATA = {
    "name": "ff0801p_anchoring_cgo_reference_gap",
    "description": (
        "Capital gains overhang (Grinblatt & Han 2005): gap between price and the "
        "turnover-weighted reference price RP_t = turnover_t*Close_t + (1-turnover_t)*RP_{t-1}, "
        "with turnover = Volume / point-in-time shares_outstanding clipped to [0,1] "
        "(fallback 0.01 * Volume / rolling-250d-mean(Volume) when SEC shares are missing). "
        "Main column cgo = (Close_t - RP_{t-1}) / Close_t uses the PRIOR reference price so the "
        "current bar does not price its own basis. Also emits cgo z-scored on its own trailing "
        "120-bar mean/std, and the An (2016) V-shaped selling-propensity combination "
        "max(cgo,0) - 0.79*min(cgo,0). Per-ticker recursive approximation of a quantity the "
        "literature builds from full float turnover history; shares coverage ~84% "
        "(ETFs/foreign fall back to the volume proxy). 60-bar burn-in, leading rows NaN."
    ),
    "requires": ["Close", "Volume"],
    "produces": [_COL_CGO, _COL_Z, _COL_V],
    "tags": [
        "anchoring",
        "behavioral",
        "disposition-effect",
        "capital-gains-overhang",
        "prospect-theory",
        "turnover",
        "pit",
    ],
    "version": "1.0.0",
    "author": (
        "Spec ff0801p_anchoring_cgo_reference_gap. Faithful per-ticker implementation of the "
        "Grinblatt & Han (2005) recursive reference-price approximation plus An (2016) V-shape. "
        "Honest proxy caveats: shares_outstanding is point-in-time SEC (backward merge_asof, "
        "~84% coverage) and, when absent, turnover is a volume-normalised stand-in scaled to a "
        "0.01 median rather than a measured float turnover."
    ),
}


def _reference_price(close: np.ndarray, turnover: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Forward recursion RP_t = to_t*Close_t + (1-to_t)*RP_{t-1}, seeded at the first
    strictly-positive finite Close. Bars with unusable turnover or price simply carry
    the previous reference price forward. Returns (rp, seed_index); seed_index is -1
    when the series never seeds. O(n) and strictly causal.
    """
    n = close.shape[0]
    rp = np.full(n, np.nan, dtype="float64")
    prev = np.nan
    seed_idx = -1

    for i in range(n):
        c = close[i]
        if seed_idx < 0:
            if np.isfinite(c) and c > 0.0:
                prev = c
                seed_idx = i
                rp[i] = prev
            continue

        t = turnover[i]
        if (not np.isfinite(t)) or (not np.isfinite(c)) or c <= 0.0:
            # no usable trade information this bar -> basis is unchanged
            rp[i] = prev
            continue

        prev = t * c + (1.0 - t) * prev
        rp[i] = prev

    return rp, seed_idx


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add the CGO level, its trailing z-score, and the An (2016) V-shape column."""

    # Every produced column exists on every path.
    df[_COL_CGO] = np.nan
    df[_COL_Z] = np.nan
    df[_COL_V] = np.nan

    n = len(df)
    if n == 0 or "Close" not in df.columns or "Volume" not in df.columns:
        return df

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype="float64")
    volume = pd.to_numeric(df["Volume"], errors="coerce").to_numpy(dtype="float64")

    # ---- turnover -------------------------------------------------------
    # Primary: Volume / point-in-time shares outstanding.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["shares_outstanding"])

    shares = pd.to_numeric(
        df.get("fund_shares_outstanding", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).to_numpy(dtype="float64")
    shares_ok = np.isfinite(shares) & (shares > 0.0)
    shares_safe = np.where(shares_ok, shares, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = np.where(shares_ok, volume / shares_safe, np.nan)

    # Fallback: volume normalised by its own trailing mean, scaled to a ~0.01 median.
    vol_s = pd.Series(volume, index=df.index)
    vol_mean = (
        vol_s.rolling(_VOL_WINDOW, min_periods=_VOL_MINP).mean().to_numpy(dtype="float64")
    )
    vol_mean_safe = np.where(np.isfinite(vol_mean) & (vol_mean > 0.0), vol_mean, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover_fb = _FALLBACK_MEDIAN_TURNOVER * (volume / vol_mean_safe)

    turnover = np.where(np.isfinite(turnover), turnover, turnover_fb)
    turnover = np.where(np.isfinite(turnover), turnover, np.nan)
    turnover = np.clip(turnover, 0.0, 1.0)

    # ---- recursive reference price -------------------------------------
    rp, seed_idx = _reference_price(close, turnover)

    # scratch fundamentals column is not part of produces
    df = df.drop(columns=[c for c in ["fund_shares_outstanding"] if c in df.columns])

    if seed_idx < 0:
        return df

    # Prior reference price: RP_{t-1}
    rp_prev = np.empty(n, dtype="float64")
    rp_prev[0] = np.nan
    if n > 1:
        rp_prev[1:] = rp[:-1]

    close_safe = np.where(np.isfinite(close) & (close > 0.0), close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cgo = (close_safe - rp_prev) / close_safe
    cgo = np.where(np.isfinite(cgo), cgo, np.nan)
    cgo = np.clip(cgo, _CGO_FLOOR, _CGO_CAP)

    # Burn-in: require _BURN_IN recursion steps after the seed bar.
    idx = np.arange(n)
    cgo = np.where(idx >= seed_idx + _BURN_IN, cgo, np.nan)

    cgo_s = pd.Series(cgo, index=df.index)

    # ---- trailing z-score ----------------------------------------------
    mu = cgo_s.rolling(_Z_WINDOW, min_periods=_Z_MINP).mean()
    sd = cgo_s.rolling(_Z_WINDOW, min_periods=_Z_MINP).std()
    sd_safe = sd.where(np.isfinite(sd) & (sd > 0.0))
    z = (cgo_s - mu) / sd_safe
    z = z.replace([np.inf, -np.inf], np.nan)

    # ---- An (2016) V-shaped selling propensity --------------------------
    gain_leg = cgo_s.clip(lower=0.0)
    loss_leg = cgo_s.clip(upper=0.0)
    v_shape = gain_leg - _LOSS_LEG_SLOPE * loss_leg
    v_shape = v_shape.replace([np.inf, -np.inf], np.nan)

    df[_COL_CGO] = cgo_s.to_numpy(dtype="float64")
    df[_COL_Z] = z.to_numpy(dtype="float64")
    df[_COL_V] = v_shape.to_numpy(dtype="float64")

    return df
