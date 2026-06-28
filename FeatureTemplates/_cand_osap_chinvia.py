"""
osap_chinvia — Change in capital investment (industry-adjusted proxy)
Spec: OpenSourceAP (Chen-Zimmermann) / Abarbanell & Bushee 1998
Predicted sign: -1 (high capex growth → lower future returns)

Industry-adjustment requires cross-sectional data; this block delivers
the per-ticker capex-growth signal only. Industry-adjustment must be
applied downstream when panel data is available.

capex source priority:
  1. fund_capex_ttm  (TTM capex from _fundamentals)
  2. fund_ppe_net    (net PP&E change as proxy when capex missing)

Growth definition (Abarbanell & Bushee):
  capex_growth = capex_t / (0.5 * (capex_{t-1} + capex_{t-2})) - 1
  Falls back to capex_t / capex_{t-1} - 1 when t-2 is missing.
  Both capex and denominator must be positive; else NaN.

Columns produced:
  osap_chinvia_raw    — raw per-ticker capex growth (unadjusted, annual frequency interpolated to daily)
  osap_chinvia_chg    — first difference of raw (acceleration / deceleration)
  osap_chinvia_sign   — sign-encoded signal: -1 * raw (so high = good, consistent with predicted sign -1)
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ── PIT fundamentals helper ────────────────────────────────────────────────
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_chinvia",
    "description": (
        "Change in capital investment (industry-adjusted proxy). "
        "Per Abarbanell & Bushee (1998): capex growth vs prior 2-year average. "
        "Uses TTM capex from PIT fundamentals; falls back to net PP&E change. "
        "Industry adjustment is NOT applied here (cross-sectional; do downstream). "
        "Predicted sign: -1 (high capex growth predicts lower future returns)."
    ),
    "requires": [],
    "produces": [
        "osap_chinvia_raw",
        "osap_chinvia_chg",
        "osap_chinvia_sign",
    ],
    "tags": ["investment", "fundamentals", "capex", "accounting", "osap"],
    "version": "1.0",
    "author": "Abarbanell and Bushee 1998; OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── 1. Pull PIT fundamentals ───────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["capex_ttm", "ppe_net"])

    # ── 2. Choose best capex proxy ─────────────────────────────────────────
    # capex_ttm is negative in accounting convention (cash outflow); take abs.
    capex_raw = df["fund_capex_ttm"].abs()

    # Where capex_ttm is missing/zero, use year-over-year change in net PP&E.
    # Change in PPE approximates capex (capex ≈ ΔPPE + depreciation; depreciation
    # not available, so this is a noisier proxy).
    ppe_chg = df["fund_ppe_net"].diff()
    # Only use ppe_chg as fallback when capex_ttm is missing and ppe_chg > 0
    capex_series = capex_raw.where(capex_raw.notna() & (capex_raw > 0))
    capex_series = capex_series.where(
        capex_series.notna(),
        other=ppe_chg.where(ppe_chg > 0),
    )

    # ── 3. Capex growth per Abarbanell & Bushee ───────────────────────────
    # Fundamentals are reported quarterly/annually; values repeat across days
    # until a new filing arrives. We detect unique (non-NaN) capex readings
    # by noticing when the value changes (i.e., a new filing landed).
    # We build a "period index" so we can reference t-1 and t-2 periods.

    # Mark rows where capex value changed (new filing) or is first non-null.
    capex_filled = capex_series.copy()
    # Forward-fill for continuity within periods (already done by as_of, but
    # make explicit to be safe).
    capex_ffilled = capex_filled.ffill()

    # Detect period boundaries: value changes from prior row (or first non-null).
    changed = capex_ffilled.ne(capex_ffilled.shift(1)) & capex_ffilled.notna()
    # Assign a monotonically increasing period counter
    period_idx = changed.cumsum()  # 0 before first value, then increments

    # For each row, we want:
    #   capex at current period (cap_t)
    #   capex at prior period   (cap_t1)
    #   capex two periods back  (cap_t2)
    # Strategy: at each period boundary, grab the period's capex value.
    # Then look up t-1 and t-2 using the period index.

    # Build a mapping: period_counter -> capex value at that period's start.
    # Use the first row of each period.
    period_cap = (
        pd.Series(capex_ffilled.values, index=period_idx.values)
        .groupby(level=0)
        .first()
    )  # index = period number, value = capex

    # Map each row to its own capex (cap_t), t-1, and t-2 period capex.
    cap_t = period_idx.map(period_cap)
    cap_t1 = (period_idx - 1).map(period_cap)
    cap_t2 = (period_idx - 2).map(period_cap)

    # ── 4. Compute growth rate ─────────────────────────────────────────────
    # Denominator: 0.5*(cap_t1 + cap_t2) when both available; else cap_t1.
    denom_2yr = 0.5 * (cap_t1 + cap_t2)
    denom_1yr = cap_t1

    # Use 2-year average where t-2 is available and both t-1 and t-2 positive.
    use_2yr = cap_t1.notna() & cap_t2.notna() & (cap_t1 > 0) & (cap_t2 > 0)
    use_1yr = cap_t1.notna() & (~use_2yr) & (cap_t1 > 0)

    denom = pd.Series(np.nan, index=df.index)
    denom = denom.where(~use_2yr, other=denom_2yr)
    denom = denom.where(~use_1yr, other=denom_1yr)

    # Guard: cap_t must be non-negative, denom must be positive.
    valid = cap_t.notna() & (cap_t >= 0) & denom.notna() & (denom > 0)

    capex_growth = pd.Series(np.nan, index=df.index)
    capex_growth[valid] = cap_t[valid] / denom[valid] - 1.0

    # Replace inf/-inf with NaN (guard for edge cases).
    capex_growth = capex_growth.replace([np.inf, -np.inf], np.nan)

    # ── 5. Produce columns ────────────────────────────────────────────────
    df["osap_chinvia_raw"] = capex_growth

    # First difference: captures acceleration/deceleration of capex growth.
    df["osap_chinvia_chg"] = capex_growth.diff()
    df["osap_chinvia_chg"] = df["osap_chinvia_chg"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Sign-encode: predicted sign is -1, so flip so "high = good signal".
    df["osap_chinvia_sign"] = -1.0 * capex_growth

    # ── 6. Drop scratch fund_ columns ─────────────────────────────────────
    df = df.drop(columns=["fund_capex_ttm", "fund_ppe_net"], errors="ignore")

    return df
