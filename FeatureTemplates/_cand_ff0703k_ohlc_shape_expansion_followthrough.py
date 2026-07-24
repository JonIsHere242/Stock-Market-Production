from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703k_ohlc_shape_expansion_followthrough",
    "description": (
        "Persistence/clustering of daily range-expansion events. Define R[t]=High-Low "
        "(NaN where <=0), med20[t]=trailing 20d median of R (min_periods=20). An 'expansion "
        "day' has R[t] > 1.3*med20[t]. Over a trailing causal 40d window (u in [t-39, t-1], "
        "requiring the successor day u+1 <= t so no lookahead), exp_followthru_40 = "
        "(count of expansion days at u whose successor u+1 is also an expansion day) / "
        "(count of expansion days at u with a valid successor in-window); denom 0 -> NaN. "
        "This measures whether volatility/range expansions tend to cluster (follow-through) "
        "or mean-revert immediately -- distinct from existing volatility-level or single-bar "
        "wick/body features. exp_followthru_updn is the same follow-through rate computed "
        "separately for trigger days that closed up (Close>Open) minus trigger days that "
        "closed down (Close<Open), each over the same 40d window (NaN if either subset is "
        "empty) -- captures directional asymmetry in expansion persistence (e.g. up-day "
        "expansions following through more/less than down-day expansions). Pure per-ticker "
        "OHLC, causal, no cross-sectional data."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703k_exp_followthru_40",
        "ff0703k_exp_followthru_updn",
    ],
    "tags": ["ohlc_shape", "range", "expansion", "persistence", "volatility", "asymmetry"],
    "version": "1.0",
    "author": (
        "ff0703k batch spec, implemented faithfully per-ticker as specified "
        "(trailing 40d causal window, no cross-sectional component needed)."
    ),
}

_MED_WIN = 20
_FT_WIN = 40
_EXP_MULT = 1.3


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out_level = np.full(n, np.nan, dtype=np.float64)
    out_asym = np.full(n, np.nan, dtype=np.float64)

    df["ff0703k_exp_followthru_40"] = out_level
    df["ff0703k_exp_followthru_updn"] = out_asym

    if n < 2:
        return df

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    openp = df["Open"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    rng = high - low
    rng_s = pd.Series(np.where(rng > 0, rng, np.nan))
    med20 = rng_s.rolling(_MED_WIN, min_periods=_MED_WIN).median().to_numpy()

    with np.errstate(invalid="ignore"):
        is_exp = (rng_s.to_numpy() > (_EXP_MULT * med20))
    is_exp = np.where(np.isnan(is_exp.astype(np.float64)), False, is_exp).astype(bool)
    # NaN comparisons already evaluate to False in numpy; guard above is a safety no-op.

    close_up = (close - openp) > 0
    close_dn = (close - openp) < 0

    level = np.full(n, np.nan, dtype=np.float64)
    asym = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        u_hi = t - 1
        u_lo = max(0, t - _FT_WIN + 1)
        if u_hi < u_lo:
            continue
        trig = is_exp[u_lo:u_hi + 1]
        follow = is_exp[u_lo + 1:u_hi + 2]

        denom = trig.sum()
        if denom > 0:
            numer = np.logical_and(trig, follow).sum()
            level[t] = numer / denom

        up_mask = np.logical_and(trig, close_up[u_lo:u_hi + 1])
        dn_mask = np.logical_and(trig, close_dn[u_lo:u_hi + 1])
        up_denom = up_mask.sum()
        dn_denom = dn_mask.sum()
        if up_denom > 0 and dn_denom > 0:
            up_rate = np.logical_and(up_mask, follow).sum() / up_denom
            dn_rate = np.logical_and(dn_mask, follow).sum() / dn_denom
            asym[t] = up_rate - dn_rate

    df["ff0703k_exp_followthru_40"] = level
    df["ff0703k_exp_followthru_updn"] = asym

    return df
