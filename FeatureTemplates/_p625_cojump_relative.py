"""
_p625_cojump_relative.py — Idiosyncratic / relative signed jump variation vs the market.

Stock's good-minus-bad (signed) jump variation in EXCESS of the market's, downside
co-jump concordance with the market, and the downside share of the beta-residual
(idiosyncratic bad variance). Index returns from SPY via the _indexes helper, aligned
to the ticker by a backward merge_asof on Date (look-ahead safe).

References:
  Bollerslev, Li & Todorov (2016), "Roughing up beta: Continuous versus discontinuous
    betas and the cross section of expected stock returns", JFE.
  Bollerslev, Patton & Quaedvlieg (2022), "Realized semicovariances".
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared index helper (skipped by framework auto-discovery)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "_p625_cojump_relative",
    "description": (
        "Relative signed jump variation, downside co-jump concordance, and "
        "idiosyncratic downside (beta-residual) variance share vs SPY "
        "(Bollerslev-Li-Todorov 2016 JFE; Bollerslev-Patton-Quaedvlieg 2022)."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "cjr_rel_signed_jump_21", "cjr_dn_cojump_21", "cjr_idio_dnvar_21",
        "cjr_rel_signed_jump_63", "cjr_dn_cojump_63", "cjr_idio_dnvar_63",
    ],
    "tags":        ["market_regime", "jump", "semivariance", "experimental"],
    "version":     "1.0",
    "author":      "paper:Bollerslev-Li-Todorov 2016 JFE; Bollerslev-Patton-Quaedvlieg 2022",
}

_WINDOWS = (21, 63)
_EPS = 1e-12


def _signed_jump_ratio(ret: pd.Series, w: int) -> pd.Series:
    """(sum up^2 - sum down^2) / (sum r^2 + eps) over a trailing window of w bars."""
    up2 = ret.clip(lower=0.0) ** 2
    dn2 = ret.clip(upper=0.0) ** 2          # negative part squared (>=0)
    r2 = ret ** 2
    mp = w
    s_up = up2.rolling(w, min_periods=mp).sum()
    s_dn = dn2.rolling(w, min_periods=mp).sum()
    s_r2 = r2.rolling(w, min_periods=mp).sum()
    denom = s_r2 + _EPS
    return (s_up - s_dn) / denom


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- stock log returns, Date-indexed (temporary; df never reindexed) ----
    df_dates = pd.to_datetime(df["Date"])
    close = df["Close"].astype(float)
    r_i = np.log(close / close.shift(1))

    # Pre-create NaN columns so the contract holds even if SPY is unavailable.
    for w in _WINDOWS:
        df[f"cjr_rel_signed_jump_{w}"] = np.nan
        df[f"cjr_dn_cojump_{w}"] = np.nan
        df[f"cjr_idio_dnvar_{w}"] = np.nan

    # --- market (SPY) log returns, aligned by backward merge_asof on Date ----
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = pd.Series(dtype="float64")
    if spy_close is None or spy_close.empty:
        return df

    spy_ret = np.log(spy_close / spy_close.shift(1))
    spy_frame = pd.DataFrame({
        "Date": pd.to_datetime(spy_ret.index),
        "_r_m": spy_ret.to_numpy(dtype="float64"),
    }).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    left = pd.DataFrame({"Date": df_dates, "_r_i": r_i.to_numpy(dtype="float64")})
    left = left.reset_index(drop=True)
    # Preserve original row order: merge_asof requires sorted keys; df is already
    # ascending by Date, so the merged frame stays row-aligned with df.
    merged = pd.merge_asof(left, spy_frame, on="Date", direction="backward")

    r_i_s = merged["_r_i"]
    r_m_s = merged["_r_m"]

    # --- per-row pieces for downside co-jump and beta-residual --------------
    dn_i2 = r_i_s.clip(upper=0.0) ** 2      # min(r_i,0)^2  >= 0
    dn_m2 = r_m_s.clip(upper=0.0) ** 2      # min(r_m,0)^2  >= 0

    for w in _WINDOWS:
        mp = w

        # 1) relative signed jump variation: SJ_i - SJ_m
        sj_i = _signed_jump_ratio(r_i_s, w)
        sj_m = _signed_jump_ratio(r_m_s, w)
        df[f"cjr_rel_signed_jump_{w}"] = (sj_i - sj_m).to_numpy(dtype="float64")

        # 2) downside co-jump concordance: rolling corr of the downside squares
        cojump = dn_i2.rolling(w, min_periods=mp).corr(dn_m2)
        cojump = cojump.clip(-1.0, 1.0)
        df[f"cjr_dn_cojump_{w}"] = cojump.to_numpy(dtype="float64")

        # 3) idiosyncratic downside variance share of the beta residual.
        #    beta_t = rolling cov(r_i,r_m) / var(r_m) over trailing w bars (causal),
        #    e_t = r_i,t - beta_t * r_m,t, then sum(down(e)^2)/(sum(e^2)+eps).
        cov_im = r_i_s.rolling(w, min_periods=mp).cov(r_m_s)
        var_m = r_m_s.rolling(w, min_periods=mp).var()
        beta = cov_im / var_m.replace(0.0, np.nan)
        e = r_i_s - beta * r_m_s
        e_dn2 = e.clip(upper=0.0) ** 2
        e2 = e ** 2
        s_e_dn2 = e_dn2.rolling(w, min_periods=mp).sum()
        s_e2 = e2.rolling(w, min_periods=mp).sum()
        idio = s_e_dn2 / (s_e2 + _EPS)
        df[f"cjr_idio_dnvar_{w}"] = idio.to_numpy(dtype="float64")

    return df
