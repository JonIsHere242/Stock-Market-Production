"""
_insider_activity.py  --  CANDIDATE feature block: insider (SEC Form 4) buying/selling signals.

Leading underscore == UNPROVEN candidate: the framework does NOT auto-discover it, so it can't leak
into the model before it earns promotion (same convention as _fundamentals_valuation.py). Promote
(drop the leading underscore) only after the gate: __diagnostics.py / __tail_screen.py for fast
triage, then the multi-seed (>=4) in-model marginal-contribution ablation that is the only signal
that has held up historically (project_tier2_feature_build_verdict_2026_06_20).

It pulls trailing-window insider primitives through _insider.windowed_primitives (a strictly BACKWARD
aggregation on filed_date -> lookahead-safe) and shares-outstanding through _fundamentals.as_of, then
composes 31 candidate variants spanning: net flow, recency, breadth/cluster, role/conviction,
magnitude-vs-ownership/liquidity, acceleration/regime, price-context, and selling/derivatives. The
`_ins_*` and `fund_*` columns are SCRATCH -- dropped at the end so only METADATA["produces"] is added.

WHY insider flow (vs another technical transform): it is genuinely orthogonal to price/volume -- it
encodes who-with-private-information is buying/selling -- which is the white-space left after OHLCV
technical features were exhausted. Open-market P (purchase) / S (sale) carry the signal; A (grant),
M (exercise), F (tax) are routine and treated separately.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_HERE = _Path(__file__).resolve().parent


def _load(modname: str):
    spec = _ilu.spec_from_file_location(modname, _HERE / f"{modname}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_insider = _load("_insider")
_fundamentals = _load("_fundamentals")

_WINDOWS = (30, 90, 180, 365)

METADATA = {
    "name":        "insider_activity",
    "description": "Insider (SEC Form 4) buying/selling signals: trailing-window net flow, recency, "
                   "breadth/cluster, role/conviction (CEO/CFO/officer/director), magnitude vs "
                   "shares-outstanding & volume, acceleration, price-context, and selling/derivatives. "
                   "All strictly point-in-time on filed_date (backward as-of).",
    "requires":    ["Close", "Volume"],
    "produces":    [
        "insdr_net_buy_ratio_30", "insdr_net_buy_ratio_90", "insdr_net_buy_ratio_180",
        "insdr_net_buy_ratio_365", "insdr_net_dollar_ratio_90", "insdr_buy_dollar_log_90",
        "insdr_buy_n_90", "insdr_sell_n_90", "insdr_days_since_buy", "insdr_days_since_sell",
        "insdr_days_since_any", "insdr_ewm_net_flow", "insdr_unique_buyers_90",
        "insdr_cluster_buy_flag", "insdr_buyer_breadth_90", "insdr_ceo_buy_90", "insdr_cfo_buy_90",
        "insdr_officer_buy_90", "insdr_dir_buy_90", "insdr_purchase_vs_grant_90",
        "insdr_buy_pct_shares_out_90", "insdr_buy_vol_ratio_90", "insdr_net_accel",
        "insdr_first_buy_flag", "insdr_flip_to_buy_flag", "insdr_buy_on_dip_flag",
        "insdr_close_vs_last_buy", "insdr_sell_dollar_log_90", "insdr_heavy_sell_flag",
        "insdr_optexer_90", "insdr_conviction_90",
    ],
    "tags":        ["insider", "sec", "form4", "informed-flow", "experimental"],
    "version":     "1.0",
    "author":      "insider-flow candidate (build_insider_panel.py + _insider.py)",
}


def _ratio(num: pd.Series, den: pd.Series) -> np.ndarray:
    den = den.to_numpy(dtype="float64")
    num = num.to_numpy(dtype="float64")
    out = np.full(len(den), np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce")

    # Lookahead-safe trailing-window insider primitives + PIT shares-outstanding.
    df = _insider.windowed_primitives(df, windows=_WINDOWS)
    df = _fundamentals.as_of(df, fields=["shares_outstanding"])

    def col(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce")

    def net_share_ratio(w: int) -> np.ndarray:
        b, s = col(f"_ins_buy_sh_{w}"), col(f"_ins_sell_sh_{w}")
        return _ratio(b - s, b + s)

    # --- net flow / intensity ---
    for w in _WINDOWS:
        df[f"insdr_net_buy_ratio_{w}"] = net_share_ratio(w)
    df["insdr_net_dollar_ratio_90"] = _ratio(col("_ins_buy_val_90") - col("_ins_sell_val_90"),
                                             col("_ins_buy_val_90") + col("_ins_sell_val_90"))
    df["insdr_buy_dollar_log_90"] = np.log1p(col("_ins_buy_val_90").clip(lower=0))
    df["insdr_buy_n_90"] = col("_ins_buy_n_90")
    df["insdr_sell_n_90"] = col("_ins_sell_n_90")

    # --- recency ---
    df["insdr_days_since_buy"] = col("_ins_days_since_buy")
    df["insdr_days_since_sell"] = col("_ins_days_since_sell")
    df["insdr_days_since_any"] = col("_ins_days_since_any")
    nv = col("_ins_ewm_net_val")
    df["insdr_ewm_net_flow"] = np.sign(nv) * np.log1p(nv.abs())

    # --- breadth / cluster ---
    ub, us = col("_ins_unique_buyers_90"), col("_ins_unique_sellers_90")
    df["insdr_unique_buyers_90"] = ub
    df["insdr_cluster_buy_flag"] = (ub >= 2).astype("float64")
    df["insdr_buyer_breadth_90"] = _ratio(ub, ub + us)

    # --- role / conviction ---
    df["insdr_ceo_buy_90"] = (col("_ins_ceo_buy_90") > 0).astype("float64")
    df["insdr_cfo_buy_90"] = (col("_ins_cfo_buy_90") > 0).astype("float64")
    df["insdr_officer_buy_90"] = col("_ins_officer_buy_90")
    df["insdr_dir_buy_90"] = col("_ins_dir_buy_90")
    df["insdr_purchase_vs_grant_90"] = _ratio(col("_ins_buy_sh_90"),
                                              col("_ins_buy_sh_90") + col("_ins_grant_sh_90"))

    # --- magnitude vs ownership / liquidity ---
    sh = col("fund_shares_outstanding").where(lambda s: s > 0)
    df["insdr_buy_pct_shares_out_90"] = col("_ins_buy_sh_90") / sh
    avgvol = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    df["insdr_buy_vol_ratio_90"] = col("_ins_buy_sh_90") / avgvol

    # --- acceleration / regime ---
    nr30, nr90, nr180 = net_share_ratio(30), net_share_ratio(90), net_share_ratio(180)
    df["insdr_net_accel"] = nr30 - nr180
    buy30, buy365 = col("_ins_buy_n_30"), col("_ins_buy_n_365")
    df["insdr_first_buy_flag"] = ((buy30 > 0) & ((buy365 - buy30) <= 0)).astype("float64")
    flip = (nr30 > 0) & (nr180 < 0)
    df["insdr_flip_to_buy_flag"] = np.where(np.isnan(nr30) | np.isnan(nr180), 0.0,
                                            flip.astype("float64"))

    # --- price-context ---
    ret21 = close / close.shift(21) - 1.0
    df["insdr_buy_on_dip_flag"] = ((buy30 > 0) & (ret21 < -0.05)).astype("float64")
    lbp = col("_ins_last_buy_price").where(lambda s: s > 0)
    df["insdr_close_vs_last_buy"] = (close - lbp) / lbp

    # --- selling / derivatives ---
    df["insdr_sell_dollar_log_90"] = np.log1p(col("_ins_sell_val_90").clip(lower=0))
    heavy = (nr90 < -0.5) & (col("_ins_sell_n_90") >= 2)
    df["insdr_heavy_sell_flag"] = np.where(np.isnan(nr90), 0.0, heavy.astype("float64"))
    df["insdr_optexer_90"] = col("_ins_optexer_n_90")

    # --- composite conviction ---
    conv = (np.nan_to_num(nr90) * np.log1p(col("_ins_buy_n_90").to_numpy())
            * (1.0 + 0.5 * col("_ins_ceo_buy_90").to_numpy() + 0.25 * col("_ins_cfo_buy_90").to_numpy()))
    df["insdr_conviction_90"] = conv

    # Drop scratch columns so only METADATA["produces"] is added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("_ins_") or c.startswith("fund_")])
    return df
