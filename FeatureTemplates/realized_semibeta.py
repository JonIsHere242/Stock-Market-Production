"""
realized_semibeta.py — Signed tail-comovement vs SPY (Tier-2 discriminator).

Bollerslev, Patton & Quaedvlieg (2022, JFE) "Realized semibetas: Disentangling
'good' and 'bad' downside risks". Standard market beta hides four very different
risks. Decompose the realized covariation of stock returns r with market m into
four signed pieces (over a rolling window), each normalized by market variance:

    ss_pp = Σ r+·m+ / Σ m²    both up        (concordant positive)
    ss_nn = Σ r-·m- / Σ m²    both down       (concordant negative)  -> PRICED +
    ss_pn = Σ r+·m- / Σ m²    stock up/mkt dn (discordant)  <= 0
    ss_np = Σ r-·m+ / Σ m²    stock dn/mkt up (discordant)  <= 0       -> PRICED -

Their finding: ss_nn predicts HIGHER future returns; ss_np predicts LOWER. The
two positive-market semibetas (ss_pp/ss_pn) are ~unpriced. This is a Tier-2
separator: it sorts WITHIN the cross-section by *which kind* of market exposure a
name carries, not by overall beta. The four sum to ordinary beta (r·m / m²), so
this is a pure decomposition (sanity: ss_pp+ss_nn+ss_pn+ss_np == beta).

Daily-window realized analog (we have daily, not intraday, bars). Two horizons.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Shared index helper (underscore-prefixed -> skipped by block auto-discovery)
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

_WINDOWS = [(21, 15), (63, 30)]   # (window, min_periods)
_MKT = "SPY"

METADATA = {
    "name":        "realized_semibeta",
    "description": "Realized semibetas vs SPY (4 signed pieces + downside concordance ratio) at 21d/63d, per Bollerslev-Patton-Quaedvlieg 2022.",
    "requires":    ["Date", "Close"],
    "produces": [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("semibeta_pp", "semibeta_nn", "semibeta_pn", "semibeta_np", "semibeta_concord_dn")
    ],
    "tags":    ["market_regime", "beta", "tail", "experimental"],
    "version": "1.0",
    "author":  "Tier-2 lit build (BPQ 2022)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(df["Close"].values, index=dates)
    r = np.log(stock_close / stock_close.shift(1))

    try:
        idx_close = _indexes.index_close(_MKT)
    except Exception:
        idx_close = pd.Series(dtype=float)

    df_dates = dates.values
    if idx_close is None or idx_close.empty:
        for c in METADATA["produces"]:
            df[c] = np.nan
        return df

    m_all = np.log(idx_close / idx_close.shift(1))
    r_a, m_a = r.align(m_all, join="inner")          # shared trading dates

    rp, rn = r_a.clip(lower=0), r_a.clip(upper=0)
    mp, mn = m_a.clip(lower=0), m_a.clip(upper=0)
    prod = {
        "pp": rp * mp,                # both up   (>=0)
        "nn": rn * mn,                # both down (>=0)
        "pn": rp * mn,                # stock up, mkt down (<=0)
        "np": rn * mp,                # stock down, mkt up (<=0)
    }
    msq = m_a * m_a

    for w, mp_ in _WINDOWS:
        denom = msq.rolling(w, min_periods=mp_).sum().replace(0, np.nan)
        sb = {k: (v.rolling(w, min_periods=mp_).sum() / denom).clip(-5, 5) for k, v in prod.items()}
        # downside concordance ratio in [0,1]: how much of the down-comovement is
        # "good" concordant-down (ss_nn) vs "bad" discordant stock-dn/mkt-up (ss_np).
        concord_dn = sb["nn"] / (sb["nn"] + sb["np"].abs() + 1e-9)
        for k in ("pp", "nn", "pn", "np"):
            df[f"semibeta_{k}_{w}"] = sb[k].reindex(df_dates).values
        df[f"semibeta_concord_dn_{w}"] = concord_dn.reindex(df_dates).values

    return df
