"""
_bvc_flow_toxicity.py  --  CANDIDATE block for the `vw` (volume-confirmed) ensemble arm.

Bulk-Volume-Classification (BVC) VPIN from OHLCV alone: split each bar's volume into buy/sell via a
Student-t CDF (df=4) of the standardized intrabar drift (Close-Open)/trailing-sigma, then emit the
20d order-flow toxicity (VPIN) level, the signed net order-imbalance fraction, and a 60d toxicity
shock-z. VPIN isolates days when a price move arrives with toxic ONE-SIDED high-volume flow -- the
exact case vw=ret*relvol amplifies and the 1-day axis cannot distinguish.

No VPIN/BVC/Student-t volume split exists anywhere in the library (vol_orderflow & vfd use crude
sign(return)*volume; _p625_kyle_lambda_shift is an unsigned lambda level). The continuous t-CDF
probability split is a strictly finer, bounded estimator than sign-counting. Toxicity is a fraction
of total volume (not a level) -> orthogonal to abs-vol / log-price / dollar-volume.

Refs: Easley, Lopez de Prado & O'Hara (2012) RFS 25(5):1457-1493; Andersen & Bondarenko (2014) JFM
(heavier-tail df note). Candidate -- NOT promoted until the multi-seed (>=4) ablation gate.
"""

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as _tdist

    def _t_cdf(z: np.ndarray) -> np.ndarray:
        out = np.full(z.shape, np.nan)
        m = np.isfinite(z)
        out[m] = _tdist.cdf(z[m], df=4)
        return out
except Exception:                      # pragma: no cover - normal fallback if scipy absent
    from math import erf as _erf
    _vphi = np.vectorize(lambda x: 0.5 * (1.0 + _erf(x / np.sqrt(2.0))) if np.isfinite(x) else np.nan)

    def _t_cdf(z: np.ndarray) -> np.ndarray:
        return _vphi(z)

METADATA = {
    "name":        "_bvc_flow_toxicity",
    "description": "Bulk-Volume-Classification VPIN from OHLCV: Student-t(4) CDF split of the "
                   "standardized intrabar body into buy/sell volume, emitting 20d toxicity (VPIN), "
                   "signed net order-imbalance fraction, and 60d VPIN shock-z. Easley-LdP-O'Hara 2012.",
    "requires":    ["Open", "Close", "Volume"],
    "produces":    ["bvc_vpin_20", "bvc_signed_oi_20", "bvc_vpin_shock_z_60"],
    "tags":        ["vw", "volume", "microstructure", "candidate"],
    "version":     "0.1",
    "author":      "alt-target-feature-research 2026-06-26 (vw VPIN white-space)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dP = df["Close"] - df["Open"]                               # intrabar drift
    sigma_dP = dP.rolling(60, min_periods=30).std()
    z = (dP / sigma_dP).where(sigma_dP > 0)

    frac_buy = pd.Series(_t_cdf(z.to_numpy(dtype=float)), index=df.index)
    buy_v = df["Volume"] * frac_buy
    sell_v = df["Volume"] - buy_v
    oi = buy_v - sell_v                                         # signed imbalance (shares)

    vol20 = df["Volume"].rolling(20, min_periods=12).sum()
    df["bvc_vpin_20"] = oi.abs().rolling(20, min_periods=12).sum() / vol20
    df["bvc_signed_oi_20"] = oi.rolling(20, min_periods=12).sum() / vol20

    m = df["bvc_vpin_20"].rolling(60, min_periods=30).mean()
    s = df["bvc_vpin_20"].rolling(60, min_periods=30).std().replace(0, np.nan)
    df["bvc_vpin_shock_z_60"] = ((df["bvc_vpin_20"] - m) / s).clip(-6.0, 6.0)
    return df
