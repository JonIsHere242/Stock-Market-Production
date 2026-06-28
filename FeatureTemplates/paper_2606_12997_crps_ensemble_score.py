"""
CRPS ensemble scoring features derived from:
  "Reliability of Probabilistic Emulation of Physical Systems"
  arXiv 2606.12997

The paper shows CRPS-trained ensembles yield better-calibrated predictive
intervals than diffusion/flow models in latent space.  The CRPS (Continuous
Ranked Probability Score) decomposes into RELIABILITY + RESOLUTION - SHARPNESS.

We apply CRPS directly to rolling return distributions:
  - At each bar, treat the trailing 20 (or 60) daily log-returns as an ensemble.
  - Score the CURRENT return against this ensemble via empirical CRPS.
  - CRPS = E[|X - x|] - 0.5 * E[|X - X'|]  (energy score form).
  - Low CRPS = realized return was predictable from the ensemble.
  - High CRPS = the return was a surprise.
  - PIT (Probability Integral Transform): fraction of ensemble < today's return.
    Under a calibrated forecast, PIT ~ Uniform[0,1].
    A persistent positive PIT bias means returns keep exceeding the ensemble.
  - Signed CRPS: CRPS with sign = direction (above median = positive).
    Captures directional predictive failure.

Feature family (7 cols):
  crps_score_20       CRPS of today's return vs 20-day trailing ensemble
  crps_score_60       CRPS of today's return vs 60-day trailing ensemble
  crps_z_20           z-score of crps_score_20 over 40-day window
  crps_signed_20      signed CRPS: positive if above ensemble median
  crps_sharpness_20   spread of 20-day ensemble (std of trailing returns)
  crps_pit_bias       rolling 20d mean of (PIT - 0.5): calibration bias signal
  crps_pit_momentum   5d change in PIT: directional PIT drift
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12997_crps_ensemble_score",
    "description": (
        "Empirical CRPS scoring of daily returns against rolling ensemble distributions: "
        "score, signed score, sharpness, and PIT calibration bias; "
        "from arXiv 2606.12997 (CRPS-trained ensembles for physical systems)."
    ),
    "requires": ["Close"],
    "produces": [
        "crps_score_20",
        "crps_score_60",
        "crps_z_20",
        "crps_signed_20",
        "crps_sharpness_20",
        "crps_pit_bias",
        "crps_pit_momentum",
    ],
    "tags": ["volatility", "statistical", "experimental"],
    "version": "1.1",
    "author": "paper:2606.12997",
}


def _empirical_crps(ensemble: np.ndarray, observation: float) -> float:
    """
    Empirical CRPS via energy score form:
      CRPS(F, x) = E_F[|X - x|] - 0.5 * E_F[|X - X'|]
    Efficient O(m log m) computation using sorted ensemble.
    """
    mask = np.isfinite(ensemble)
    m = mask.sum()
    if m < 3 or not np.isfinite(observation):
        return np.nan
    ens = ensemble[mask]
    term1 = np.mean(np.abs(ens - observation))
    ens_sorted = np.sort(ens)
    idx = np.arange(1, m + 1, dtype=np.float64)
    weights = 2.0 * idx - m - 1.0
    term2 = np.dot(ens_sorted, weights) / (m * m)
    return float(term1 - 0.5 * term2)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values
    n = len(df)

    crps_20  = np.full(n, np.nan)
    crps_60  = np.full(n, np.nan)
    signed20 = np.full(n, np.nan)
    sharp_20 = np.full(n, np.nan)
    pit_20   = np.full(n, np.nan)  # PIT = fraction of ensemble < today's return

    for i in range(1, n):
        x = log_ret[i]
        if not np.isfinite(x):
            continue

        # 20-day trailing ensemble (excludes current bar — no lookahead)
        s20   = max(0, i - 20)
        ens20 = log_ret[s20:i]
        c20   = _empirical_crps(ens20, x)
        crps_20[i] = c20

        mask20 = np.isfinite(ens20)
        if mask20.sum() >= 3:
            ens20_v = ens20[mask20]
            sharp_20[i] = float(np.std(ens20_v))
            pit_val     = float(np.mean(ens20_v < x))
            pit_20[i]   = pit_val
            # Signed CRPS: positive if obs above ensemble median, negative below
            if np.isfinite(c20):
                sign = 1.0 if pit_val >= 0.5 else -1.0
                signed20[i] = sign * c20

        # 60-day trailing ensemble
        if i >= 60:
            s60   = max(0, i - 60)
            ens60 = log_ret[s60:i]
            crps_60[i] = _empirical_crps(ens60, x)

    crps_20_s  = pd.Series(crps_20,  index=df.index)
    crps_60_s  = pd.Series(crps_60,  index=df.index)
    signed20_s = pd.Series(signed20, index=df.index)
    sharp_20_s = pd.Series(sharp_20, index=df.index)
    pit_20_s   = pd.Series(pit_20,   index=df.index)

    # z-score of CRPS_20 over 40-day rolling window
    mu40  = crps_20_s.rolling(40, min_periods=10).mean()
    sig40 = crps_20_s.rolling(40, min_periods=10).std()
    crps_z_20 = (crps_20_s - mu40) / sig40.replace(0.0, np.nan)

    # PIT bias: rolling 20d mean of (PIT - 0.5)
    # Persistent positive = returns keep beating ensemble → momentum signal
    # Persistent negative = returns keep missing ensemble → mean-reversion
    pit_bias = (pit_20_s - 0.5).rolling(20, min_periods=5).mean()

    # PIT momentum: 5-day change in rolling PIT (directional drift)
    pit_roll5 = pit_20_s.rolling(5, min_periods=2).mean()
    pit_momentum = pit_roll5.diff(5)

    df["crps_score_20"]   = crps_20_s
    df["crps_score_60"]   = crps_60_s
    df["crps_z_20"]       = crps_z_20
    df["crps_signed_20"]  = signed20_s
    df["crps_sharpness_20"] = sharp_20_s
    df["crps_pit_bias"]   = pit_bias
    df["crps_pit_momentum"] = pit_momentum

    return df
