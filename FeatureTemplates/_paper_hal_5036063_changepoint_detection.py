"""
Bayesian Online Change-Point Detection (BOCD) Feature Pack
Paper: "Enhancing financial crisis prediction: Integrating change point detection
        for exogenous event identification"  — HAL open archive id 5036063

The paper advocates using CPD to flag moments when the data-generating process
for a financial series shifts (mean/variance regime change), enabling adaptive
position management. This block implements the Adams & MacKay (2007) Bayesian
Online Changepoint Detection (BOCD) algorithm from scratch using numpy/scipy
only — NO external CPD libraries.

BOCD maintains a posterior over the current "run length" (bars since last
change point) using a conjugate Normal-Gamma model for the mean and variance
of log-returns. At each bar t the algorithm computes:
  P(r_t = k | x_{1:t})  for all k = 0 … t   (run-length posterior)

Key outputs derived from this posterior at every bar (online, causal):
  cpd_run_length          — MAP run length (bars since most probable change point)
  cpd_cp_prob             — P(change point at t | data) = P(r_t = 0)
  cpd_cp_smooth           — EWM-smoothed change-point probability (noise reduction)
  cpd_variance_stat       — evidence of a VARIANCE shift: log-ratio of predictive
                            variances under run-length-1 vs the MAP run-length
                            (positive = current variance exceeds baseline)
  cpd_mean_stat           — evidence of a MEAN shift: |predictive mean under
                            run-length-1 minus predictive mean under MAP run length|
                            normalised by the MAP predictive std
  cpd_regime_age_norm     — cpd_run_length / its own rolling 252-bar mean
                            (>1 = unusually long regime; <1 = recent instability)

Six columns total, all prefixed "cpd_", all causal (no lookahead).

NOTE: This is an UNPROVEN candidate block (leading underscore) — per-ticker
online change-point proxy; performance vs the live model is not yet validated.
"""

import math
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_hal_5036063_changepoint_detection",
    "description": (
        "Bayesian Online Changepoint Detection (BOCD) on per-ticker log-returns: "
        "run-length posterior, change-point probability, mean/variance shift statistics, "
        "and regime-age normalisation — inspired by HAL 5036063."
    ),
    "requires": ["Close"],
    "produces": [
        "cpd_run_length",
        "cpd_cp_prob",
        "cpd_cp_smooth",
        "cpd_variance_stat",
        "cpd_mean_stat",
        "cpd_regime_age_norm",
    ],
    "tags": ["market_regime", "volatility", "experimental"],
    "version": "1.0",
    "author": "paper:hal-5036063  (BOCD: Adams & MacKay 2007, Conjugate Normal-Gamma)",
}


# ---------------------------------------------------------------------------
# BOCD internals
# ---------------------------------------------------------------------------

def _bocd_run(
    x: np.ndarray,
    hazard: float = 1.0 / 100.0,   # prior P(change point at any bar) ≈ 1/expected_run
    mu0: float = 0.0,               # prior mean for Normal-Gamma
    kappa0: float = 1.0,            # prior pseudo-count on mean
    alpha0: float = 1.0,            # prior shape (half-df for variance)
    beta0: float = 0.01,            # prior rate (controls baseline variance)
    max_run: int = 300,             # cap run-length dimension for speed
) -> tuple:
    """
    Online BOCD over scalar observations x[0..n-1].

    Uses a Normal-Gamma conjugate prior: the sufficient stats for run length r are:
      mu_r, kappa_r, alpha_r, beta_r  (updated exactly at each bar)

    At each step the predictive distribution p(x_t | r_t = k, data) is Student-t
    with:
      nu    = 2 * alpha_r
      loc   = mu_r
      scale = sqrt(beta_r * (kappa_r + 1) / (alpha_r * kappa_r))

    Returns
    -------
    run_lengths     : (n,) int32  — MAP run length at each bar
    cp_probs        : (n,) float  — P(r_t = 0 | x_{1:t}) = change-point prob
    pred_means_rl1  : (n,) float  — predictive mean under run_length = 1 (just after a CP)
    pred_stds_rl1   : (n,) float  — predictive std under run_length = 1
    pred_means_map  : (n,) float  — predictive mean under MAP run length
    pred_stds_map   : (n,) float  — predictive std under MAP run length
    """
    n = len(x)
    # Storage
    run_lengths   = np.zeros(n, dtype=np.int32)
    cp_probs      = np.zeros(n, dtype=np.float64)
    pred_means_rl1 = np.full(n, np.nan)
    pred_stds_rl1  = np.full(n, np.nan)
    pred_means_map = np.full(n, np.nan)
    pred_stds_map  = np.full(n, np.nan)

    # Run-length posterior log-probs  (log R[r] for r = 0..t)
    # We keep a 1-D array of size max_run+1 and overwrite each step.
    # At t=0, r=0 with prob 1.
    log_R = np.full(max_run + 1, -np.inf)
    log_R[0] = 0.0  # log P(r=0) = 0 initially

    # Sufficient statistics for each hypothesis r=0..max_run
    # After k observations under a run of length k:
    #   kappa_r = kappa0 + k
    #   mu_r    = (kappa0*mu0 + sum_x) / kappa_r
    #   alpha_r = alpha0 + k/2
    #   beta_r  = beta0 + 0.5*(sum_x2 - kappa_r*mu_r^2 + kappa0*mu0^2)
    # We track: kappa_r, mu_r, alpha_r, beta_r as arrays indexed by run length.
    kappa = np.full(max_run + 1, kappa0)
    mu    = np.full(max_run + 1, mu0)
    alpha = np.full(max_run + 1, alpha0)
    beta  = np.full(max_run + 1, beta0)

    log_h  = math.log(hazard)
    log_1h = math.log(1.0 - hazard)

    for t in range(n):
        xt = x[t]
        if not math.isfinite(xt):
            # Propagate NaN; keep posterior unchanged
            run_lengths[t]    = run_lengths[t - 1] if t > 0 else 0
            cp_probs[t]       = np.nan
            pred_means_rl1[t] = np.nan
            pred_stds_rl1[t]  = np.nan
            pred_means_map[t] = np.nan
            pred_stds_map[t]  = np.nan
            continue

        # --- Step 1: Compute log predictive p(x_t | r_{t-1} = r, data) ------
        # Student-t with nu=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa))
        nu    = 2.0 * alpha                              # (max_run+1,)
        scale2 = beta * (kappa + 1.0) / (alpha * kappa) # predictive variance
        scale  = np.sqrt(np.maximum(scale2, 1e-30))

        # log Student-t PDF (scipy-free, log-gamma via math.lgamma via loop is slow;
        # use the log-formula directly with numpy's log/gammaln stand-in)
        # log p(x|nu,loc,scale) = lgamma((nu+1)/2) - lgamma(nu/2)
        #                          - 0.5*log(nu*pi) - log(scale)
        #                          - (nu+1)/2 * log(1 + ((x-mu)/scale)^2 / nu)
        # We compute lgamma via a fast vectorised approximation using
        # the log-factorial/Stirling if needed — but numpy has np.special... wait,
        # only scipy has gammaln. scipy IS allowed per contract. Use scipy.special.gammaln.
        from scipy.special import gammaln  # within function: imported once per bar would be slow
        # (import is cached by Python after first call — cost is O(1) after first bar)

        half_nu1 = 0.5 * (nu + 1.0)
        half_nu  = 0.5 * nu
        z        = (xt - mu) / scale
        log_pred = (
            gammaln(half_nu1)
            - gammaln(half_nu)
            - 0.5 * np.log(nu * math.pi)
            - np.log(scale)
            - half_nu1 * np.log1p(z * z / nu)
        )

        # --- Step 2: Joint log prob  log R[r] + log_pred[r] ------------------
        log_joint = log_R + log_pred  # (max_run+1,) — only finite where log_R > -inf

        # --- Step 3: New run-length posterior ---------------------------------
        # r_t = 0  (change point): sum over all r_{t-1} of R[r]*h * pred[r]
        log_cp = _log_sum_exp(log_joint + log_h)   # scalar

        # r_t = r+1 (no change): shift joint by 1, multiply by (1-h)
        log_new_R = np.full(max_run + 1, -np.inf)
        log_new_R[0] = log_cp

        # Shift: log_new_R[r+1] = log_joint[r] + log_1h  for r = 0..max_run-1
        src = log_joint[:-1] + log_1h             # length max_run
        log_new_R[1:] = src

        # Normalise
        log_Z = _log_sum_exp(log_new_R)
        if math.isfinite(log_Z):
            log_new_R -= log_Z
        log_R = log_new_R

        # --- Step 4: MAP run length and posterior summaries -------------------
        # Use only the finite (active) portion
        active = np.isfinite(log_R)
        if active.any():
            map_r = int(np.argmax(log_R))
        else:
            map_r = 0
        run_lengths[t] = map_r
        cp_probs[t]    = math.exp(log_R[0]) if math.isfinite(log_R[0]) else np.nan

        # Predictive stats under run_length=1 (one bar into a new regime)
        # and under MAP run length
        if map_r <= max_run and kappa[map_r] > 0 and alpha[map_r] > 0:
            sc2_map = beta[map_r] * (kappa[map_r] + 1.0) / (alpha[map_r] * kappa[map_r])
            pred_means_map[t] = mu[map_r]
            pred_stds_map[t]  = math.sqrt(max(sc2_map, 0.0))

        if kappa[1] > 0 and alpha[1] > 0:
            sc2_rl1 = beta[1] * (kappa[1] + 1.0) / (alpha[1] * kappa[1])
            pred_means_rl1[t] = mu[1]
            pred_stds_rl1[t]  = math.sqrt(max(sc2_rl1, 0.0))
        elif kappa[0] > 0 and alpha[0] > 0:
            sc2_rl0 = beta[0] * (kappa[0] + 1.0) / (alpha[0] * kappa[0])
            pred_means_rl1[t] = mu[0]
            pred_stds_rl1[t]  = math.sqrt(max(sc2_rl0, 0.0))

        # --- Step 5: Update sufficient statistics for next step ---------------
        # Run-length r stats are updated: new obs xt belongs to ALL hypotheses.
        # For r=0 (just-reset run), the stats are the prior (already in kappa[0] etc.)
        # For r>=1, kappa[r], mu[r], alpha[r], beta[r] accumulate xt.
        # After the shift above, log_R[r] represents a run that has now seen r obs.
        # The conjugate update for each active run length:
        #   kappa_new = kappa + 1
        #   mu_new    = (kappa*mu + xt) / kappa_new
        #   alpha_new = alpha + 0.5
        #   beta_new  = beta + kappa*(xt - mu)^2 / (2*(kappa+1))
        # We update for all positions >= 1 (position 0 stays at prior for next step's CP).

        # Shift the sufficient stats array to match the new run-length indexing.
        # kappa[r] should now hold the stats for a run that is r long (after this obs).
        # After the run-length shift: position r in the NEW R corresponds to the old
        # position r-1 having seen one more observation = xt.
        # So we shift stats arrays rightward by 1 first, then update positions 1..max_run.

        diff  = xt - mu         # (max_run+1,)
        kappa_new = kappa + 1.0
        mu_new    = (kappa * mu + xt) / kappa_new
        alpha_new = alpha + 0.5
        beta_new  = beta + kappa * diff * diff / (2.0 * kappa_new)

        # Shift: position r+1 gets the updated stats of position r
        kappa[1:]  = kappa_new[:-1]
        mu[1:]     = mu_new[:-1]
        alpha[1:]  = alpha_new[:-1]
        beta[1:]   = beta_new[:-1]

        # Reset position 0 to prior (for the CP hypothesis)
        kappa[0] = kappa0
        mu[0]    = mu0
        alpha[0] = alpha0
        beta[0]  = beta0

    return (
        run_lengths,
        cp_probs,
        pred_means_rl1,
        pred_stds_rl1,
        pred_means_map,
        pred_stds_map,
    )


def _log_sum_exp(log_p: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1D array (ignores -inf)."""
    finite = log_p[np.isfinite(log_p)]
    if len(finite) == 0:
        return -math.inf
    m = float(finite.max())
    return m + math.log(float(np.exp(finite - m).sum()))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker Bayesian Online Changepoint Detection over log-returns.

    All features are strictly causal: at row t, only x[0..t] is used.
    """
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Log returns (NaN at t=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret = np.empty(n, dtype=np.float64)
        log_ret[0] = np.nan
        log_ret[1:] = np.log(
            np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
        )

    # Run BOCD
    (
        run_lengths,
        cp_probs,
        pred_means_rl1,
        pred_stds_rl1,
        pred_means_map,
        pred_stds_map,
    ) = _bocd_run(log_ret)

    idx = df.index

    # --- cpd_run_length: MAP run length (int, 0 = just-detected change point) -
    df["cpd_run_length"] = pd.array(run_lengths, dtype="Int32")

    # --- cpd_cp_prob: raw change-point probability P(r_t=0 | data) ------------
    df["cpd_cp_prob"] = pd.Series(cp_probs, index=idx)

    # --- cpd_cp_smooth: EWM-smoothed change-point probability -----------------
    df["cpd_cp_smooth"] = (
        pd.Series(cp_probs, index=idx)
        .ewm(span=5, min_periods=3, adjust=False)
        .mean()
    )

    # --- cpd_variance_stat: log-ratio of predictive stds ----------------------
    # Positive = current predictive uncertainty (run-length-1) exceeds MAP regime.
    # Signals a VARIANCE shift: the model under a fresh-CP hypothesis sees
    # much more variance than the established-regime model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_stat = np.log(
            np.where(
                (pred_stds_map > 1e-12) & (pred_stds_rl1 > 1e-12),
                pred_stds_rl1 / pred_stds_map,
                np.nan,
            )
        )
    df["cpd_variance_stat"] = pd.Series(var_stat, index=idx)

    # --- cpd_mean_stat: |mean shift| normalised by MAP predictive std ---------
    # Signals a MEAN shift: |rl-1 mean - MAP mean| / MAP std.
    # Large values = the regime-just-after-a-CP expects a very different mean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_stat = np.where(
            pred_stds_map > 1e-12,
            np.abs(pred_means_rl1 - pred_means_map) / pred_stds_map,
            np.nan,
        )
    df["cpd_mean_stat"] = pd.Series(mean_stat, index=idx)

    # --- cpd_regime_age_norm: run_length / rolling mean of run_length ----------
    # >1 = unusually long quiet regime; <1 = recent change-point instability.
    rl_float = run_lengths.astype(np.float64)
    rl_series = pd.Series(rl_float, index=idx)
    # Use past 252 bars (excluding current) to avoid forward bias
    rl_rolling_mean = rl_series.shift(1).rolling(252, min_periods=30).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        regime_age_norm = np.where(
            rl_rolling_mean > 1e-6,
            rl_float / rl_rolling_mean.values,
            np.nan,
        )
    df["cpd_regime_age_norm"] = pd.Series(regime_age_norm, index=idx)

    return df
