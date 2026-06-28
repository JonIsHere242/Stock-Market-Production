"""
Heterogeneous Autoregressive Realized Variance (HAR-RV) features derived from:
  "On options-driven realized volatility forecasting: Information gains via rough
   volatility model" (arXiv 2604.02743).

The paper augments the HAR-RV model (Corsi, 2009) with options-derived rough
volatility estimates. We implement the CORE HAR-RV model itself — the paper's main
baseline and primary contribution mechanism — using only daily OHLCV.

SIGNAL MECHANISM (HAR-RV):
  Realized variance (RV) is approximated from daily data as the squared daily
  log-return or, more accurately, with the Yang-Zhang high-frequency estimator:
    RV_t ~ (log(High/Low))^2 / (4 * ln(2))   [Parkinson approximation]

  The HAR model then regresses RV onto:
    - Yesterday's RV (daily component)
    - Past 5-day average RV (weekly component)
    - Past 22-day average RV (monthly component)

  We don't fit the regression (that's a training step); instead we expose the
  three HAR components directly as features, plus the normalized current RV
  residual (deviation from the HAR-implied expected level), and a roughness
  proxy (ratio of daily RV to weekly RV, capturing the Hurst-like rough property).

These components are the inputs to the HAR forecast and carry genuine predictive
content for volatility regimes — useful as standalone features for tree models.

Options data (the paper's augmentation) is not available per-ticker, so that
extension is omitted; see METADATA author note.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2604_02743_har_rv",
    "description": (
        "HAR-RV components (daily/weekly/monthly realized variance, residual from "
        "HAR-implied level, roughness ratio) from arXiv 2604.02743; implements the "
        "core HAR-RV baseline only — options-derived rough-vol augmentation omitted "
        "(needs options surface data)."
    ),
    "requires":    ["High", "Low", "Close"],
    "produces": [
        "har_rv_daily",
        "har_rv_weekly",
        "har_rv_monthly",
        "har_rv_resid",
        "har_rv_roughness",
        "har_rv_log_daily",
        "har_rv_log_weekly",
        "har_rv_log_monthly",
    ],
    "tags":        ["volatility", "mean_reversion", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2604.02743 — HAR-RV baseline only; options rough-vol extension omitted",
}

# HAR horizons (daily, weekly, monthly)
_D  = 1
_W  = 5
_M  = 22


def _parkinson_rv(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Parkinson (1980) realized variance estimator from daily High/Low:
      RV_t = (ln(H_t / L_t))^2 / (4 * ln(2))
    This is an efficient, unbiased estimator of daily variance using only OHLCV.
    Annualization factor NOT applied — we keep daily units for HAR structure.
    """
    eps = 1e-10
    hl_log = np.log((high + eps) / (low + eps))
    return hl_log ** 2 / (4.0 * np.log(2.0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HAR-RV features.

    Features:
      har_rv_daily    : Parkinson daily RV (yesterday's RV, the daily HAR component)
      har_rv_weekly   : 5-day rolling mean of RV (weekly HAR component)
      har_rv_monthly  : 22-day rolling mean of RV (monthly HAR component)
      har_rv_resid    : current RV minus the HAR-implied level (0.3*d + 0.4*w + 0.3*m)
                        as a z-score over a 60-day rolling window
      har_rv_roughness: ratio RV_daily / (RV_weekly + eps) — rough if > 1, smooth if < 1
      har_rv_log_*    : log-transformed versions (log(RV+eps)), which stabilize
                        the HAR model (log-HAR is empirically more Gaussian)
    """
    high  = df["High"].values.astype(np.float64)
    low   = df["Low"].values.astype(np.float64)
    close = df["Close"].values.astype(np.float64)
    n     = len(df)
    eps   = 1e-14

    # --- Parkinson daily RV ---
    rv = _parkinson_rv(high, low)
    rv_s = pd.Series(rv, index=df.index)

    # --- HAR components (shifted by 1 to avoid using today to predict today) ---
    # HAR is typically: RV_{t+1} = c + b_D * RV_t + b_W * RV^(W)_t + b_M * RV^(M)_t
    # As FEATURES (not a fitted model), we expose the components:
    #   rv_daily   = yesterday's RV (i.e., current day's lagged RV)
    #   rv_weekly  = past 5-day average (including today) — the within-window mean
    #   rv_monthly = past 22-day average (including today)
    # We do NOT shift here — these are per-bar observable summaries of past RV,
    # analogous to what a HAR predictor would use at time t.
    rv_daily   = rv_s.shift(1)                               # lag 1 = yesterday's RV
    rv_weekly  = rv_s.rolling(_W,  min_periods=2).mean()     # 5-day rolling mean
    rv_monthly = rv_s.rolling(_M,  min_periods=5).mean()     # 22-day rolling mean

    # --- Log-transformed versions (log-HAR; Corsi et al. find this more Gaussian) ---
    log_rv        = np.log(rv_s + eps)
    log_rv_daily  = log_rv.shift(1)
    log_rv_weekly = log_rv.rolling(_W,  min_periods=2).mean()
    log_rv_monthly= log_rv.rolling(_M,  min_periods=5).mean()

    # --- HAR-implied level (equal-weight HAR mixing the three components) ---
    # Roughly captures the "expected" RV level; residual = current RV above/below this
    har_implied = 0.333 * rv_daily + 0.333 * rv_weekly + 0.333 * rv_monthly

    # Z-score the residual over a 60-day expanding window to avoid lookahead
    raw_resid = rv_s - har_implied
    resid_mu  = raw_resid.expanding(min_periods=10).mean()
    resid_std = raw_resid.expanding(min_periods=10).std()
    rv_resid  = (raw_resid - resid_mu) / (resid_std + eps)

    # --- Roughness ratio: daily RV / weekly RV ---
    # > 1: current day's vol spike exceeds recent average (rough / jumpy)
    # < 1: day calmer than recent history (smooth persistence)
    rv_roughness = rv_daily / (rv_weekly + eps)

    # --- Write outputs ---
    df["har_rv_daily"]     = rv_daily.values
    df["har_rv_weekly"]    = rv_weekly.values
    df["har_rv_monthly"]   = rv_monthly.values
    df["har_rv_resid"]     = rv_resid.values
    df["har_rv_roughness"] = rv_roughness.values
    df["har_rv_log_daily"]   = log_rv_daily.values
    df["har_rv_log_weekly"]  = log_rv_weekly.values
    df["har_rv_log_monthly"] = log_rv_monthly.values

    return df
