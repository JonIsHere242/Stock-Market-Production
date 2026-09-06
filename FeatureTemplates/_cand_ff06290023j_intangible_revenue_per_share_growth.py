from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load helper: point-in-time SEC fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06290023j_intangible_revenue_per_share_growth",
    "description": (
        "YoY revenue-per-share (RPS) growth, dilution-adjusted. "
        "rps = revenue_ttm / shares_outstanding at the PIT date. "
        "Growth = (rps_now - rps_lag252) / max(|rps_lag252|, 1e-9), clipped [-2,2]. "
        "Captures organic top-line strength net of share dilution; a negative growth "
        "rate when revenue grows but share count dilutes faster flags value destruction. "
        "Also produces a 63-day slope of the RPS level (trend within-year) and a "
        "sign-asymmetry flag (positive vs negative growth). "
        "Per-ticker PIT proxy; intangible/quality vein orthogonal to price-momentum."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290023j_rps_growth_yoy",   # YoY RPS growth clipped [-2,2]
        "ff06290023j_rps_slope_63d",    # OLS slope of RPS level over trailing 63 bars (normalised)
        "ff06290023j_rps_growth_pos",   # 1 if growth>0, -1 if <0, 0 if =0, NaN if missing
    ],
    "tags": ["fundamentals", "intangible", "quality", "dilution", "revenue"],
    "version": "1.0.0",
    "author": "ff06290023j-codegen",
}

# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (required by gate)
    df["ff06290023j_rps_growth_yoy"] = np.nan
    df["ff06290023j_rps_slope_63d"] = np.nan
    df["ff06290023j_rps_growth_pos"] = np.nan

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # Pull PIT fundamentals
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=["revenue_ttm", "shares_outstanding"])

    rev = df["fund_revenue_ttm"].to_numpy(dtype=float)
    shr = df["fund_shares_outstanding"].to_numpy(dtype=float)
    n = len(df)

    # ------------------------------------------------------------------
    # Compute RPS at every bar
    # ------------------------------------------------------------------
    rps = np.where(
        (np.isfinite(rev) & np.isfinite(shr) & (shr > 0)),
        rev / np.maximum(shr, 1e-9),
        np.nan,
    )

    # ------------------------------------------------------------------
    # Feature 1: YoY RPS growth (lag 252 bars)
    # ------------------------------------------------------------------
    lag = 252
    rps_prev = np.full(n, np.nan)
    if n > lag:
        rps_prev[lag:] = rps[: n - lag]

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.maximum(np.abs(rps_prev), 1e-9)
        growth = (rps - rps_prev) / denom

    # Require both present and shares > 0 at both points; else NaN
    valid = np.isfinite(rps) & np.isfinite(rps_prev)
    growth = np.where(valid, growth, np.nan)
    growth = np.clip(growth, -2.0, 2.0)
    df["ff06290023j_rps_growth_yoy"] = growth

    # ------------------------------------------------------------------
    # Feature 2: 63-bar OLS slope of RPS level (normalised by |mean|)
    # Vectorised via cumsum trick for linear regression slope.
    # ------------------------------------------------------------------
    win = 63
    if n >= win:
        # Use pandas rolling to get slope via Pearson correlation trick:
        # slope = corr(x, y) * std(y) / std(x)  where x=bar_index [0..win-1]
        rps_s = pd.Series(rps)
        roll_mean = rps_s.rolling(win, min_periods=win).mean()
        roll_std = rps_s.rolling(win, min_periods=win).std(ddof=0)

        # x within window: 0,1,...,win-1; stats are fixed
        x_mean = (win - 1) / 2.0
        x_std = np.sqrt(((win - 1) * (2 * win - 1)) / 6.0 - x_mean ** 2)

        # rolling cov(x, y) via cumsum: E[x*y] - E[x]*E[y]
        # x[i] within window = (i - (start)) => positional index
        # We need rolling cov between position-within-window and rps value.
        # Efficient approach: use weighted sum with triangular weights.
        # cov(x,y) = (1/win) * sum(k * rps[t-win+1+k]) - x_mean * roll_mean
        # Compute sum(k * rps[t-win+1+k]) with a rolling sum of k-weighted series.
        k_weights = np.arange(win, dtype=float)  # 0,1,...,win-1
        # Pad and convolve (finite IR filter)
        rps_arr = rps_s.to_numpy(dtype=float)
        # Use np.convolve with reversed weights for rolling dot product
        conv = np.convolve(rps_arr, k_weights[::-1], mode="full")[: n]
        # conv[t] = sum_{k=0}^{win-1} k * rps[t - (win-1) + k]  when fully covered
        # valid from index win-1 onward
        cov_xy = np.full(n, np.nan)
        cov_xy[win - 1:] = (
            conv[win - 1:] / win - x_mean * roll_mean.to_numpy()[win - 1:]
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where(
                (x_std > 0) & np.isfinite(cov_xy),
                cov_xy / (x_std ** 2),  # OLS slope = cov(x,y)/var(x)
                np.nan,
            )
            # Normalise by rolling |mean| of RPS to make scale-invariant
            abs_mean = np.abs(roll_mean.to_numpy())
            slope_norm = np.where(
                abs_mean > 1e-9,
                slope / abs_mean,
                np.nan,
            )

        df["ff06290023j_rps_slope_63d"] = slope_norm

    # ------------------------------------------------------------------
    # Feature 3: sign of YoY growth
    # ------------------------------------------------------------------
    g = df["ff06290023j_rps_growth_yoy"].to_numpy(dtype=float)
    sign = np.where(np.isfinite(g), np.sign(g), np.nan)
    df["ff06290023j_rps_growth_pos"] = sign

    # ------------------------------------------------------------------
    # Drop scratch fund_ columns
    # ------------------------------------------------------------------
    df.drop(
        columns=[c for c in df.columns if c.startswith("fund_")],
        inplace=True,
    )

    return df
