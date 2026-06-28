"""
Volatility Network Features  —  DOI:10.1155/2020/7051402
"The Volatility Forecasting Power of Financial Network Analysis"

The paper uses Minimum Spanning Tree (MST) length across 26 country indices to forecast
realized volatility — a fundamentally cross-asset signal. Per-ticker proxy implemented here:

  * The MST length captures how "clustered" the cross-sectional correlation matrix is.
    On a single series, the analogous measure is the INTERNAL structure of the series itself:
    how coupled is the return at lag t to lags t-1..t-k? A tightly-coupled (high-autocorr)
    series mirrors a "short" MST (concentrated network); a noisy series mirrors a long/sparse MST.

  * Proxy 1 (vpn_autocorr_coupling_20 / _60): rolling sum of |autocorrelation| at lags 1..5
    on the return series — a scalar measure of intra-series coupling (proxy for MST "density").

  * Proxy 2 (vpn_rv_har_ratio_5_22): ratio of 5-day realized vol to 22-day realized vol.
    Captures the HAR (Heterogeneous Autoregressive) multi-scale structure the paper uses as
    its baseline forecast model.

  * Proxy 3 (vpn_vol_regime_zscore_60): z-score of realized vol relative to its own 60-day
    history — operationalizes the paper's finding that network-length forecasts are most
    powerful during volatility "shocks."

NOTE: The cross-sectional MST across many markets cannot be replicated per-ticker; this is
an honest per-ticker proxy. See METADATA["author"] for attribution.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_doaj_7051402_volatility_power_network",
    "description": (
        "Per-ticker proxy for the MST-length volatility-network signal (DOI:10.1155/2020/7051402): "
        "rolling intra-series autocorrelation coupling, HAR vol ratio, and realized-vol regime "
        "z-score. Cross-sectional MST element dropped — single-series structural coupling proxy used."
    ),
    "requires": ["Close"],
    "produces": [
        "vpn_autocorr_coupling_20",
        "vpn_autocorr_coupling_60",
        "vpn_rv_har_ratio_5_22",
        "vpn_vol_regime_zscore_60",
        "vpn_rv_22d",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": (
        "paper:10.1155/2020/7051402 (per-ticker proxy — cross-sectional MST dropped; "
        "intra-series autocorrelation coupling used as structural analog)"
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker proxies for MST-length volatility-network signal.

    All features are causal (rolling over past data only). Leading NaNs during
    warm-up are expected.
    """
    close = df["Close"].astype(np.float64)

    # Daily log returns
    ret = np.log(close / close.shift(1))

    # ---- 1. Realized volatility (22-day and 5-day) --------------------------
    rv_22 = ret.rolling(22, min_periods=10).std() * np.sqrt(252)
    rv_5  = ret.rolling(5,  min_periods=3).std()  * np.sqrt(252)

    df["vpn_rv_22d"] = rv_22

    # ---- 2. HAR vol ratio: 5d / 22d -----------------------------------------
    # High values = short-horizon spike relative to longer baseline
    har_ratio = rv_5 / rv_22.replace(0, np.nan)
    df["vpn_rv_har_ratio_5_22"] = har_ratio

    # ---- 3. Realized-vol regime z-score: (rv_22 - mean_rv) / std_rv ---------
    rv_mean_60 = rv_22.rolling(60, min_periods=20).mean()
    rv_std_60  = rv_22.rolling(60, min_periods=20).std()
    df["vpn_vol_regime_zscore_60"] = (rv_22 - rv_mean_60) / rv_std_60.replace(0, np.nan)

    # ---- 4. Autocorrelation coupling: rolling sum |ACF| at lags 1..5 --------
    # This is the per-ticker analog of MST "length" — a tightly autocorrelated
    # series (concentrated energy) ~ short MST; noisy series ~ long/sparse MST.
    n = len(ret)
    ret_arr = ret.values.astype(np.float64)

    def _rolling_acf_sum(window: int, min_p: int) -> np.ndarray:
        out = np.full(n, np.nan)
        lags = [1, 2, 3, 4, 5]
        for i in range(window - 1, n):
            start = i - window + 1
            seg = ret_arr[start: i + 1]
            finite = seg[np.isfinite(seg)]
            if len(finite) < min_p:
                continue
            mu = np.mean(finite)
            var = np.var(finite)
            if var < 1e-15:
                out[i] = 0.0
                continue
            acf_sum = 0.0
            for lag in lags:
                if lag >= len(finite):
                    break
                cov = np.mean((finite[lag:] - mu) * (finite[:-lag] - mu))
                acf_sum += abs(cov / var)
            out[i] = acf_sum
        return out

    df["vpn_autocorr_coupling_20"] = _rolling_acf_sum(20, 12)
    df["vpn_autocorr_coupling_60"] = _rolling_acf_sum(60, 30)

    return df
