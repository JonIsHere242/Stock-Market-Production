import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_kyle_lambda_shift",
    "description": "Kyle's-lambda regime shift: fast(10d) vs slow(60d) price-impact slope ratio/accel + shock z (Kyle 1985 Econometrica 53(6); Hasbrouck 2009 JF 64(3):1445-1477; Brennan,Huh&Subrahmanyam 2013 RFS 26(5)).",
    "requires":    [],
    "produces":    ["klsh_lambda_fast_10", "klsh_lambda_accel", "klsh_lambda_ratio", "klsh_lambda_shock_z_120"],
    "tags":        ["liquidity", "microstructure", "regime", "experimental"],
    "version":     "1.0",
    "author":      "paper:Kyle (1985) Econometrica 53(6); Hasbrouck (2009) JF 64(3); Brennan,Huh&Subrahmanyam (2013) RFS 26(5)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # daily simple return and signed dollar-volume (in millions)
    ret = close.pct_change()
    dvol = close * vol
    signed_dvol_m = np.sign(ret) * dvol / 1e6

    def _lambda(w: int) -> pd.Series:
        mp = w // 2
        cov = ret.rolling(w, min_periods=mp).cov(signed_dvol_m)
        var = signed_dvol_m.rolling(w, min_periods=mp).var()
        lam = cov / (var + 1e-12)
        return lam.clip(-1.0, 1.0)

    lam_10 = _lambda(10)
    lam_60 = _lambda(60)

    df["klsh_lambda_fast_10"] = lam_10
    df["klsh_lambda_accel"] = (lam_10 - lam_60).clip(-2.0, 2.0)
    df["klsh_lambda_ratio"] = (lam_10.abs() / (lam_60.abs() + 1e-6)).clip(0.0, 10.0)

    # trailing 120d z-score of the fast lambda
    roll = lam_10.rolling(120, min_periods=60)
    mu = roll.mean()
    sd = roll.std()
    z = (lam_10 - mu) / sd.replace(0.0, np.nan)
    df["klsh_lambda_shock_z_120"] = z.clip(-6.0, 6.0)

    return df
