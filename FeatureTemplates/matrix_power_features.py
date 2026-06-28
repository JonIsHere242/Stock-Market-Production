import json as _mp_json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "matrix_power_features",
    "description": (
        "14 mp_* matrix-power features built from sign-weighted, panel-normalised "
        "OHLCV primitives. Each feature applies expm (scaling-and-squaring + "
        "Taylor K=16) to a variant (raw/sym/skew) DxD matrix and extracts a "
        "scalar (trace, Frobenius norm, or top-left element)."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces":    [
        "mp_d4_b185_sym_trace",
        "mp_d4_b65_sym_frob",
        "mp_d5_b108_sym_trace",
        "mp_d4_b306_raw_top_left",
        "mp_d5_b169_sym_top_left",
        "mp_d4_b267_sym_trace",
        "mp_d4_b291_sym_frob",
        "mp_d4_b173_sym_top_left",
        "mp_d4_b50_raw_top_left",
        "mp_d4_b364_skew_top_left",
        "mp_d4_b396_sym_trace",
        "mp_d4_b291_sym_top_left",
        "mp_d4_b477_skew_trace",
        "mp_d5_b39_sym_trace",
    ],
    "tags":        ["experimental", "gp"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.add_matrix_power_features",
}

# ---------------------------------------------------------------------------
# Spec file — resolved relative to the repo root (parent of FeatureTemplates/)
# ---------------------------------------------------------------------------
_MP_SPEC_PATH = Path(__file__).resolve().parent.parent / "Data" / "matrix_power_spec.json"
_MP_SPEC_CACHE = None


# ---------------------------------------------------------------------------
# Private helpers (inlined from 3__AlphaSensitivity.py)
# ---------------------------------------------------------------------------

def _mp_load_spec():
    global _MP_SPEC_CACHE
    if _MP_SPEC_CACHE is None:
        with open(_MP_SPEC_PATH, "r") as _f:
            _MP_SPEC_CACHE = _mp_json.load(_f)
    return _MP_SPEC_CACHE


def _mp_compute_primitives(df, window):
    """14 causal OHLCV primitives. Returns ndarray (N, 14) in spec order."""
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    C = df["Close"].astype(np.float64)
    V = df["Volume"].astype(np.float64)
    log_ret    = np.log(C).diff()
    log_vol    = np.log(V.replace(0, np.nan)).diff()
    range_pct  = (H - L) / C
    body_pct   = (C - O) / C
    upper_wick = (H - np.maximum(O, C)) / C
    lower_wick = (np.minimum(O, C) - L) / C
    rv         = log_ret.rolling(window).std()
    z_close    = (C - C.rolling(window).mean()) / (C.rolling(window).std() + 1e-9)
    z_vol      = (V - V.rolling(window).mean()) / (V.rolling(window).std() + 1e-9)
    skew_p     = log_ret.rolling(window).skew()
    kurt_p     = log_ret.rolling(window).kurt()
    mom5       = np.log(C / C.shift(5))
    mom20      = np.log(C / C.shift(window))
    sign_last  = np.sign(log_ret)
    # Column order MUST match spec['primitive_names'].
    return np.column_stack([
        log_ret.to_numpy(),    log_vol.to_numpy(),
        range_pct.to_numpy(),  body_pct.to_numpy(),
        upper_wick.to_numpy(), lower_wick.to_numpy(),
        rv.to_numpy(),         z_close.to_numpy(),    z_vol.to_numpy(),
        skew_p.to_numpy(),     kurt_p.to_numpy(),
        mom5.to_numpy(),       mom20.to_numpy(),
        sign_last.to_numpy(),
    ])


def _mp_expm_batch(M_batch, K=16):
    """Batched matrix exponential via scaling-and-squaring + Taylor K=16.

    M_batch : (N, D, D)
    Returns  : (N, D, D)
    """
    N, D, _ = M_batch.shape
    norms = np.linalg.norm(M_batch, ord="fro", axis=(1, 2))
    nmax  = float(norms.max()) if norms.size else 0.0
    s     = max(0, int(np.ceil(np.log2(max(nmax, 1e-9)))))
    scale = 2.0 ** s
    M_s   = M_batch / scale
    I      = np.broadcast_to(np.eye(D, dtype=M_batch.dtype), (N, D, D)).copy()
    result = I.copy()
    term   = I.copy()
    for k in range(1, K + 1):
        term   = (term @ M_s) / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 14 mp_* features to df. Rows before the rolling window are NaN.

    Each feature builds a DxD matrix M[t] from sign-weighted, panel-normalised
    OHLCV primitives over the rolling window defined in the JSON spec, applies
    a variant (raw/sym/skew), computes expm(M[t]) via scaling-and-squaring +
    Taylor K=16, and extracts a scalar (trace, Frobenius norm, or top-left
    element). Rows where any primitive is non-finite produce NaN output.
    """
    spec   = _mp_load_spec()
    window = spec["window"]
    scales = np.asarray(spec["primitive_scales"], dtype=np.float64)
    scales = np.where(scales > 0, scales, 1.0)

    prim_arr = _mp_compute_primitives(df, window)          # (N, 14)

    # Rows where any primitive is non-finite must produce NaN output; otherwise
    # the zero-replaced values below would yield expm(0) = I — bogus signal.
    nan_mask  = ~np.isfinite(prim_arr).all(axis=1)

    # Panel-normalise and bound entries to [-1, 1] via tanh (matches EDA).
    prim_norm = np.tanh(prim_arr / scales)
    prim_norm = np.nan_to_num(prim_norm, nan=0.0, posinf=0.0, neginf=0.0)

    N = prim_norm.shape[0]
    for feat in spec["features"]:
        D       = int(feat["d"])
        idx_mat = np.asarray(feat["idx_matrix"],  dtype=np.int64)
        sgn_mat = np.asarray(feat["sign_matrix"], dtype=np.float64)

        # M[t, i, j] = sgn_mat[i, j] * prim_norm[t, idx_mat[i, j]]
        M_all   = sgn_mat[None, :, :] * prim_norm[:, idx_mat]

        variant = feat["variant"]
        if variant == "sym":
            M_all = 0.5 * (M_all + M_all.transpose(0, 2, 1))
        elif variant == "skew":
            M_all = 0.5 * (M_all - M_all.transpose(0, 2, 1))
        elif variant != "raw":
            raise ValueError(f"Unknown matrix-power variant: {variant!r}")

        try:
            E_all = _mp_expm_batch(M_all)
        except Exception:
            E_all = np.broadcast_to(np.eye(D, dtype=np.float64), (N, D, D)).copy()

        extractor = feat["extractor"]
        if extractor == "trace":
            vals = E_all.trace(axis1=1, axis2=2).real
        elif extractor == "frob":
            vals = np.linalg.norm(E_all, ord="fro", axis=(1, 2))
        elif extractor == "top_left":
            v00  = E_all[:, 0, 0]
            vals = v00.real if np.iscomplexobj(v00) else v00
        else:
            raise ValueError(f"Unknown matrix-power extractor: {extractor!r}")

        vals = np.where(nan_mask, np.nan, vals)
        df[f"mp_{feat['name']}"] = vals

    return df
