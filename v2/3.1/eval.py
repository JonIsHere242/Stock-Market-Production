"""Multi-objective screening, the 10k -> 300 selection funnel, and correctness gates.

This is where breadth gets earned back as edge. Three pieces:

  * screen_multi  — scores every feature against EVERY ensemble target arm (not just
                    next-day return). Per (feature, arm): cross-sectional IC, IC-IR,
                    tail-lift, fold sign-stability, and a p-value.
  * select_features — the funnel: Benjamini-Hochberg FDR across ALL trials, a tail-lift
                    floor, and a multi-fold sign-stability gate. 10k candidates in,
                    a deflation-corrected shortlist out. More features is only good
                    if the selector is honest; this is the honest selector.
  * gates         — verify_bit_exact (your maxdiff<1e-9 vectorization oracle) and
                    causality_check (the truncation look-ahead test), as reusable fns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# stats primitives (no scipy dependency)
# --------------------------------------------------------------------------- #
def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float((x @ x) * (y @ y)))
    return float(x @ y) / d if d > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def tail_lift_xs(x: np.ndarray, y: np.ndarray, q: float = 0.1) -> float:
    """Cross-sectional top-q lift: P(target winner | feature in top q) / q."""
    n = len(x)
    k = max(1, int(round(q * n)))
    top_feat = np.argsort(-x)[:k]
    winner = y >= np.quantile(y, 1 - q)
    base = winner.mean()
    if base <= 0:
        return 1.0
    return float(winner[top_feat].mean() / base)


# --------------------------------------------------------------------------- #
# multi-objective screen
# --------------------------------------------------------------------------- #
def screen_multi(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    date_col: str = "Date",
    q: float = 0.1,
    n_folds: int = 5,
) -> pd.DataFrame:
    dates = np.sort(panel[date_col].unique())
    fold_of = {d: min(n_folds - 1, int(i / len(dates) * n_folds)) for i, d in enumerate(dates)}
    groups = list(panel.groupby(date_col))

    rows = []
    for f in feature_cols:
        for t in target_cols:
            per_day_ic, per_day_lift, fold_ic = [], [], [[] for _ in range(n_folds)]
            for d, g in groups:
                x = g[f].to_numpy(float)
                y = g[t].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 5:
                    continue
                ic = spearman(x[m], y[m])
                per_day_ic.append(ic)
                per_day_lift.append(tail_lift_xs(x[m], y[m], q))
                fold_ic[fold_of[d]].append(ic)
            if len(per_day_ic) < n_folds:
                continue
            ic = float(np.mean(per_day_ic))
            ir = ic / (np.std(per_day_ic) + 1e-12) * math.sqrt(len(per_day_ic))
            p = 2.0 * (1.0 - _norm_cdf(abs(ir)))
            fold_means = [np.mean(fi) for fi in fold_ic if fi]
            pos = sum(1 for fm in fold_means if fm > 0)
            sign_stability = max(pos, len(fold_means) - pos) / len(fold_means)
            rows.append(
                {
                    "feature": f,
                    "arm": t,
                    "ic": ic,
                    "ic_ir": ir,
                    "lift": float(np.mean(per_day_lift)),
                    "sign_stability": sign_stability,
                    "p": p,
                    "n_days": len(per_day_ic),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# selection funnel (FDR + tail floor + multi-fold stability)
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed_sorted = pvals[order] <= thresh
    kmax = np.where(passed_sorted)[0]
    mask = np.zeros(n, bool)
    if len(kmax):
        cutoff = order[: kmax.max() + 1]
        mask[cutoff] = True
    return mask


def select_features(
    screen: pd.DataFrame,
    *,
    n_target: int = 300,
    fdr_alpha: float = 0.10,
    min_lift: float = 1.05,
    min_sign_stability: float = 0.8,
) -> tuple[pd.DataFrame, dict]:
    s = screen.copy()
    funnel = {"trials": len(s), "features_in": s["feature"].nunique()}
    if s.empty:
        return s, funnel

    s["fdr_pass"] = benjamini_hochberg(s["p"].to_numpy(), fdr_alpha)
    s["pass"] = s["fdr_pass"] & (s["lift"] >= min_lift) & (s["sign_stability"] >= min_sign_stability)
    funnel["after_fdr"] = int(s["fdr_pass"].sum())
    funnel["after_all_gates"] = int(s["pass"].sum())

    survivors = s[s["pass"]].sort_values("lift", ascending=False).drop_duplicates("feature")
    funnel["features_surviving"] = len(survivors)
    shortlist = survivors.head(n_target).reset_index(drop=True)
    funnel["shortlisted"] = len(shortlist)
    return shortlist, funnel


# --------------------------------------------------------------------------- #
# correctness gates
# --------------------------------------------------------------------------- #
def verify_bit_exact(compute_a, compute_b, df: pd.DataFrame, cols: list[str], tol: float = 1e-9):
    a = compute_a(df.copy())
    b = compute_b(df.copy())
    maxdiff = 0.0
    for c in cols:
        va = np.asarray(a[c], float)
        vb = np.asarray(b[c], float)
        m = np.isfinite(va) & np.isfinite(vb)
        if m.any():
            maxdiff = max(maxdiff, float(np.max(np.abs(va[m] - vb[m]))))
        if (np.isfinite(va) != np.isfinite(vb)).any():
            return False, float("inf")
    return maxdiff < tol, maxdiff


def causality_check(block, df: pd.DataFrame, fracs=(0.6, 0.75, 0.9), tol: float = 1e-6) -> bool:
    """Recompute on a truncated prefix and assert past values are unchanged.
    Catches look-ahead: a causal feature must not depend on future bars."""
    full = block.run(df.copy())
    n = len(df)
    for fr in fracs:
        k = int(n * fr)
        if k < 5:
            continue
        trunc = block.run(df.iloc[:k].copy())
        for c in block.produces:
            a = np.asarray(full[c].to_numpy()[:k], float)
            b = np.asarray(trunc[c].to_numpy(), float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any() and np.max(np.abs(a[m] - b[m])) > tol:
                return False
    return True
