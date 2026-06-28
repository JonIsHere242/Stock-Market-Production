"""Marginal, conditional feature selection (tail discrimination given incumbents).

The existing selector (eval.select_features) ranks each feature INDEPENDENTLY by its
cross-sectional tail-lift / IC, so five correlated views of the same signal all clear the
gate and you ship five copies of one idea. This module instead does greedy FORWARD
selection: at each step it orthogonalizes every remaining candidate against the
already-chosen features (within each date's cross-section) and keeps whichever residual
adds the most NEW top-q tail discrimination. A redundant twin scores ~1.0 once its parent
is in and is rejected. Mirrors eval.tail_lift_xs exactly; numpy + pandas only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def orthogonalize(candidate: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Residualize ``candidate`` (n,) against ``basis`` (n, m) by least squares.

    Returns ``candidate - basis_aug @ lstsq(basis_aug, candidate)`` where ``basis_aug``
    carries an intercept column, so the residual is orthogonal to the span of the basis
    plus a constant. An empty basis (m == 0) returns the candidate unchanged. lstsq's
    ``rcond`` guards rank-deficient / collinear bases.
    """
    candidate = np.asarray(candidate, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    n = candidate.shape[0]
    if basis.size == 0 or basis.shape[1] == 0:
        return candidate.copy()
    aug = np.column_stack([np.ones(n), basis])
    coef, *_ = np.linalg.lstsq(aug, candidate, rcond=None)
    return candidate - aug @ coef


def xs_tail_lift(feature: np.ndarray, target: np.ndarray, q: float = 0.1) -> float:
    """Cross-sectional top-q lift within one date: P(target winner | feature top-q) / q.

    Identical definition to eval.tail_lift_xs: pick the top-k feature names, count how
    many are also target winners (top-q of target), normalize by the base winner rate.
    Returns 1.0 (neutral) when the cross-section is degenerate.
    """
    x = np.asarray(feature, dtype=float)
    y = np.asarray(target, dtype=float)
    n = x.shape[0]
    if n < 2:
        return 1.0
    k = max(1, int(round(q * n)))
    top_feat = np.argsort(-x)[:k]
    winner = y >= np.quantile(y, 1 - q)
    base = winner.mean()
    if base <= 0:
        return 1.0
    return float(winner[top_feat].mean() / base)


# --------------------------------------------------------------------------- #
# marginal (conditional) score
# --------------------------------------------------------------------------- #
def marginal_score(
    panel: pd.DataFrame,
    candidate_col: str,
    selected_cols: list[str],
    target_col: str,
    date_col: str = "Date",
    q: float = 0.1,
) -> float:
    """Mean-over-dates tail-lift of the candidate's RESIDUAL given the incumbents.

    For each date: orthogonalize the candidate against the selected columns within that
    date's cross-section, then take xs_tail_lift of the residual vs the target. Averaging
    over dates yields the NEW tail discrimination the candidate contributes beyond the
    incumbents. With an empty ``selected_cols`` this is just the raw mean tail-lift.
    """
    lifts: list[float] = []
    for _, g in panel.groupby(date_col, sort=False):
        cand = g[candidate_col].to_numpy(dtype=float)
        y = g[target_col].to_numpy(dtype=float)
        if selected_cols:
            basis = g[selected_cols].to_numpy(dtype=float)
            m = np.isfinite(cand) & np.isfinite(y) & np.isfinite(basis).all(axis=1)
        else:
            basis = np.empty((cand.shape[0], 0))
            m = np.isfinite(cand) & np.isfinite(y)
        if m.sum() < 5:
            continue
        resid = orthogonalize(cand[m], basis[m])
        lifts.append(xs_tail_lift(resid, y[m], q))
    if not lifts:
        return 1.0
    return float(np.mean(lifts))


# --------------------------------------------------------------------------- #
# greedy forward selection
# --------------------------------------------------------------------------- #
def greedy_marginal_select(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str = "Date",
    k_max: int = 50,
    min_gain: float = 1.01,
    q: float = 0.1,
) -> dict:
    """Greedily add the feature with the highest CONDITIONAL marginal tail-lift.

    Start empty; each round score every remaining candidate's residual-vs-incumbents
    tail-lift and add the best one if its marginal lift >= ``min_gain``. Stop at
    ``k_max`` selections or when no candidate clears the floor.

    Cost is O(k_max * n_features * n_dates) tail-lift evaluations, each an lstsq over a
    single date's cross-section. To bound runtime on very large feature sets, candidates
    are pre-ranked once by raw (unconditional) marginal lift and at most ``MAX_CONSIDER``
    of them are carried into the greedy loop; the rest cannot lead and are dropped. Lower
    ``k_max`` / ``MAX_CONSIDER`` if the candidate pool is enormous.
    """
    MAX_CONSIDER = 400

    candidates = list(dict.fromkeys(feature_cols))  # de-dup, preserve order
    n_considered_total = len(candidates)

    # Pre-rank once by raw marginal lift; cap the working pool defensively.
    if len(candidates) > MAX_CONSIDER:
        raw = {c: marginal_score(panel, c, [], target_col, date_col, q) for c in candidates}
        candidates = sorted(candidates, key=lambda c: raw[c], reverse=True)[:MAX_CONSIDER]

    selected: list[str] = []
    trace: list[dict] = []
    remaining = list(candidates)

    while remaining and len(selected) < k_max:
        best_col = None
        best_lift = -np.inf
        for col in remaining:
            lift = marginal_score(panel, col, selected, target_col, date_col, q)
            if lift > best_lift:
                best_lift, best_col = lift, col
        if best_col is None or best_lift < min_gain:
            break
        selected.append(best_col)
        remaining.remove(best_col)
        trace.append(
            {"step": len(selected), "feature": best_col, "marginal_lift": float(best_lift)}
        )

    return {"selected": selected, "trace": trace, "n_considered": n_considered_total}


# --------------------------------------------------------------------------- #
# redundancy diagnostic
# --------------------------------------------------------------------------- #
def redundancy_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "Date",
) -> pd.DataFrame:
    """Average pooled absolute correlation of each feature with all the others.

    A high value flags a feature that duplicates information already present elsewhere in
    the set. Returns columns [feature, mean_abs_corr] sorted descending (most redundant
    first). Correlation is pooled across the whole panel (the ``date_col`` is ignored for
    the correlation itself but excluded from the feature matrix).
    """
    cols = [c for c in feature_cols if c != date_col]
    mat = panel[cols].to_numpy(dtype=float)
    corr = np.abs(np.corrcoef(mat, rowvar=False))
    np.fill_diagonal(corr, np.nan)
    mean_abs = np.nanmean(corr, axis=1)
    out = pd.DataFrame({"feature": cols, "mean_abs_corr": mean_abs})
    return out.sort_values("mean_abs_corr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# demo / smoke test
# --------------------------------------------------------------------------- #
def _build_synthetic_panel(seed: int = 7, n_dates: int = 30, n_tickers: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        signal_a = rng.standard_normal(n_tickers)   # first latent driver
        signal_c = rng.standard_normal(n_tickers)   # independent second driver
        target = 1.2 * signal_a + 1.0 * signal_c + rng.standard_normal(n_tickers)
        feat_a = signal_a + 0.30 * rng.standard_normal(n_tickers)
        feat_b = 0.95 * feat_a + 0.02 * rng.standard_normal(n_tickers)  # redundant twin of A
        feat_c = signal_c + 0.30 * rng.standard_normal(n_tickers)       # independent signal
        block = {
            "Date": np.full(n_tickers, d),
            "Ticker": np.arange(n_tickers),
            "fwd_ret": target,
            "feat_A": feat_a,
            "feat_B": feat_b,
            "feat_C": feat_c,
        }
        for j in range(4):  # pure-noise distractors
            block[f"noise_{j}"] = rng.standard_normal(n_tickers)
        rows.append(pd.DataFrame(block))
    return pd.concat(rows, ignore_index=True)


def _naive_top_by_lift(panel: pd.DataFrame, feature_cols, target_col, date_col="Date", q=0.1):
    """Independent ranking that double-counts: each feature scored on its own."""
    scored = [(c, marginal_score(panel, c, [], target_col, date_col, q)) for c in feature_cols]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


if __name__ == "__main__":
    panel = _build_synthetic_panel()
    feats = ["feat_A", "feat_B", "feat_C", "noise_0", "noise_1", "noise_2", "noise_3"]
    target = "fwd_ret"

    print("=== naive independent top-by-lift (double-counts A and B) ===")
    naive = _naive_top_by_lift(panel, feats, target)
    naive_top3 = [c for c, _ in naive[:3]]
    for c, lift in naive:
        print(f"  {c:9s} raw_lift={lift:.3f}")
    print(f"  naive top-3: {naive_top3}")

    print("\n=== greedy MARGINAL selection (conditional on incumbents) ===")
    res = greedy_marginal_select(panel, feats, target, k_max=5, min_gain=1.20, q=0.1)
    for row in res["trace"]:
        print(f"  step {row['step']}: {row['feature']:9s} marginal_lift={row['marginal_lift']:.3f}")
    print(f"  selected: {res['selected']}  (n_considered={res['n_considered']})")

    print("\n=== redundancy report ===")
    print(redundancy_report(panel, feats).to_string(index=False))

    sel = set(res["selected"])
    not_both_ab = not ({"feat_A", "feat_B"} <= sel)
    has_c = "feat_C" in sel
    naive_picked_both_ab = {"feat_A", "feat_B"} <= set(naive_top3)
    print("\n=== assertions ===")
    print(f"  greedy did NOT pick both A and B : {not_both_ab}")
    print(f"  greedy picked C                  : {has_c}")
    print(f"  naive WOULD pick both A and B    : {naive_picked_both_ab}")

    assert not_both_ab, "FAIL: greedy selected the redundant twin (both A and B)"
    assert has_c, "FAIL: greedy missed the independent signal C"
    print("\nPASS: marginal selection rejected the redundant twin and kept the orthogonal signal")
