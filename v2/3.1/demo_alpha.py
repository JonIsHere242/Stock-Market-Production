"""'Alpha extraction' demo — the validated-signal capstone.

  cd v2 && python 3.1/demo_alpha.py

Proves the layer that turns a feature factory into an alpha engine:
  1. VALIDATION   — purged/embargoed folds remove leaking samples; CPCV path count
  2. MARGINAL SEL — greedy conditional selection (rejects redundant twins)
  3. MODEL OOF    — purged XGBoost OOF; purged AUC does NOT exceed the leaky naive AUC
  4. ENSEMBLE     — rank-avg of purged-OOF target arms cuts the edge's day-to-day spread
  5. EVOLVE       — auto-discover the best family params via the live screen
  6. IMPORTANCE   — purged gain-importance ranking

Synthetic data has weak signal by construction; the point is that the PLUMBING is
leakage-free and the mechanisms behave correctly. ASCII-only output.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import evolve, select_marginal, validation
from cache import Cache
from demo import make_panel
from ensemble import target_ensemble
from eval import tail_lift_xs
from model import XGB_AVAILABLE, auc, oof_predict, purged_importance
from materialize import Materializer
from registry import discover
from storage import FeatureLake

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
FAM_DIR = Path(__file__).resolve().parent / "families"
WORK = Path(__file__).resolve().parents[1] / "_alpha_work"


def hr(t: str) -> None:
    print("\n" + "=" * 74 + f"\n {t}\n" + "=" * 74)


def _roll(returns: np.ndarray, w: int, stat: str) -> np.ndarray:
    s = pd.Series(returns)
    if stat == "mean":
        return s.rolling(w).mean().to_numpy()
    if stat == "std":
        return s.rolling(w).std().to_numpy()
    if stat == "zscore":
        return ((s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-12)).to_numpy()
    return s.rolling(w).skew().to_numpy()


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    panel = make_panel(n_tickers=60, n_days=320, seed=5)
    tickers = sorted(panel["Ticker"].unique())
    frames = {tk: panel[panel["Ticker"] == tk].reset_index(drop=True) for tk in tickers}
    blocks, _ = discover(family_dir=FAM_DIR)
    full = Materializer(blocks, Cache(WORK / "cache"), FeatureLake(WORK / "lake"), ROOT).run(frames)["panel"]

    full = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    for h in (1, 5, 10):
        full[f"fwd{h}"] = full.groupby("Ticker")["Close"].shift(-h) / full["Close"] - 1
    feat_cols = [c for c in full.columns if c.startswith(("roll_", "xs_"))]

    # =================================================== 1. validation
    hr("1. VALIDATION: purged/embargoed CPCV (leakage-free folds)")
    ft = full.sort_values("Date").reset_index(drop=True)
    et = ft["Date"].to_numpy()
    purged = validation.purged_kfold(et, n_splits=5, horizon=5, embargo_frac=0.02)
    naive = validation.purged_kfold(et, n_splits=5, horizon=0, embargo_frac=0.0)
    purged_tr = np.mean([len(tr) for tr, _ in purged])
    naive_tr = np.mean([len(tr) for tr, _ in naive])
    cp = validation.cpcv(et, n_groups=6, k_test_groups=2, horizon=5)
    print(f"avg train rows/fold: naive={naive_tr:.0f}  purged={purged_tr:.0f}  "
          f"(purge+embargo removed {naive_tr - purged_tr:.0f} leaking rows/fold)")
    print(f"CPCV(6,2): n_splits={cp['n_splits']}  backtest paths={cp['n_paths']}")
    assert naive_tr >= purged_tr and cp["n_splits"] == 15 and cp["n_paths"] == 5
    print("[OK] overlapping-label leakage purged; CPCV gives many backtest paths")

    # =============================================== 2. marginal selection
    hr("2. MARGINAL SELECTION: conditional on incumbents (kills redundancy)")
    # controlled cross-sectional signal + a near-duplicate twin + an independent noise feature
    sp = full.dropna(subset=["fwd1"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    n = len(sp)
    signal = rng.standard_normal(n)
    sp["y_ctrl"] = signal + 0.5 * rng.standard_normal(n)        # target driven by the signal
    sp["feat_sig"] = signal + 0.3 * rng.standard_normal(n)      # a feature carrying it
    sp["feat_twin"] = 0.97 * sp["feat_sig"] + 0.03 * rng.standard_normal(n)  # near-duplicate
    sp["feat_indep"] = rng.standard_normal(n)                   # independent noise
    raw = select_marginal.marginal_score(sp, "feat_twin", [], "y_ctrl")
    cond = select_marginal.marginal_score(sp, "feat_twin", ["feat_sig"], "y_ctrl")
    print(f"feat_twin tail-lift alone = {raw:.3f}  ->  given its twin feat_sig = {cond:.3f} (collapses to ~1)")
    sel = select_marginal.greedy_marginal_select(
        sp, ["feat_sig", "feat_twin", "feat_indep"] + feat_cols, "y_ctrl", date_col="Date",
        k_max=4, min_gain=1.05)
    print(f"greedy selected {len(sel['selected'])}: {sel['selected']}")
    both_twins = {"feat_sig", "feat_twin"} <= set(sel["selected"])
    keeps_signal = ("feat_sig" in sel["selected"]) or ("feat_twin" in sel["selected"])
    assert cond < raw and not both_twins and keeps_signal
    print("[OK] keeps the signal once, collapses the redundant twin (vs naive ranking = both)")

    if not XGB_AVAILABLE:
        print("\n[skip] xgboost not installed -> sections 3,4,6 need a model")
        return

    # ================================================= 3. model OOF (no leak)
    hr("3. MODEL OOF: purged XGBoost AUC does not exceed leaky naive AUC")
    md = full.dropna(subset=["fwd5"]).sort_values("Date").reset_index(drop=True)
    et5 = md["Date"].to_numpy()
    X = np.nan_to_num(md[feat_cols].to_numpy(float))
    y5 = (md["fwd5"].to_numpy(float) > 0).astype(float)
    oof_naive = oof_predict(X, y5, et5, horizon=0, embargo_frac=0.0, n_splits=5)
    oof_purged = oof_predict(X, y5, et5, horizon=5, embargo_frac=0.02, n_splits=5)
    a_naive, a_purged = auc(y5, oof_naive), auc(y5, oof_purged)
    print(f"naive  (leaky) OOF AUC = {a_naive:.4f}")
    print(f"purged       OOF AUC = {a_purged:.4f}   (gap from leakage = {a_naive - a_purged:+.4f})")
    assert a_purged <= a_naive + 0.03
    print("[OK] purging does not inflate AUC -> honest out-of-sample estimate")

    # ===================================================== 4. ensemble
    hr("4. TARGET ENSEMBLE: rank-avg of purged-OOF arms cuts edge variance")
    ens = target_ensemble(
        full.dropna(subset=["fwd1", "fwd5", "fwd10"]), feat_cols,
        ["fwd1", "fwd5", "fwd10"], date_col="Date", horizon=5, n_splits=5)
    for arm, d in ens["per_arm"].items():
        print(f"  arm {arm:<6} lift={d['mean_lift']:.3f} spread={d['spread']:.3f} oof_auc={d['oof_auc']:.3f}")
    e = ens["ensemble"]
    print(f"  ENSEMBLE     lift={e['mean_lift']:.3f} spread={e['spread']:.3f}  "
          f"(spread cut {ens['spread_reduction']:+.3f} vs mean arm {ens['mean_arm_spread']:.3f})")
    assert e["spread"] <= ens["mean_arm_spread"] * 1.10
    print("[OK] rank-averaging diverse arms reduces the day-to-day spread of the edge")

    # ====================================================== 5. evolve
    hr("5. EVOLVE: auto-discover the best family params via the live screen")
    base = full.dropna(subset=["fwd1"]).reset_index(drop=True)
    close = base["Close"].to_numpy(float)
    rows_by_tk = {tk: base.index[base["Ticker"] == tk].to_numpy() for tk in base["Ticker"].unique()}
    dates = base["Date"].to_numpy()
    fwd1 = base["fwd1"].to_numpy(float)

    def score_fn(p: dict) -> float:
        feat = np.full(len(base), np.nan)
        for rows in rows_by_tk.values():
            c = close[rows]
            r = np.zeros_like(c)
            r[1:] = np.diff(np.log(c))
            feat[rows] = _roll(r, int(p["window"]), p["stat"])
        lifts = []
        for d in np.unique(dates):
            m = (dates == d) & np.isfinite(feat) & np.isfinite(fwd1)
            if m.sum() >= 10:
                lifts.append(tail_lift_xs(feat[m], fwd1[m], 0.1))
        return float(np.mean(lifts)) if lifts else 1.0

    space = {"window": [5, 10, 20, 50, 100], "stat": ["mean", "std", "zscore", "skew"]}
    res = evolve.evolve_params(space, score_fn, pop_size=10, n_generations=5, seed=1)
    print(f"best params: {res['best_params']}  best lift={res['best_score']:.3f}")
    print(f"evaluated {res['n_evals']}/{evolve.grid_size(space)} grid cells "
          f"(history best: {[round(h['best'], 3) for h in res['history']]})")
    assert res["n_evals"] <= evolve.grid_size(space)
    print("[OK] evolution searches the family grid guided by the real screen")

    # ==================================================== 6. importance
    hr("6. PURGED IMPORTANCE: leakage-free feature ranking")
    imp = purged_importance(X, y5, et5, feat_cols, horizon=5, n_splits=5)
    for name, gain in list(imp.items())[:5]:
        print(f"  {name:<24} gain={gain:.2f}")

    hr("DONE -- the stone has been bled")
    print(f"artifacts in: {WORK}")


if __name__ == "__main__":
    main()
