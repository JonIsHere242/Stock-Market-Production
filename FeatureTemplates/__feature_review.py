"""
__feature_review.py  --  thin runner for the Step-2 + Step-3 feature review.

WHY
---
The full feature review is two tools that must run in order:
  STEP 2  __relatedness_map.py  -- builds the ONE topo-correct panel (cached), the per-day
          codependence clusters, the stability/breadth headline, and -- critically -- the
          de-collinearized incumbent BASIS (Data/_eda_review/basis.json) that Step 3 needs.
  STEP 3  __tail_screen.py --neut_basis ... --rw_fdr  -- conditional tail-screen: residualize
          every candidate against the basis (own cluster keeper excluded), multi-seed
          cond_edge mean+/-std, within-day null, Romano-Wolf step-down FDR -> conditional.csv.

This runner just chains them with one consistent set of knobs and the courteous single-core
env. It is auto-skipped by the framework (leading __).

USAGE
-----
  stock_env\\Scripts\\python.exe FeatureTemplates\\__feature_review.py                  # FULL (all blocks)
  ...\\python.exe FeatureTemplates\\__feature_review.py --smoke                          # 15-block RAM-safe self-test
  ...\\python.exe FeatureTemplates\\__feature_review.py --n 350 --seeds 4 --null_shuffles 200

KNOBS:  --n tickers(350)  --seeds(4)  --null_shuffles(200)  --smoke
        --skip_map (reuse an existing basis.json, run Step 3 only)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
EDA_DIR = ROOT / "Data" / "_eda_review"
BASIS = EDA_DIR / "basis.json"


def discover_active_blocks():
    """All active (non-underscore) blocks -- the full candidate universe for Step 3."""
    return sorted(p.stem for p in HERE.glob("*.py") if not p.stem.startswith("_"))


SMOKE_BLOCKS = [
    "returns", "momentum_score", "price_differental_ratio",
    "price_differential_signal_pack", "volatility_indicators", "rsi",
    "amihud_size_illiquidity", "price_structure_metrics", "asym_updown_moves",
    "anchoring_52w_gh", "residual_momentum", "beta_dynamics", "variance_ratio",
    "capital_gains_overhang", "salience_theory_value",
]


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], env=os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(f"step failed (rc={r.returncode}): {cmd[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--null_shuffles", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_map", action="store_true",
                    help="reuse existing basis.json; run Step 3 only")
    args = ap.parse_args()

    if args.smoke:
        args.n = min(args.n, 120)
        args.seeds = min(args.seeds, 2)
        # NULL-FLOOR HONESTY FIX: the old smoke used 50 shuffles, which is BELOW the stable
        # regime -- a clean random column intermittently clears the (noisy) null floor at 50 but
        # is reliably rejected at >=200. __tail_screen only TRUSTS the standalone null floor at
        # >=200 shuffles; keep the smoke at the trusted floor so its `robust` verdict is honest.
        args.null_shuffles = max(args.null_shuffles, 200)
        args.folds = min(args.folds, 3)

    # ---- STEP 2: relatedness map (builds basis.json) ----------------------
    if not args.skip_map:
        cmd = [PY, str(HERE / "__relatedness_map.py"),
               "--n", args.n, "--seeds", args.seeds]
        if args.smoke:
            cmd.append("--smoke")
        run(cmd)

    if not BASIS.exists():
        raise SystemExit(f"no basis.json at {BASIS} -- run Step 2 first (drop --skip_map)")

    # ---- STEP 3: conditional tail-screen + Romano-Wolf --------------------
    if args.smoke:
        blocks = ",".join(SMOKE_BLOCKS)
    else:
        blocks = ",".join(discover_active_blocks())
    cmd = [PY, str(HERE / "__tail_screen.py"),
           "--blocks", blocks,
           "--n", args.n, "--folds", args.folds,
           "--seeds", args.seeds,
           "--null_shuffles", args.null_shuffles,
           "--neut_basis", str(BASIS),
           "--rw_fdr"]
    run(cmd)

    print(f"\n  feature review complete. outputs in {EDA_DIR}:")
    for f in ("clusters.csv", "basis.json", "block_map.csv", "conditional.csv"):
        p = EDA_DIR / f
        print(f"    {'OK ' if p.exists() else 'MISS'} {p}")


if __name__ == "__main__":
    main()
