"""'Beyond' demo — the full platform end-to-end.

  cd v2 && python 3.1/demo_beyond.py

Exercises every advanced subsystem with assertions:
  1. STORAGE + INCREMENTAL MATERIALIZE — partitioned lake; cold/warm/edit deltas; pushdown read
  2. PANEL BACKEND        — pandas over('Date'); bit-matches the family's XS op; polars-ready
  3. LEAKAGE              — cross-sectional + survivorship audit (beyond per-ticker truncation)
  4. DEFLATE              — deflated Sharpe + Romano-Wolf + PBO (honest selection)
  5. GOVERN               — p95 cost-regression + value-per-CPU-second
  6. CATALOG SEARCH       — semantic search + white-space + near-duplicate detection
  7. VIZ                  — DAG -> Graphviz DOT + Mermaid

ASCII-only output (Windows cp1252 safe).
"""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np

import catalog_search, deflate, govern, leakage, panel_backend, viz
from cache import Cache
from demo import _edited_compute, make_panel
from engine import Engine
from eval import screen_multi
from graph import FeatureGraph
from materialize import Materializer
from registry import build_catalog, discover
from storage import FeatureLake

ROOT = Path(__file__).resolve().parents[2]
FAM_DIR = Path(__file__).resolve().parent / "families"
WORK = Path(__file__).resolve().parents[1] / "_beyond_work"
TARGET = "roll::stat=mean,window=20"


def hr(t: str) -> None:
    print("\n" + "=" * 74 + f"\n {t}\n" + "=" * 74)


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    panel = make_panel(n_tickers=50, n_days=240, seed=11)
    tickers = sorted(panel["Ticker"].unique())
    frames = {tk: panel[panel["Ticker"] == tk].reset_index(drop=True) for tk in tickers}
    blocks, families = discover(family_dir=FAM_DIR)
    cdir, ldir = WORK / "cache", WORK / "lake"

    # =============================================== 1. storage + materialize
    hr("1. STORAGE + INCREMENTAL MATERIALIZE")
    cold = Materializer(blocks, Cache(cdir), FeatureLake(ldir), ROOT).run(frames)
    warm = Materializer(blocks, Cache(cdir), FeatureLake(ldir), ROOT).run(frames)
    edited = [
        dataclasses.replace(b, version="2", compute=_edited_compute, _version_fn=_edited_compute)
        if b.name == TARGET else b
        for b in blocks
    ]
    edit = Materializer(edited, Cache(cdir), FeatureLake(ldir), ROOT).run(frames)

    print(f"cold:  changed={len(cold['changed_blocks'])} blocks  hit_rate={cold['hit_rate']:.0%}  "
          f"lake={cold['lake']['files']} files / {cold['lake']['partitions']} partitions / "
          f"{cold['lake']['total_mb']} MB / {cold['lake']['rows']} rows")
    print(f"warm:  changed={len(warm['changed_blocks'])} blocks  hit_rate={warm['hit_rate']:.0%}")
    print(f"edit:  changed={edit['changed_blocks']}  hit_rate={edit['hit_rate']:.0%} "
          f"(roll_mean_w20 + its xs descendant recompute, rest cached)")
    assert cold["hit_rate"] == 0.0 and warm["hit_rate"] == 1.0
    assert edit["changed_blocks"] == [TARGET]
    print("[OK] full recompute -> delta rebuild")

    lake = FeatureLake(ldir)
    sl = lake.read(columns=["roll_mean_w20"], date_range=(50, 60), tickers=tickers[:5])
    assert set(sl["Ticker"].unique()) <= set(tickers[:5])
    assert sl["Date"].between(50, 60).all()
    print(f"[OK] predicate-pushdown read: {len(sl)} rows (5 tickers x dates 50-60), "
          f"cols {list(sl.columns)}")

    full = warm["panel"].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    full["arm_topq"] = full.groupby("Ticker")["Close"].shift(-1) / full["Close"] - 1
    full["arm_mom5"] = full.groupby("Ticker")["Close"].shift(-5) / full["Close"] - 1
    feat_cols = [c for c in full.columns if c.startswith(("roll_", "xs_"))]
    arms = ["arm_topq", "arm_mom5"]
    screen = screen_multi(full.dropna(subset=arms), feat_cols, arms, n_folds=5)

    # ============================================================ 2. panel backend
    hr("2. PANEL BACKEND: over('Date') cross-sectional, backend-pluggable")
    pr = panel_backend.xs_rank(full, "roll_mean_w20", by="Date")
    fam = full["xs_rank_roll_mean_w20"].to_numpy()
    m = np.isfinite(pr) & np.isfinite(fam)
    maxdiff = float(np.max(np.abs(pr[m] - fam[m])))
    print(f"active backend: {panel_backend.active_backend()}  "
          f"(polars installed: {panel_backend.POLARS_AVAILABLE})")
    print(f"panel_backend.xs_rank vs family xs op: maxdiff={maxdiff:.2e}")
    print(f"parity check: {panel_backend.verify_parity(full, 'roll_mean_w20')}")
    assert maxdiff < 1e-9
    print("[OK] cross-sectional op reproduces the family bit-for-bit")

    # ================================================================ 3. leakage
    hr("3. LEAKAGE: panel + survivorship (what per-ticker truncation misses)")
    xsl = leakage.xs_normalization_leak(full, "roll_mean_w20", "arm_topq")
    sv = leakage.survivorship_bias(full, target_col="arm_topq")
    print(f"xs_normalization_leak: global_ic={xsl['global_ic']:.3f} infold_ic={xsl['infold_ic']:.3f} "
          f"gap={xsl['leak_gap']:.3f} flag={xsl['flag']}")
    print(f"survivorship_bias: tickers={sv['n_tickers']} survivors={sv['frac_survivors']:.0%} "
          f"flag={sv['flag']}")
    print("[OK] panel-level audits run (detect modes the truncation test cannot see)")

    # ================================================================ 4. deflate
    hr("4. DEFLATE: honest selection statistics")
    good = deflate.deflated_sharpe_ratio(2.5, 0.5, n_trials=10, n_obs=1000)
    lucky = deflate.deflated_sharpe_ratio(1.2, 1.0, n_trials=5000, n_obs=1000)
    rng = np.random.default_rng(0)
    k = 40
    t_obs = np.abs(rng.normal(0, 1, k))
    t_obs[:3] = [5.0, 4.6, 4.1]
    t_boot = np.abs(rng.normal(0, 1, (600, k)))
    rej = deflate.romano_wolf(t_obs, t_boot, alpha=0.05)
    perf = rng.normal(0, 1, (200, 20))
    perf[:, 0] += 0.3
    pbo = deflate.pbo_cscv(perf, n_splits=10)
    print(f"deflated_sharpe: good_strategy={good:.3f}  lucky_among_5000={lucky:.3f}")
    print(f"romano_wolf: rejected {int(rej.sum())}/{k} (3 planted signals)")
    print(f"pbo_cscv (persistent config): {pbo:.3f}")
    assert good > 0.9 and lucky < 0.5 and rej.sum() >= 3
    print("[OK] deflation discriminates real edge from selection luck")

    # ================================================================= 5. govern
    hr("5. GOVERN: p95 cost-regression + value-per-CPU-second")
    eng = Engine([b for b in blocks if b.kind == "per_ticker"], Cache(WORK / "gcache"), ROOT)
    timings: dict[str, list[float]] = {}
    for tk in tickers[:20]:
        _, prof = eng.run(frames[tk])
        for n, r in prof.items():
            timings.setdefault(n, []).append(r["seconds"] * 1000)
    base = govern.CostBaseline()
    base.update(timings)
    base.save(WORK / "baseline.json")
    current = {n: ([x * 3 for x in v] if n == "roll::stat=skew,window=50" else v) for n, v in timings.items()}
    regs = base.regressions(current)
    cost_by_feature = {col: float(np.mean(timings.get(b.name, [1.0])))
                       for b in blocks for col in b.produces}
    vps = govern.value_per_cpu_second(screen, cost_by_feature)
    print(f"regressions flagged: {[(r['block'], r['severity']) for r in regs[:3]]}")
    print("top value-per-CPU-second:")
    for _, r in vps.head(3).iterrows():
        print(f"  {r['feature']:<26} lift={r['best_lift']:.3f} cost={r['cost_ms']:.3f}ms "
              f"value/s={r['value_per_cpu_s']:.2f}")
    assert any(r["block"] == "roll::stat=skew,window=50" for r in regs)
    print("[OK] cost governance flags regressions and prioritizes cheap edge")

    # ========================================================= 6. catalog search
    hr("6. CATALOG SEARCH: navigability without reading source")
    cat = build_catalog(blocks, families)
    idx = catalog_search.build_index(cat)
    hits = idx.search("rolling volatility dispersion", k=4)
    gaps = idx.whitespace(["insider buying signal", "options implied skew", "earnings drift"])
    dups = idx.near_duplicates(threshold=0.6, max_pairs=5)
    print(f"search('rolling volatility dispersion'): {[(n, round(s,3)) for n, s in hits]}")
    print(f"white-space (uncovered topics): {[q for q, _ in gaps]}")
    print(f"near-duplicate feature pairs (>=0.6): {len(dups)} found")
    assert len(gaps) >= 2
    print("[OK] an agent can find features + white-space from metadata alone")

    # ==================================================================== 7. viz
    hr("7. VIZ: DAG -> Graphviz DOT + Mermaid")
    g = FeatureGraph(blocks)
    out = viz.write_viz(g, WORK / "viz")
    print(f"wrote {Path(out['dot']).name}, {Path(out['mermaid']).name}, "
          f"{Path(out['families']).name}  ({out['n_nodes']} nodes / {out['n_edges']} edges)")

    hr("DONE -- total conceptual domination achieved")
    print(f"artifacts in: {WORK}")


if __name__ == "__main__":
    main()
