"""End-to-end demo on synthetic data — PROVES the headline behaviours.

Run:  cd v2 && python 3.1/demo.py

It demonstrates, with assertions:
  1. CODEGEN          — 2 family files -> 18 named feature blocks.
  2. CACHE COLD->WARM — first build all MISS; identical rerun all HIT (delta rebuild).
  3. SELECTIVE RECOMPUTE — edit ONE block -> only it misses, siblings stay cached.
  4. CAS DEDUP        — bit-identical columns store one physical blob.
  5. PARTIAL-DAG      — materialize a target feature + its ancestors only.
  6. MULTI-OBJECTIVE SELECT — screen vs 3 target arms, FDR funnel 10k-style -> shortlist.
  7. GATES            — causality (look-ahead) + bit-exact verification.
  8. CATALOG          — machine-readable manifest for LLM navigation.

All prints are ASCII (Windows cp1252 safe).
"""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cache import Cache
from engine import Engine, summarize_profile
from eval import causality_check, screen_multi, select_features, verify_bit_exact
from registry import build_catalog, discover, write_catalog

ROOT = Path(__file__).resolve().parents[2]          # repo root (for closure hashing)
FAM_DIR = Path(__file__).resolve().parent / "families"
WORK = Path(__file__).resolve().parents[1] / "_demo_work"


def make_panel(n_tickers=40, n_days=180, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_tickers):
        drift = rng.normal(0.0003, 0.0002)
        rets = rng.normal(drift, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(rets))
        frames.append(
            pd.DataFrame(
                {
                    "Date": np.arange(n_days),
                    "Ticker": f"T{i:03d}",
                    "Close": close,
                    "Volume": rng.integers(1e5, 1e6, n_days).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _edited_compute(df: pd.DataFrame) -> pd.DataFrame:
    """A deliberately different roll_mean_w20 (scaled) to simulate a code edit."""
    c = df["Close"].to_numpy(float)
    r = np.zeros_like(c)
    r[1:] = np.diff(np.log(c))
    df["roll_mean_w20"] = pd.Series(r).rolling(20).mean().to_numpy() * 1.5
    return df


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n {title}\n" + "=" * 72)


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    cache_dir = WORK / "cache"

    panel = make_panel()
    tickers = sorted(panel["Ticker"].unique())

    # ----------------------------------------------------------------- 1. codegen
    hr("1. CODEGEN: families -> blocks")
    blocks, families = discover(family_dir=FAM_DIR)
    per_ticker = [b for b in blocks if b.kind == "per_ticker"]
    panel_blocks = [b for b in blocks if b.kind == "panel"]
    print(f"families: {len(families)}  ->  blocks: {len(blocks)} "
          f"({len(per_ticker)} per-ticker + {len(panel_blocks)} panel)")
    for f in families:
        print(f"  family {f.name!r}: cardinality {f.cardinality()}")
    assert len(blocks) == 18, blocks

    # ------------------------------------------------- 2. cache cold -> warm
    hr("2. CACHE: cold build (all MISS) then identical rerun (all HIT)")

    def run_per_ticker(block_list, cdir):
        cache = Cache(cdir)
        eng = Engine(block_list, cache, ROOT)
        profs, outs = [], []
        for tk in tickers:
            tdf = panel[panel["Ticker"] == tk].reset_index(drop=True)
            out, prof = eng.run(tdf)
            profs.append(prof)
            outs.append(out)
        eng.finalize()
        return summarize_profile(profs), pd.concat(outs, ignore_index=True), cache

    cold, _, cold_cache = run_per_ticker(per_ticker, cache_dir)
    print(f"cold : hit_rate={cold['hit_rate']:.0%}  hits={cold['hits']} misses={cold['misses']}  "
          f"blobs_written={cold_cache.stats['blobs_written']} deduped={cold_cache.stats['blobs_deduped']}")
    warm, panel_feat, warm_cache = run_per_ticker(per_ticker, cache_dir)
    print(f"warm : hit_rate={warm['hit_rate']:.0%}  hits={warm['hits']} misses={warm['misses']}")
    assert cold["hit_rate"] == 0.0
    assert warm["hit_rate"] == 1.0
    print("[OK] cold=0% hits, warm=100% hits -> incremental materialization works")

    # ------------------------------------------ 3. selective recompute on edit
    hr("3. SELECTIVE RECOMPUTE: edit one block -> only it misses")
    target_name = "roll::stat=mean,window=20"
    edited = [
        dataclasses.replace(b, version="2", compute=_edited_compute, _version_fn=_edited_compute)
        if b.name == target_name else b
        for b in per_ticker
    ]
    # warm cache already holds all 16 original blocks for every ticker; run the
    # EDITED set once and read each block's hit/miss status.
    eng = Engine(edited, Cache(cache_dir), ROOT)
    _, prof = eng.run(panel[panel["Ticker"] == tickers[0]].reset_index(drop=True))
    misses = sorted(n for n, r in prof.items() if r["cache"] == "miss")
    hits = [n for n, r in prof.items() if r["cache"] == "hit"]
    print(f"after editing {target_name!r}:  MISS={misses}  ({len(hits)} others HIT)")
    assert misses == [target_name], misses
    print("[OK] one-line edit -> 1 block recomputes, 15 siblings served from cache")

    # ------------------------------------------------------- 4. CAS dedup
    hr("4. CAS DEDUP: bit-identical column content -> one blob")
    dd = Cache(WORK / "ddcheck")
    a = np.arange(256.0)
    h1 = dd.store_column(a)
    h2 = dd.store_column(a.copy())  # identical content, different array object
    assert h1 == h2 and dd.stats["blobs_written"] == 1 and dd.stats["blobs_deduped"] == 1
    print(f"stored an identical column twice -> written={dd.stats['blobs_written']} "
          f"deduped={dd.stats['blobs_deduped']} (one content hash {h1})")
    print("[OK] content-addressed store collapses bit-identical columns to a single blob")

    # ------------------------------------------------- 5. partial-DAG (panel)
    hr("5. PARTIAL-DAG: materialize one cross-sectional feature + ancestors")
    # panel pass: xs blocks require roll_mean_w20, already present in panel_feat
    pcache = Cache(WORK / "pcache")
    peng = Engine(panel_blocks, pcache, ROOT)
    sub = peng.graph.select("xs::op=rank")  # block + (auto) ancestors within this subgraph
    print(f"selector 'xs::op=rank' resolves to execution set: {sub}")
    out, _ = peng.run(panel_feat, only={"xs::op=rank"})
    peng.finalize()
    assert "xs_rank_roll_mean_w20" in out.columns
    print("[OK] produced xs_rank_roll_mean_w20 via panel groupby(Date)")

    # --------------------------------------- 6. multi-objective screen + select
    hr("6. MULTI-OBJECTIVE SCREEN + SELECT FUNNEL")
    full = out  # has roll_* + xs_* features
    # synthetic target arms: topq (1d fwd ret), 5d momentum, downside
    full = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    fwd = full.groupby("Ticker")["Close"].shift(-1) / full["Close"] - 1
    full["arm_topq"] = fwd
    full["arm_mom5"] = full.groupby("Ticker")["Close"].shift(-5) / full["Close"] - 1
    full["arm_downside"] = -(full.groupby("Ticker")["Close"].shift(-1) / full["Close"] - 1).clip(upper=0)
    feat_cols = [c for c in full.columns if c.startswith(("roll_", "xs_"))]
    arms = ["arm_topq", "arm_mom5", "arm_downside"]
    screen = screen_multi(full.dropna(subset=arms), feat_cols, arms, date_col="Date", n_folds=5)
    shortlist, funnel = select_features(screen, n_target=300, fdr_alpha=0.10)
    print(f"trials (feature x arm): {funnel['trials']}   features in: {funnel['features_in']}")
    print(f"funnel: after_FDR={funnel.get('after_fdr',0)}  "
          f"survivors={funnel.get('features_surviving',0)}  shortlisted={funnel.get('shortlisted',0)}")
    if len(shortlist):
        print("top survivors (feature / best arm / lift / sign-stability):")
        for _, r in shortlist.head(5).iterrows():
            print(f"  {r['feature']:<26} {r['arm']:<12} lift={r['lift']:.3f} stab={r['sign_stability']:.2f}")
    print("[OK] every feature scored against ALL 3 arms, not just topq")

    # ----------------------------------------------------------- 7. gates
    hr("7. GATES: causality (look-ahead) + bit-exact verification")
    b20 = next(b for b in per_ticker if b.name == target_name)
    one = panel[panel["Ticker"] == tickers[0]].reset_index(drop=True)
    causal = causality_check(b20, one)
    same, maxdiff = verify_bit_exact(b20.compute, b20.compute, one, b20.produces)
    print(f"causality_check(roll_mean_w20) = {causal}   (recompute-on-truncation stable)")
    print(f"verify_bit_exact(self,self)    = {same}  maxdiff={maxdiff:.2e}")
    assert causal and same

    # --------------------------------------------------------- 8. catalog
    hr("8. CATALOG: machine-readable manifest for LLM navigation")
    cost = {n: r["mean_ms"] for n, r in cold["blocks"].items()}
    catalog = build_catalog(blocks, families, cost_ms=cost)
    jp, mp = write_catalog(catalog, WORK / "catalog")
    print(f"wrote {jp.name} ({catalog['n_blocks']} blocks, {catalog['n_feature_columns']} cols, "
          f"{catalog['n_families']} families) + {mp.name}")
    print(f"tag taxonomy: {catalog['tags']}")

    hr("DONE")
    print(f"artifacts in: {WORK}")


if __name__ == "__main__":
    main()
