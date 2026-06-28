# 3.1__FeatureFramework

**A content-addressed, codegen-first feature engine for quantitative research.**
*The v3.1 successor to `3__FeatureFramework.py` — same shape (a numbered orchestrator
script `3.1__FeatureFramework.py` + a sibling code folder `3.1/`, mirroring
`3__FeatureFramework.py` + `FeatureTemplates/`).*
Apache Hamilton's spine — block/function DAG, topological scheduling, content-addressed
caching — plus the four things Hamilton doesn't attempt, built for a feature library
heading to 10TB of data and millions of features.

> Drop-in compatible with the existing `FeatureTemplates/` contract (`METADATA + compute(df)`).
> It ingests the current library **unchanged** — verified live on **206 blocks / 1451 columns**.

---

## Why this exists

Hamilton proved the architecture: model each feature as a node, infer a DAG, cache on
`hash(code + upstream data versions)`, execute only what's needed. This framework adopts
that spine and then closes the gaps that bite at scale:

| Hamilton stops at | 3.1__FeatureFramework adds |
|---|---|
| hashes a node's OWN source | **transitive helper-closure hashing** — edit `Util.py`/`_marketcap.py` and dependents correctly invalidate (Hamilton's silent stale-cache hole, closed) |
| executes features you wrote | **parametrized feature-family codegen** — 10k features from hundreds of LOC, each still named, profiled, cached, verifiable |
| single-target lineage | **multi-objective screening** across every ensemble target arm, not just next-day return |
| viz/lineage from code | a **machine-readable catalog** so an LLM maps white-space without reading the source |
| — | a **10k -> 300 selection funnel**: Benjamini-Hochberg FDR + tail-lift floor + multi-fold sign-stability |

The thesis: **breadth is cheap, selection is everything.** Generate features with codegen,
then filter brutally. Code volume is a cost; edge comes from honest selection.

---

## Quick start

```bash
cd v2
python 3.1__FeatureFramework.py demo                    # synthetic end-to-end proof
python 3.1__FeatureFramework.py list                    # show families -> blocks
python 3.1__FeatureFramework.py catalog --legacy-dir ../FeatureTemplates --out _catalog
# any module also runs directly (Python auto-adds 3.1/ to sys.path):
python 3.1/demo_alpha.py        # or: python 3.1/deflate.py , etc.
```

## What the demo proves (with assertions)

```
1. CODEGEN            2 family files            -> 18 named feature blocks
2. CACHE cold->warm   first build all MISS      -> identical rerun 100% HIT
3. SELECTIVE RECOMPUTE edit ONE block           -> only it misses, 15 siblings cached
4. CAS DEDUP          bit-identical column      -> one physical blob
5. PARTIAL-DAG        materialize feature X      -> runs X + ancestors only
6. MULTI-OBJECTIVE    51 (feature x arm) trials -> FDR funnel -> shortlist (scored vs 3 arms)
7. GATES              causality + bit-exact verification pass
8. CATALOG            machine-readable manifest for LLM navigation
```

---

## Architecture

```
3.1__FeatureFramework.py   orchestrator entry script (mirrors 3__FeatureFramework.py)
3.1/                       flat code folder, path-loaded (mirrors FeatureTemplates/)
  hashing.py        content + transitive-closure + env fingerprint + data versioning
  block.py          Block (the unit) + FeatureFamily (codegen) + legacy adapter
  graph.py          dependency DAG, Kahn topo-sort, partial-DAG selection (modified+ / +target)
  cache.py          content-addressed cache: manifest + CAS (bit-identical columns dedupe)
  engine.py         schedule -> cache-or-execute -> profile (per-block hit/miss timing)
  registry.py       family-first discovery + the auto-generated catalog
  eval.py           multi-objective screen, FDR/stability selection, causality + bit-exact gates
  storage.py        date-partitioned feature lake (pyarrow, zstd, float32, predicate pushdown)
  materialize.py    incremental watermark orchestrator (engine + cache + lake -> delta rebuild)
  panel_backend.py  cross-sectional over('Date') ops, pandas now / polars when installed (parity-checked)
  deflate.py        deflated Sharpe + Romano-Wolf + PBO (honest selection at scale)
  leakage.py        panel + survivorship leakage (what per-ticker truncation misses)
  govern.py         p95 cost-regression alerts + value-per-CPU-second prioritization
  catalog_search.py TF-IDF semantic search + white-space + near-duplicate detection
  viz.py            DAG -> Graphviz DOT + Mermaid (+ family roll-up)
  --- alpha-extraction layer ---
  validation.py     purged/embargoed CPCV (Lopez de Prado) -- leakage-free folds
  select_marginal.py greedy conditional selection (orthogonalize vs incumbents)
  model.py          purged out-of-fold XGBoost + purged gain-importance
  ensemble.py       OOF target-ensemble stacking (rank-avg -> variance reduction)
  evolve.py         evolutionary family-param search guided by the live screen
  families/         example codegen families (rolling stats, cross-sectional rank/zscore)
  cli.py            list / catalog / viz / search / demo / demo-beyond / demo-alpha
  demo.py (core, 8 sections)  demo_beyond.py (platform, 7 subsystems)  demo_alpha.py (alpha, 6 sections)
```

Three proofs: `demo` (engine), `demo_beyond` (platform), `demo_alpha` (validated alpha).
~4,600 LOC, 28 modules, **13/13 verification checks green**, every module self-tests on import.

### The cache key (the load-bearing idea)

```
code_version(block) = sha( AST-normalized source of compute (or family builder + params)
                           + sources of every PROJECT-LOCAL module it transitively imports
                           + environment fingerprint (python + numpy/pandas/polars/xgboost) )

cache_key(block)    = sha( code_version + sorted(data_version(c) for c in block.requires) )
```

A block is a cache **hit** iff its code, its helper closure, its library env, and every
input column are unchanged — then it is never recomputed. A full 28GB rebuild becomes a
delta rebuild. Identical produced columns share one blob in the CAS (free dedup at 10TB).

### Feature families (breadth without sprawl)

```python
FeatureFamily(
    name="roll",
    param_grid={"window": [5, 10, 20, 50], "stat": ["mean", "std", "zscore", "skew"]},
    builder=_builder,                 # params -> (produces, compute)
    requires=["Close"], tags=["price", "rolling"], kind="per_ticker",
)
```

`.expand()` yields 16 individually-addressable `Block`s. Discovery imports the *family*,
not 16 files — so 10k features cost a handful of module imports, not 10k `importlib` calls.
Each emitted feature keeps per-block profiling and bit-identical verification.

---

## Preserved invariants (non-negotiable, all kept)

1. Modular units — one block/family per file.
2. Per-block profiling — wall-clock per block, now annotated with cache hit/miss.
3. Bit-identical optimizability — `verify_bit_exact` (`maxdiff < 1e-9`) is the speedup oracle.
4. `_` / `__` naming convention — production / candidate / infra.
5. Multi-seed/fold promotion gate — selection requires sign-stability across folds + FDR.
6. PIT safety — causality (look-ahead) gate via recompute-on-truncation.

## Roadmap (incremental, no rewrite)

- [x] content-addressed cache + manifest + CAS
- [x] partial-DAG selection (`modified+`, `+target`)
- [x] family codegen + family-first discovery
- [x] multi-objective screen + FDR/stability selection
- [x] auto catalog
- [x] incremental watermark materialization (delta rebuild)
- [x] date-partitioned storage lake (zstd / float32 / predicate pushdown)
- [x] panel `over('Date')` backend (pandas now, polars-ready, parity-checked)
- [x] p95 cost-regression alerts + value-per-CPU-second governance
- [x] panel-level leakage detector (cross-sectional + survivorship)
- [x] deflated-metric selection (deflated Sharpe / Romano-Wolf / PBO)
- [x] semantic catalog search + white-space + near-duplicate detection
- [x] DAG visualization (Graphviz DOT + Mermaid)
- [x] purged/embargoed CPCV validation (leakage-free folds)
- [x] marginal/conditional selection (orthogonalize vs incumbents)
- [x] purged out-of-fold XGBoost + purged gain-importance
- [x] OOF target-ensemble stacking (rank-avg, variance-reducing)
- [x] evolutionary family-param search via the live screen
- [ ] `pip install polars` to activate the lazy out-of-core backend (parity gate already wired)
- [ ] Lance table format for as-of/random-access reads at 10TB
- [ ] wire `ensemble.target_ensemble` OOF to the live `_target_ensemble.py` signal

See `FEATURE_FRAMEWORK_SCALING_SPEC.md` for the full requirements (R1–R16) and rationale.
