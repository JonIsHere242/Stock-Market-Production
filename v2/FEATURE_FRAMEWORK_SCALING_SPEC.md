# Feature Framework Scaling Spec — Current State, Hamilton Parity, and Beyond

> **Purpose.** A self-contained handoff spec for designing the next generation of a bespoke
> quantitative feature-engineering framework. The receiving model has NO prior context — everything
> needed is below. Goal of the work: scale the feature library and data lake by ~1–2 orders of
> magnitude (toward ~10TB data, and feature *breadth* equivalent to ~1M–10M LOC of hand-written
> features) **without** losing the properties that make the current system fast to iterate on.

---

## 0. Domain primer (read first)

- **What it is:** a daily US-equity ML system. An XGBoost classifier predicts next-day up-move
  probability per (stock, day). Universe ≈ 4000 tickers × ~2000–3000 daily bars each ≈ ~4.9M rows.
- **Current direction:** moving from a single target to an **ensemble of diverse prediction targets**
  (rank-averaged) to cut variance — this is a confirmed real lift (rank-avg of 4 targets beat the
  single topq target on 4/4 seeds). This matters for the spec: feature *evaluation* must now serve
  **multiple target "arms,"** not just next-day log-return.
- **Hard-won statistical truths from this project (these constrain the whole design):**
  - **Feature SELECTION > feature COUNT.** Going 708 → 935 features once *diluted* the top-1%
    selection and collapsed live-equivalent returns from ~131% to ~6–14%. More features is not free.
  - **Single-seed "winners" are noise.** The only trusted promotion gate is a **multi-seed (≥4)
    ablation.** Most candidates die there.
  - **Raw OHLCV is ~tapped.** The remaining alpha white-space is new *information* (fundamentals,
    insider/EDGAR, short-volume, cross-asset), not the N-thousandth OHLCV transform.

These three facts are why "just write more feature code" is the wrong scaling axis, and why the spec
emphasizes generation efficiency, selection rigor, and data breadth over raw LOC.

---

## 1. What we have today (current-state inventory)

**Block contract (the unit of work).** One Python file in `FeatureTemplates/` = one "block." Each block exposes:
- `METADATA = {name, description, requires:[cols], produces:[cols], tags, version}`
- `def compute(df) -> df` — **per-ticker, stateless**: receives one ticker's OHLCV + any
  already-produced columns; may only **ADD** columns; must never drop, reshape, or reorder rows.

**Naming convention (LOCKED — preserve exactly).**
- no prefix → production block (runs in `--all`)
- single underscore `_foo` → candidate block (opt-in, excluded from `--all`)
- double underscore `__foo` → infrastructure/tooling (never run as a block)

**Orchestrator — `3__FeatureFramework.py` (~777 ln).**
- `discover_blocks(include_candidates)` — imports every non-`__` file via `importlib`.
- `resolve_order()` — Kahn BFS **topological sort** over `requires`/`produces` dependency edges.
- `run_pipeline_timed()` — runs blocks in order, **wall-clock times each block**, returns a timing dict
  (`{n_rows, total_s, blocks:{name:secs}, order, skipped, skip_reasons, meta}`).
- `process_all()` — `ProcessPoolExecutor` fan-out, **one parquet output per ticker**.
- **FULL RECOMPUTE every run** (no caching of unchanged blocks).
- `--save_timing` exports a per-ticker × per-block CSV matrix.

**Materialization / data.**
- Input `Data/PriceData/` (per-ticker parquet, ~30 KB ea).
- Output `Data/ProcessedData_v2/` ≈ **28 GB**, per-ticker parquet 6–8 MB ea, 600–800 feature cols, float64.
- `Data/ProcessedData_v2_f32/` — bit-identical float32 variant (~49% RAM cut). float32 is safe:
  XGBoost `hist` bins every column to 256 buckets, so float64→float32 collapses to the same bin.

**Shared-panel caching — `_marketcap.py`.** Loads a 2.68M-row panel once per subprocess and stashes it
on `sys` (`sys._MARKETCAP_PANEL_CACHE`) so fresh `importlib` module instances reuse it. PIT-safe via
backward `merge_asof`. (This is the only cross-block cache today, and it caches *input data*, not
block outputs.)

**Correctness gates.**
- `verify_block.py` — bit-exact gate: a refactored/vectorized block must match ground truth with
  `maxdiff < 1e-9`. This is what lets an LLM speed up a block while guaranteeing identical output.
- `__smoke_new.py` — contract + **causality** check: recompute on series truncated at 60/75/90% and
  assert past values are unchanged (catches look-ahead), plus < 150 ms/ticker speed bar.
- `import_scan` — import hygiene.

**Screening / evaluation tooling.**
- `__tail_screen.py` — offline (no-retrain) cross-sectional top-decile **tail-lift** triage, with a
  trend+vol-neutralized "neut" variant and K date-folds. **Now supports `--target <arm>`** to screen
  any ensemble arm in <10 min.
- `__diagnostics.py` — per-block Spearman IC + speed tiers. **Still scores 1-day-ahead return ONLY**
  (not yet multi-arm — a known gap).
- `__feature_lab.py` — monotonic-transform battery on "meh" features; ranks by tail lift; **has an
  `--emit` codegen path** (writes out a new block from a chosen transform). Seed of a codegen system.
- `__trial_ledger.py` — append-only trial log enabling Romano-Wolf / FDR deflation across trials.
- `__live_xcheck.py` — cross-checks feature ranks vs realized trade PnL.
- `__relatedness_map.py` — block-level clustering → `basis.json` (seed of a navigability artifact).
- **Promotion gate:** multi-seed (≥4) ablation. Single-seed results are not trusted.

---

## 2. What Apache Hamilton provides (verified) — and where it stops

Hamilton (Stitch Fix → Apache; function-as-node dataflow library) is the closest existing analog to
this framework. All claims below are from primary docs and were adversarially verified.

**Hamilton HAS (worth importing the *design* of):**
- **Function-as-node DAG.** Function name = output; parameter names = dependencies. This is the
  *signature-inferred* equivalent of our explicit `METADATA(requires/produces)` + `resolve_order()`.
- **Content-addressed caching.** `cache_key = node_name + code_version + dependencies_data_versions`,
  where `code_version` = hash of the node's source **ignoring docstrings/comments**, and
  `dependencies_data_versions` = hashes of each upstream output. A node recomputes **only** when its
  code or an upstream input changed. Root nodes key on code hash alone.
- **Deferred data loading.** It passes small `data_version` strings through the DAG instead of the
  actual data until a node must execute — the trick that makes caching cheap at scale.
- **Partial-DAG execution.** The Driver computes only the minimum subgraph needed for requested
  outputs ("materialize feature X and its ancestors").
- **Auto lineage + visualization.** Fine-grained/column-level lineage and a DAG rendered *from code
  without running it* (visualization never drifts from source).

**Hamilton does NOT attempt (the gaps WE must own):**
1. **Feature generation / combinatorics / dedup.** It executes features you wrote; it has no concept
   of parametrized feature families, sweeps, or near-duplicate detection.
2. **Financial correctness.** No point-in-time discipline, no look-ahead/leakage detection, no
   survivorship/PIT-universe handling. (Our `verify_block.py`/`__smoke_new.py` are bespoke.)
3. **Multiple-testing control / feature selection.** Nothing like `__trial_ledger.py` FDR or the
   multi-seed gate. More nodes → more spurious winners; Hamilton is silent on this.
4. **Cache-key transitive closure.** `code_version` hashes a node's OWN source but **NOT** nested
   helper calls (`_marketcap.py`, `_indexes.py`, `Util.py`) or library/Python upgrades — a silent
   stale-cache correctness hole we must close ourselves.
5. **Storage / data-lake management.** It caches node outputs; it does not lay out a 10TB lake
   (partitioning, file sizing, compaction, table formats).
6. **"DAG too big to load."** It builds the full FunctionGraph in memory — same wall our
   `importlib`-over-every-file discovery hits at hundreds of thousands of files.
7. **Distributed scheduling.** Backfills/retries/cross-machine orchestration are out of scope
   (adapters for Ray/Dask/Spark exist, but it doesn't own scheduling).

**Verdict:** adopt Hamilton's caching + partial-DAG + auto-code-hash **design** into
`3__FeatureFramework.py`; do **not** rewrite blocks as Hamilton nodes (a Hamilton node is
one-function-one-output, but our blocks add many columns per `compute(df)` — analogy, not 1:1 fit).

---

## 3. What we build beyond Hamilton (gap map)

| # | Capability | Hamilton? | Why we need it |
|---|---|---|---|
| G1 | Content-addressed incremental cache (with **transitive helper-closure** hashing) | partial (no closure) | kill 28GB→10TB full recompute, safely |
| G2 | Partial-DAG selective rebuild (`--select state:modified+`) | yes (design) | one edit rebuilds 1 column, not the lake |
| G3 | Panel-wide compute engine (Polars `over()`), keep row contract | no | cross-sectional features + drop 4000-subprocess fan-out |
| G4 | **Parametrized feature-family codegen registry** | no | get breadth WITHOUT 10M LOC of hand files; dodge importlib wall |
| G5 | Machine-readable navigable registry for LLM agents | partial (viz/lineage) | white-space discovery without burning tokens on 75k+ LOC |
| G6 | **Multi-objective evaluation** across all ensemble arms | no | screening currently blind to non-topq targets |
| G7 | Panel/cross-sectional **leakage detection** | no | per-ticker truncation test cannot see XS/survivorship leaks |
| G8 | Statistical **selection at scale** (FDR across huge trial counts) | no | the real ceiling: more candidates → harsher bar |
| G9 | Storage layer for the materialized matrix at 10TB | no | one-parquet-per-ticker is a small-files anti-pattern |
| G10 | Cost governance (tail-lift-per-CPU-second, p95 regression) | no | avoid death-by-a-thousand-slow-features |

---

## 4. Requirements (concrete, prioritized, with acceptance criteria)

> Priority: **P0** = load-bearing for any scale-up; **P1** = needed within the scale-up; **P2** = needed at the far edge.

### P0 — Incremental materialization & cache (replaces FULL RECOMPUTE)

- **R1.** A per-block **content-addressed cache**. Cache key MUST = `hash(block compute() source)` +
  `hash(transitive helper-module closure it imports)` + `pinned env/lib version` +
  `data_version(each required column)`. Root blocks (raw OHLCV only) key on code+env hash alone.
  - *Accept:* editing one block (or a helper it calls) recomputes that block and its descendants and
    NOTHING else; editing nothing → a run reads 100% from cache. Stale-on-helper-edit must be
    impossible (test: edit `_marketcap.py`, confirm dependents recompute).
- **R2.** A persisted **manifest/registry** mapping `block → {source_hash, helper_closure_hash,
  produces, last_materialized_date_range}`. Tracked **per block**, not globally (so a new candidate
  block materializes only itself over only new rows).
  - *Accept:* manifest round-trips across runs; adding a block touches only that block's state.
- **R3.** A **content-addressable store (CAS)** for produced column blobs keyed by content hash, so
  bit-identical columns dedupe across tickers/blocks/runs.
  - *Accept:* two blocks producing an identical column store one physical blob.

### P0 — Selective execution & correctness preservation

- **R4.** A selector flag (`--select`) supporting `state:modified+` (changed blocks + descendants) and
  `+target` (a target column + its ancestors), driven by the existing `resolve_order()` adjacency.
  - *Accept:* `--select +feature_X` runs exactly feature_X's ancestor set.
- **R5.** All existing gates remain wired and become **incremental**: `verify_block.py` (bit-exact
  maxdiff<1e-9) is the cache-correctness oracle; `__smoke_new.py` causality gate runs on changed
  blocks; the multi-seed (≥4) promotion gate is untouched.
  - *Accept:* a cached run produces byte-identical `ProcessedData_v2` to a full recompute.

### P1 — Compute engine evolution

- **R6.** Hot and **cross-sectional** blocks may be authored as **Polars window expressions**:
  `over("Ticker")` for per-ticker (preserves the add-columns-never-reshape contract via default
  `mapping_strategy="group_to_rows"`), `over("Date")` for cross-sectional rank/z-score.
  - *Accept:* converted block passes `verify_block.py` against the pandas reference. MUST use
    `order_by="Date"` for order-sensitive ops (cum/diff/rolling); MUST NOT use `explode` strategy in
    `with_columns`.
- **R7.** Out-of-core path for panel-wide ops (DuckDB or Polars streaming) so a full-universe panel
  build never requires the whole matrix in RAM.

### P1 — Parametrized feature-family codegen (the "breadth without sprawl" answer)

- **R8.** A **feature-family** primitive: one generator module declares a parameter grid (windows ×
  transforms × neutralizations × universes) and emits many concrete features. Generalizes
  `__feature_lab.py --emit`.
  - *Accept:* a single ~hundreds-LOC generator can stand in for thousands of hand-written blocks;
    each emitted feature is individually addressable, profilable, and bit-identical-verifiable.
- **R9.** Per-**family** profiling + per-**feature** lineage, so the "point an LLM at the slowest unit
  and optimize it bit-identically" workflow survives at the family level.
  - *Accept:* timing report attributes cost per family AND per emitted feature.
- **R10.** The registry MUST avoid the `importlib`-over-N-files wall: discovery scales sub-linearly in
  emitted-feature count (load generators, not one file per feature).
  - *Accept:* discovery time for 100k logical features ≈ discovery time for the generators that
    define them, not 100k file imports.

### P1 — Navigability for LLM/agent operation (stated pain point)

- **R11.** An auto-generated, **machine-readable feature catalog** (from METADATA + family defs):
  name, produces, tags, family, cost, last evaluation scores per arm, lineage edges. Plus a compact
  index an agent can query to find white-space WITHOUT reading source.
  - *Accept:* an agent can answer "do we already have X / what's adjacent to X / what's unexplored"
    from the catalog alone; extends `__relatedness_map.py`/`basis.json`.

### P1 — Multi-objective evaluation (ensemble arms)

- **R12.** `__diagnostics.py` and all screeners MUST score against **every ensemble target arm**, not
  just 1-day log-return (close the known `__diagnostics` 1-day-only gap; `__tail_screen --target`
  already does this — bring diagnostics to parity).
  - *Accept:* a single screen run reports per-feature lift/IC for each arm; a feature useful only to a
    non-topq arm is visible.

### P1/P2 — Correctness & statistics at scale

- **R13.** A **panel-level leakage detector** beyond the per-ticker truncation test: catch
  cross-sectional (rank/z-score) leakage, survivorship, and point-in-time-universe drift. XS
  normalization/selection MUST be refit inside each walk-forward fold.
  - *Accept:* a deliberately-leaky XS feature is flagged; a clean one passes.
- **R14.** **Selection at scale:** FDR/Romano-Wolf deflation (via `__trial_ledger.py`) MUST scale to
  the full candidate count, and the promotion bar MUST tighten as breadth grows.
  - *Accept:* promoting from a 100k-candidate batch yields a deflation-corrected shortlist, not raw
    top-rank.

### P2 — Storage & cost governance at 10TB

- **R15.** Replace one-parquet-per-ticker with a layout sized for both access patterns: date-partitioned
  (cheap cross-sectional/append-only) + ticker-sorted within (cheap per-entity scans); ZSTD,
  row-groups ~100–256k rows; evaluate Lance for the random-access/as-of side.
  - *Accept:* an XGBoost trainer streams the panel without per-file open overhead dominating; a
    single-date cross-section reads without touching all files.
- **R16.** Cost governance: per-block baselines + **p95 runtime-regression alerts**, and a
  **tail-lift-per-CPU-second** prioritization so expensive features face a higher promotion bar.
  - *Accept:* a block whose p95 regresses >X% alerts; a 600ms block must clear a higher gate than a 6ms one.

---

## 5. Cherished invariants (NON-NEGOTIABLE — every design must preserve)

1. **Modular units.** One block/family = one self-contained file. No god-modules.
2. **Per-unit profiling.** Every unit has its own wall-clock cost, exposed in the timing report.
3. **Bit-identical optimizability.** Any speedup/refactor is provable via `verify_block.py`
   (`maxdiff < 1e-9`). This is what makes "hand an LLM the slowest unit" safe.
4. **`_` / `__` naming convention.** production / candidate / infra — keep exactly.
5. **Multi-seed (≥4) promotion gate.** No feature ships on a single-seed result.
6. **PIT safety.** No look-ahead; helper merges stay backward/`as_of`.

---

## 6. Why this is needed (rationale)

The current framework is, structurally, a hand-rolled Apache Hamilton: file-per-feature nodes, a
declared `requires`/`produces` dependency graph, a topological scheduler, and bit-exact verification.
That design is sound and validated by Hamilton's independent convergence on the same shape — but it
was built for a few hundred blocks and a 28 GB lake, and it **recomputes everything on every run**.
At the stated trajectory (~10TB of data and an order-of-magnitude more feature breadth), full
recompute is fatal and one-file-per-feature collides with a hard `importlib`/in-memory-graph wall.
The first job of this spec is therefore to import the proven survival mechanics — content-addressed
incremental caching, partial-DAG selective rebuild, and panel-wide vectorized compute — *without a
rewrite*, so that adding the next feature or the next trading day costs delta work, not full work.

But the deeper point is that **code volume is the wrong scaling axis, and this project's own data
proves it.** More features have measurably *hurt* (708→935 once collapsed returns by diluting the
top-1% selection), single-seed winners are noise, and raw OHLCV is largely tapped. So the goal is not
"10M LOC of hand-written features" — it is the *breadth* such a number implies, achieved through
parametrized feature-family codegen (thousands of features from hundreds of LOC), while shifting the
real investment to where the ceiling actually binds: **evaluation and selection.** As candidate count
explodes, the binding constraint becomes statistical signal-to-noise — honest multiple-testing
control, multi-objective screening across every ensemble arm, and panel-aware leakage detection — not
the ability to author or execute more code. Hamilton helps with none of that; it is purely an
execution/lineage substrate.

Finally, scale changes *who* operates the system. At 75k LOC an engineer can hold the library in their
head; at the target scale, the primary operator is an LLM agent, and its effectiveness is gated by
navigability. A machine-readable feature catalog and relatedness index — so an agent can find
white-space without reading the source — is not a nicety but a throughput multiplier. Build the cache
and the codegen registry to remove the *compute* and *authoring* walls; build the selection harness
and the catalog to remove the *statistical* and *navigability* walls. Together they are what turn
"just do more, push harder" into compounding edge instead of compounding variance and maintenance.
