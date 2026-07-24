---
name: feature-factory
description: Spammable quant-feature generator+gater for the Stock-Market FeatureTemplates pipeline. One run = author a batch of fresh orthogonal feature-idea specs (biased to productive veins) → fan out subagents to codegen them as _cand_*.py blocks matching the contract → gate at high power (n=120) → log survivors. Built to be fired repeatedly to burn Max-plan usage on real feature discovery. Invoke when the user says things like "feature factory", "run the factory", "make/generate more features", "spam features", "MOREEEE features", "burn usage making features", "keep generating features", or asks to mass-produce/gate candidate features.
---

# Feature Factory

A spammable loop that mass-produces and gates candidate features for the FeatureTemplates
pipeline. Each invocation runs **one batch**: generate fresh idea specs → spawn subagents to
codegen them into contract-correct `_cand_*.py` blocks → gate them at high power → record
survivors. Fire it again (or run several rounds) to keep going. The gate is the quality filter,
so casting wide is safe — junk is thrown out, survivors accrue.

**Files (all in `.claude/skills/feature-factory/`):**
- `factory.py` — engine. Solo: `specs`, `gate`, `status`, `veins`. Producer: `queue-status`,
  `topup`, `enqueue`. Consumer/forge: `claim` (gate auto-releases the claimed batch).
- `workflow.js` — the codegen Workflow (1 subagent per spec, sonnet, writes the block).
- `queue/` — the producer→consumer handoff: `pending/` (authored, waiting) → `claimed/` (in flight)
  → `done/` (gated); `manifests/` maps a forge batch to its claimed concepts. Filled by the
  separate **`feature-research`** skill; drained in *forge mode* (see below).
- `specs/` — materialised contract spec `.txt` the workflow reads. `concepts_done.txt` — dedup ledger.
- `ledger.csv` — cumulative gated results. `REPORT.md` — appended human-readable summaries.

## Why these design choices (lessons baked in — do NOT regress)
- **Gate at n=120, not n=40.** Low-ticker IC is noise — a 0.14 OOS "winner" at n=40 evaporated at n=120. `factory.py gate` uses n=120 by default.
- **Redundancy is the real bar.** ~190 live features exist; most new ideas are redundant. Value = `maxcorr ≈ 0` (orthogonal) features. The catalog is biased toward orthogonal veins.
- **Orthogonal-but-weak is GOOD.** The intangible-capital vein keeps producing maxcorr-0 features with low *standalone* OOS IC. Their value is marginal-in-model — don't discard them for low univariate IC; flag them for the multivariate/backtest gate.
- **Causal striding must be fixed-from-start** (`i % 5 == 0`), never anchored to the last bar (re-anchors under truncation → leaks). The contract in `workflow.js` states this.
- **Don't trust agent self-reports** (speed/bit-exactness) — the gate is the arbiter.
- **Most productive veins:** `beta_risk`, `factor` (multi-index), `freq_stability`, then `intangible` (orthogonality goldmine). **Dead vein:** per-ticker calendar/seasonality (leaks, sparse) — not in the catalog.

## Procedure for ONE batch

1. **Pick parameters** from the user's request (defaults in brackets):
   - `vein` [`rotate`] — `rotate` (least-covered first), `all`, or a specific vein (`factory.py veins` lists them).
   - `n` [`30`] — features this batch. Bigger = more usage burned per run.
   - `rounds` [`1`] — how many batches to chain this invocation.

2. **Make a unique batch tag** (lowercase alnum), e.g. run in Bash:
   `BATCH=$(date +%m%d%H%M)`  (or append a letter if firing several within a minute).

3. **Generate specs** and capture the printed ids JSON (stdout):
   `python .claude/skills/feature-factory/factory.py specs --vein <vein> --n <n> --batch $BATCH`
   The ids JSON (e.g. `["ff... ", ...]`) is what you pass to the workflow. (Progress notes go to stderr.)
   - If it prints "No fresh concepts left", switch to **Creative mode** (below).

4. **Launch the codegen workflow** with the `Workflow` tool:
   `Workflow({ scriptPath: "<repo>/.claude/skills/feature-factory/workflow.js", args: <the ids JSON array> })`
   (Pass the array value. The workflow tolerates a JSON string too.) It runs in the background;
   wait for the completion notification (1 subagent per spec writes `FeatureTemplates/_cand_<id>.py`).

5. **Gate the batch** at high power:
   `python .claude/skills/feature-factory/factory.py gate --batch $BATCH`
   This runs `validate_feature.py --batch "_cand_ff$BATCH_*.py" --n 120 --keep_fail`, appends to
   `ledger.csv` + `REPORT.md`, and prints the PASS / novel (maxcorr<0.4) summary.

6. **Report** the PASS list and the novel (maxcorr<0.4) list to the user. Note that candidates are
   hidden `_cand_*` files (the framework's discovery count is unchanged — nothing leaks into the model
   until promoted by dropping the leading underscore, and only after the heavy multi-seed backtest).

7. **If `rounds` > 1 or the user said "keep going / MOREEEE":** go back to step 2 with a new tag and
   (for `rotate`) the next least-covered vein. Repeat.

## Creative mode (when the catalog is exhausted, or the user wants net-new veins)
Author fresh spec files yourself into `.claude/skills/feature-factory/specs/` (filenames
`ff<batch>_<vein>_<slug>.txt`, same format as the generated ones: `SPEC ID / VEIN / METHOD-DEFINITION`),
biasing toward orthogonal structure and the productive veins above. Then continue at step 4 with those
ids, and step 5 to gate. Add the new angles to `CATALOG` in `factory.py` so autopilot can reuse them.

## Forge mode (queue consumer — the parallel / "code-monkey" path)
Use this instead of step 3's `specs` when a separate **`feature-research`** agent is producing ideas
into the shared queue (`queue/pending/`). You become the pure **consumer**: claim → codegen → gate.
This lets ideation (another Max session) and codegen run **in parallel** with no file collisions —
single-writer-per-file + atomic claim. **Only ONE forge runs the gate at a time** (it shares
`battery_results.csv`/`ledger.csv`); don't run two forges concurrently.

1. Make a batch tag: `BATCH=$(date +%m%d%H%M)` (append a letter if firing several a minute).
2. **Claim** up to N pending specs (atomically moves them to `queue/claimed/` and materialises
   `specs/<id>.txt`), capturing the printed ids JSON:
   `python .claude/skills/feature-factory/factory.py claim --n 40 --batch $BATCH`
   (If it prints `[]`, the queue is empty — the producer is behind; wait or ping it. Don't generate
   your own specs in forge mode.)
3. **Codegen** exactly as normal: `Workflow({ scriptPath: "<repo>/.claude/skills/feature-factory/workflow.js", args: <ids JSON> })`.
4. **Gate**: `python .claude/skills/feature-factory/factory.py gate --batch $BATCH`. Gate
   auto-**releases** the claimed specs to `queue/done/` when it finishes (via the batch manifest).
5. Report PASS + novel (maxcorr<0.4) as usual, then loop back to step 1 to drain more. Pair with
   `/loop` to keep draining unattended.

(Solo mode is unchanged: with no producer running, just use `specs` → workflow → `gate` as below.
`topup` can also seed the queue from the residual catalog if you want to prime forge mode yourself.)

## Spamming it (the point)
- **Many rounds in one go:** ask for `rounds 5` (or just keep saying "more") — each batch ≈ `n`
  subagents + a gate. `n=40, rounds=5` ≈ 200 subagent codegens per invocation.
- **Unattended:** pair with `/loop` (e.g. `/loop feature-factory rotate n=40`) to fire batches on a
  cadence. Each fires a fresh batch, rotating veins, deduped against everything already done.
- **Throughput note:** generation is cheap and the bottleneck is the user's *downstream* evaluation
  (multi-seed backtest / model marginal-contribution). Periodically run
  `python .claude/skills/feature-factory/factory.py status` for the cumulative scoreboard and feed the
  top PASS + the orthogonal (maxcorr-0) group into the heavy gate rather than letting candidates pile up.

## Outputs
- Candidate blocks: `FeatureTemplates/_cand_ff*.py` (hidden from discovery).
- Cumulative results: `.claude/skills/feature-factory/ledger.csv`, human report `REPORT.md`.
- To promote a winner (after the heavy backtest gate): drop the leading `_` so
  `3__FeatureFramework.py` discovers it. To discard: delete the `_cand_*` files (harmless; hidden).
