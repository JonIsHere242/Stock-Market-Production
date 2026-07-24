---
name: feature-research
description: Ideation/producer half of the feature factory — keeps the code-monkey's candidate queue topped up. One run checks queue depth and, only if it's running low (pending < 50), generates fresh ORTHOGONAL feature-idea specs (residual catalog first, then RESEARCH mode that invents net-new veins/angles) and enqueues them for a separate forge agent to codegen+gate in parallel. Does NOT codegen or gate. Invoke when the user says things like "run ideation", "feature research", "research mode", "top up the queue", "feed the code monkey", "keep the queue full", "generate feature ideas", "ideate features", or wants the parallel ideation agent to refill candidates.
---

# Feature Research (the ideation producer)

This is the **producer** half of the split feature factory. It runs as its *own agent* (a second
Max-usage session) and its only job is to keep the **queue** full of high-quality, orthogonal,
codegen-able feature-idea specs. A separate **forge / code-monkey** agent (the existing
`feature-factory` skill in *forge mode*) drains that queue, codegens each spec into a `_cand_*.py`
block, and gates it. The two run **in parallel** and never touch the same files.

```
ideation (THIS skill)            queue/                     forge (feature-factory)
  topup / RESEARCH  ──writes──▶  pending/  ──claims──▶      claim → codegen → gate → done/
```

**Engine:** the shared `python .claude/skills/feature-factory/factory.py` (NOT a separate copy).
**Hard rule — you are the producer:** you ONLY add specs to the queue. You **never** run
`claim`, the codegen workflow, or `gate`. That is the forge agent's job. If you codegen/gate you
will collide with it.

## Demand-driven: only refill when the queue is low

The bottleneck is the forge (codegen+gate is slow; ideation is cheap), so do **not** spam the queue
unbounded. Refill on a threshold with hysteresis:
- **min = 50** — if `pending` ≥ 50, the forge has plenty; **do nothing** (report and stop / wait).
- **target = 100** — when below min, refill back up to ~100, then stop.

## Procedure for ONE run

1. **Check depth:**
   `python .claude/skills/feature-factory/factory.py queue-status --min 50 --json`
   → `{"pending":N,"claimed":..,"done":..,"catalog_fresh":F,"needs_topup":bool}`.
   If `needs_topup` is false → say "queue healthy (N pending), nothing to do" and stop.

2. **Free refill from the residual catalog first** (deterministic, zero ideation cost):
   `python .claude/skills/feature-factory/factory.py topup --target 100`
   → `{"added":.., "pending":.., "catalog_fresh":.., "catalog_exhausted":bool, "still_below_target":bool}`.
   If this gets `pending` to ≥ target, you're done — stop.

3. **RESEARCH mode** (only if `still_below_target` is true, i.e. the catalog is dry): invent
   **net-new** specs. Fan out ideation subagents (Agent tool, or a small Workflow if you want to
   burn more usage) — each takes a distinct theme and returns spec objects. Bias hard toward the
   productive + orthogonal veins (below). Then **enqueue** them:
   - Write the collected specs to a JSON file: an array of `{"vein","slug","definition"}`.
   - `python .claude/skills/feature-factory/factory.py enqueue --json <that_file>`
     → `{"added":.., "skipped_dups":.., "pending":..}`. Dedup is automatic (already-queued or
     already-done concepts are skipped). Repeat until `pending` ≥ target.

4. **Report** the new `pending` depth and a one-line summary of the themes you added. Stop.

## What makes a good spec (so it survives the forge's gate)

Each enqueued spec is one JSON object:
- `vein` — bucket label (use an existing vein when it fits; a new short token is fine for a net-new
  theme). `slug` — short unique name for the idea. `definition` — the METHOD/DEFINITION the codegen
  subagent will implement (precise, self-contained, a few sentences; a level + a dynamic/asymmetry
  variant is good).
- The codegen runs under a strict contract, so only propose things that fit it:
  **per-ticker; OHLCV + the `_indexes` (SPY/QQQ/IWM/DIA/VIX) helper + the `_fundamentals.as_of`
  PIT helper only; causal/no-lookahead; vectorizable < 100ms; guard divisions.** No cross-sectional
  ranks, no training, no external data. Expensive metrics must use a CAUSAL fixed-from-start stride
  (`i % 5 == 0`), never anchored to the last bar.
- **Orthogonality is the real bar.** ~190 live features exist; the gate rejects near-duplicates
  (high maxcorr). Value = `maxcorr ≈ 0`. Orthogonal-but-weak (low standalone IC) is GOOD — it earns
  its keep on marginal-in-model contribution; don't avoid it.

## Where to look for non-redundant ideas (read before inventing)
- `concepts_done.txt` — everything already tried; don't repeat (enqueue also auto-skips dups).
- `ledger.csv` — what PASSed and the orthogonal (maxcorr<0.4) group; mine adjacent untried angles.
- `FeatureTemplates/*.py` METADATA `produces` — the live feature surface to be orthogonal to.
- **Productive veins:** `beta_risk`, `factor` (multi-index), `freq_stability`, then `intangible`
  (the orthogonality goldmine — capitalized SG&A/R&D, organizational capital, intangible-adjusted
  ratios via `_fundamentals.as_of`). Net-new veins worth mining: option-free PIT-fundamental
  dynamics, accruals/quality, cross-asset (VIX/breadth) conditioning, path/sequence geometry.
- **Dead vein (do NOT propose):** per-ticker calendar/seasonality (leaks, sparse).

## Spamming / unattended
- Fire repeatedly, or pair with `/loop` (e.g. `/loop feature-research`) so it wakes, checks depth,
  and only refills when the forge has drained below 50 — a self-throttling idea pump.
- Run this in a *different* agent/session than the forge so the two burn usage in parallel.
