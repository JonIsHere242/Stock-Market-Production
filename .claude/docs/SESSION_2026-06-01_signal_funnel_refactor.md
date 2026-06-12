# Session Chronicle — Signals → Broker Pipeline Refactor

**Date:** 2026-06-01 (overnight)
**Scope:** Rework how daily buy signals flow from the backtester to the live IBKR broker —
introduce a real candidate→book funnel, separate the state ledger, harden the broker, and
fix the morning timing.

> Written as a handoff. If you're a future session picking this up cold, read the
> **TL;DR**, then **Architecture**, then **Gotchas / Landmines** before changing anything.

---

## TL;DR

- `_Buy_Signals.parquet` is now the **final broker book** (≤4 narrowed names), **not** a backtest
  ledger. The 766-row ledger that used to live there has been **retired**.
- Daily flow: backtester writes a **~12-name candidate pool** to `Data/0__signals.parquet` →
  **`7__MacroFilter.py` funnel** narrows it to **4** → broker reads `_Buy_Signals.parquet`.
- `7__MacroFilter.py` was **rewritten clean-sheet** into a cost-ordered funnel
  (free mechanical screens → paid LLM only on survivors), is **idempotent** (skips, $0, if you
  hand-funneled via the skill), **self-waits to 9:35 ET** so its research sees post-open news,
  and **auto-skips the LLM** if the API key is unfunded.
- The broker (`9_SuperFastBroker.py`) now reads `_Buy_Signals.parquet` with a **fail-safe guard**:
  it refuses to trade an un-narrowed file (no `Status` column, or >12 pending rows).
- **Tonight's live book** (hand-funneled before this work): **DRS, CVI, AU, AG**.

---

## Where we started (the problem)

The user said "my system now uses `_Buy_Signals.parquet`; narrow it to 4 and point the broker at it."
Investigation found a tangle:

- `_Buy_Signals.parquet` was a **766-row per-ticker state ledger** (`LastBuySignalDate`,
  `IsCurrentlyBought`, `ConsecutiveLosses`, …) — *not* a list of today's signals. Only **one** row
  (DRS) was actually fresh.
- The real candidate file with today's signals was **`Data/0__signals.parquet`** (20 Pending rows,
  rich schema with `StopPrice`/`TargetPrice`/`ATR`/`CapBucket`/`UpProbability`).
- The broker read `Util.SIGNALS_FILE = 'Z_signals.parquet'` — **which didn't exist**.
- The whole thing was mid-migration and inconsistent.

So the "narrow to 4" couldn't happen against the ledger; the candidate pool lived elsewhere; and
the broker was pointed at a missing file.

---

## Architecture (after this session)

```
EVENING (5:00 PM MT, scheduled task)
  trading_system.ps1  →  1__Ticker → 2__Price → 3__Alpha → 4__Predictor → 5__NightlyBackTester
                                                                              │
                                       writes ~12-name candidate POOL ────────┘
                                                                              ▼
                                                          Data/0__signals.parquet   (Util.SIGNALS_FILE)

MORNING (7:28 AM MT = 9:28 ET, scheduled task → morning mode)
  trading_system.ps1  →  7__MacroFilter.py (FUNNEL)  →  9_SuperFastBroker.py
                              │                                │
            reads 0__signals pool                   reads _Buy_Signals.parquet
            narrows to ≤4                            guard: Status col + ≤12 rows
            writes _Buy_Signals.parquet              waits to 10:00 ET, SPY gate,
                                                     per-stock gap filter → LOCKS IN

  trade_history.parquet  = completed trades (1,112 rows) — untouched, separate concern
  Data/_retired_trading_ledger.parquet = where the old per-ticker ledger writes go now (retired)
```

### Three files, three roles (the mental model)
| File | Role | Written by | Read by |
|---|---|---|---|
| `Data/0__signals.parquet` | recent-day **candidate pool** (~12) | backtester | funnel + (Util.read_signals) |
| `_Buy_Signals.parquet` | final **narrowed book** (≤4) | funnel (or the skill, by hand) | broker |
| `trade_history.parquet` | **completed trades** (1,112) | trade logger | analysis only |

---

## The funnel — `7__MacroFilter.py` (clean-sheet rewrite)

Cost-ordered so the **paid LLM only ever sees names that survive the free screens**:

- **Stage 0 — load & align.** Read `0__signals`, filter `Status=='Pending'`, keep rows whose
  `TargetDate == get_next_trading_day()` (NYSE calendar). Falls back to all-pending if the date
  filter empties it. Loads each ticker's `Data/PriceData/<TKR>.parquet` (the price source of truth).
- **Stage 1 — HARD mechanical exclusions (free):** price<$5, micro-cap<$952M, **weekly-vol>5%**,
  RSI(14) in the 30–40 death-zone, ideological quarantine. Dropped with reasons.
- **Stage 2 — SOFT delisting/merger flags (free):** penny/illiquid, and a big-gap-then-vol-collapse
  "deal-peg" signature. These **deprioritize** (don't drop) and become the LLM's priority list.
- **Stage 3 — LLM judgement (paid, survivors only):** `claude-opus-4-8`, adaptive thinking +
  `effort:max`, `web_search_20260209`, rubric prompt-cached. Confirms **active-M&A target** /
  **material crisis** → hard drop. **Auto-skips** (neutral) on unfunded/invalid key.
- **Stage 4 — rank & select:** by `UpProbability`, **clean names first, then relax soft-flags to
  fill 4** (keep capital deployed). Writes the book (rich schema, `Status=Pending`, stop/target
  nulled so the broker anchors to live mid). Backs up the prior book first.

### Behaviors that matter
- **Idempotent.** If `_Buy_Signals.parquet` already holds a `Status`-bearing book of ≤12 Pending
  rows **all dated for the next trading day**, the funnel **exits immediately, $0** — so hand-running
  the `trade-signals` skill is detected and not redone. `--force` overrides.
- **Self-waits to 9:35 ET** before the LLM (post-open news) — but only if launched within 45 min of
  it, so off-hours/manual runs proceed immediately. `--no-wait` skips. Mechanical screens run first
  (prior-close data, so their timing is irrelevant).
- **Bounded LLM:** ranks first, web-searches top-down only until the book fills (typically ~4 calls,
  capped at `LLM_MAX_CHECKS=6`, `LLM_CALL_TIMEOUT=150s`). Finishes before the 10:00 ET lock-in and
  costs less.
- **Never writes an empty book** over an existing one (refuses on 0 survivors) — a bad screen can't
  wipe a good book.

### Flags
```
python 7__MacroFilter.py                 # normal funnel (idempotent; waits to 9:35 ET for LLM)
python 7__MacroFilter.py --skip-llm      # mechanical only (no API calls)
python 7__MacroFilter.py --dry-run       # report selection, don't write
python 7__MacroFilter.py --no-wait       # don't self-wait to the research window
python 7__MacroFilter.py --force         # re-run even if already funneled
```

---

## Timing design (morning)

The flaw was the funnel running **pre-open** (9:28 ET). The broker already self-waits to **10:00 ET**
to lock in (EDA-calibrated entry, Sharpe 1.11 vs 0.31 at open — **do not move it**).

| ET | Event |
|---|---|
| 9:28 | task fires → funnel launches → idempotency check (skip $0 if hand-funneled) → mechanical screens |
| **9:35** | funnel self-waits to here, then web-searches (post-open news) |
| ~9:42 | book written (LLM bounded) |
| **10:00** | broker locks in orders |

≈18 min between book finalized and lock-in. Machine is **Mountain time** (MDT in summer = ET−2), so
locally: funnel 7:28, research 7:35, broker 8:00.

---

## File-by-file changes

| File | Change |
|---|---|
| `7__MacroFilter.py` | **Clean-sheet rewrite** into the funnel (stages 0–4, idempotency, self-wait, bounded LLM, empty-guard). Was a per-ticker yfinance+Sonnet quality gate pointed at the missing `Z_signals.parquet`. |
| `9_SuperFastBroker.py` | Reads `_Buy_Signals.parquet` (was `read_signals()` → `Z_signals.parquet`); added `MAX_BOOK=12` fail-safe guard (refuse if no `Status` col or >12 pending). Removed unused `read_signals` import. |
| `Util.py` | `SIGNALS_FILE` → `Data/0__signals.parquet` (was missing `Z_signals.parquet`). Non-live `read/write_trading_data` ledger path → `Data/_retired_trading_ledger.parquet` (was `_Buy_Signals.parquet`). |
| `5__NightlyBackTester.py` | `save_guaranteed_signals_to_parquet` writes `Data/0__signals.parquet` (was `Z_signals.parquet`); writes a **pool of 12** (`SIGNAL_POOL_SIZE`) not `max_positions`(4); `read_trading_data()` retired (returns empty in-memory, touches no file). |
| `trading_system.ps1` | Morning mode (hour 7) now runs the **funnel before the broker**. |

---

## Gotchas / Landmines (read before touching the pipeline)

1. **`signal_filter.py` is MISSING.** `Util.annotate_signals_mechanical_filter` is therefore `None`,
   so the backtester's `Mech*` annotation is a **silent no-op** — `0__signals` has **no `Mech*`
   columns**. The funnel recomputes the mechanical screens itself from `Data/PriceData`. If you ever
   restore `signal_filter.py`, the funnel could read `Mech*` instead of recomputing.
2. **`0__signals` prices are ~2× stale** for some names (AU CurrentPrice 46.53 vs live $96.84) while
   the cap column matches live — looks like a price-feed/split bug in `2__PriceDownloader`. The funnel
   computes price/vol/RSI from `Data/PriceData` (fresh) and **nulls StopPrice/TargetPrice/ATR** on
   write so the broker anchors stop/target/trail to **live mid**. Worth investigating upstream: if the
   *features* fed to the model were also stale, the `UpProbability` values themselves are suspect.
3. **Weekly-vol must be `rng5`** — mean of daily `(High−Low)/Close` over the last 5 sessions ×100.
   This matches FinViz "Volatility W" within ~0.3%. A `std×√5` formula ran **~2× hot** and excluded
   the entire pool. Don't "fix" it back.
4. **`trade_history.parquet` is completed trades (1,112 rows), NOT the state ledger.** Do not route
   the per-ticker ledger there — it would clobber real history. The retired ledger goes to
   `Data/_retired_trading_ledger.parquet`.
5. **The scheduled task `\Trading_System` needs admin to modify** (`Set-ScheduledTask` → Access
   denied from a normal shell). The funnel's self-wait makes the trigger time non-critical, so this
   isn't blocking — but if you want to retime the 7:28 trigger, run the elevated command in the
   "Operations" section.
6. **Mechanical funnel ≠ the skill's discretionary pick.** The funnel ranks purely by
   `UpProbability` (no FilterRubric tiering, no diversification/HR-hostile overlay). On 2026-06-01 it
   would pick **AU, KGC, AG, DRS** (3 precious-metals miners by conviction) where the human skill kept
   **CVI** over a third gold miner for diversification. The `trade-signals` skill is the manual
   override. (A correlation/sector cap in Stage 4 is an open idea — not built.)
7. **API key is currently $0** → the LLM stage auto-skips; the mechanical funnel alone ships 4. To
   activate Stage 3, fund the account or drop a working key in `Claud-API-KEY.txt`.
8. **Date-labeling quirk.** Signals are tagged with `TargetDate = next trading day`, computed via the
   NYSE calendar. The broker does **not** gate on date — it trades whatever Pending rows exist.
9. **Don't run two backtests at once** (shared parquet writes can corrupt). Still applies.

---

## Operations

```powershell
# Manual narrow (the human-judgment path; HR-hostile + diversification overlay):
#   invoke the `trade-signals` skill in Claude Code → writes _Buy_Signals.parquet (4)
#   The morning funnel then detects this and exits $0.

# Mechanical/automated funnel (what the scheduler runs):
python 7__MacroFilter.py            # idempotent; waits to 9:35 ET for the LLM

# Live broker (port 7496 = LIVE; real money):
python 9_SuperFastBroker.py         # waits to 10:00 ET, SPY gate, gap filter, locks in

# OPTIONAL — retime the morning task 7:28 → 7:35 MT (run as Administrator):
$t = Get-ScheduledTask -TaskName 'Trading_System'
foreach($trg in $t.Triggers){ if($trg.StartBoundary -like '*T07:28:00'){ $trg.StartBoundary = $trg.StartBoundary -replace 'T07:28:00','T07:35:00' } }
Set-ScheduledTask -TaskName 'Trading_System' -Trigger $t.Triggers
```

**Scheduled task `\Trading_System`** (Daily): 5:00 PM MT → evening data pipeline; 7:28 AM MT →
morning mode (funnel → broker).

---

## Open items / TODOs

- [ ] **Fund the API key** (or place a working one in `Claud-API-KEY.txt`) to activate the LLM
      M&A/crisis stage. Until then the funnel is mechanical-only.
- [ ] **Investigate the stale `0__signals` prices** (~2× off) in `2__PriceDownloader` — possible
      split/adjustment bug; may also affect model features → `UpProbability` quality.
- [ ] **Decide whether to restore `signal_filter.py`** (so the backtester bakes `Mech*` columns and
      the funnel can read instead of recompute) or leave the logic in the funnel.
- [ ] **Optional:** add a correlation/sector cap to funnel Stage 4 so it won't stack 3 correlated
      miners on its own (close the mechanical-vs-discretionary gap).
- [ ] **Optional:** retime the scheduled task trigger to 7:35 MT (elevated) to avoid ~7 min of idle.
- [ ] **Optional:** give the broker a freshness check (refuse a book dated for a past session) — the
      current `MAX_BOOK`/`Status` guard catches un-narrowed files but not a stale-but-narrowed book.

---

## Decisions log (why, not just what)

- **`_Buy_Signals.parquet` = final book, ledger retired** — user's call; resolves the collision where
  the evening backtester's ledger dump would feed the morning broker garbage.
- **Funnel ranks by UpProbability** — the model's conviction is the core edge; the skill remains the
  discretionary override.
- **Keep broker entry at 10:00 ET** — EDA-calibrated; the gap comes from running the funnel earlier,
  not the broker later.
- **Self-wait instead of retiming the task** — no admin needed, robust to trigger-time drift, and
  short-circuited by idempotency when you've already hand-funneled.
- **Null stops/targets on write** — `0__signals` prices are stale; the broker's live-mid-anchored
  fallback (1.75% stop / 2R target / dynamic trail) is safer than a stale-derived level.
```
