# Handoff — finish the mechanical signal pre-filter

The harness was lagging tool output badly in the previous session, so picking this
up fresh. Everything below is the remaining work; the hard part (the filter logic)
is already written and tested.

## Context / goal

Automate the **non-web** hard exclusions from `FilterRubric.txt` so the live broker
reads a pre-screened signals file and we don't hand-filter through ~20 trash signals
every morning. Decision already made: **annotate, don't drop** (add verdict columns),
**bake the logic into `Util.py`**, and run it as the **very last step** of signal
generation before the broker.

The 4 automated checks (FilterRubric Step-1, the ones that need no web research):
1. Price < $2.00
2. Market cap < $952M (`CapMillions` column)
3. Weekly volatility > 5% (avg daily High-Low / Close over last 5 sessions)
4. RSI(14) in [30, 40] (the "death zone")

M&A + crisis-event exclusions and the Step-2 soft flags stay manual (web research).

## What's already DONE (verified working)

- **`signal_filter.py`** created at repo root. Standalone CLI + importable functions:
  - `annotate_signals_mechanical_filter(df, ...)` — adds columns `MechRSI14`,
    `MechWeeklyVolPct`, `MechExclude` (bool), `MechReasons` (str). No rows dropped.
  - `prefilter_signals_file(path, write=, drop=, backup=)` — reads/annotates/writes
    `Data/0__Signals.parquet`, timestamped backup.
  - `compute_rsi14()` (Wilder), `compute_weekly_vol_pct()` — pure, unit-testable.
  - CLI: `python signal_filter.py` (annotate in place + backup),
    `--no-write` (report only), `--drop` (also physically remove excluded rows).
  - Reads price history from `Data/RFpredictions/<SYM>.parquet` (cols Date/High/Low/Close).
    Has an ATR/price fallback for weekly-vol when history is missing.
- **Dry-run passed** on the live 20-row `Data/0__Signals.parquet`. Result: 17 PASS,
  3 EXCLUDE — AVO (cap $800M < $952M), HII (RSI 35.75), CALX (RSI 39.81). Output is a
  clean green/red PASS/EXCLUDE table. Logic confirmed correct.

## What's LEFT to do

1. **Bake into Util.py** (the user's explicit ask — "bake this into the util file").
   Decide one of:
   - (a) `from signal_filter import annotate_signals_mechanical_filter, prefilter_signals_file`
     near the top of `Util.py` and re-export, keeping the impl in `signal_filter.py`, OR
   - (b) move the function bodies into `Util.py` directly and have `signal_filter.py`
     import from `Util`.
   Recommendation: (a) — smaller diff, keeps Util.py (already 3960 lines) from growing,
   single source of truth. Confirm with user.
   NOTE: `Util.py`'s real header name is `0__Util.py` per its docstring; it already has
   `read_signals()`/`write_signals()` (lines ~717/771) — reuse those for IO consistency.

2. **Wire it as the LAST step of signal generation** so the broker always gets the
   annotated file. Candidates to call `prefilter_signals_file()` at the very end:
   - `5__NightlyBackTester.py` (and/or the experimental copy) after it writes
     `Data/0__Signals.parquet` via `save_guaranteed_signals_to_parquet()`, OR
   - prepend a call inside the `trade-signals` skill flow, OR
   - `9_SuperFastBroker.py` / `9_DailyBrokerFast.py` on startup before it reads signals.
   Ask the user which entry point they consider "the very very last signals" step.

3. **Make the broker respect `MechExclude`.** Even with columns present, confirm the
   broker filters them out (either we `--drop` at write time, or the broker skips rows
   where `MechExclude == True`). Check how `9_SuperFastBroker.py` reads the file.

4. **Update `.claude/skills/trade-signals/SKILL.md`** — add a Step (before the manual
   research) that runs the mechanical pre-filter so the analyst only researches PASS
   rows. The 4 automated checks can be removed from the manual Step-3 hard-exclusion
   list (or marked "auto-handled").

5. **Optional:** unit-test `compute_rsi14` / `compute_weekly_vol_pct` against a known
   series; decide whether `--drop` should be the default in production.

## Watch-outs

- `Data/0__Signals.parquet` is PRODUCTION. Always back up before writing (the code does).
- Signals-file price column priority: `CurrentPrice` → `SignalPrice` → `EntryPrice`.
- Two rubric copies exist: `FilterRubric.txt` (root) and a duplicate; thresholds match.
- The previous session also created `5__NightlyBackTester_experimental.py` with the
  `--signal_count` knob + ticker-tape output — unrelated to this task, leave as-is.
