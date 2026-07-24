---
name: trade-signals
description: Refresh today's signals file, run the FilterRubric (with HR-hostile discretionary overlay) on each ticker, rank the candidates down to the best 4 positions, then launch the live IBKR broker. Invoke when the user says things like "run the trading code", "generate the signals", "do the signals", "run today's trades", "filter the signals and trade", or similar phrasing referring to the morning trading workflow.
---

# Trade Signals Workflow

This is the morning workflow for the Stock-Market project. It cleans the signals file, screens each candidate against `FilterRubric.txt`, applies the user's discretionary "HR-hostile" overlay, ranks the candidates down to the best 4 positions, and runs the live IBKR broker. The model now emits a surplus of candidates, so the rubric is a funnel: the hard exclusions are auto-applied upstream (the `Mech*` columns), and your job is the soft-filter ranking that fills the 4-position book.

**Treat the live file as production.** Always back up before modifying. Always confirm before launching the broker.

## The two signal files — READ THIS FIRST

The pipeline uses **two** parquet files and they are NOT interchangeable:

- **`Data/0__signals.parquet`** — the wide **candidate pool** (≈12 names) the nightly
  pipeline refreshes. This is the file you READ and run the rubric against.
- **`_Buy_Signals.parquet`** (repo root) — the **narrowed book** (≤ `MAX_BOOK`, target 4)
  the broker ACTUALLY trades. `9_SuperFastBroker.py` hard-codes
  `BUY_SIGNALS_FILE = _Buy_Signals.parquet` and reads **nothing else**. Editing the pool
  has **zero** effect on what the broker trades until you write your picks into this file.

So the whole point of the workflow is: read the pool → pick the 4 → **write them to
`_Buy_Signals.parquet`** → verify that file → launch. A stale `_Buy_Signals.parquet` left
over from a previous day gets traded as-is if you don't overwrite it — this exact bug
shipped a 4-day-old book on 2026-06-26 (broker filled NEM/B/EL/OR/GPK off 06-22 signals
while the day's real picks sat unused in the pool). The Step 6 write + Step 7 guardrail
below exist specifically to prevent that.

## Inputs and outputs

- **Read:** `Data/0__signals.parquet` (candidate pool), `FilterRubric.txt`
- **Write:** `_Buy_Signals.parquet` (the broker's book — your selected ≤4), plus
  timestamped backups of **both** files before any change.
- **Run:** `python 9_SuperFastBroker.py` (live IBKR, port 7496) — reads `_Buy_Signals.parquet`.

## Step 1 — Inspect and relabel the candidate pool

This operates on the **candidate pool** `Data/0__signals.parquet` (not the broker book).
Use a single Python one-liner via Bash. Always timestamp the backup so old runs aren't clobbered.

```python
import pandas as pd, shutil
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.copy('Data/0__signals.parquet', f'Data/0__signals_backup_{ts}.parquet')
df = pd.read_parquet('Data/0__signals.parquet')
today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
now = pd.Timestamp.now()
df['_cd'] = pd.to_datetime(df['CreatedDate']).dt.normalize()
stale = df['_cd'] < today
df_fresh = df.loc[~stale].drop(columns=['_cd']).copy()
df_fresh['TargetDate'] = today
df_fresh['SignalDate'] = today
df_fresh['LastUpdated'] = now
df_fresh['LastUpdate']  = now
df_fresh.to_parquet('Data/0__signals.parquet', index=False)
```

Report: backup path, rows dropped as stale (with symbols), rows kept. **If EVERY row is
stale (all `CreatedDate < today`), the nightly pipeline did not run — stop and tell the
user; do not run the rubric or trade on a stale pool.**

## Step 2 — Research each remaining ticker

For every Symbol in the file run these in **parallel** (one tool message, multiple calls):

1. **FinViz quote** — `https://finviz.com/quote.ashx?t=<SYMBOL>` via WebFetch. Extract: Price, Market Cap, Beta, Volatility W, RSI (14), SMA200 %, Perf Quarter, Debt/Eq, Recom, Sector, Industry, EPS Q/Q, Short Float, Earnings Date.
2. **News scan** — WebSearch for `"<TICKER> <current month year> news merger acquisition lawsuit"` (substitute the actual month/year). Looking for: pending M&A, SEC actions, bankruptcy/liquidity warnings, fraud allegations, recalls, executive turmoil, layoffs.

If a ticker shape suggests a specific structural risk (biotech with trial readout, asset manager with redemption gates, miner with MSHA history, HR-tech vendor with TAM compression, etc.), add one targeted search for that risk.

## Step 3 — Apply FilterRubric.txt mechanically

Walk every ticker through the rubric **in order**:

1. **Step 1 hard exclusions** — any one triggers immediate Exclude:
   - Price < $2.00 — *auto-computed* (see `MechExclude` below)
   - Market cap < $952M (micro-cap) — *auto-computed*
   - Active M&A target (last 60 days) — **manual / web research**
   - Material crisis event (last 30 days that alters 5-day risk) — **manual / web research**
   - Weekly volatility > 5.0% — **the sharpest cliff edge; non-negotiable** — *auto-computed*
   - RSI between 30 and 40 (the death zone) — *auto-computed*

   **The four *auto-computed* checks were historically baked into the signals file** as
   columns `MechRSI14`, `MechWeeklyVolPct`, `MechExclude` (bool), `MechReasons` (str),
   written by `signal_filter.prefilter_signals_file()`. **As of 2026-06-26 `signal_filter.py`
   has been removed and the `Mech*` columns are usually ABSENT** — the import in `Util.py`
   fails silently, so don't count on them being there. Check for the columns first; if they
   exist and `MechExclude == True`, record the reason and move on. **If they're missing (the
   normal case now), compute the four checks yourself from the FinViz data you already pulled
   in Step 2:**
   - Price < $2.00
   - Market cap < $952M (micro-cap)
   - Weekly Vol (Volatility W) > 5.0%
   - RSI(14) between 30 and 40
   Any one true ⇒ the same hard exclusion. Don't try to re-run `signal_filter.py`; it's gone.

2. **Step 2 risk flags** — count them: A Beta>1.25, B small-cap with negative Perf Quarter, C below SMA200, D Recom<1.5, E D/E>3.0, F Real Estate or Consumer Cyclical sector, G Perf Quarter<-5%.

3. **Step 3 positive signals** — count them: 1 Mid/Large cap, 2 positive Perf Quarter (especially top-quartile), 3 Beta 0.5-0.75, 4 RSI 50-70, 5 EPS Q/Q>25%, 6 Basic Materials/Healthcare/Consumer Defensive/Utilities, 7 D/E 0.5-2.0.

4. **Step 4 — rank and select the best 4** (the deliverable is a ranked shortlist, not a per-ticker verdict; the book targets **4 active positions** and the model emits a surplus to choose from):
   - **Pass 1 — disqualify:** drop anything with `MechExclude == True`, a confirmed active M&A target, or a material crisis event. These are out and cannot rank back in.
   - **Pass 2 — tier the survivors** (flags weigh more than positives):
     - Tier 1: 0-1 flags AND ≥2 positives
     - Tier 2: 0-1 flags AND 0-1 positives
     - Tier 3 (marginal, 50% size only): exactly 2 flags AND ≥1 positive
     - Disqualified: 3+ flags, OR 2 flags with 0 positives
   - **Pass 3 — fill 4 slots top-down** (Tier 1 → 2 → 3). Within a tier rank by strongest positives: Signal 2 (top-quartile Perf Quarter) > Signal 4 (RSI 50-70) > Signal 1 (Mid/Large cap) > Signal 3 (Beta 0.5-0.75) > rest; break further ties on fewer flags, then `UpProbability`. List **2 ranked alternates** as broker fallbacks.
   - If fewer than 4 survive (thin day): ship only what qualifies — never backfill from the disqualified pool. An empty slot beats a hard-excluded trade.

## Step 4 — Apply the HR-hostile discretionary overlay

**This is the user's stated edge.** Read [[feedback_predictor_focus]] context: the user actively avoids "what HR would approve of" thinking and quarantines ideologically-exposed tickers (see [[project_ideological_ticker_exclusion]] — HIMS, DJT, Israel-HQ'd names). Apply the same skepticism here.

For each ticker, briefly answer:
- **What is the actual core business?** Not the SIC code — what they sell, to whom, and whether that customer base is structurally growing or shrinking.
- **What would an ESG/HR-aligned analyst miss or mis-rate?**
  - Companies where HR/DEI funds *can't own* but the underlying cash flows are improving (ESG-banned commodity producers, defense, tobacco-adjacent, etc.) are often undervalued — lean toward keep.
  - Companies where +EPS growth is mark-to-model or cost-out (private credit asset managers, restructuring software cos) deserve extra skepticism even if the rubric is silent.
  - Companies selling INTO the HR/compliance/DEI complex face TAM compression in the current administration's environment. Flag this.
  - Country-of-domicile and ADR structure matter (Cayman/Taiwan/Israel parents). Note even if not on the quarantine list.

This overlay can **add caution** to a mechanical Include (downgrade to Conditional or watch-only) but should not override a mechanical hard exclusion. Be explicit about which findings are mechanical-rubric vs discretionary-overlay.

## Step 5 — Output per the rubric's format

For every ticker, produce the block from `FilterRubric.txt` lines 156-172:

```
Ticker: [SYMBOL]
Price: $[XX.XX]  |  Market Cap: $[X]B/M  |  Cap Tier: [Micro/Small/Mid/Large]
Beta: [X.X]  |  Weekly Vol: [X.X%]  |  RSI: [XX]  |  SMA200: [above/below, X%]
Perf Quarter: [X%]  |  Debt/Eq: [X.X]  |  Analyst Recom: [X.X — label]
Sector: [sector]  |  EPS Q/Q: [X%]

HARD EXCLUSIONS TRIGGERED: [list or "None"]
RISK FLAGS ACTIVE: [list flags A–G or "None"]
POSITIVE SIGNALS PRESENT: [list 1–7 or "None"]

M&A STATUS: [None identified / Active: details]
CRISIS EVENTS: [None identified / Flag: details]

CORE BUSINESS / HR-HOSTILE NOTE: [1–3 sentences on what they do, who buys, and the discretionary read]

RECOMMENDATION: [Include / Conditional / Exclude]
POSITION SIZING: [Standard / Reduced 50% / No Position]
KEY REASON: [1–2 sentences — the most important factor]
```

End with the **selection table** (the deliverable): rank | ticker | UpProb | tier | flags | positives | size | key reason — the top 4 first, then 2 alternates below a `-- alternates --` divider, then a `SELECTED FOR THE BOOK:` line listing the 4 symbols (or fewer on a thin day).

## Step 6 — Write the selected 4 to the broker's book (`_Buy_Signals.parquet`)

**This is the step that actually matters — it's the only file the broker reads.** Pull the
selected rows from the pool (they carry the full rich schema the broker wants: `StopPrice`,
`TargetPrice`, `ATR`, etc.) and **overwrite** `_Buy_Signals.parquet`. Always overwrite, never
append — a leftover stale book is exactly the failure mode this prevents.

**Confirm with the user before writing.** They are the final authority on the final 4 —
including whether a Tier-3 (50% size) name ships or a slot is left empty on a thin day.

```python
import pandas as pd, shutil, os
from datetime import datetime
ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
now   = pd.Timestamp.now()

# Back up BOTH files before touching anything.
shutil.copy('Data/0__signals.parquet', f'Data/0__signals_backup_pretrim_{ts}.parquet')
if os.path.exists('_Buy_Signals.parquet'):
    shutil.copy('_Buy_Signals.parquet', f'Data/_Buy_Signals_backup_{ts}.parquet')

pool = pd.read_parquet('Data/0__signals.parquet')
keep = [<the selected symbols, in rank order>]          # e.g. ['CMBT','RHI','AEO','TOST']

book = pool[pool['Symbol'].isin(keep)].copy()
book['_o'] = book['Symbol'].map({s: i for i, s in enumerate(keep)})
book = book.sort_values('_o').drop(columns='_o').reset_index(drop=True)
book['Status']      = 'Pending'
book['TargetDate']  = today
book['SignalDate']  = today
book['LastUpdated'] = now
book['LastUpdate']  = now
book['VetSource']   = 'manual'   # REQUIRED: marks the book hand-vetted; 7__MacroFilter.py
book['VetTime']     = now        # will never auto-overwrite a same-session 'manual' book
book.to_parquet('_Buy_Signals.parquet', index=False)    # <-- the file the broker reads
print('Wrote', len(book), 'to _Buy_Signals.parquet:', list(book['Symbol']))
```

**Do not omit the `VetSource='manual'` stamp.** It is what stops the automated funnel
(`7__MacroFilter.py`, launched by `trading_system.ps1` every morning and by the overnight
pipeline) from clobbering your hand-vetted book with its mechanical fallback — that exact
clobber happened at 07:35 on 2026-07-02 and the broker traded the unvetted 8-name book.

Then run a final QC **against `_Buy_Signals.parquet`** (not the pool): row count == number
selected, no nulls in `Symbol/UpProbability/CurrentPrice/StopPrice/TargetPrice`, all
`Status == 'Pending'`, all dated today, `StopPrice < CurrentPrice < TargetPrice`, all
`VetSource == 'manual'`.

## Step 7 — Verify the broker's input, then launch

`9_SuperFastBroker.py` defaults to **port 7496 (LIVE IBKR)**. It waits until 10:00 ET, then gates on SPY direction (abort if SPY ≤ −0.5%) and per-stock gap (skip if gapped > +2% at open — the script has its own intraday filter that can veto a rubric-Include). It reads **`_Buy_Signals.parquet`**, filters to `Status == 'Pending'`, and **aborts if more than `MAX_BOOK` (12) pending rows are present**.

**Guardrail — run this BEFORE launch.** It reads the broker's actual input and confirms it is today's book and nothing stale. This is the check that would have caught the 2026-06-26 incident:

```python
import pandas as pd
from datetime import datetime
EXPECTED = [<the selected symbols>]          # same list you wrote in Step 6
b = pd.read_parquet('_Buy_Signals.parquet')
today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
assert set(b['Symbol']) == set(EXPECTED), f"book {sorted(b['Symbol'])} != selected {sorted(EXPECTED)}"
assert (b['Status'] == 'Pending').all(), "non-Pending rows present"
assert (pd.to_datetime(b['TargetDate']).dt.normalize() == today).all(), "STALE rows — _Buy_Signals.parquet not refreshed today"
assert len(b) <= 12, "exceeds MAX_BOOK — broker will abort"
assert 'VetSource' in b.columns and (b['VetSource'] == 'manual').all(), "book not stamped manual — the automated funnel may overwrite it"
print('OK to launch:', list(b['Symbol']))
```

If any assertion fails, **do not launch** — go back to Step 6. A stale or mismatched
`_Buy_Signals.parquet` is the one failure mode that silently trades the wrong names.

**Confirm with the user one more time before launch.** This is real money.

After confirmation, run in the background and tee to a log:

```bash
python 9_SuperFastBroker.py 2>&1 | tee -a "broker_run_$(date +%Y%m%d_%H%M%S).log"
```

When the background task completes, read the log and report:
- Connection status and account NAV
- SPY gate result
- For each symbol: bid/ask/mid/spread, open gap, skipped vs ordered
- For each placed bracket: shares, limit, hard stop, trail %, take profit, OCA group

## Notes on common surprises

- **The broker reads `_Buy_Signals.parquet`, NOT `Data/0__signals.parquet`.** This is the
  single most important thing in this skill. The pool is for picking; the book is for
  trading. If you only edit the pool, the broker trades whatever stale book is already on
  disk. Always write Step 6 and run the Step 7 guardrail.
- **`_Buy_Signals.parquet` can be a stale leftover.** The nightly narrowing step (in
  `7__MacroFilter.py`) does not reliably refresh it — on 2026-06-26 the pool was fresh but
  `_Buy_Signals.parquet` was 4 days old, and the broker traded the old names. Overwrite it
  every run; never trust its existing contents.
- **The automated funnel can also overwrite YOUR book — unless it is stamped.** On
  2026-07-02 a hand-vetted 3-name book was clobbered at 07:35 by the morning
  `7__MacroFilter.py` run (its session-date guard was broken and its LLM stage silently
  degraded to mechanical-only on an unfunded API key); the broker then traded the unvetted
  8-name mechanical book. Both bugs are fixed (session now derived from the pool's
  TargetDate; provenance guard added), but the protection **only recognizes books with
  `VetSource == 'manual'`** — always write the stamp in Step 6. Overriding a manual book
  requires running `7__MacroFilter.py --force` by hand.
- **The broker also writes its own log** to `Data/logging/FastExecutor.log` regardless of
  the tee'd `broker_run_*.log`. If the tee file is missing or empty, read
  `Data/logging/FastExecutor.log` to confirm what the broker actually did (it records the
  exact symbol batch it read, so it's the ground truth for which file/book was traded).
- **The signals file may be empty or pre-filled with stale rows.** If empty, say so and stop — don't fabricate signals.
- **The rubric's RSI death zone is 30-40, not <30.** RSI just above 40 (e.g. 42) is close-but-allowed; flag it as a watch.
- **Strong Buy consensus (Recom < 1.5) is a flag, not a positive.** Counterintuitive — historical worst-WR band.
- **The broker's own gap filter is a feature, not a bug.** It can override a rubric-Include intraday and that's fine; report it as the system working.
- **Backups accumulate fast.** Don't auto-delete them — they're the only undo path.
