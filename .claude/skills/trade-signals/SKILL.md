---
name: trade-signals
description: Refresh today's signals file, run the FilterRubric (with HR-hostile discretionary overlay) on each ticker, rank the candidates down to the best 4 positions, then launch the live IBKR broker. Invoke when the user says things like "run the trading code", "generate the signals", "do the signals", "run today's trades", "filter the signals and trade", or similar phrasing referring to the morning trading workflow.
---

# Trade Signals Workflow

This is the morning workflow for the Stock-Market project. It cleans the signals file, screens each candidate against `FilterRubric.txt`, applies the user's discretionary "HR-hostile" overlay, ranks the candidates down to the best 4 positions, and runs the live IBKR broker. The model now emits a surplus of candidates, so the rubric is a funnel: the hard exclusions are auto-applied upstream (the `Mech*` columns), and your job is the soft-filter ranking that fills the 4-position book.

**Treat the live file as production.** Always back up before modifying. Always confirm before launching the broker.

## Inputs and outputs

- **Read:** `Data/0__signals.parquet`, `FilterRubric.txt`
- **Write:** `Data/0__signals.parquet` (in place, after backup), `Data/0__signals_backup_<TIMESTAMP>.parquet`
- **Run:** `python 9_SuperFastBroker.py` (live IBKR, port 7496)

## Step 1 — Inspect and relabel the signals file

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

Report: backup path, rows dropped as stale (with symbols), rows kept.

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

   **The four *auto-computed* checks are already baked into the signals file.** The
   nightly backtester runs `signal_filter.prefilter_signals_file()` as its last step,
   adding columns `MechRSI14`, `MechWeeklyVolPct`, `MechExclude` (bool), `MechReasons`
   (str) to `Data/0__Signals.parquet`. Rows are **kept, not dropped** — just flagged.
   Read these columns first; for any row with `MechExclude == True`, the price / cap /
   weekly-vol / RSI verdict is done — only confirm the two web-research exclusions
   (M&A, crisis) and the soft flags. If the file is missing the `Mech*` columns (e.g.
   hand-edited mid-cycle), re-run `python signal_filter.py` to regenerate them.

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

## Step 6 — Trim the file to the selected 4

**Confirm with the user before modifying the file.** They are the final authority on the final 4 — including whether a Tier-3 (50% size) name ships or a slot is left empty on a thin day.

After confirmation, keep the selected symbols (the 4, ranked; alternates only ship if the user swaps one in):

```python
import pandas as pd, shutil
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.copy('Data/0__signals.parquet', f'Data/0__signals_backup_pretrim_{ts}.parquet')
df = pd.read_parquet('Data/0__signals.parquet')
keep = [<the selected symbols, in rank order>]
df[df['Symbol'].isin(keep)].to_parquet('Data/0__signals.parquet', index=False)
```

Then run a final QC: row count, no nulls in `Symbol/UpProbability/CurrentPrice/StopPrice/TargetPrice`, all `Status == 'Pending'`, all dated today, `StopPrice < CurrentPrice < TargetPrice`.

## Step 7 — Launch the live broker

`9_SuperFastBroker.py` defaults to **port 7496 (LIVE IBKR)**. It waits until 10:00 ET, then gates on SPY direction (abort if SPY ≤ −0.5%) and per-stock gap (skip if gapped > +2% at open — the script has its own intraday filter that can veto a rubric-Include).

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

- **The signals file may be empty or pre-filled with stale rows.** If empty, say so and stop — don't fabricate signals.
- **The rubric's RSI death zone is 30-40, not <30.** RSI just above 40 (e.g. 42) is close-but-allowed; flag it as a watch.
- **Strong Buy consensus (Recom < 1.5) is a flag, not a positive.** Counterintuitive — historical worst-WR band.
- **The broker's own gap filter is a feature, not a bug.** It can override a rubric-Include intraday and that's fine; report it as the system working.
- **Backups accumulate fast.** Don't auto-delete them — they're the only undo path.
