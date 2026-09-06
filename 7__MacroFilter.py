#!/usr/bin/env python
"""
7__MacroFilter.py - the signal FUNNEL (pool -> final book).

*** OUT OF THE LIVE PATH SINCE 2026-08-28. READ THIS BEFORE TRUSTING ANYTHING BELOW. ***
The broker now runs trigger entry by DEFAULT and arms straight off the WIDE pool
(Data/0__Signals.parquet). It does not read _Buy_Signals.parquet at all unless it is
launched with --no-trigger-entry. So this module still runs and still writes a narrowed
book, but nothing trades that book on the default path: the narrowing is dead output.
The discretionary layer that replaced it is a shallow veto recorded ON the pool by
rubric_veto.py (RubricExclude, capped at 4 names). Keep this file for the rollback path
and for its Stage-1 mechanical screens; do not extend it expecting to steer live trades.

Reads the candidate POOL (Data/0__Signals.parquet, cap 24 names written by the
nightly backtester) and narrows it to the final book of <=TARGET_BOOK_SIZE names (10)
that the live broker (9_SuperFastBroker.py) trades from _Buy_Signals.parquet ONLY under
--no-trigger-entry. Cost-ordered, so the expensive LLM step only ever sees the few names
that survive the free mechanical screens:

  Stage 0  Load pool, align to the next NYSE trading day, attach price history.
  Stage 1  HARD mechanical exclusions (free, no API): price floor, micro-cap,
           weekly-vol cliff, UpProbability floor, stuck-price flatline, ideological
           quarantine. Dropped.
  Stage 2  SOFT mechanical flags (free, no API): penny/illiquid, a big-gap-then-vol-
           collapse "deal-peg" signature, and overbought (RSI>80) overextension. These
           DEPRIORITIZE (and hand the name to the LLM); they do not drop on their own.
  Stage 3  LLM summary judgement (paid, best model, max effort) on survivors ONLY:
           claude-opus-4-8 + web_search confirms active M&A target / material
           crisis. Auto-SKIPS cleanly if the API key is unfunded/invalid - the
           mechanical funnel alone still produces a book.
  Stage 4  Rank survivors by UpProbability and take the top TARGET_BOOK_SIZE, subject to
           an industry/sector CONCENTRATION cap (clean first, then relax soft-flags AND
           the cap to fill the book so capital stays deployed), write the book.

  NOTE (2026-07-24): the Stage-1 MICRO-CAP GATE HAD BEEN SILENTLY DEAD. FinViz's
  ticker_fundament() started raising, 5__NightlyBackTester swallowed it with a bare
  `except`, and CapMillions arrived null for every pool row - and `pd.notna(cap)` turns a
  null into "passes". The gate is a validated keeper, so it now falls back to the PIT panel
  (Data/MarketCaps/historical_market_caps.parquet). A GATE CENSUS now prints per-gate
  evaluated / no-input / fired counts every run so the next such failure is visible in one
  line instead of invisible for weeks. Slot-replay 2022-07..2026-07 values the restored
  gate at +11.1pp (pool-12) / +5.2pp (pool-25), positive in every year, both halves and
  every book size tested. See experimental/macrofilter_v2/FINDINGS.md.

  NOTE (2026-06-24): the old Stage-1 RSI "death-zone" [30,40] HARD exclusion was REMOVED.
  A 17.6k-candidate study (analysis_output/analyze_macro_filter.py) showed [30,40] is the
  BEST RSI band (oversold-bounce, +0.173% vs +0.060% kept) - the gate was dropping winners
  and was the only screen net-negative at book level (-0.029%/day). Overextension risk
  (RSI>80) is now a soft de-prioritization, and the user-requested concentration cap was
  added in Stage 4. See analysis_output/MACRO_FILTER_FINDINGS.md.

IDEMPOTENT + PROVENANCE-GUARDED: the target session is taken from the POOL's
TargetDate, not the calendar - get_next_trading_day(today) returns the day AFTER
today, so any run on the session day itself (the overnight pipeline finishing at
~02:00, the 07:00 morning task) used to compute the wrong session, the guard never
matched, and every rerun clobbered the book (2026-07-02: overwrote a hand-vetted
book with the mechanical fallback while the API key was unfunded).
If _Buy_Signals.parquet already holds a narrowed book for the target session:
  - VetSource='manual' (hand-vetted via the trade-signals skill): NEVER replaced
    automatically; --force is the only override.
  - VetSource='llm': already vetted - exits immediately, spends $0.
  - VetSource='mechanical' (or an older unstamped book): re-runs ONLY if this run
    has a working LLM stage to upgrade it; a mechanical rerun keeps the book.
The guard is re-checked right before the write (the research-window wait is up to
~35 min), and a run whose LLM died mid-flight (0 successful checks) refuses to
overwrite an existing same-session book. Use --force to re-run anyway.
"""



import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Util import get_logger, get_next_trading_day

# Diagnostics sidecars (diagnostics/hooks.py). No-op unless DIAG_OUT is set; the
# stand-in below keeps this script independent of the diagnostics folder.
try:
    from diagnostics import hooks as _diag
except Exception:
    class _diag:
        enabled = staticmethod(lambda: False)
        dump_json = dump_parquet = append_jsonl = stamp = staticmethod(lambda *a, **k: None)

# Shared industry classifier (FinViz Industry -> fine group, + coarse sector rollup). Used
# for the Stage-4 concentration cap. Degrades gracefully if the module is unavailable.
try:
    from build_data_panels import group_from_text, coarse_sector, COARSE_FROM_GROUP
except Exception:
    COARSE_FROM_GROUP = {}
    def group_from_text(_):       # type: ignore
        return None
    def coarse_sector(_):         # type: ignore
        return "Other"

logger = get_logger(script_name="7__MacroFilter")

# ── Files ─────────────────────────────────────────────────────────────────────
POOL_FILE = "Data/0__signals.parquet"      # input: candidate pool (cap 24 since 2026-08-28)
BOOK_FILE = "_Buy_Signals.parquet"         # output: narrowed book. The broker reads this
                                           # ONLY under --no-trigger-entry (rollback path).
PRICE_DIR = "Data/PriceData"               # per-ticker daily OHLCV (price source of truth)
QUARANTINE_DIR = "Data/_quarantine_ideological"
API_KEY_FILE = os.path.join("auxiliary", "Claud-API-KEY.txt")
MCAP_FILE = "Data/MarketCaps/historical_market_caps.parquet"   # PIT cap fallback (see below)

# ── Book sizing ─────────────────────────────────────────────────────────────────
TARGET_BOOK_SIZE = 3   # 4->8->10->3 (2026-08-23). The k10 choice came from a joint
                       # seed x config x book grid (Data/_jointstress) run on a GROSS
                       # backtest that carried NO cost model, so it could not see the
                       # only thing book size strongly controls: per-trade cost. On the
                       # 362-fill calibration, 98.9% of orders pay the $1.00 Fixed floor,
                       # making commission percentage a pure function of notional. That
                       # grid is superseded on cost, not refuted on signal - if a cost-
                       # aware book-size sweep is ever run, re-read it before moving this.
                       # Broker PositionSizer (9_SuperFastBroker, max_positions) must match.
MAX_BOOK = 4           # mirrors the broker's fail-safe guard (TARGET_BOOK_SIZE + 1)

# ── Stage 1 hard-exclusion thresholds (FilterRubric Step-1) ──────────────────────
PRICE_FLOOR = 5.00         # exclude if latest close < $5
MICRO_CAP_MAX_M = 952.0    # exclude if market cap < $952M (micro). 2026-07-24: this gate had
                           # been SILENTLY DEAD - finvizfinance.ticker_fundament() started
                           # raising, 5__NightlyBackTester swallowed it with a bare except, and
                           # CapMillions came through null for every pool row, so the
                           # `pd.notna(cap)` guard below never fired. Now falls back to the PIT
                           # panel (MCAP_FILE). Slot-replay 2022-07..2026-07: restoring it is
                           # worth +11.1pp (pool-12) / +5.2pp (pool-25), positive in every year,
                           # both halves and every book size. See
                           # experimental/macrofilter_v2/FINDINGS.md.
WEEKLY_VOL_MAX_PCT = 5.0   # exclude if weekly volatility > 5.0% (the sharp cliff). VALIDATED
                           # both ways (2026-06-24): excluded names -0.08% net vs +0.32% kept
                           # on real intraday fills; +candidate panel monotonic. Keep.
# RSI(14) overbought de-prioritization (SOFT, see soft_flags). The old [30,40] HARD
# "death-zone" exclusion was REMOVED 2026-06-24 - it was dropping the best RSI band.
RSI_OVERBOUGHT_HI = 80.0   # soft-flag if RSI(14) > 80 (panel: RSI>80 = -0.43%, worst band)
# Cross-sectional UpProbability floor. map_pct_rank_to_upprob puts the model's
# non-top-fraction names in [0.30,0.44]; an EDA on 944 real backtest trades
# (2026-06-18) found [0.30,0.38) is a stable money-loser (neg PnL in both 2025-H2
# and 2026-H1, ~42% win) while [0.42,0.44) is the profitable workhorse. Flooring at
# 0.40 cuts the junk band only: +$4.7k total PnL, per-trade Sharpe 0.19->0.32. A
# 0.45 floor was REJECTED (would cut the workhorse band, -$9k / -67% of profit).
UPPROB_FLOOR = 0.40        # exclude if entry-day UpProbability < 0.40

# Hardcoded ideological quarantine (union'd with QUARANTINE_DIR contents at runtime).
# TOST: operator do-not-buy, added 2026-08-07 per instruction "no more buying TOST".
QUARANTINE_SEED = {"HIMS", "DJT", "ODD", "TOST"}

# ── Stage 2 soft (delisting / merger) heuristic params ───────────────────────────
MERGER_LOOKBACK = 40       # sessions to scan for a deal-gap
MERGER_GAP_PCT = 15.0      # a single-day move this big...
MERGER_POSTVOL_MAX = 1.0   # ...followed by daily realized vol below this % => deal-peg
DELIST_PRICE = 3.00        # penny-ish
DELIST_DOLLAR_VOL = 1_000_000.0   # median 20d dollar volume below this => illiquid

# ── STUCK-PRICE gate (HARD, 2026-08-18) ──────────────────────────────────────────
# A name that barely moves cannot reach the broker's +6% target inside the 5-day clock,
# so the slot is dead capital for a week. Measured on the 18,361-row unfiltered candidate
# panel (2022-2026) scored at the LIVE 3.0/6.0 bracket: candidates whose LARGEST intraday
# range over the last 10 sessions stayed under 1.0% of price hit the target 0.0% of the time
# (0 of 123 cases) and timed out 100.0% of the time, vs a 23.3% baseline timeout rate
# (t=-2.36). Zero target-hits in EVERY year the gate fired (2024 n=93, 2025 n=30).
#
# PERCENT of price, NOT absolute cents, and that distinction is load-bearing. An absolute
# "daily range < $0.03" variant was tested and REJECTED: it flags NIO, a $5 stock that
# ranges 2-7% a day, purely because cheap stocks have small absolute ranges. Absolute cents
# measures price level; percent measures whether the stock actually moves.
#
# HARD rather than soft, on the delisting precedent: this is tail insurance. It fires on
# 0.84% of candidates and would have blocked 0 of the 813 trades in the live record, so it
# costs nothing on a normal book and exists for the halted / deal-pegged / flatlined name.
# THRESHOLD SWEEP 2026-08-18 (Data/_stuckgrid/: sweep.csv, heatmap.png, cliff.png).
# 240-cell grid over measure x window x threshold. The surface is SMOOTH and MONOTONE -
# no bumps, no knife-edge - with one hard CLIFF: below 1.0-1.25% intraday range, ZERO cut
# names ever reach +6%, and above it the target-hit rate climbs steadily (1.25% -> 0.5%,
# 1.5% -> 1.7%, 2.0% -> 3.7%, 3.0% -> 10.6%). So the gate is set by walking BACK from the
# cliff to the last threshold where nothing tradeable is lost.
#
# MAX-over-window, not a count of tiny days: one number, no counting, and it catches more
# names at the same zero-target-hit safety (123 at 10d/1.0% with 100% timeout, vs 154 for
# the old ">=5 of 10 under 0.5%" form at 96.1% timeout).
#
# WHY INTRADAY RANGE AND NOT CLOSE-TO-CLOSE: a flat close-to-close can hide a large
# intraday swing. Measured: of 254 candidates whose 3-day close-to-close move stayed under
# 0.5%, 104 still ranged >=1% INTRADAY - tradeable names a close-to-close gate would have
# wrongly cut. The sweep shows it directly: the close-to-close panel starts hitting targets
# at a 0.2% threshold, while intraday range holds at exactly 0.0% all the way to 1.25%.
STUCK_RANGE_PCT = 1.0      # if the LARGEST (High-Low)/Close over the window is under this %...
STUCK_LOOKBACK_DAYS = 10   # ...across this many trailing sessions => the name is flatlined



# ── Rule 12b-25 LATE-FILING notices (NT 10-Q / NT 10-K / NT 20-F) ────────────────
# SOFT, and gated on OCCURRENCE - deliberately NOT on days-late. Measured 2026-07-28 on
# Data/PriceDataFull, 900 tickers, 2010-01..2026-05 (2.66M ticker-days, 5 folds, tie-matched
# ticker-permuted placebo, ranked on placebo-NET excess):
#
#     gate on days-late > T      T=0(none)  5     10    17    25    40
#     16-year placebo-net excess    0.49   0.39  0.32  0.18  0.17  0.14   Spearman(T,excess) = -1.00
#      3-year placebo-net excess    0.38   0.33  0.32  0.34  0.33  0.40   Spearman(T,excess) = +0.23
#
# The literature prior (12b-25 grants +5d for a 10-Q and +15d for a 10-K, so a delay past ~17d
# means the filer used the extension and STILL missed) is endorsed ONLY by the 3-year window.
# At 16 years the gate erodes the edge MONOTONICALLY - no plateau, no knife-edge, just loss.
# A power-matched control settles it: sfn_nt_maxlate_log_3y grades severity continuously and so
# discards no episodes at all, and it still scores 0.50 against the ungated 3-year COUNT's 0.65.
# Severity is not the mechanism; the ACT of filing a late notice is. So the flag fires on
# occurrence, and days-late is carried in the flag TEXT as description for Stage 3 / the human.
#
# SOFT and not a hard cut, on this repo's own precedent: soft_flags deprioritizes (clean names
# sort ahead of flagged ones, both by UpProb) but still lets a flagged name fill a thin book.
# The earnings-gate probe found a hard gate wrong where a size-down was right, and the micro-cap
# gate's whole failure mode was a hard gate that went silently inert. Base rates on the 3,839
# covered tickers: >=1 notice in 60d = 0.49% (rare, high-conviction), >=2 in 3y = 6.5%.
FILING_CAL_DIR = "Data/SEC/filing_calendar"   # built by build_filing_calendar.py
NT_RECENT_DAYS = 60        # soft-flag if a 12b-25 notice landed within this many days
NT_CHRONIC_COUNT_3Y = 2    # ...or if this many notices landed in the trailing 3 years
NT_OVERDUE_STALE_DAYS = 365   # an "outstanding" notice older than this is an abandoned shell,
                              # not a pending filing (measured: ABEO carried four unresolved
                              # 2005-2009 notices, which a raw clock read as 7,735 days overdue)

# ── Concentration risk (industry / sector crowding) ──────────────────────────────
# The book is equal-weight (~1/TARGET_BOOK_SIZE per name, see 9_SuperFastBroker's
# PositionSizer), so a 0.50 group cap == "don't let >50% of the portfolio sit in one
# industry" - exactly the user's biotech / gold-miner example. A 17.6k-candidate study
# (analysis_output/concentration_risk.py) found dangerous single-industry crowding in the
# daily top-10 is RARE (fires on ~2% of book-days) and capping it costs ~0.00%/day of mean
# return - near-free tail insurance against a correlated single-theme drawdown. The caps
# are SOFT: if honoring them would leave the book unfilled, they relax (capital deployed).
SECTOR_MAP_FILE = "Data/SectorMap.parquet"   # built by build_data_panels.py sector (SEC SIC codes)
CONC_GROUP_CAP_FRAC  = 0.50   # max book fraction in one fine IndustryGroup (Biotech, GoldMiner, ...)
CONC_SECTOR_CAP_FRAC = 0.70   # looser cap on the coarse Sector (Healthcare, Tech, Energy, ...)

# ── Stage 3 LLM config (see claude-api skill) ────────────────────────────────────
LLM_MODEL = "claude-opus-4-8"   # best model - real money
LLM_EFFORT = "max"              # maximum effort (Opus-tier); dial to "high" to save tokens
LLM_MAX_TOKENS = 16000          # streamed, so well clear of the non-streaming timeout guard
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
LLM_SLEEP_BETWEEN = 2.0         # gentle spacing between the handful of calls
LLM_MAX_CHECKS = TARGET_BOOK_SIZE + 2   # cap calls: research top names until book full (+2 buffer)
LLM_CALL_TIMEOUT = 150.0        # seconds/call - hard deadline: broker locks in at 10:00 ET

# ── Research timing ──────────────────────────────────────────────────────────────
# Run the LLM AFTER the open for fresher news, but finish well before the broker
# locks in at 10:00 ET. When launched within MAX_PREOPEN_WAIT_MIN of the window the
# funnel self-waits to RESEARCH_START_ET before web-searching; otherwise (off-hours
# / manual run, or already past) it proceeds immediately. The cheap mechanical
# screens always run first - they use prior-close data, so their timing is moot.
ET = ZoneInfo("America/New_York")
RESEARCH_START_ET = (9, 35)     # 5 min after the 9:30 ET open
MAX_PREOPEN_WAIT_MIN = 45       # never idle-wait longer than this (guards off-hours runs)

LLM_SYSTEM_RUBRIC = (
    "You are a risk screener for a short-horizon (about 5 trading days), long-only US "
    "equity strategy. For the single ticker given, use web search to determine TWO things "
    "about events in roughly the last 30-60 days:\n"
    "1) ACTIVE M&A TARGET: is the company itself being acquired / taken private / merged "
    "INTO another company (announced or pending)? The ACQUIRER in a deal does NOT count. "
    "Closed/abandoned deals do NOT count.\n"
    "2) MATERIAL CRISIS: a recent, confirmed event that raises 5-day downside risk - SEC "
    "enforcement or fraud allegations, bankruptcy/liquidity warning, going-concern doubt, "
    "major product recall, accounting restatement, exchange delisting notice, or a CEO/CFO "
    "ouster amid scandal.\n"
    "Only flag MATERIAL, RECENT, CONFIRMED events from reputable sources. Routine "
    "litigation, analyst rating changes, normal price volatility, guidance tweaks, and "
    "ordinary news do NOT count.\n"
    "Respond with ONLY a JSON object, no prose before or after:\n"
    "{\n"
    '  "has_active_ma": true|false,\n'
    '  "ma_details": "one sentence or null",\n'
    '  "has_crisis": true|false,\n'
    '  "crisis_details": "one sentence or null",\n'
    '  "summary": "2-3 sentence justification citing what you found"\n'
    "}"
)

NEUTRAL_VERDICT = {
    "has_active_ma": False, "ma_details": None,
    "has_crisis": False, "crisis_details": None,
    "summary": "skipped", "skipped": True,
}



# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════
def _next_trading_day():
    ntd = get_next_trading_day(datetime.now().date())
    return pd.Timestamp(ntd).date()


def load_quarantine():
    q = set(QUARANTINE_SEED)
    if os.path.isdir(QUARANTINE_DIR):
        for p in glob.glob(os.path.join(QUARANTINE_DIR, "*.parquet")):
            q.add(os.path.splitext(os.path.basename(p))[0].upper())
    return q


class GateCensus:
    """Per-gate accounting: how many pool names each gate could evaluate, how many it had
    to skip for a missing input, and how many it actually excluded.

    This exists because the micro-cap gate died silently for weeks: its input went null,
    `pd.notna(cap)` turned that into "passes", and nothing in the log said otherwise. A
    gate that cannot see its input must be loud about it."""

    GATES = ("quarantine", "upprob_floor", "micro_cap", "price_floor", "weekly_vol",
             "stuck_price", "nt_late")

    def __init__(self):
        self.evaluated = {g: 0 for g in self.GATES}
        self.skipped = {g: 0 for g in self.GATES}
        self.fired = {g: 0 for g in self.GATES}
        self.cap_source = {"pool": 0, "pit": 0, "none": 0}

    def note(self, gate, evaluable, fired):
        if evaluable:
            self.evaluated[gate] += 1
            if fired:
                self.fired[gate] += 1
        else:
            self.skipped[gate] += 1

    def report(self, n_pool):
        logger.info("-" * 70)
        logger.info(f"GATE CENSUS (pool of {n_pool})")
        for g in self.GATES:
            ev, sk, fi = self.evaluated[g], self.skipped[g], self.fired[g]
            line = f"  {g:14s} evaluated {ev:3d}  no-input {sk:3d}  fired {fi:3d}"
            if n_pool and ev == 0:
                logger.warning(line + "   <<< GATE COMPLETELY INERT")
            elif n_pool and sk > n_pool * 0.5:
                logger.warning(line + "   <<< GATE MOSTLY BLIND - input missing")
            else:
                logger.info(line)
        cs = self.cap_source
        logger.info(f"  market-cap source: pool/FinViz {cs['pool']}, PIT panel {cs['pit']}, "
                    f"unavailable {cs['none']}")
        if cs["pool"] == 0 and cs["pit"] > 0:
            logger.warning("  FinViz market cap is dead for the WHOLE pool - running on the "
                           "PIT panel. Check finvizfinance / 5__NightlyBackTester.")
        logger.info("-" * 70)
        _diag.append_jsonl("M_gate_census", {"n_pool": n_pool, "evaluated": dict(self.evaluated),
                                             "skipped": dict(self.skipped), "fired": dict(self.fired),
                                             "cap_source": dict(cs)})


def load_pit_marketcaps(symbols, session):
    """ticker -> market cap in $M as of `session` (backward as-of; no lookahead).

    Fallback for the micro-cap gate when the pool's FinViz CapMillions is null. Source is
    Data/MarketCaps/historical_market_caps.parquet (auxiliary/0__ApproximateMarketCaps.py:
    a static FinViz share-count anchor x the daily Close series). Approximate, but a $952M
    threshold does not need better - and an approximate cap beats a dead gate.
    Returns {} if the panel is unavailable."""
    if not os.path.exists(MCAP_FILE):
        logger.warning(f"No {MCAP_FILE} - micro-cap gate falls back to the pool's "
                       f"CapMillions (currently null; gate will be blind). "
                       f"Build it: python auxiliary/0__ApproximateMarketCaps.py")
        return {}
    try:
        mc = pd.read_parquet(MCAP_FILE)
        mc["Date"] = pd.to_datetime(mc["Date"])
        up = mc["Ticker"].str.upper()
        syms = {str(s).upper() for s in symbols}
        sub = mc[up.isin(syms) & (mc["Date"] <= pd.Timestamp(session))]
        if sub.empty:
            logger.warning("PIT market-cap panel has no rows at/behind the session date.")
            return {}
        key = sub["Ticker"].str.upper()
        last = sub.sort_values("Date").groupby(key)["MarketCap"].last()
        asof = sub.groupby(key)["Date"].max()
        stale = int(((pd.Timestamp(session) - asof).dt.days > 120).sum())
        if stale:
            logger.info(f"PIT cap: {stale} ticker(s) priced >120d before the session "
                        f"(still used; refresh with auxiliary/0__ApproximateMarketCaps.py).")
        return {k: float(v) / 1e6 for k, v in last.items() if np.isfinite(v)}
    except Exception as e:
        logger.warning(f"Could not load PIT market caps ({e}) - micro-cap gate degraded.")
        return {}


FINVIZ_CACHE_FILE = "Data/finviz_sector_cache.parquet"


def load_sector_map():
    """ticker -> (Sector, IndustryGroup) from Data/SectorMap.parquet (build_data_panels.py sector).
    Returns {} (concentration cap silently DISABLED) if the map is missing/unreadable."""
    if not os.path.exists(SECTOR_MAP_FILE):
        logger.warning(f"No {SECTOR_MAP_FILE} - concentration cap DISABLED "
                       f"(run: python build_data_panels.py sector to enable).")
        return {}
    try:
        df = pd.read_parquet(SECTOR_MAP_FILE)
        return {str(r.Ticker).upper(): (str(r.Sector), str(r.IndustryGroup))
                for r in df.itertuples()}
    except Exception as e:
        logger.warning(f"Could not load sector map ({e}) - concentration cap DISABLED.")
        return {}


def load_finviz_cache():
    """ticker -> (FvSector, FvIndustry) cached from prior FinViz pulls (build_data_panels.py
    sector --finviz, and live top-ups below). FinViz Industry is the cleanest source - it tags
    foreign gold miners (Barrick/Osisko/Triple Flag) that SEC SIC codes miss."""
    if not os.path.exists(FINVIZ_CACHE_FILE):
        return {}
    try:
        df = pd.read_parquet(FINVIZ_CACHE_FILE)
        out = {}
        for r in df.itertuples():
            sec, ind = str(r.FvSector), str(r.FvIndustry)
            # Skip blank/poisoned entries so they self-heal on the next live pull instead
            # of short-circuiting the lookup forever (see _live_finviz below).
            if not ind or ind in ("", "nan", "None"):
                continue
            out[str(r.Ticker).upper()] = (sec, ind)
        return out
    except Exception:
        return {}


_LIVE_FINVIZ_STATS = {"fail": 0, "nodata": 0}


def _live_finviz(symbol, fv_cache):
    """Best-effort single-ticker FinViz Sector/Industry; updates fv_cache on SUCCESS only.
    Never raises - a FinViz outage must not block the broker's 10:00 ET lock-in.
    Returns (sec, industry) or None.

    2026-07-24: this used to cache ("","") on failure. That tuple is TRUTHY, so it was
    written to Data/finviz_sector_cache.parquet and then short-circuited every future
    lookup for that ticker - a permanent poison entry. Failures are no longer cached.

    2026-07-28: switched off finvizfinance's own ticker_fundament(), which RAISES against
    FinViz's current markup (it looks for `div.quote-links`, renamed to
    `a.quote-header_category` links; 1.3.0 has the identical bug, so there is no upgrade
    to wait for). Every call here had been failing, meaning this whole live-top-up leg was
    dead and the concentration cap silently ran on the SIC sector map alone. The failure
    is now counted so a repeat is visible instead of invisible."""
    try:
        from auxiliary.finviz_compat import sector_industry
        sec, ind = sector_industry(symbol)
        if ind:
            fv_cache[symbol.upper()] = (sec, ind)
            return sec, ind
        _LIVE_FINVIZ_STATS["nodata"] += 1
    except Exception as e:                                    # noqa: BLE001
        _LIVE_FINVIZ_STATS["fail"] += 1
        if _LIVE_FINVIZ_STATS["fail"] == 1:
            logger.warning(f"Live FinViz Sector/Industry lookup failed on {symbol} "
                           f"({type(e).__name__}: {e}). The concentration cap falls back "
                           f"to the SIC sector map. If FinViz changed their markup again, "
                           f"fix finviz_compat.py (check: python auxiliary/finviz_compat.py AAPL).")
    return None


def industry_group(symbol, row, sector_map, fv_cache):
    """Resolve (coarse_sector, fine_group) for the concentration cap, in priority order:
       1. pool-row FinViz columns (written by 5__NightlyBackTester's signal pull),
       2. the FinViz cache, 3. a best-effort live FinViz call, 4. the SIC sector map.
    'Unknown' names (ETFs/ADRs/unclassifiable) are a heterogeneous grab-bag, NOT a real
    single-industry bet, so the caller gives them a unique key (they never pool to a cap)."""
    sym = symbol.upper()
    # The coarse sector is always rolled up from the fine group (coarse_sector) so the
    # sector-level cap pools consistently regardless of which source named the group
    # (FinViz says "Basic Materials", SIC says "Materials" - both must count as one sector).
    # 1. pool row carries FinViz fields (fresh, no network)
    for col in ("FinvizIndustry", "Industry"):
        val = row.get(col) if hasattr(row, "get") else None
        if val is not None and pd.notna(val):
            g = group_from_text(str(val))
            if g:
                return coarse_sector(g), g
    # 2. FinViz cache  /  3. live FinViz top-up
    fv = fv_cache.get(sym) or _live_finviz(sym, fv_cache)
    if fv and fv[1]:
        g = group_from_text(fv[1])
        if g:
            return coarse_sector(g), g
    # 4. SIC sector map (its Sector is already a coarse_sector label)
    sec, grp = sector_map.get(sym, ("Unknown", "Unknown"))
    if grp not in ("", "Unknown", "nan", "None"):
        return coarse_sector(grp) if grp in COARSE_FROM_GROUP else sec, grp
    return "Unknown", "Unknown"


def _conc_keys(symbol, row, sector_map, fv_cache):
    """Concentration counting keys. Unknown/unmapped names get a unique key so they never
    pool toward a cap."""
    sec, grp = industry_group(symbol, row, sector_map, fv_cache)
    if grp in ("", "Unknown", "nan", "None"):
        grp = f"_uniq_{symbol.upper()}"
    if sec in ("", "Unknown", "nan", "None"):
        sec = f"_uniq_{symbol.upper()}"
    return sec, grp


def _save_finviz_cache(fv_cache):
    """Persist FinViz top-ups gathered during the run (best-effort). Blank industries are
    never written - that is what poisoned the cache before."""
    if not fv_cache:
        return
    try:
        rows = [{"Ticker": k, "FvSector": v[0], "FvIndustry": v[1]}
                for k, v in fv_cache.items() if v and v[1]]
        if rows:
            pd.DataFrame(rows).to_parquet(FINVIZ_CACHE_FILE, index=False)
    except Exception:
        pass


def load_price_history(ticker):
    path = os.path.join(PRICE_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.sort_values("Date")
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"[{ticker}] could not read price history: {e}")
        return None


def compute_rsi14(close):
    """Wilder's RSI(14); returns the latest value or None if too short."""
    if close is None or len(close) < 15:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def compute_weekly_vol_pct(price_df, window=5):
    """Approx FinViz 'Volatility (Week)': mean daily (High-Low)/Close over the last
    ~5 sessions, in %. Calibrated to within ~0.3% of FinViz across test names
    (AU/DRS/KGC/AG/CVI/TIGO/RBLX/MSTR). A std*sqrt(5) scaling ran ~2x hot and
    nuked the whole pool - do not use it."""
    if price_df is None or len(price_df) < 3:
        return None
    if {"High", "Low", "Close"}.issubset(price_df.columns):
        rng = ((price_df["High"] - price_df["Low"]) / price_df["Close"]).dropna().tail(window)
        if len(rng) < 3:
            return None
        return float(rng.mean() * 100.0)
    rets = price_df["Close"].pct_change().abs().dropna().tail(window)   # fallback: close-to-close
    if len(rets) < 3:
        return None
    return float(rets.mean() * 100.0)


def max_intraday_range_pct(price_df, lookback=STUCK_LOOKBACK_DAYS):
    """The LARGEST (High-Low)/Close, in %, over the last `lookback` sessions.

    Intraday range rather than close-to-close on purpose: a name can close flat every day
    and still swing intraday, and such names DO reach the target (see the STUCK_* note).

    Returns None when the history is too short to judge - which the census records as
    no-input rather than a pass, so a blind gate is visible instead of silently permissive
    (the micro-cap failure mode)."""
    if price_df is None or not {"High", "Low", "Close"}.issubset(price_df.columns):
        return None
    tail = price_df.tail(lookback)
    if len(tail) < lookback:
        return None
    c = tail["Close"].replace(0, np.nan)
    rng_pct = (((tail["High"] - tail["Low"]) / c) * 100.0).dropna()
    if len(rng_pct) < lookback:
        return None
    return float(rng_pct.max())


# ════════════════════════════════════════════════════════════════════════════════
# Idempotency - has the funnel already produced today's book?
# ════════════════════════════════════════════════════════════════════════════════
def already_funneled(next_td):
    """(done, symbols, vet_source) - does _Buy_Signals.parquet already hold a narrowed
    book for next_td, and who produced it ('manual' / 'llm' / 'mechanical')?
    Books written before provenance stamping report 'mechanical' (lowest tier), so
    they stay upgradeable. Any 'manual' row marks the whole book manual - that is
    the protective direction."""
    if not os.path.exists(BOOK_FILE):
        return False, None, None
    try:
        df = pd.read_parquet(BOOK_FILE)
    except Exception:
        return False, None, None
    if "Status" not in df.columns:          # raw ledger / unnarrowed
        return False, None, None
    pend = df[df["Status"] == "Pending"]
    if pend.empty or len(pend) > MAX_BOOK:   # empty or still pool-sized
        return False, None, None
    if "TargetDate" not in pend.columns:
        return False, None, None
    dates = set(pd.to_datetime(pend["TargetDate"], errors="coerce").dt.date.dropna())
    if dates == {next_td}:                   # narrowed AND dated for the upcoming session
        src = "mechanical"
        if "VetSource" in pend.columns:
            vals = {str(v).strip().lower() for v in pend["VetSource"].dropna()}
            if "manual" in vals:
                src = "manual"
            elif "llm" in vals:
                src = "llm"
        return True, pend["Symbol"].tolist(), src
    return False, None, None


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1 - hard mechanical exclusions
# ════════════════════════════════════════════════════════════════════════════════
def hard_exclude(symbol, row, price_df, quarantine, pit_caps, census):
    reasons = []

    q = symbol.upper() in quarantine
    census.note("quarantine", True, q)
    if q:
        reasons.append("ideological quarantine")

    # Cross-sectional UpProbability floor (see UPPROB_FLOOR note). Cuts the model's
    # stable-negative low band before it can fill the book on thin days.
    up = row.get("UpProbability")
    ok = pd.notna(up)
    fired = bool(ok and float(up) < UPPROB_FLOOR)
    census.note("upprob_floor", ok, fired)
    if fired:
        reasons.append(f"UpProb {float(up):.3f} < floor {UPPROB_FLOOR:.2f}")

    # Market cap: the pool's FinViz snapshot first, then the PIT panel. The pool value has
    # been null for every name since FinViz's fundamentals scrape broke, which is exactly
    # how this gate went dead - see the MICRO_CAP_MAX_M note.
    cap = row.get("CapMillions")
    cap_val, src = (float(cap), "pool") if pd.notna(cap) else (None, None)
    if cap_val is None:
        p = pit_caps.get(symbol.upper())
        if p is not None and np.isfinite(p):
            cap_val, src = float(p), "pit"
    census.cap_source[src or "none"] += 1
    fired = bool(cap_val is not None and cap_val < MICRO_CAP_MAX_M)
    census.note("micro_cap", cap_val is not None, fired)
    if fired:
        reasons.append(f"micro-cap ${cap_val:.0f}M < ${MICRO_CAP_MAX_M:.0f}M [{src}]")

    # Price floor - prefer the live price-history close over the (sometimes stale) pool price
    price = None
    if price_df is not None and "Close" in price_df.columns and len(price_df):
        price = float(price_df["Close"].iloc[-1])
    elif pd.notna(row.get("CurrentPrice")):
        price = float(row["CurrentPrice"])
    fired = bool(price is not None and price < PRICE_FLOOR)
    census.note("price_floor", price is not None, fired)
    if fired:
        reasons.append(f"price ${price:.2f} < ${PRICE_FLOOR:.2f}")

    # Weekly volatility cliff (computed from price history). The old RSI(14) [30,40]
    # "death-zone" HARD exclusion was REMOVED here 2026-06-24 - the 17.6k-candidate study
    # showed [30,40] is the BEST RSI band (oversold bounce), so the gate dropped winners
    # (only net-negative screen at book level). Overextension (RSI>80) is now a SOFT flag.
    wv = None
    if price_df is not None and "Close" in price_df.columns:
        wv = compute_weekly_vol_pct(price_df)
    else:
        logger.info(f"[{symbol}] no price history - vol check skipped")
    fired = bool(wv is not None and wv > WEEKLY_VOL_MAX_PCT)
    census.note("weekly_vol", wv is not None, fired)
    if fired:
        reasons.append(f"weekly vol {wv:.1f}% > {WEEKLY_VOL_MAX_PCT:.1f}%")

    # Stuck price: a flatlined name cannot reach the +6% target inside the 5-day clock,
    # so the slot is dead capital. Tail insurance - fires on <1% of candidates.
    mx_rng = max_intraday_range_pct(price_df)
    fired = bool(mx_rng is not None and mx_rng < STUCK_RANGE_PCT)
    census.note("stuck_price", mx_rng is not None, fired)
    if fired:
        reasons.append(f"stuck price (max intraday range {mx_rng:.2f}% over "
                       f"{STUCK_LOOKBACK_DAYS} sessions < {STUCK_RANGE_PCT:.1f}%)")

    return (len(reasons) > 0), reasons


# ── Rule 12b-25 late-filing lookup (see the NT_* constants for the measurement) ──
_NT_CAL = {"mod": None, "warned": False}


def _filingcal_module():
    """The FeatureTemplates/_filingcal.py PIT loader, imported by path (it is underscore-prefixed
    so the feature framework skips it, and FeatureTemplates is not a package)."""
    if _NT_CAL["mod"] is None:
        import importlib.util
        p = os.path.join("FeatureTemplates", "_filingcal.py")
        spec = importlib.util.spec_from_file_location("_filingcal", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _NT_CAL["mod"] = mod
    return _NT_CAL["mod"]


def nt_late_status(symbol, session):
    """
    (evaluable, recent_n, count_3y, live_overdue_days) for Rule 12b-25 notices as of `session`.

    `evaluable` is FALSE when we have no filing calendar for this ticker - which is NOT the same
    as a clean filing record and must never be scored as one. The block-level family makes exactly
    this distinction (no coverage -> NaN, never filed -> 0.0) and it is load-bearing here for the
    same reason: this signal is one-sided (clean filers are the winners), so quietly treating
    "unknown" as "clean" parks every uncovered name in the favourable bucket. The census reports
    those as no-input so a blind gate is visible instead of silently permissive.

    Point-in-time: every quantity is derived from notice dates <= `session` and elapsed calendar
    time. `live_overdue_days` is how long a notice has been outstanding RIGHT NOW (capped, see
    NT_OVERDUE_STALE_DAYS) - knowable today, and reported for description only; the flag itself
    gates on occurrence, because days-late lost the 16-year dose-response.
    """
    try:
        cal = _filingcal_module()
    except Exception as exc:
        if not _NT_CAL["warned"]:
            _NT_CAL["warned"] = True
            logger.warning(f"12b-25 late-filing gate DISABLED - cannot load _filingcal "
                           f"({exc}). Late filers will NOT be flagged.")
        return False, 0, 0, 0.0
    try:
        epi = cal.nt_episodes(str(symbol).upper())
    except Exception as exc:
        logger.warning(f"[{symbol}] 12b-25 lookup failed ({exc}) - gate blind for this name")
        return False, 0, 0, 0.0
    if epi is None:
        return False, 0, 0, 0.0          # no calendar for this ticker => UNKNOWN, not clean

    t0, t1, _ext = epi
    asof = pd.Timestamp(session).value
    if t0.size == 0:
        return True, 0, 0, 0.0           # has a calendar, never filed an NT => a real 0.0

    ns_day = 86_400_000_000_000.0
    age = (asof - t0[t0 <= asof]) / ns_day
    recent_n = int((age <= NT_RECENT_DAYS).sum())
    count_3y = int((age <= 1095).sum())
    live = [(asof - t0[i]) / ns_day for i in range(t0.size)
            if t0[i] <= asof and t1[i] > asof and (asof - t0[i]) / ns_day <= NT_OVERDUE_STALE_DAYS]
    return True, recent_n, count_3y, (max(live) if live else 0.0)


def check_filing_calendar_freshness():
    """Loud, once-per-run staleness check. A frozen derived lake is this project's most expensive
    known failure mode (Data/Indexes froze and ~88 panel columns went all-NaN for 17 sessions with
    nothing to catch it). A stale filing calendar makes this gate UNDER-fire - it cannot produce a
    false exclusion, only a silently missed late filer - so it stays enabled and stays loud."""
    if not os.path.isdir(FILING_CAL_DIR):
        logger.warning(f"No {FILING_CAL_DIR} - the 12b-25 late-filing flag is BLIND for every "
                       f"name. Build it: python builders/build_filing_calendar.py")
        return
    try:
        f = _filingcal_module().freshness()
    except Exception as exc:
        logger.warning(f"12b-25 lake freshness unknown ({exc}) - flag may be under-firing")
        return
    if f.get("stale"):
        logger.warning(f"12b-25 filing calendar is STALE ({f.get('reason')}) - recent late "
                       f"filers will NOT be flagged. Run builders/build_filing_calendar.py.")
    else:
        logger.info(f"12b-25 filing calendar: {f.get('n_tickers')} tickers, "
                    f"newest filing {f.get('max_filed_date')} ({f.get('age_days')}d old).")


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2 - soft delisting / merger flags
# ════════════════════════════════════════════════════════════════════════════════
def soft_flags(symbol, price_df, session=None, census=None):
    flags = []

    # Rule 12b-25 late filer - SOFT (deprioritize, do not exclude). Runs before the
    # price-history early-return below: it needs no price history, and gating it behind one
    # would make it silently inert for exactly the thin/illiquid names most likely to file late.
    if session is not None:
        evaluable, recent_n, count_3y, overdue = nt_late_status(symbol, session)
        fired = bool(evaluable and (recent_n >= 1 or count_3y >= NT_CHRONIC_COUNT_3Y))
        if census is not None:
            census.note("nt_late", evaluable, fired)
        if fired:
            bits = []
            if recent_n >= 1:
                bits.append(f"{recent_n} in last {NT_RECENT_DAYS}d")
            if count_3y >= NT_CHRONIC_COUNT_3Y:
                bits.append(f"{count_3y} in 3y")
            # days-overdue is DESCRIPTION, not the trigger - the >17d gate lost the 16-year
            # dose-response (see the NT_* constants). It is surfaced because a currently
            # delinquent filer is the same category as the Stage-3 rubric's "accounting
            # restatement / delisting notice", and the human reading this log wants it.
            if overdue > 0:
                bits.append(f"one still outstanding {overdue:.0f}d")
            flags.append(f"late filer (12b-25: {', '.join(bits)})")
    if price_df is None or "Close" not in price_df.columns or len(price_df) < 11:
        return flags
    close = price_df["Close"]

    # Delisting / illiquidity
    last = float(close.iloc[-1])
    if last < DELIST_PRICE:
        flags.append(f"penny (${last:.2f})")
    if "Volume" in price_df.columns:
        dv = (close * price_df["Volume"]).tail(20)
        if len(dv) >= 10 and float(dv.median()) < DELIST_DOLLAR_VOL:
            flags.append(f"illiquid (med $vol ${float(dv.median())/1e6:.1f}M)")

    # Merger / deal-peg: a big single-day gap recently, then collapsed daily vol
    rets = close.pct_change().tail(MERGER_LOOKBACK)
    if len(rets) >= 15:
        max_move = float(rets.abs().max() * 100.0)
        recent_vol = float(rets.tail(10).std() * 100.0)
        if max_move >= MERGER_GAP_PCT and recent_vol < MERGER_POSTVOL_MAX:
            flags.append(f"deal-peg signature (gap {max_move:.0f}%, vol {recent_vol:.1f}%)")

    # Overextension: very overbought names mean-revert (panel RSI>80 = -0.43%, worst band).
    # SOFT - deprioritizes the name but still lets it fill the book on a thin day. (Kept
    # soft rather than a hard cut because the few RSI>80 names that were actually TAKEN
    # historically did fine - a real-fill survivorship signal that argues against dropping.)
    rsi = compute_rsi14(close)
    if rsi is not None and rsi > RSI_OVERBOUGHT_HI:
        flags.append(f"overbought (RSI {rsi:.0f} > {RSI_OVERBOUGHT_HI:.0f})")
    return flags


# ════════════════════════════════════════════════════════════════════════════════
# Stage 3 - LLM summary judgement (auto-skips on unfunded/invalid key)
# ════════════════════════════════════════════════════════════════════════════════
def _resolve_api_key():
    """Explicit key only. Returning None is NOT a failure: the SDK client can still
    authenticate from ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN or an `ant auth login`
    OAuth profile, so the key file is optional once one of those exists."""
    if os.path.exists(API_KEY_FILE):
        try:
            k = open(API_KEY_FILE).read().strip()
            if k and not k.startswith("#"):
                return k
        except Exception:
            pass
    return None


def _parse_verdict(text):
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    elif not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1:
            t = t[i:j + 1]
    return json.loads(t)


def _is_fatal_key_error(exc, anthropic):
    """Key-level failure (unfunded/invalid/unreachable) -> skip the whole stage."""
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                        anthropic.APIConnectionError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        msg = (getattr(exc, "message", "") or str(exc)).lower()
        if any(s in msg for s in ("credit", "billing", "balance", "quota")):
            return True
    return False


def _llm_call(client, system_blocks, user_text):
    """One streamed Opus call with web_search; resumes through pause_turn."""
    messages = [{"role": "user", "content": user_text}]
    msg = None
    client = client.with_options(timeout=LLM_CALL_TIMEOUT)
    for _ in range(4):
        with client.messages.stream(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=system_blocks,
            thinking={"type": "adaptive"},
            output_config={"effort": LLM_EFFORT},
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "pause_turn":            # server tool loop paused - resume
            messages.append({"role": "assistant", "content": msg.content})
            continue
        break
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text


class LLMJudge:
    """Stage 3 judge. Lazily inits the Opus client and judges ONE ticker per call, so
    the funnel can research candidates top-down and stop once the book is full (bounds
    cost AND wall-clock - the broker locks in at 10:00 ET). Disables itself permanently
    on a key-level error (unfunded/invalid/unreachable) so the rest of the run is
    mechanical-only."""

    def __init__(self, skip=False):
        self.enabled = False
        self.anthropic = None
        self.client = None
        self.system_blocks = None
        if skip:
            logger.info("Stage 3 (LLM): skipped by flag - mechanical only.")
            return
        try:
            import anthropic
        except ImportError:
            logger.warning("Stage 3 (LLM): anthropic SDK not installed - mechanical only.")
            return
        key = _resolve_api_key()
        try:
            # With key=None the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
            # an `ant auth login` OAuth profile on its own; raises if nothing exists.
            self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        except Exception as e:
            logger.warning(f"Stage 3 (LLM): no API key, env credential, or OAuth "
                           f"profile - mechanical only. ({e})")
            return
        self.anthropic = anthropic
        # Stable rubric -> cache_control (engages once the prefix exceeds Opus's
        # 4096-token cache minimum; harmless and correct below that).
        self.system_blocks = [{"type": "text", "text": LLM_SYSTEM_RUBRIC,
                               "cache_control": {"type": "ephemeral"}}]
        self.enabled = True
        logger.info(f"Stage 3 (LLM): {LLM_MODEL} effort={LLM_EFFORT}, web-searching up to "
                    f"{LLM_MAX_CHECKS} candidate(s) on demand.")

    def judge(self, symbol, row):
        """Return a verdict dict (has_active_ma/ma_details/has_crisis/crisis_details/
        summary/skipped). Never raises."""
        if not self.enabled:
            return dict(NEUTRAL_VERDICT)
        cap = row.get("CapMillions")
        cap_s = f"${float(cap):.0f}M" if pd.notna(cap) else "unknown"
        price = row.get("CurrentPrice")
        price_s = f"${float(price):.2f}" if pd.notna(price) else "unknown"
        user_text = (f"Ticker: {symbol}\n"
                     f"Context: price {price_s}, market cap {cap_s}.\n"
                     "Search recent news and return the JSON verdict.")
        try:
            text = _llm_call(self.client, self.system_blocks, user_text)
            v = _parse_verdict(text)
            verdict = {
                "has_active_ma": bool(v.get("has_active_ma", False)),
                "ma_details": v.get("ma_details"),
                "has_crisis": bool(v.get("has_crisis", False)),
                "crisis_details": v.get("crisis_details"),
                "summary": v.get("summary", ""),
                "skipped": False,
            }
            tag = []
            if verdict["has_active_ma"]:
                tag.append(f"M&A: {verdict['ma_details']}")
            if verdict["has_crisis"]:
                tag.append(f"CRISIS: {verdict['crisis_details']}")
            logger.info(f"[{symbol}] LLM: {'; '.join(tag) if tag else 'clear'}")
            time.sleep(LLM_SLEEP_BETWEEN)
            return verdict
        except Exception as e:
            if _is_fatal_key_error(e, self.anthropic):
                logger.warning(f"Stage 3 (LLM): key-level error ({type(e).__name__}) - disabling "
                               f"LLM for the rest of the run (mechanical only). Detail: {e}")
                self.enabled = False
            else:
                logger.error(f"[{symbol}] LLM error - treating as neutral: {e}")
            return dict(NEUTRAL_VERDICT)


# ════════════════════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════════════════════
def write_book(pool_df, selected_symbols, dry_run, vet_source="mechanical"):
    """Write the chosen symbols (rich pool schema) to _Buy_Signals.parquet.
    vet_source stamps provenance ('llm' when >=1 successful LLM check ran this run,
    else 'mechanical'); the trade-signals skill stamps 'manual'. The guard in main()
    uses this to decide what may ever be overwritten."""
    book = pool_df[pool_df["Symbol"].isin(selected_symbols)].copy()
    # Preserve rank order
    book["Symbol"] = pd.Categorical(book["Symbol"], categories=selected_symbols, ordered=True)
    book = book.sort_values("Symbol").reset_index(drop=True)
    book["Symbol"] = book["Symbol"].astype(str)

    if "Status" in book.columns:
        book["Status"] = "Pending"
    # Null stale price-derived risk levels - broker anchors stop/target/trail to live mid.
    for c in ("StopPrice", "TargetPrice", "ATR"):
        if c in book.columns:
            book[c] = pd.NA
    book["VetSource"] = vet_source
    book["VetTime"] = pd.Timestamp.now()

    if dry_run:
        logger.info("DRY RUN - not writing the book.")
        return book

    if os.path.exists(BOOK_FILE):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join("Data", "_backups", f"_Buy_Signals_backup_{ts}.parquet")
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        try:
            pd.read_parquet(BOOK_FILE).to_parquet(backup, index=False)
            logger.info(f"Backed up existing book -> {backup}")
        except Exception as e:
            logger.warning(f"Could not back up existing book: {e}")
    book.to_parquet(BOOK_FILE, index=False)
    logger.info(f"Wrote {len(book)} rows to {BOOK_FILE}: {selected_symbols}")
    return book


def wait_for_research_window(no_wait=False):
    """Sleep until RESEARCH_START_ET so the LLM sees post-open news - but only if that
    window is a short hop away (scheduler fires ~9:28 ET). Skips the wait if already
    past it, if it's too far off (off-hours/manual run), or if --no-wait is given."""
    if no_wait:
        return
    now = datetime.now(ET)
    target = now.replace(hour=RESEARCH_START_ET[0], minute=RESEARCH_START_ET[1],
                         second=0, microsecond=0)
    delta = (target - now).total_seconds()
    if delta <= 0:
        logger.info(f"Past the {target:%H:%M} ET research window - researching now.")
        return
    if delta > MAX_PREOPEN_WAIT_MIN * 60:
        logger.info(f"Research window {target:%H:%M} ET is {delta/60:.0f} min away "
                    f"(> {MAX_PREOPEN_WAIT_MIN} min) - not waiting (off-hours/manual run).")
        return
    logger.info(f"Waiting {delta/60:.1f} min until {target:%H:%M} ET so research sees "
                f"post-open news (broker locks in at 10:00 ET). Use --no-wait to skip.")
    time.sleep(delta)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Signal funnel: pool -> final narrowed book.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a narrowed book already exists for the next session.")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip the paid Stage 3 LLM check (mechanical funnel only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the selection but do not write _Buy_Signals.parquet.")
    ap.add_argument("--no-wait", action="store_true",
                    help="Skip the self-wait to the post-open research window (run LLM now).")
    args = ap.parse_args()

    logger.info("=" * 70)
    logger.info("MACRO FILTER - SIGNAL FUNNEL")
    logger.info("=" * 70)

    # ── Stage 0: load the pool FIRST - it defines the session this run targets ────
    # (get_next_trading_day(today) returns the day AFTER today, so a run on the
    # session day itself would compute the wrong date; the pool's TargetDate is the
    # canonical session, written by the nightly pipeline. Calendar is fallback only.)
    if not os.path.exists(POOL_FILE):
        logger.error(f"Pool file not found: {POOL_FILE}. Nothing to funnel.")
        return
    pool = pd.read_parquet(POOL_FILE)
    if "Status" in pool.columns:
        pool = pool[pool["Status"] == "Pending"].copy()
    logger.info(f"Pool: {len(pool)} pending candidate(s).")

    session = None
    if "TargetDate" in pool.columns:
        td = pd.to_datetime(pool["TargetDate"], errors="coerce").dt.date
        valid = td.dropna()
        if not valid.empty:
            session = valid.max()
            pool = pool[td == session].copy()
    if session is None:
        session = _next_trading_day()
        logger.warning(f"Pool has no usable TargetDate - falling back to the calendar "
                       f"next trading day: {session}.")
    logger.info(f"Target session: {session}")

    today_et = datetime.now(ET).date()
    if session < today_et:
        logger.error(f"Pool is STALE (dated {session}, today {today_et} ET) - the nightly "
                     f"pipeline did not produce fresh signals. Leaving the existing book "
                     f"UNTOUCHED; refusing to funnel stale candidates.")
        return
    if pool.empty:
        logger.info("No candidates to funnel. Exiting.")
        return

    # ── Idempotency + provenance guard ────────────────────────────────────────────
    # A book already narrowed for this session is only replaced when this run can
    # genuinely improve it; a MANUALLY vetted book is never replaced automatically.
    judge = LLMJudge(skip=args.skip_llm)

    done, syms, vet_src = already_funneled(session)
    if done and not args.force:
        if vet_src == "manual":
            logger.info(f"Book for {session} is MANUALLY VETTED ({syms}) - protected; "
                        f"not touching it (--force is the only override). Spent $0.")
            logger.info("FUNNEL COMPLETE")
            return
        if vet_src == "llm":
            logger.info(f"Book already LLM-vetted for {session}: {syms}. "
                        f"Nothing to do (use --force to re-run). Spent $0.")
            logger.info("FUNNEL COMPLETE")
            return
        if not judge.enabled:
            logger.info(f"Book for {session} ({syms}) is mechanical-vetted and this run "
                        f"has no working LLM stage - a rerun could only produce the same "
                        f"or worse. Keeping the existing book (--force to override). Spent $0.")
            logger.info("FUNNEL COMPLETE")
            return
        logger.info(f"Existing book for {session} ({syms}) is mechanical-only - "
                    f"re-funneling WITH LLM vetting to upgrade it.")
    if done and args.force:
        if vet_src == "manual":
            logger.warning(f"--force: OVERRIDING a MANUALLY VETTED book for {session} "
                           f"({syms}). If this is an automated run, something is wrong.")
        else:
            logger.info(f"Existing {vet_src} book for {session} found ({syms}) - "
                        f"--force given, re-running.")

    quarantine = load_quarantine()
    price_cache = {s: load_price_history(s) for s in pool["Symbol"].unique()}
    pit_caps = load_pit_marketcaps(pool["Symbol"].unique(), session)
    check_filing_calendar_freshness()
    census = GateCensus()

    # ── Stages 1 & 2 ─────────────────────────────────────────────────────────────
    survivors = []   # list of dicts: symbol, row, up_prob, soft
    for _, row in pool.iterrows():
        sym = str(row["Symbol"])
        pdf = price_cache.get(sym)
        excluded, reasons = hard_exclude(sym, row, pdf, quarantine, pit_caps, census)
        if excluded:
            logger.info(f"[{sym}] EXCLUDED (hard): {'; '.join(reasons)}")
            continue
        flags = soft_flags(sym, pdf, session=session, census=census)
        if flags:
            logger.info(f"[{sym}] soft-flag: {'; '.join(flags)}")
        survivors.append({
            "symbol": sym, "row": row.to_dict(),
            "up_prob": float(row.get("UpProbability", 0.0) or 0.0),
            "soft": flags,
        })
    census.report(len(pool))
    logger.info(f"After mechanical screens: {len(survivors)} survivor(s).")
    _diag.stamp("7__MacroFilter")
    _diag.append_jsonl("M_funnel", {"session": str(session), "n_pool": int(len(pool)),
                                    "n_survivors": len(survivors),
                                    "n_soft": sum(1 for s in survivors if s["soft"]),
                                    "hard": [{"symbol": str(r["Symbol"])} for _, r in pool.iterrows()
                                             if str(r["Symbol"]) not in {s["symbol"] for s in survivors}],
                                    "soft": [{"symbol": s["symbol"], "flags": s["soft"]} for s in survivors if s["soft"]]})
    if not survivors:
        logger.warning("No survivors after mechanical screens - leaving the existing book "
                       "UNTOUCHED (refusing to write an empty book). Investigate the pool or "
                       "thresholds; the broker will trade whatever the current book holds.")
        return

    # ── Stage 4 priority order, computed BEFORE the LLM so we only research names
    #    that can actually make the book: clean by UpProb, then soft-flagged by UpProb.
    clean = sorted([s for s in survivors if not s["soft"]], key=lambda x: x["up_prob"], reverse=True)
    flagged = sorted([s for s in survivors if s["soft"]], key=lambda x: x["up_prob"], reverse=True)
    ordered = clean + flagged

    # ── Stage 3: web-search top-down ONLY until the book is full (bounds cost + time;
    #    must finish before the broker locks in at 10:00 ET). Confirmed active-M&A /
    #    crisis is dropped and we research the next-ranked name instead.
    if judge.enabled:
        wait_for_research_window(args.no_wait)   # hold for post-open news before web-searching

    # Concentration cap setup (Stage 4). Caps are slot counts derived from the book size.
    sector_map = load_sector_map()
    fv_cache = load_finviz_cache()
    conc_enabled = bool(sector_map) or bool(fv_cache)
    grp_cap = max(1, int(TARGET_BOOK_SIZE * CONC_GROUP_CAP_FRAC))
    sec_cap = max(1, int(TARGET_BOOK_SIZE * CONC_SECTOR_CAP_FRAC))
    if conc_enabled:
        logger.info(f"Concentration cap: <= {grp_cap}/{TARGET_BOOK_SIZE} per industry, "
                    f"<= {sec_cap}/{TARGET_BOOK_SIZE} per sector (soft; relaxes to fill the book).")
    grp_counts, sec_counts = {}, {}

    chosen = []          # list of (survivor-dict, verdict, was_checked)
    deferred_conc = []   # LLM-cleared names held back by a concentration cap (relax-fill later)
    checks = 0           # LLM checks attempted
    checks_ok = 0        # LLM checks that actually returned a verdict (key alive)
    for cand in ordered:
        if len(chosen) >= TARGET_BOOK_SIZE:
            break
        if judge.enabled and checks < LLM_MAX_CHECKS:
            verdict = judge.judge(cand["symbol"], cand["row"])
            checks += 1
            checked = not verdict.get("skipped", True)
            checks_ok += int(checked)
            if verdict["has_active_ma"] or verdict["has_crisis"]:
                why = verdict["ma_details"] if verdict["has_active_ma"] else verdict["crisis_details"]
                logger.info(f"[{cand['symbol']}] DROPPED (LLM): {why}")
                continue
        else:
            verdict, checked = dict(NEUTRAL_VERDICT), False

        # Concentration cap (soft): hold a name back if its industry/sector slot is full.
        if conc_enabled:
            sec, grp = _conc_keys(cand["symbol"], cand["row"], sector_map, fv_cache)
            grp_full = grp_counts.get(grp, 0) >= grp_cap
            sec_full = sec_counts.get(sec, 0) >= sec_cap
            if grp_full or sec_full:
                where = grp if grp_full else sec
                logger.info(f"[{cand['symbol']}] held back: {('industry' if grp_full else 'sector')} "
                            f"concentration cap ({where}) already at limit")
                deferred_conc.append((cand, verdict, checked, sec, grp))
                continue
            grp_counts[grp] = grp_counts.get(grp, 0) + 1
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        chosen.append((cand, verdict, checked))

    # Relax the concentration cap ONLY if needed to keep capital deployed (thin-book days).
    # STEPWISE: widen every cap by one slot at a time and re-test admission. The previous
    # version refilled blindly from deferred_conc[:need] without re-checking counts, which
    # could restore an all-one-industry book on exactly the crowded days the cap exists for.
    # (Measured inert on the panel and on all archived real pools - this is a correctness
    # fix, not a return fix.)
    if len(chosen) < TARGET_BOOK_SIZE and deferred_conc:
        slack = 0
        while len(chosen) < TARGET_BOOK_SIZE and slack < TARGET_BOOK_SIZE and deferred_conc:
            slack += 1
            still = []
            for cand, verdict, checked, sec, grp in deferred_conc:
                if len(chosen) >= TARGET_BOOK_SIZE:
                    still.append((cand, verdict, checked, sec, grp))
                    continue
                if (grp_counts.get(grp, 0) >= grp_cap + slack
                        or sec_counts.get(sec, 0) >= sec_cap + slack):
                    still.append((cand, verdict, checked, sec, grp))
                    continue
                grp_counts[grp] = grp_counts.get(grp, 0) + 1
                sec_counts[sec] = sec_counts.get(sec, 0) + 1
                logger.info(f"[{cand['symbol']}] added back (concentration cap relaxed "
                            f"+{slack} slot)")
                chosen.append((cand, verdict, checked))
            deferred_conc = still

    _save_finviz_cache(fv_cache)   # persist any live FinViz top-ups gathered this run
    selected = [c["symbol"] for c, _, _ in chosen]
    if not selected:
        logger.warning("Nothing survived to selection - leaving the existing book UNTOUCHED "
                       "(refusing to write an empty book).")
        return

    logger.info("-" * 70)
    logger.info(f"SELECTED ({len(selected)}/{TARGET_BOOK_SIZE}): {selected}  [{checks} LLM check(s)]")
    for rank, (c, v, checked) in enumerate(chosen, 1):
        bits = ["clean" if not c["soft"] else f"soft-relaxed: {'; '.join(c['soft'])}",
                "LLM-cleared" if checked else "LLM-unchecked"]
        logger.info(f"  {rank}. {c['symbol']:6s} UpProb={c['up_prob']:.3f}  [{'; '.join(bits)}]")
    if len(selected) < TARGET_BOOK_SIZE:
        logger.info(f"  (thin day - only {len(selected)} qualified; "
                    f"{TARGET_BOOK_SIZE - len(selected)} slot(s) left empty)")
    logger.info("-" * 70)
    _diag.append_jsonl("M_selection", {"session": str(session), "ordered": [s["symbol"] for s in ordered],
                                       "chosen": list(selected), "llm_checks": checks, "llm_ok": checks_ok,
                                       "deferred_conc": [c["symbol"] for c, *_ in deferred_conc],
                                       "target_book": TARGET_BOOK_SIZE})

    # ── Final guard, re-checked at WRITE time ─────────────────────────────────────
    # The research-window wait above can be ~35 min, so a manually vetted book may
    # have landed on disk since the launch check; and a run whose LLM died mid-flight
    # (unfunded key: enabled at construction, fails on first call) must not replace
    # an existing same-session book with a mechanical-only rewrite.
    vet_source = "llm" if checks_ok > 0 else "mechanical"
    done_now, syms_now, src_now = already_funneled(session)
    if done_now and not args.force:
        if src_now == "manual":
            logger.info(f"WRITE ABORTED: a MANUALLY VETTED book for {session} ({syms_now}) "
                        f"is on disk - keeping it. (This run would have written: {selected})")
            logger.info("FUNNEL COMPLETE")
            return
        if checks_ok == 0:
            logger.info(f"WRITE ABORTED: zero successful LLM checks this run (key unfunded/"
                        f"errored?) and a {src_now} book for {session} ({syms_now}) already "
                        f"exists - a mechanical rewrite adds nothing over it. Keeping the "
                        f"existing book. (This run would have written: {selected})")
            logger.info("FUNNEL COMPLETE")
            return

    write_book(pool, selected, args.dry_run, vet_source)
    logger.info("FUNNEL COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
