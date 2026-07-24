#!/usr/bin/env python
"""
7__MacroFilter.py — the signal FUNNEL (pool -> final book).

Reads the candidate POOL (Data/0__signals.parquet, ~12 names written by the
nightly backtester) and narrows it to the final book of <=4 names that the live
broker (9_SuperFastBroker.py) trades from _Buy_Signals.parquet. Cost-ordered, so
the expensive LLM step only ever sees the few names that survive the free
mechanical screens:

  Stage 0  Load pool, align to the next NYSE trading day, attach price history.
  Stage 1  HARD mechanical exclusions (free, no API): price floor, micro-cap,
           weekly-vol cliff, UpProbability floor, ideological quarantine. Dropped.
  Stage 2  SOFT mechanical flags (free, no API): penny/illiquid, a big-gap-then-vol-
           collapse "deal-peg" signature, and overbought (RSI>80) overextension. These
           DEPRIORITIZE (and hand the name to the LLM); they do not drop on their own.
  Stage 3  LLM summary judgement (paid, best model, max effort) on survivors ONLY:
           claude-opus-4-8 + web_search confirms active M&A target / material
           crisis. Auto-SKIPS cleanly if the API key is unfunded/invalid — the
           mechanical funnel alone still produces a book.
  Stage 4  Rank survivors by UpProbability and take the top TARGET_BOOK_SIZE, subject to
           an industry/sector CONCENTRATION cap (clean first, then relax soft-flags AND
           the cap to fill the book so capital stays deployed), write the book.

  NOTE (2026-06-24): the old Stage-1 RSI "death-zone" [30,40] HARD exclusion was REMOVED.
  A 17.6k-candidate study (analysis_output/analyze_macro_filter.py) showed [30,40] is the
  BEST RSI band (oversold-bounce, +0.173% vs +0.060% kept) — the gate was dropping winners
  and was the only screen net-negative at book level (-0.029%/day). Overextension risk
  (RSI>80) is now a soft de-prioritization, and the user-requested concentration cap was
  added in Stage 4. See analysis_output/MACRO_FILTER_FINDINGS.md.

IDEMPOTENT + PROVENANCE-GUARDED: the target session is taken from the POOL's
TargetDate, not the calendar — get_next_trading_day(today) returns the day AFTER
today, so any run on the session day itself (the overnight pipeline finishing at
~02:00, the 07:00 morning task) used to compute the wrong session, the guard never
matched, and every rerun clobbered the book (2026-07-02: overwrote a hand-vetted
book with the mechanical fallback while the API key was unfunded).
If _Buy_Signals.parquet already holds a narrowed book for the target session:
  - VetSource='manual' (hand-vetted via the trade-signals skill): NEVER replaced
    automatically; --force is the only override.
  - VetSource='llm': already vetted — exits immediately, spends $0.
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
POOL_FILE = "Data/0__signals.parquet"      # input: ~12-name candidate pool
BOOK_FILE = "_Buy_Signals.parquet"         # output: final narrowed book (broker reads this)
PRICE_DIR = "Data/PriceData"               # per-ticker daily OHLCV (price source of truth)
QUARANTINE_DIR = "Data/_quarantine_ideological"
API_KEY_FILE = "Claud-API-KEY.txt"

# ── Book sizing ─────────────────────────────────────────────────────────────────
TARGET_BOOK_SIZE = 10  # 4->8->10 (2026-06-08): joint seed x config x book grid (Data/
                       # _jointstress) — k10 most robust marginally (best mean Sharpe 2.90,
                       # lowest sd 0.92, best mean rank 2.20 across model+config noise).
                       # 6-12 is the plateau; 4 robustly worst. Broker PositionSizer
                       # (9_SuperFastBroker line ~73, max_positions) must match (=10).
MAX_BOOK = 12          # mirrors the broker's fail-safe guard

# ── Stage 1 hard-exclusion thresholds (FilterRubric Step-1) ──────────────────────
PRICE_FLOOR = 5.00         # exclude if latest close < $5
MICRO_CAP_MAX_M = 952.0    # exclude if market cap < $952M (micro)
WEEKLY_VOL_MAX_PCT = 5.0   # exclude if weekly volatility > 5.0% (the sharp cliff). VALIDATED
                           # both ways (2026-06-24): excluded names -0.08% net vs +0.32% kept
                           # on real intraday fills; +candidate panel monotonic. Keep.
# RSI(14) overbought de-prioritization (SOFT, see soft_flags). The old [30,40] HARD
# "death-zone" exclusion was REMOVED 2026-06-24 — it was dropping the best RSI band.
RSI_OVERBOUGHT_HI = 80.0   # soft-flag if RSI(14) > 80 (panel: RSI>80 = -0.43%, worst band)
# Cross-sectional UpProbability floor. map_pct_rank_to_upprob puts the model's
# non-top-fraction names in [0.30,0.44]; an EDA on 944 real backtest trades
# (2026-06-18) found [0.30,0.38) is a stable money-loser (neg PnL in both 2025-H2
# and 2026-H1, ~42% win) while [0.42,0.44) is the profitable workhorse. Flooring at
# 0.40 cuts the junk band only: +$4.7k total PnL, per-trade Sharpe 0.19->0.32. A
# 0.45 floor was REJECTED (would cut the workhorse band, -$9k / -67% of profit).
UPPROB_FLOOR = 0.40        # exclude if entry-day UpProbability < 0.40

# Hardcoded ideological quarantine (union'd with QUARANTINE_DIR contents at runtime).
QUARANTINE_SEED = {"HIMS", "DJT", "ODD"}

# ── Stage 2 soft (delisting / merger) heuristic params ───────────────────────────
MERGER_LOOKBACK = 40       # sessions to scan for a deal-gap
MERGER_GAP_PCT = 15.0      # a single-day move this big...
MERGER_POSTVOL_MAX = 1.0   # ...followed by daily realized vol below this % => deal-peg
DELIST_PRICE = 3.00        # penny-ish
DELIST_DOLLAR_VOL = 1_000_000.0   # median 20d dollar volume below this => illiquid

# ── Concentration risk (industry / sector crowding) ──────────────────────────────
# The book is equal-weight (~1/TARGET_BOOK_SIZE per name, see 9_SuperFastBroker's
# PositionSizer), so a 0.50 group cap == "don't let >50% of the portfolio sit in one
# industry" — exactly the user's biotech / gold-miner example. A 17.6k-candidate study
# (analysis_output/concentration_risk.py) found dangerous single-industry crowding in the
# daily top-10 is RARE (fires on ~2% of book-days) and capping it costs ~0.00%/day of mean
# return — near-free tail insurance against a correlated single-theme drawdown. The caps
# are SOFT: if honoring them would leave the book unfilled, they relax (capital deployed).
SECTOR_MAP_FILE = "Data/SectorMap.parquet"   # built by build_data_panels.py sector (SEC SIC codes)
CONC_GROUP_CAP_FRAC  = 0.50   # max book fraction in one fine IndustryGroup (Biotech, GoldMiner, ...)
CONC_SECTOR_CAP_FRAC = 0.70   # looser cap on the coarse Sector (Healthcare, Tech, Energy, ...)

# ── Stage 3 LLM config (see claude-api skill) ────────────────────────────────────
LLM_MODEL = "claude-opus-4-8"   # best model — real money
LLM_EFFORT = "max"              # maximum effort (Opus-tier); dial to "high" to save tokens
LLM_MAX_TOKENS = 16000          # streamed, so well clear of the non-streaming timeout guard
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
LLM_SLEEP_BETWEEN = 2.0         # gentle spacing between the handful of calls
LLM_MAX_CHECKS = TARGET_BOOK_SIZE + 2   # cap calls: research top names until book full (+2 buffer)
LLM_CALL_TIMEOUT = 150.0        # seconds/call — hard deadline: broker locks in at 10:00 ET

# ── Research timing ──────────────────────────────────────────────────────────────
# Run the LLM AFTER the open for fresher news, but finish well before the broker
# locks in at 10:00 ET. When launched within MAX_PREOPEN_WAIT_MIN of the window the
# funnel self-waits to RESEARCH_START_ET before web-searching; otherwise (off-hours
# / manual run, or already past) it proceeds immediately. The cheap mechanical
# screens always run first — they use prior-close data, so their timing is moot.
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
    "2) MATERIAL CRISIS: a recent, confirmed event that raises 5-day downside risk — SEC "
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


FINVIZ_CACHE_FILE = "Data/finviz_sector_cache.parquet"


def load_sector_map():
    """ticker -> (Sector, IndustryGroup) from Data/SectorMap.parquet (build_data_panels.py sector).
    Returns {} (concentration cap silently DISABLED) if the map is missing/unreadable."""
    if not os.path.exists(SECTOR_MAP_FILE):
        logger.warning(f"No {SECTOR_MAP_FILE} — concentration cap DISABLED "
                       f"(run: python build_data_panels.py sector to enable).")
        return {}
    try:
        df = pd.read_parquet(SECTOR_MAP_FILE)
        return {str(r.Ticker).upper(): (str(r.Sector), str(r.IndustryGroup))
                for r in df.itertuples()}
    except Exception as e:
        logger.warning(f"Could not load sector map ({e}) — concentration cap DISABLED.")
        return {}


def load_finviz_cache():
    """ticker -> (FvSector, FvIndustry) cached from prior FinViz pulls (build_data_panels.py
    sector --finviz, and live top-ups below). FinViz Industry is the cleanest source — it tags
    foreign gold miners (Barrick/Osisko/Triple Flag) that SEC SIC codes miss."""
    if not os.path.exists(FINVIZ_CACHE_FILE):
        return {}
    try:
        df = pd.read_parquet(FINVIZ_CACHE_FILE)
        return {str(r.Ticker).upper(): (str(r.FvSector), str(r.FvIndustry)) for r in df.itertuples()}
    except Exception:
        return {}


def _live_finviz(symbol, fv_cache):
    """Best-effort single-ticker FinViz Sector/Industry; updates fv_cache. Never raises —
    a FinViz outage must not block the broker's 10:00 ET lock-in. Returns (sec, industry)."""
    try:
        from finvizfinance.quote import finvizfinance
        f = finvizfinance(symbol).ticker_fundament()
        sec, ind = str(f.get("Sector", "") or ""), str(f.get("Industry", "") or "")
        fv_cache[symbol.upper()] = (sec, ind)
        return sec, ind
    except Exception:
        return "", ""


def industry_group(symbol, row, sector_map, fv_cache):
    """Resolve (coarse_sector, fine_group) for the concentration cap, in priority order:
       1. pool-row FinViz columns (written by 5__NightlyBackTester's signal pull),
       2. the FinViz cache, 3. a best-effort live FinViz call, 4. the SIC sector map.
    'Unknown' names (ETFs/ADRs/unclassifiable) are a heterogeneous grab-bag, NOT a real
    single-industry bet, so the caller gives them a unique key (they never pool to a cap)."""
    sym = symbol.upper()
    # The coarse sector is always rolled up from the fine group (coarse_sector) so the
    # sector-level cap pools consistently regardless of which source named the group
    # (FinViz says "Basic Materials", SIC says "Materials" — both must count as one sector).
    # 1. pool row carries FinViz fields (fresh, no network)
    for col in ("FinvizIndustry", "Industry"):
        val = row.get(col) if hasattr(row, "get") else None
        if val is not None and pd.notna(val):
            g = group_from_text(str(val))
            if g:
                return coarse_sector(g), g
    # 2. FinViz cache  /  3. live FinViz top-up
    fv = fv_cache.get(sym) or (_live_finviz(sym, fv_cache) if fv_cache is not None else None)
    if fv:
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
    """Persist FinViz top-ups gathered during the run (best-effort)."""
    if not fv_cache:
        return
    try:
        pd.DataFrame([{"Ticker": k, "FvSector": v[0], "FvIndustry": v[1]}
                      for k, v in fv_cache.items()]).to_parquet(FINVIZ_CACHE_FILE, index=False)
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
    nuked the whole pool — do not use it."""
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


# ════════════════════════════════════════════════════════════════════════════════
# Idempotency — has the funnel already produced today's book?
# ════════════════════════════════════════════════════════════════════════════════
def already_funneled(next_td):
    """(done, symbols, vet_source) — does _Buy_Signals.parquet already hold a narrowed
    book for next_td, and who produced it ('manual' / 'llm' / 'mechanical')?
    Books written before provenance stamping report 'mechanical' (lowest tier), so
    they stay upgradeable. Any 'manual' row marks the whole book manual — that is
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
# Stage 1 — hard mechanical exclusions
# ════════════════════════════════════════════════════════════════════════════════
def hard_exclude(symbol, row, price_df, quarantine):
    reasons = []

    if symbol.upper() in quarantine:
        reasons.append("ideological quarantine")

    # Cross-sectional UpProbability floor (see UPPROB_FLOOR note). Cuts the model's
    # stable-negative low band before it can fill the book on thin days.
    up = row.get("UpProbability")
    if pd.notna(up) and float(up) < UPPROB_FLOOR:
        reasons.append(f"UpProb {float(up):.3f} < floor {UPPROB_FLOOR:.2f}")

    # Market cap (from the pool's FinViz snapshot at signal-generation time)
    cap = row.get("CapMillions")
    if pd.notna(cap) and float(cap) < MICRO_CAP_MAX_M:
        reasons.append(f"micro-cap ${float(cap):.0f}M < ${MICRO_CAP_MAX_M:.0f}M")

    # Price floor — prefer the live price-history close over the (sometimes stale) pool price
    price = None
    if price_df is not None and "Close" in price_df.columns and len(price_df):
        price = float(price_df["Close"].iloc[-1])
    elif pd.notna(row.get("CurrentPrice")):
        price = float(row["CurrentPrice"])
    if price is not None and price < PRICE_FLOOR:
        reasons.append(f"price ${price:.2f} < ${PRICE_FLOOR:.2f}")

    # Weekly volatility cliff (computed from price history). The old RSI(14) [30,40]
    # "death-zone" HARD exclusion was REMOVED here 2026-06-24 — the 17.6k-candidate study
    # showed [30,40] is the BEST RSI band (oversold bounce), so the gate dropped winners
    # (only net-negative screen at book level). Overextension (RSI>80) is now a SOFT flag.
    if price_df is not None and "Close" in price_df.columns:
        wv = compute_weekly_vol_pct(price_df)
        if wv is not None and wv > WEEKLY_VOL_MAX_PCT:
            reasons.append(f"weekly vol {wv:.1f}% > {WEEKLY_VOL_MAX_PCT:.1f}%")
    else:
        logger.info(f"[{symbol}] no price history — vol check skipped")

    return (len(reasons) > 0), reasons


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2 — soft delisting / merger flags
# ════════════════════════════════════════════════════════════════════════════════
def soft_flags(symbol, price_df):
    flags = []
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
    # SOFT — deprioritizes the name but still lets it fill the book on a thin day. (Kept
    # soft rather than a hard cut because the few RSI>80 names that were actually TAKEN
    # historically did fine — a real-fill survivorship signal that argues against dropping.)
    rsi = compute_rsi14(close)
    if rsi is not None and rsi > RSI_OVERBOUGHT_HI:
        flags.append(f"overbought (RSI {rsi:.0f} > {RSI_OVERBOUGHT_HI:.0f})")
    return flags


# ════════════════════════════════════════════════════════════════════════════════
# Stage 3 — LLM summary judgement (auto-skips on unfunded/invalid key)
# ════════════════════════════════════════════════════════════════════════════════
def _resolve_api_key():
    if os.path.exists(API_KEY_FILE):
        try:
            k = open(API_KEY_FILE).read().strip()
            if k:
                return k
        except Exception:
            pass
    return os.environ.get("ANTHROPIC_API_KEY")


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
        if msg.stop_reason == "pause_turn":            # server tool loop paused — resume
            messages.append({"role": "assistant", "content": msg.content})
            continue
        break
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text


class LLMJudge:
    """Stage 3 judge. Lazily inits the Opus client and judges ONE ticker per call, so
    the funnel can research candidates top-down and stop once the book is full (bounds
    cost AND wall-clock — the broker locks in at 10:00 ET). Disables itself permanently
    on a key-level error (unfunded/invalid/unreachable) so the rest of the run is
    mechanical-only."""

    def __init__(self, skip=False):
        self.enabled = False
        self.anthropic = None
        self.client = None
        self.system_blocks = None
        if skip:
            logger.info("Stage 3 (LLM): skipped by flag — mechanical only.")
            return
        try:
            import anthropic
        except ImportError:
            logger.warning("Stage 3 (LLM): anthropic SDK not installed — mechanical only.")
            return
        key = _resolve_api_key()
        if not key:
            logger.warning("Stage 3 (LLM): no API key — mechanical only.")
            return
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
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
                logger.warning(f"Stage 3 (LLM): key-level error ({type(e).__name__}) — disabling "
                               f"LLM for the rest of the run (mechanical only). Detail: {e}")
                self.enabled = False
            else:
                logger.error(f"[{symbol}] LLM error — treating as neutral: {e}")
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
    # Null stale price-derived risk levels — broker anchors stop/target/trail to live mid.
    for c in ("StopPrice", "TargetPrice", "ATR"):
        if c in book.columns:
            book[c] = pd.NA
    book["VetSource"] = vet_source
    book["VetTime"] = pd.Timestamp.now()

    if dry_run:
        logger.info("DRY RUN — not writing the book.")
        return book

    if os.path.exists(BOOK_FILE):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"_Buy_Signals_backup_{ts}.parquet"
        try:
            pd.read_parquet(BOOK_FILE).to_parquet(backup, index=False)
            logger.info(f"Backed up existing book -> {backup}")
        except Exception as e:
            logger.warning(f"Could not back up existing book: {e}")
    book.to_parquet(BOOK_FILE, index=False)
    logger.info(f"Wrote {len(book)} rows to {BOOK_FILE}: {selected_symbols}")
    return book


def wait_for_research_window(no_wait=False):
    """Sleep until RESEARCH_START_ET so the LLM sees post-open news — but only if that
    window is a short hop away (scheduler fires ~9:28 ET). Skips the wait if already
    past it, if it's too far off (off-hours/manual run), or if --no-wait is given."""
    if no_wait:
        return
    now = datetime.now(ET)
    target = now.replace(hour=RESEARCH_START_ET[0], minute=RESEARCH_START_ET[1],
                         second=0, microsecond=0)
    delta = (target - now).total_seconds()
    if delta <= 0:
        logger.info(f"Past the {target:%H:%M} ET research window — researching now.")
        return
    if delta > MAX_PREOPEN_WAIT_MIN * 60:
        logger.info(f"Research window {target:%H:%M} ET is {delta/60:.0f} min away "
                    f"(> {MAX_PREOPEN_WAIT_MIN} min) — not waiting (off-hours/manual run).")
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
    logger.info("MACRO FILTER — SIGNAL FUNNEL")
    logger.info("=" * 70)

    # ── Stage 0: load the pool FIRST — it defines the session this run targets ────
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
        logger.warning(f"Pool has no usable TargetDate — falling back to the calendar "
                       f"next trading day: {session}.")
    logger.info(f"Target session: {session}")

    today_et = datetime.now(ET).date()
    if session < today_et:
        logger.error(f"Pool is STALE (dated {session}, today {today_et} ET) — the nightly "
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
            logger.info(f"Book for {session} is MANUALLY VETTED ({syms}) — protected; "
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
                        f"has no working LLM stage — a rerun could only produce the same "
                        f"or worse. Keeping the existing book (--force to override). Spent $0.")
            logger.info("FUNNEL COMPLETE")
            return
        logger.info(f"Existing book for {session} ({syms}) is mechanical-only — "
                    f"re-funneling WITH LLM vetting to upgrade it.")
    if done and args.force:
        if vet_src == "manual":
            logger.warning(f"--force: OVERRIDING a MANUALLY VETTED book for {session} "
                           f"({syms}). If this is an automated run, something is wrong.")
        else:
            logger.info(f"Existing {vet_src} book for {session} found ({syms}) — "
                        f"--force given, re-running.")

    quarantine = load_quarantine()
    price_cache = {s: load_price_history(s) for s in pool["Symbol"].unique()}

    # ── Stages 1 & 2 ─────────────────────────────────────────────────────────────
    survivors = []   # list of dicts: symbol, row, up_prob, soft
    for _, row in pool.iterrows():
        sym = str(row["Symbol"])
        pdf = price_cache.get(sym)
        excluded, reasons = hard_exclude(sym, row, pdf, quarantine)
        if excluded:
            logger.info(f"[{sym}] EXCLUDED (hard): {'; '.join(reasons)}")
            continue
        flags = soft_flags(sym, pdf)
        if flags:
            logger.info(f"[{sym}] soft-flag: {'; '.join(flags)}")
        survivors.append({
            "symbol": sym, "row": row.to_dict(),
            "up_prob": float(row.get("UpProbability", 0.0) or 0.0),
            "soft": flags,
        })
    logger.info(f"After mechanical screens: {len(survivors)} survivor(s).")
    if not survivors:
        logger.warning("No survivors after mechanical screens — leaving the existing book "
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
                deferred_conc.append((cand, verdict, checked))
                continue
            grp_counts[grp] = grp_counts.get(grp, 0) + 1
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        chosen.append((cand, verdict, checked))

    # Relax the concentration cap ONLY if needed to keep capital deployed (thin-book days).
    if len(chosen) < TARGET_BOOK_SIZE and deferred_conc:
        need = TARGET_BOOK_SIZE - len(chosen)
        for cand, verdict, checked in deferred_conc[:need]:
            logger.info(f"[{cand['symbol']}] added back (relaxing concentration cap to fill book)")
            chosen.append((cand, verdict, checked))

    _save_finviz_cache(fv_cache)   # persist any live FinViz top-ups gathered this run
    selected = [c["symbol"] for c, _, _ in chosen]
    if not selected:
        logger.warning("Nothing survived to selection — leaving the existing book UNTOUCHED "
                       "(refusing to write an empty book).")
        return

    logger.info("─" * 70)
    logger.info(f"SELECTED ({len(selected)}/{TARGET_BOOK_SIZE}): {selected}  [{checks} LLM check(s)]")
    for rank, (c, v, checked) in enumerate(chosen, 1):
        bits = ["clean" if not c["soft"] else f"soft-relaxed: {'; '.join(c['soft'])}",
                "LLM-cleared" if checked else "LLM-unchecked"]
        logger.info(f"  {rank}. {c['symbol']:6s} UpProb={c['up_prob']:.3f}  [{'; '.join(bits)}]")
    if len(selected) < TARGET_BOOK_SIZE:
        logger.info(f"  (thin day — only {len(selected)} qualified; "
                    f"{TARGET_BOOK_SIZE - len(selected)} slot(s) left empty)")
    logger.info("─" * 70)

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
                        f"is on disk — keeping it. (This run would have written: {selected})")
            logger.info("FUNNEL COMPLETE")
            return
        if checks_ok == 0:
            logger.info(f"WRITE ABORTED: zero successful LLM checks this run (key unfunded/"
                        f"errored?) and a {src_now} book for {session} ({syms_now}) already "
                        f"exists — a mechanical rewrite adds nothing over it. Keeping the "
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
