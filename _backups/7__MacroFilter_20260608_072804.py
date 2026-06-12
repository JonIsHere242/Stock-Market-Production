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
           weekly-vol cliff, RSI death-zone, ideological quarantine. Dropped.
  Stage 2  SOFT mechanical delisting / merger flags (free, no API): penny/illiquid,
           and a big-gap-then-vol-collapse "deal-peg" signature. These DEPRIORITIZE
           and hand the name to the LLM to confirm; they do not drop on their own.
  Stage 3  LLM summary judgement (paid, best model, max effort) on survivors ONLY:
           claude-opus-4-8 + web_search confirms active M&A target / material
           crisis. Auto-SKIPS cleanly if the API key is unfunded/invalid — the
           mechanical funnel alone still produces a book.
  Stage 4  Rank survivors by UpProbability, take the top 4 (clean first, then relax
           soft-flags to fill 4 so capital stays deployed), write the book.

IDEMPOTENT: if _Buy_Signals.parquet already holds a narrowed book dated for the
next trading day (e.g. you funneled by hand via the trade-signals skill), this
exits immediately and spends $0. Use --force to re-run anyway.
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

logger = get_logger(script_name="7__MacroFilter")

# ── Files ─────────────────────────────────────────────────────────────────────
POOL_FILE = "Data/0__signals.parquet"      # input: ~12-name candidate pool
BOOK_FILE = "_Buy_Signals.parquet"         # output: final narrowed book (broker reads this)
PRICE_DIR = "Data/PriceData"               # per-ticker daily OHLCV (price source of truth)
QUARANTINE_DIR = "Data/_quarantine_ideological"
API_KEY_FILE = "Claud-API-KEY.txt"

# ── Book sizing ─────────────────────────────────────────────────────────────────
TARGET_BOOK_SIZE = 4   # narrow down to 4 so capital stays deployed (not 1-2)
MAX_BOOK = 12          # mirrors the broker's fail-safe guard

# ── Stage 1 hard-exclusion thresholds (FilterRubric Step-1) ──────────────────────
PRICE_FLOOR = 5.00         # exclude if latest close < $5
MICRO_CAP_MAX_M = 952.0    # exclude if market cap < $952M (micro)
WEEKLY_VOL_MAX_PCT = 5.0   # exclude if weekly volatility > 5.0% (the sharp cliff)
RSI_DEATH_LO, RSI_DEATH_HI = 30.0, 40.0   # exclude if RSI(14) in the death zone

# Hardcoded ideological quarantine (union'd with QUARANTINE_DIR contents at runtime).
QUARANTINE_SEED = {"HIMS", "DJT", "ODD"}

# ── Stage 2 soft (delisting / merger) heuristic params ───────────────────────────
MERGER_LOOKBACK = 40       # sessions to scan for a deal-gap
MERGER_GAP_PCT = 15.0      # a single-day move this big...
MERGER_POSTVOL_MAX = 1.0   # ...followed by daily realized vol below this % => deal-peg
DELIST_PRICE = 3.00        # penny-ish
DELIST_DOLLAR_VOL = 1_000_000.0   # median 20d dollar volume below this => illiquid

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
    """True if _Buy_Signals.parquet already holds a narrowed book for next_td."""
    if not os.path.exists(BOOK_FILE):
        return False, None
    try:
        df = pd.read_parquet(BOOK_FILE)
    except Exception:
        return False, None
    if "Status" not in df.columns:          # raw ledger / unnarrowed
        return False, None
    pend = df[df["Status"] == "Pending"]
    if pend.empty or len(pend) > MAX_BOOK:   # empty or still pool-sized
        return False, None
    if "TargetDate" not in pend.columns:
        return False, None
    dates = set(pd.to_datetime(pend["TargetDate"], errors="coerce").dt.date.dropna())
    if dates == {next_td}:                   # narrowed AND dated for the upcoming session
        return True, pend["Symbol"].tolist()
    return False, None


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1 — hard mechanical exclusions
# ════════════════════════════════════════════════════════════════════════════════
def hard_exclude(symbol, row, price_df, quarantine):
    reasons = []

    if symbol.upper() in quarantine:
        reasons.append("ideological quarantine")

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

    # Weekly volatility cliff + RSI death-zone (computed from price history)
    if price_df is not None and "Close" in price_df.columns:
        close = price_df["Close"]
        wv = compute_weekly_vol_pct(price_df)
        if wv is not None and wv > WEEKLY_VOL_MAX_PCT:
            reasons.append(f"weekly vol {wv:.1f}% > {WEEKLY_VOL_MAX_PCT:.1f}%")
        rsi = compute_rsi14(close)
        if rsi is not None and RSI_DEATH_LO <= rsi <= RSI_DEATH_HI:
            reasons.append(f"RSI {rsi:.0f} in death-zone [{RSI_DEATH_LO:.0f},{RSI_DEATH_HI:.0f}]")
    else:
        logger.info(f"[{symbol}] no price history — vol/RSI checks skipped")

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
def write_book(pool_df, selected_symbols, dry_run):
    """Write the chosen symbols (rich pool schema) to _Buy_Signals.parquet."""
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
    ap = argparse.ArgumentParser(description="Signal funnel: pool -> final book (<=4).")
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

    next_td = _next_trading_day()
    logger.info(f"Next trading day: {next_td}")

    # ── Idempotency: skip (spend $0) if already funneled for this session ─────────
    done, syms = already_funneled(next_td)
    if done and not args.force:
        logger.info(f"Book already narrowed for {next_td}: {syms}. "
                    f"Nothing to do (use --force to re-run). Spent $0.")
        return
    if done and args.force:
        logger.info(f"Existing book for {next_td} found ({syms}) — --force given, re-running.")

    # ── Stage 0: load + align ────────────────────────────────────────────────────
    if not os.path.exists(POOL_FILE):
        logger.error(f"Pool file not found: {POOL_FILE}. Nothing to funnel.")
        return
    pool = pd.read_parquet(POOL_FILE)
    if "Status" in pool.columns:
        pool = pool[pool["Status"] == "Pending"].copy()
    logger.info(f"Pool: {len(pool)} pending candidate(s).")

    if "TargetDate" in pool.columns:
        td = pd.to_datetime(pool["TargetDate"], errors="coerce").dt.date
        day = pool[td == next_td].copy()
        if day.empty:
            logger.warning(f"No pool rows dated {next_td}; falling back to all pending rows.")
            day = pool.copy()
        pool = day
    if pool.empty:
        logger.info("No candidates to funnel. Exiting.")
        return

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
    judge = LLMJudge(skip=args.skip_llm)
    if judge.enabled:
        wait_for_research_window(args.no_wait)   # hold for post-open news before web-searching
    chosen = []   # list of (survivor-dict, verdict, was_checked)
    checks = 0
    for cand in ordered:
        if len(chosen) >= TARGET_BOOK_SIZE:
            break
        if judge.enabled and checks < LLM_MAX_CHECKS:
            verdict = judge.judge(cand["symbol"], cand["row"])
            checks += 1
            checked = not verdict.get("skipped", True)
            if verdict["has_active_ma"] or verdict["has_crisis"]:
                why = verdict["ma_details"] if verdict["has_active_ma"] else verdict["crisis_details"]
                logger.info(f"[{cand['symbol']}] DROPPED (LLM): {why}")
                continue
        else:
            verdict, checked = dict(NEUTRAL_VERDICT), False
        chosen.append((cand, verdict, checked))

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

    write_book(pool, selected, args.dry_run)
    logger.info("FUNNEL COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
