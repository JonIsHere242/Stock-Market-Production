#!/usr/bin/env python
import argparse
import asyncio
import csv
import json
import os
import sys
import traceback
from datetime import datetime

from zoneinfo import ZoneInfo

import ib_insync as ibi
import numpy as np
import pandas as pd

from Util import get_logger, PositionSizer

# Diagnostics sidecars (diagnostics/hooks.py). No-op unless DIAG_OUT is set; every
# call is wrapped so bookkeeping can never touch the trading loop.
try:
    from diagnostics import hooks as _diag
except Exception:
    class _diag:
        enabled = staticmethod(lambda: False)
        dump_json = dump_parquet = append_jsonl = stamp = staticmethod(lambda *a, **k: None)

# ── Entry filter thresholds - derived from EDA_IntradayEntry.ipynb ────────────
# N = 1,231 signal-day observations, all exit at close.
#
#  Buy Open  → Sharpe 0.31  |  WinRate 49.4%
#  Buy 10:00 → Sharpe 1.11  |  WinRate 50.9%   (+256% Sharpe improvement)
#
#  SPY condition at 10:00   Mean%   WinRate%  Sharpe
#  All days                 +0.16     50.9    +1.11
#  SPY green day            +0.74     62.0    +5.34
#  SPY red day              -0.50     38.3    -3.62
#  SPY > +0.5%              +1.00     68.2    +6.77
#  SPY < -0.5%              -0.83     31.5    -6.50
# ─────────────────────────────────────────────────────────────────────────────
ENTRY_HOUR          = 10      # Wait until 10:00 ET before placing orders
ENTRY_MINUTE        = 0
SPY_ABORT_THRESHOLD = -0.5    # Hard skip ALL trades if SPY ≤ -0.5% from open
SPY_WARN_THRESHOLD  =  0.0    # Warn (but proceed) if SPY in (-0.5%, 0%)
SPY_STRONG_THRESHOLD=  0.5    # Log "strong market" if SPY > +0.5%
STOCK_GAP_SKIP      =  4.0    # Skip stock if it gapped up > 4.0% from open.
                              # 1.5 -> 4.0 (2026-07-30): the 1.5 gate was fitted to the
                              # retired 1.9/3.5 bracket. Refereed at the slot level:
                              # 6/6 paired 90%-universe seeds, expR +0.137, Calmar +3.09
                              # vs the 1.5 gate (analysis_output/gap_gate_referee/), on
                              # top of the 9/9 per-signal read and the account audit
                              # (13 gap-vetoed real trades, 77% winners). Rollback: 1.5.
STOCK_DIP_GOOD_LO   = -1.5    # Favorable dip-entry band lower bound
STOCK_DIP_GOOD_HI   = -0.5    # Favorable dip-entry band upper bound
# The bracket now comes from bracket_config.py, which the backtester and the signals
# file also read - so the backtest finally simulates what this broker actually sends.
# The VALUES are unchanged (1.9% / 3.5%), so live behaviour does not move; only the
# place they are defined does. 2026-06-10 note kept for provenance: HARD_STOP_PCT was
# raised 0.5 -> 1.9 because a ~2%-daily-vol name entered at 10:00 has a 60-80% chance of
# touching -0.5% before the close even on green days, so the old 0.5% exited most trades
# on noise and the backtest expectancy never described that system.
from auxiliary import bracket_config as BRACKET
HARD_STOP_PCT       = BRACKET.HARD_STOP_PCT      # 1.9

SLIPPAGE_LOG = os.path.join(os.path.dirname(__file__), "Data", "slippage_log.csv")

# Final narrowed book the broker trades. This is the post-FilterRubric shortlist
# (≤4 names), written in the 0__signals rich schema. Replaces the legacy
# read_signals() -> Z_signals.parquet path.
BUY_SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "_Buy_Signals.parquet")
# Fail-safe: the broker only trades a NARROWED book. If _Buy_Signals.parquet has no
# Status column (looks like the raw ledger) or more than MAX_BOOK pending rows (not
# narrowed), the broker refuses to place any orders. 12->4 (2026-08-23) to track the
# 3-name book: at 12 this guard could never fire against a 10-name un-narrowed file.
MAX_BOOK = 4

# ── TRIGGER-ENTRY DEFAULTS (2026-08-26, the whole mode is default OFF) ────────
# Reproduce FULL_reach24.parquet: BT_LIMIT_ENTRY_K=1.5, BT_LIMIT_OVERSUB=8 at 3 slots.
# Reach is a convenience, not an optimum: 24 vs 15 is 0.23 pooled sd and 24 vs 30 is
# 0.49 across ten shuffle seeds. The RETURN edge over the shipped control is +0.0370pp
# at t 0.53 (5 of 10 runs) and must NOT be sized as real. The DRAWDOWN difference is
# what this mode is actually for: 17.83-23.92% across all ten seeds against the
# control's 33.25%, with zero overlap, on ~three quarters of the capital.
TRIGGER_K     = 1.5
# REACH 12 (was 24, changed 2026-08-28 on a full-universe width sweep).
# Same 4 shuffle seeds at each width, --sample 100, against a deterministic control:
#
#   arm         Ann   Sharpe   MaxDD   Ulcer     Vol    CVaR  Trades  CapDep  capBreach
#   CONTROL  101.38     1.42   32.85   13.71   53.11   -7.39     250   58.19     -
#   reach 12 164.72     2.53   15.04    4.99   35.99   -4.33     185   33.28   0.00
#   reach 24 153.57     1.95   18.16    6.71   47.94   -5.33     243   44.72   1.75
#   reach 36 244.64     2.31   22.58    7.46   49.65   -5.93     269   50.46   2.75
#
# Reach 12 wins EVERY risk metric, the gradient is monotone in width, and its drawdown
# is the stablest quantity in the study (sd 0.45 across seeds vs 2.97 at 24, 5.42 at 36).
# It is also the only width that NEVER breached the 3-slot cap, which is the live
# overfill hazard. Return cannot separate the widths (sd 53/50/163, all overlapping), so
# there is no return case for going wider - reach 36's higher mean carries 3x the spread.
# Narrower reach = fewer simultaneous limits = fewer fills = less capital deployed
# (33% vs 45% vs 50%) = less risk. The cost is participation: 185 trades vs control's 250.
# Convenient side effect: 12 is the existing pool width, so BT_SIGNAL_POOL_SIZE does not
# need raising for production.
TRIGGER_REACH = 12
# The trigger arm's known live hazard, unmodelled everywhere. On 2026-08-25 the paper
# shadow armed DRUG at a 214.3bp median spread against an edge of roughly 43bp, and
# nothing in the arming step looked at spread. Default 0 = OFF, byte-identical to the
# measured arm; set a ceiling in bps to refuse to rest on names wider than it.
TRIGGER_MAX_SPREAD_BPS = 0.0
# Arming mode. 'virtual' holds the trigger table in memory and sends ONE real order the
# moment a name reaches its trigger, so at most free_slots orders ever exist at once.
# 'resting' puts a live limit on every eligible name at 10:00 and leaves them there,
# which is what the backtest modelled: a resting bid fills AT the trigger and earns the
# half-spread, while an order sent on touch pays it. Virtual is the default because a
# book full of simultaneous live orders is not what most operators expect, but it is
# NOT the arm that was measured. See the block above stage_trigger_entries.
TRIGGER_ARM  = 'virtual'
TRIGGER_POLL = 5.0   # seconds between quote polls in virtual mode

# ── MONITOR KNOBS (2026-08-28, the reach-24 changeover) ──────────────────────
# Virtual arming turns every pool name into a POTENTIAL position that has to be
# tracked from armed -> working -> filled, because "an order was sent" is not "a
# position was taken". Before this, the watcher counted SENT orders against the book
# cap: three touches that all failed to fill disarmed the whole pool and the day ended
# flat with the counter reading full. The monitor below tracks real fills instead.
#
# TRIGGER_UNFILL_MIN: minutes a working (sent, unfilled) entry may rest before it is
# cancelled and its name returned to the armed pool. 0 = never cancel, leave it resting
# to the cutoff. 0 IS THE MEASURED ARM -- the backtested trigger rests its bid for the
# whole session -- so this defaults off and anything above 0 is a deviation.
TRIGGER_UNFILL_MIN = 0.0
# Minutes between watch-table heartbeats. Set 0 to log only on state changes.
TRIGGER_STATUS_MIN = 15.0
# Ceiling on how many names the morning rubric may veto out of the pool. The rubric is
# a SHALLOW cut over a wide book now, not a narrowing to 3: if it ever flags more than
# this, that is a rubric malfunction (or a stale verdict column), and silently arming a
# 4-name book would look exactly like a normal day. Excess vetoes are refused, loudly.
TRIGGER_RUBRIC_MAX_CUT = 4
# Crash breadcrumb. The monitor holds the only record of what was armed and what fired;
# a taskkill at 14:00 used to lose the entire session's intent. Rewritten on every state
# transition so a restart (or a post-mortem) can see exactly where the day got to.
TRIGGER_MONITOR_FILE = os.path.join(os.path.dirname(__file__), 'Data', 'trigger_monitor.json')

# Fail-safe: STALENESS of the narrowed book. The book's TargetDate is the trading day it
# was built FOR; the broker runs on that day, so a fresh book is dated today. Staleness is
# measured in TRADING days (np.busday_count, so weekends don't count against it):
#   0 trading days   → fresh, trade normally
#   >= STALE_ABORT_TDAYS → ABORT: place NO orders, print a big red banner
# 2026-06-26 incident: the broker filled a 4-trading-day-old book (TargetDate 06-22)
# because the morning narrowing left a stale _Buy_Signals.parquet in place and nothing
# checked the date. Bump STALE_ABORT_TDAYS if you ever want to tolerate older books.
# 2026-07-10 incident: the 1-trading-day "warn but proceed" path traded the stale 07-06
# book on 07-07 (STAA/CRK) after the nightly pipeline silently stopped running. A book
# that is not dated for TODAY's session is untested by the morning funnel - abort.
STALE_WARN_TDAYS  = 1
STALE_ABORT_TDAYS = 1

ET = ZoneInfo('America/New_York')

# ── MAX-HOLD EXIT ────────────────────────────────────────────────────────────────
# 2026-07-24: the broker had NO age-based exit. It placed a bracket at entry and then
# never touched the position again, so the only automatic exit was the hard stop.
# Measured consequences over the 30 days to 2026-07-24 (real IBKR fills):
#     44 automated stop exits ..... 0 winners ..... -$663
#     12 discretionary exits ...... 11 winners .... +$826
#      0 max-hold exits
# and positions aged well past any sane horizon (INSP 20 days, RYTM/VKTX 35+).
#
# That is the money-losing half of the strategy running without the money-making half.
# Across the 1,112-trade live record (2025-06-23 -> 2026-05-27), by exit reason:
#     Manual Exit (days 2-4) ... n=867 .... -$4,324 ... -36% of net
#     Max Hold Time (day 5+) ... n=240 ... +$14,816 ... +122% of net   <-- the strategy
#     Take Profit .............. n=5 ..... +$1,607 .... +13% of net
# Max-hold was net positive in ELEVEN consecutive months and then stopped firing.
# Util.check_position_exit() still contains the "Max Hold Time" logic but is called by
# NOTHING - it was orphaned. This restores it in the broker.
#
# 5 CALENDAR days, matching Util.STRATEGY_PARAMS['position_timeout'] exactly. Evaluated
# each trading morning, which reproduces the 2-8 calendar-day spread seen in the live
# record (weekends inflate the count). analysis_output/COMPREHENSIVE_VERDICT.md swept
# 1d-12d: flat above 3d, 1d hurts - 5 is the validated, non-sensitive value.
MAX_HOLD_DAYS      = BRACKET.MAX_HOLD_DAYS       # 5
POSITION_LEDGER    = os.path.join(os.path.dirname(__file__), 'Data', 'position_ledger.parquet')
MAX_EXITS_PER_RUN  = 12    # sanity cap; a run wanting more than this is a bug, not a signal

# ── Runner exits (2026-07-29, bracket_config.USE_RUNNER_EXITS) ────────────────
# When ON: no take-profit leg is placed; each morning every resting stop is re-pegged
# UP to prior close * (1 - TRAIL_EOD_PCT), never down (an EOD ratchet, deliberately
# NOT an exchange TRAIL order - the 1-min referee refuted intraday-HWM trailing 0/9
# books); positions up >= RUNNER_TRIG_PCT are scaled out to RUNNER_KEEP_FRAC, and the
# kept runner rides the repegged stop up to RUNNER_MAX_HOLD_DAYS instead of
# MAX_HOLD_DAYS. Evidence + residuals: memory memo
# project_trail_runner_exit_package_2026_07_29.
RUNNER_MODE          = BRACKET.USE_RUNNER_EXITS
RUNNER_TRIG_PCT      = BRACKET.RUNNER_TRIG_PCT
RUNNER_KEEP_FRAC     = BRACKET.RUNNER_KEEP_FRAC
RUNNER_MAX_HOLD_DAYS = BRACKET.RUNNER_MAX_HOLD_DAYS
TRAIL_EOD_PCT        = BRACKET.TRAIL_EOD_PCT

# Take-profit repair. The bracket builds a TP for every name but they are not all landing
# (2026-07-24 live: 5 stops, 1 take-profit across 5 positions). Root cause not yet proven,
# so rather than guess at OCA/orderId semantics this re-attaches a missing TP to the
# EXISTING stop's OCA group - self-healing regardless of why the leg went missing.
# Worth little in PnL (0.4% of historical trades exit on TP) but it is free and correct.
TAKE_PROFIT_PCT    = BRACKET.TAKE_PROFIT_PCT   # 3.5 = 2:1 on the 1.75% risk; validated live.
                           # COMPREHENSIVE_VERDICT swept 1.8%-9%: wider targets strictly
                           # worsen drawdown (maxDD 13%->37%). Do NOT widen without a
                           # multi-seed real-backtester run.


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL EXIT ORDER TYPES - INACTIVE unless --exp-stop-limit / --exp-moc-exit
# ══════════════════════════════════════════════════════════════════════════════
# Evidence: analysis_output/EXEC_ORDER_TYPE_REPORT.md (2026-07-28), rig 8_3__OrderTypeLab.py.
# 855 signals replayed on 1-min bars, 8 seeded books, n=4677 on the 5-min lake:
#   STP LMT @ 4 half-spreads + give-up + MOC timeouts = +0.0309 pp/signal,
#   p<0.001, 8/8 books sign-consistent, and every left-tail statistic improves.
#
# WHY: orderType='STP' releases a MARKET order the instant the stop is touched. On a
# gap-down open that market order sells at the bottom of the move. A stop-LIMIT refuses
# to sell below a floor and waits to be lifted; on this book the bounce nearly always
# came. Only the ~9% of trades that GAP through the stop are affected at all.
#
# ── THE HONEST IMPLEMENTATION GAP ────────────────────────────────────────────────
# The sim's give-up leg converts to market after 3 BARS. This process is a one-shot
# that runs at 10:00 and exits, so nothing is watching a GTC stop-limit that triggers
# overnight. Live, the give-up can only be a NEXT-MORNING sweep (sweep_stranded_stops
# below), which is far coarser. Two mitigations, in order of preference:
#   1. Keep the limit WIDE. At 4 half-spreads the sim showed only 0.41% unfilled even
#      with NO give-up leg at all. The floor is protection against the gap print, not
#      an attempt to get a good price.
#   2. Run the morning sweep, which catches a stranded position within one session.
# If a persistent watcher ever exists, tighten this to match the sim.
STOP_LIMIT_HALF_SPREADS = 4.0   # limit sits this many half-spreads under the stop trigger
STOP_LIMIT_MIN_BPS      = 12.0  # ...but never tighter than this, when the spread is unknown
STOP_LIMIT_MAX_BPS      = 60.0  # ...and never wider than this (a floor that low is no floor)






def build_hard_stop_order(order_id, size, stop_price, parent_id, oca_group,
                          half_spread=None, use_stop_limit=False, logger=None):
    """The hard-stop leg of the entry bracket.

    Default path returns exactly what the live system has always sent (STP, a market
    order on trigger). With use_stop_limit=True it returns STP LMT with a price floor
    below the trigger, which is the change measured in EXEC_ORDER_TYPE_REPORT.md.

    `half_spread` is in DOLLARS (ask-bid)/2 at signal time. When it is unknown the
    floor falls back to STOP_LIMIT_MIN_BPS so the order is still well-formed.
    """
    common = dict(orderId=order_id, action='SELL', totalQuantity=size, tif='GTC',
                  outsideRth=True, parentId=parent_id, ocaGroup=oca_group,
                  ocaType=1, transmit=True)

    if not use_stop_limit:
        return ibi.Order(orderType='STP', auxPrice=stop_price, **common)

    # Floor = stop − 4 half-spreads, clamped so it is neither cosmetic nor a cliff.
    off = (half_spread or 0.0) * STOP_LIMIT_HALF_SPREADS
    lo = stop_price * STOP_LIMIT_MIN_BPS / 1e4
    hi = stop_price * STOP_LIMIT_MAX_BPS / 1e4
    off = min(max(off, lo), hi)
    lmt = round(stop_price - off, 2)
    if logger:
        logger.info(f"    EXP STP LMT: trigger ${stop_price:.2f} -> floor ${lmt:.2f} "
                    f"(-{off / stop_price * 1e4:.0f} bps){'' if half_spread else ' [no spread, used min]'}")
    return ibi.Order(orderType='STP LMT', auxPrice=stop_price, lmtPrice=lmt, **common)


def build_timeout_exit_order(size, use_moc=False):
    """The max-hold / timeout exit.

    Default is the live MKT+Adaptive order. With use_moc=True it becomes Market-on-Close,
    which executes in the closing auction: one clearing price, no continuous spread to
    cross. Worth +0.021 pp/signal in the sim -- but note that is the LARGER and SOFTER
    half of the win: the auction's advantage is structural, its magnitude was assumed
    rather than measured. It also changes WHEN you exit (at the bell, not on contact),
    so paper-trade it before believing the sim's number.
    """
    if not use_moc:
        return ibi.Order(action='SELL', totalQuantity=size, orderType='MKT', tif='DAY',
                         outsideRth=False, algoStrategy='Adaptive',
                         algoParams=[ibi.TagValue('adaptivePriority', 'Normal')],
                         transmit=True)
    return ibi.Order(action='SELL', totalQuantity=size, orderType='MOC', tif='DAY',
                     outsideRth=False, transmit=True)


class AsyncFastExecutor:

    # Live port = 7496  |  Paper port = 7497
    def __init__(self, host='127.0.0.1', port=7496, client_id=99,
                 entry_hour=ENTRY_HOUR, entry_minute=ENTRY_MINUTE,
                 spy_abort=SPY_ABORT_THRESHOLD, skip_wait=False,
                 no_exits=False, dry_run_exits=False, model_exits=False,
                 exp_stop_limit=False, exp_moc_exit=False,
                 exp_entry_escalate_min=0.0,
                 trigger_entry=True, trigger_k=TRIGGER_K, trigger_reach=TRIGGER_REACH,
                 trigger_max_spread_bps=0.0, trigger_cutoff='15:45',
                 trigger_arm=TRIGGER_ARM, trigger_poll=TRIGGER_POLL,
                 trigger_unfill_min=TRIGGER_UNFILL_MIN,
                 trigger_status_min=TRIGGER_STATUS_MIN,
                 trigger_rubric_max_cut=TRIGGER_RUBRIC_MAX_CUT,
                 dry_run=False):
        self.ib           = ibi.IB()
        # Order rejections used to vanish: nothing listened to TWS error events, so a
        # bracket rejected for funds just never filled and never logged (found via
        # statement forensics 2026-07-30). Everything TWS reports now hits the log.
        self.ib.errorEvent += self._on_ib_error
        self.host         = host
        self.port         = port
        self.client_id    = client_id
        self.entry_hour   = entry_hour
        self.entry_minute = entry_minute
        self.spy_abort    = spy_abort
        self.skip_wait    = skip_wait
        self.no_exits     = no_exits        # kill switch for the max-hold/TP machinery
        # MODEL EXITS. Default OFF: this is a NEW exit family the broker has never run,
        # so unset reproduces the shipped broker exactly. See auxiliary/model_exits.py
        # for the measurement, the leak gate and the sentinel guard.
        self.model_exits  = model_exits
        self.dry_run_exits = dry_run_exits  # log what would be exited, place nothing
        # FULL dry run: no order of ANY kind is transmitted, entries included.
        # --dry-run-exits deliberately leaves entries live ("Entries still run
        # normally"), and --no-exits only gates the exit machinery, so before this
        # flag existed there was NO way to exercise the entry path without risking
        # the account. The handoff's "dry structural check" was in fact a live-order
        # command that happened to be blocked by a stale pool; regenerating the pool
        # (its own step 2) armed it. Entry dry-run implies exit dry-run.
        self.dry_run      = bool(dry_run)
        if self.dry_run:
            self.dry_run_exits = True
        # EXPERIMENTAL, default OFF. Nothing below changes unless these are passed.
        self.exp_stop_limit = exp_stop_limit
        self.exp_moc_exit   = exp_moc_exit
        # Latency fix (2026-07-30): minutes to let the Adaptive entry work before the
        # ceiling is raised to the ask. The account-fill replay measured the cost of
        # unbounded patience: 1 in 3 real entries filled after 10:05 at +55 bps vs the
        # sim, while on-time fills matched the sim to the basis point. The 1-min engine
        # models a 3-bar walk; this makes live match that assumption. 0 disables.
        self.exp_entry_escalate_min = float(exp_entry_escalate_min)
        # ── TRIGGER ENTRY (2026-08-26; DEFAULT ON since the 2026-08-28 changeover) ──
        # Replaces the marketable Adaptive entry with resting BUY LIMITs at a
        # per-name dip depth, sent to `trigger_reach` candidates so that
        # ~max_positions of them fill. See auxiliary/trigger_entry.py for the formula
        # and the evidence. This is now what the broker does with no flags at all;
        # --no-trigger-entry restores the marketable-Adaptive path byte-for-byte.
        self.trigger_entry   = bool(trigger_entry)
        self.trigger_k       = float(trigger_k)
        self.trigger_reach   = int(trigger_reach)
        self.trigger_max_spread_bps = float(trigger_max_spread_bps)
        self.trigger_cutoff  = trigger_cutoff
        self.trigger_arm     = (trigger_arm or TRIGGER_ARM).strip().lower()
        self.trigger_poll    = float(trigger_poll)
        self.trigger_unfill_min = float(trigger_unfill_min)
        # Seconds to let streaming quotes populate before the first touch test. A poll
        # against an empty ticker reads ask=None and silently skips every name, so the
        # first pass must not run on a cold book. Named rather than inline so the
        # offline monitor tests can zero it (tests/test_trigger_monitor.py).
        self.trigger_md_settle  = 4.0
        self.trigger_status_min = float(trigger_status_min)
        self.trigger_rubric_max_cut = int(trigger_rubric_max_cut)
        if self.trigger_arm not in ('virtual', 'resting'):
            raise ValueError('trigger_arm must be virtual or resting, got %r' % trigger_arm)
        self._armed          = []   # POTENTIAL positions: one record per pool name
        self._held_at_start  = 0    # book size frozen before the first arm fires
        self._fill_log       = []   # every confirmed entry fill, in order
        self._trigger_orders = []   # (contract, parent_order, symbol) real entries
        self._staged_parents = []   # (contract, parent_order, symbol) from execute_batch
        self.logger       = get_logger("FastExecutor")
        self.logger.info(f"BRACKET: {BRACKET.describe()}")
        _bw = BRACKET.live_warning()
        if _bw:
            for _ in range(3):
                self.logger.warning("=" * 100)
            self.logger.warning(f"  {_bw}")
            for _ in range(3):
                self.logger.warning("=" * 100)
        if exp_stop_limit or exp_moc_exit:
            self.logger.warning(
                f"EXPERIMENTAL EXIT ORDERS ACTIVE - stop_limit={exp_stop_limit} "
                f"moc_exit={exp_moc_exit}. See analysis_output/EXEC_ORDER_TYPE_REPORT.md. "
                f"These have NOT been live-validated; watch the fills.")

        self.position_sizer = PositionSizer(
            cash_buffer_pct=10.0,
            max_positions=3   # 5->8->10->3 (2026-08-23). The 10 came from a joint
                              # seed x config x book grid on a GROSS backtest with no
                              # cost model. Calibrated on 362 real IBKR fills, commission
                              # is a flat $1.00 floor on 98.9% of orders, so cost as a
                              # percentage is a pure function of position size: at the
                              # measured $649 median notional that is 0.308% round trip.
                              # Concentrating the same capital into 3 names cuts the
                              # per-trade cost drag by the same factor it raises notional.
                              # Must match MacroFilter TARGET_BOOK_SIZE.
        )

        self.account_value    = 0.0
        self.available_cash   = 0.0
        self.current_positions = {}

        # mid price at order time keyed by symbol - used for slippage measurement
        self._signal_mids: dict[str, float] = {}

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self):
        try:
            self.logger.info("Connecting to IB Gateway/TWS...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.ib.execDetailsEvent += self._on_fill
            self.logger.info("Connected.")
        except Exception as e:
            self.logger.critical(f"Could not connect: {e}")
            sys.exit(1)

    # ── Fill handler - logs actual fill vs mid-at-signal for slippage tracking ─

    def _on_fill(self, _trade, fill):
        symbol    = fill.contract.symbol
        fill_px   = fill.execution.price
        side      = fill.execution.side      # 'BOT' or 'SLD'
        shares    = fill.execution.shares
        mid_ref   = self._signal_mids.get(symbol)
        slippage  = (fill_px - mid_ref) if (mid_ref and side == 'BOT') else None
        slip_bps  = (slippage / mid_ref * 10_000) if slippage is not None else None
        now_str   = datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')

        self.logger.info(
            f"[{symbol}] FILL: {side} {shares} @ ${fill_px:.2f} | "
            f"mid-ref: ${mid_ref:.2f}" if mid_ref else
            f"[{symbol}] FILL: {side} {shares} @ ${fill_px:.2f}"
            + (f" | slippage: {slip_bps:+.1f} bps" if slip_bps is not None else "")
        )

        # Append to CSV for calibration - builds ground truth over time
        try:
            os.makedirs(os.path.dirname(SLIPPAGE_LOG), exist_ok=True)
            write_header = not os.path.exists(SLIPPAGE_LOG)
            with open(SLIPPAGE_LOG, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['timestamp', 'symbol', 'side', 'shares',
                                'fill_price', 'mid_at_signal', 'slippage_bps'])
                w.writerow([now_str, symbol, side, shares,
                            fill_px, mid_ref, slip_bps])
        except Exception as e:
            self.logger.warning(f"Could not write slippage log: {e}")

    # ── Step 1: Wait until entry time ─────────────────────────────────────────

    async def wait_until_ready(self):
        """
        Sleep until entry_hour:entry_minute ET, then return.
        If already past that time, returns immediately.

        Why 10:00?  EDA shows opening-minute noise causes the worst fill prices.
        Every extra minute you wait improves Sharpe - 9:30→10:00 is +256%.
        """
        now_et    = datetime.now(ET)
        target_et = now_et.replace(
            hour=self.entry_hour, minute=self.entry_minute,
            second=0, microsecond=0
        )
        wait_secs = (target_et - now_et).total_seconds()

        if wait_secs > 0:
            self.logger.info(
                f"Waiting {wait_secs:.0f}s until {target_et.strftime('%H:%M')} ET  "
                f"[EDA: 10:00 entry Sharpe=1.11 vs 0.31 at open]"
            )
            await asyncio.sleep(wait_secs)
        else:
            self.logger.info(
                f"Already past {target_et.strftime('%H:%M')} ET "
                f"({now_et.strftime('%H:%M:%S')}) - proceeding immediately."
            )

    # ── Step 2: SPY market-condition gate ─────────────────────────────────────

    async def check_market_conditions(self) -> dict:
        """
        Snapshot SPY at entry time. Returns a dict:
            spy_open, spy_current, spy_move_pct, market_ok, spy_green, spy_strong

        EDA results (N=1,231 signal days, Buy-10:00 entry, exit at close):
            SPY green day   mean +0.74%  WinRate 62%  Sharpe +5.34
            SPY red day     mean -0.50%  WinRate 38%  Sharpe -3.62
            SPY < -0.5%     mean -0.83%  WinRate 31%  Sharpe -6.50
            SPY > +0.5%     mean +1.00%  WinRate 68%  Sharpe +6.77
        """
        self.logger.info("Checking SPY market conditions...")

        result = {
            'spy_open':     None,
            'spy_current':  None,
            'spy_move_pct': None,
            'market_ok':    True,
            'spy_green':    True,
            'spy_strong':   False,
        }

        try:
            spy_contract = ibi.Stock('SPY', 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(spy_contract)
            [spy_ticker] = await self.ib.reqTickersAsync(spy_contract)

            spy_open    = spy_ticker.open
            spy_current = (spy_ticker.last if (spy_ticker.last and spy_ticker.last > 0)
                           else spy_ticker.ask if (spy_ticker.ask and spy_ticker.ask > 0)
                           else None)

            if not spy_open or spy_open <= 0 or not spy_current:
                self.logger.warning("SPY market data unavailable - proceeding without SPY filter.")
                return result

            spy_move = (spy_current / spy_open - 1) * 100

            result.update({
                'spy_open':     spy_open,
                'spy_current':  spy_current,
                'spy_move_pct': spy_move,
                'spy_green':    spy_move > 0,
                'spy_strong':   spy_move > SPY_STRONG_THRESHOLD,
            })

            if spy_move <= self.spy_abort:
                result['market_ok'] = False
                self.logger.warning(
                    f"SPY ABORT: SPY {spy_move:+.2f}% from open "
                    f"(threshold {self.spy_abort:+.1f}%). "
                    f"EDA: Sharpe={-6.50:.2f} on days this weak - no orders placed."
                )
            elif spy_move < SPY_WARN_THRESHOLD:
                self.logger.warning(
                    f"SPY CAUTION: SPY {spy_move:+.2f}% - market slightly red. "
                    f"EDA: WinRate drops to ~38% on red days. Proceeding with caution."
                )
            elif result['spy_strong']:
                self.logger.info(
                    f"SPY STRONG: SPY {spy_move:+.2f}% from open. "
                    f"EDA: WinRate 68%, Sharpe +6.77 on days SPY > +0.5%."
                )
            else:
                self.logger.info(f"SPY OK: {spy_move:+.2f}% from open.")

        except Exception as e:
            self.logger.warning(f"SPY check failed ({e}) - proceeding without filter.")

        return result

    # ── Account snapshot ───────────────────────────────────────────────────────

    async def prepare_account_data(self):
        """
        Fetches account summary ONE time to be used for all calculations.
        Called AFTER wait_until_ready() so values are fresh at entry time.
        """
        self.logger.info("Snapshotting Account Data...")

        positions = self.ib.positions()
        self.current_positions = {
            p.contract.symbol: p.position
            for p in positions if p.position != 0
        }

        for tag in self.ib.accountValues():
            if tag.tag == 'NetLiquidation':
                self.account_value = float(tag.value)
            elif tag.tag == 'AvailableFunds':
                self.available_cash = float(tag.value)

        self.logger.info(
            f"Account Ready. NAV: ${self.account_value:,.0f} | "
            f"Cash: ${self.available_cash:,.0f} | "
            f"Open Pos: {len(self.current_positions)}"
        )

    # ── Position ledger (entry dates for the max-hold clock) ───────────────────
    # IBKR positions carry no entry date, and TWS reqExecutions only reaches back about
    # a day, so the age clock needs a local record. The ledger is reconciled against the
    # broker's own position list on every run: new symbols are stamped, closed symbols
    # are dropped. It is therefore self-healing and cannot drift for long.

    def _load_ledger(self) -> dict:
        # Side effect: populates self._scaled, the set of symbols already scaled out
        # (runner mode). Missing column = legacy ledger = nothing scaled.
        self._scaled = set()
        if not os.path.exists(POSITION_LEDGER):
            return {}
        try:
            df = pd.read_parquet(POSITION_LEDGER)
            if 'Scaled' in df.columns:
                self._scaled = {str(r.Symbol).upper() for r in df.itertuples()
                                if bool(getattr(r, 'Scaled', False))}
            return {str(r.Symbol).upper(): pd.to_datetime(r.EntryDate).date()
                    for r in df.itertuples()}
        except Exception as e:
            self.logger.warning(f"Could not read {POSITION_LEDGER} ({e}) - age-exit will "
                                f"re-stamp every position TODAY and exit nothing this run.")
            return {}

    def _save_ledger(self, ledger: dict):
        try:
            os.makedirs(os.path.dirname(POSITION_LEDGER), exist_ok=True)
            scaled = getattr(self, '_scaled', set())
            pd.DataFrame(
                [{'Symbol': s, 'EntryDate': pd.Timestamp(d), 'Scaled': s in scaled}
                 for s, d in sorted(ledger.items())]
            ).to_parquet(POSITION_LEDGER, index=False)
        except Exception as e:
            self.logger.error(f"Could not write {POSITION_LEDGER} ({e}) - the max-hold clock "
                              f"will not advance. FIX THIS: positions will be held forever.")

    def _reconcile_ledger(self) -> dict:
        """Sync the ledger to the positions IBKR actually reports.

        FAIL-SAFE DIRECTION: a position whose entry date we do not know is stamped TODAY,
        never force-exited on this run. We would rather hold something an extra few days
        than liquidate a position we cannot account for.
        """
        ledger = self._load_ledger()
        today = datetime.now(ET).date()
        held = {s.upper() for s in self.current_positions}

        for sym in sorted(held):
            if sym not in ledger:
                ledger[sym] = today
                self.logger.warning(
                    f"[{sym}] held but absent from the position ledger - stamping entry "
                    f"{today}. It will not be age-exited for {MAX_HOLD_DAYS} more days."
                )
        for sym in [s for s in ledger if s not in held]:
            self.logger.info(f"[{sym}] no longer held - removed from the position ledger.")
            del ledger[sym]
        self._scaled = getattr(self, '_scaled', set()) & held

        self._save_ledger(ledger)
        return ledger

    # ── Max-hold exit ──────────────────────────────────────────────────────────

    async def close_aged_positions(self, dry_run: bool = False) -> list:
        """Force-exit every position held >= MAX_HOLD_DAYS calendar days.

        Runs BEFORE the entry batch so the freed slots recycle the same morning - that
        recycling is where the edge lives (exit-to-recycle dominates exit-to-cash, see
        the 2026-07-13 exit verdict). Returns the list of symbols sold.

        Order of operations per name, and it matters:
          1. cancel the resting GTC bracket legs (an live stop would otherwise race the
             exit and could double-sell),
          2. verify the cancels settled,
          3. sell the exact held quantity, MKT via the Adaptive algo.
        """
        if not self.current_positions:
            return []

        ledger = self._reconcile_ledger()
        today = datetime.now(ET).date()

        aged = []
        for sym, qty in self.current_positions.items():
            sym = sym.upper()
            if qty is None or qty <= 0:            # never touch a short/flat line
                continue
            entry = ledger.get(sym)
            if entry is None:
                continue
            days = (today - entry).days
            # Runner mode: a scaled-out position earns the longer runner clock.
            limit = (RUNNER_MAX_HOLD_DAYS
                     if RUNNER_MODE and sym in getattr(self, '_scaled', set())
                     else MAX_HOLD_DAYS)
            if days >= limit:
                aged.append((sym, int(qty), entry, days))

        if not aged:
            ages = ', '.join(f"{s}:{(today - d).days}d" for s, d in sorted(ledger.items()))
            self.logger.info(f"Max-hold: nothing aged >= {MAX_HOLD_DAYS}d. Held ages [{ages}]")
            return []

        if len(aged) > MAX_EXITS_PER_RUN:
            self.logger.error(
                f"ABORT max-hold exits: {len(aged)} positions qualify, which exceeds "
                f"MAX_EXITS_PER_RUN={MAX_EXITS_PER_RUN}. That is a ledger bug, not a signal. "
                f"No exit orders placed. Symbols: {[a[0] for a in aged]}"
            )
            return []

        self.logger.info("=" * 64)
        self.logger.info(f"MAX-HOLD EXIT: {len(aged)} position(s) at/over {MAX_HOLD_DAYS} days")
        for sym, qty, entry, days in aged:
            self.logger.info(f"   {sym:6s} {qty:>6d} sh   entered {entry}   held {days}d")
        self.logger.info("=" * 64)

        if dry_run:
            self.logger.info("--dry-run-exits: no exit orders placed.")
            return []

        sold = []
        for sym, qty, entry, days in aged:
            try:
                contract = ibi.Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)

                # 1. cancel resting bracket legs for this symbol
                cancelled = 0
                for tr in self.ib.openTrades():
                    if (tr.contract.symbol.upper() == sym
                            and tr.orderStatus.status not in ('Filled', 'Cancelled',
                                                              'ApiCancelled', 'Inactive')):
                        self.ib.cancelOrder(tr.order)
                        cancelled += 1
                if cancelled:
                    self.logger.info(f"[{sym}] cancelled {cancelled} resting order(s) "
                                     f"before the age exit.")
                    await asyncio.sleep(1.5)   # 2. let the cancels settle at IB

                # 3. sell exactly what is held
                # Default MKT/Adaptive; --exp-moc-exit routes it to the closing auction.
                exit_order = build_timeout_exit_order(abs(int(qty)),
                                                      use_moc=self.exp_moc_exit)
                self.ib.placeOrder(contract, exit_order)
                self.logger.info(f"[{sym}] MAX-HOLD SELL {qty} sh @ "
                                 f"{'MOC (closing auction)' if self.exp_moc_exit else 'MKT/Adaptive'} "
                                 f"(held {days}d, entered {entry}).")
                sold.append(sym)
            except Exception as e:
                self.logger.error(f"[{sym}] max-hold exit FAILED ({e}) - position left open "
                                  f"with its bracket possibly cancelled. CHECK THIS MANUALLY.")
                self.logger.error(traceback.format_exc())

        if sold:
            await asyncio.sleep(8)   # give the fills a chance before the account refresh
        return sold

    async def close_model_exits(self, dry_run: bool = False, skip: set = None) -> list:
        """Exit held names whose UpProbability signal has decayed (auxiliary/model_exits).

        The one exit family the broker has never run. It reads Data/RFpredictions/,
        written by the nightly 4__Predictor from the prior close, which is the same bar
        the backtest exits on, so live and sim act on identical information.

        Ordered AFTER close_aged_positions (a name already being flattened is skipped)
        and BEFORE the entry batch, so a freed slot recycles the same morning. Note this
        inverts the backtest's internal order, where the momentum rule is evaluated
        before the age clock. The position is sold either way; only the attributed
        reason differs, and the broker does not attribute exits by rule.

        Refuses to act on stale predictions. The nightly runner has died mid-pipeline
        before, which leaves yesterday's probabilities looking like today's.
        """
        if not self.current_positions:
            return []

        from auxiliary import model_exits as ME

        skip = {x.upper() for x in (skip or set())}
        ledger = self._reconcile_ledger()
        today = datetime.now(ET).date()

        firing, held_lines, skipped = [], [], []
        for sym, qty in self.current_positions.items():
            sym = sym.upper()
            if qty is None or qty <= 0 or sym in skip:
                continue
            entry = ledger.get(sym)
            if entry is None:
                skipped.append(f'{sym}: not in the ledger, cannot age it')
                continue
            try:
                probs, last_pred = ME.load_probs(sym)
            except (FileNotFoundError, ValueError) as e:
                # No prediction is NOT "no exit signal". Say so and leave it alone.
                skipped.append(f'{sym}: no usable prediction ({e.__class__.__name__})')
                continue
            stale = ME.staleness_tdays(last_pred, today)
            if stale > ME.STALE_ABORT_TDAYS:
                skipped.append(f'{sym}: predictions {stale} trading day(s) stale '
                               f'(last {last_pred}), refusing to act')
                continue
            days_held = (today - entry).days
            d = ME.evaluate(probs, days_held)
            if d.exit:
                firing.append((sym, int(qty), d.reason, d.detail, days_held))
            else:
                held_lines.append(f'{sym}: {d.detail}')

        self.logger.info("=" * 64)
        self.logger.info(f"MODEL EXITS: {len(firing)} firing, {len(held_lines)} held, "
                         f"{len(skipped)} not evaluated")
        for line in held_lines:
            self.logger.info(f"   HOLD  {line}")
        for line in skipped:
            self.logger.warning(f"   SKIP  {line}")
        for sym, qty, reason, detail, days in firing:
            self.logger.info(f"   EXIT  {sym:6s} {qty:>6d} sh  [{reason}]  {detail}")
        self.logger.info("=" * 64)

        if not firing:
            return []

        # Same sanity cap as the age exit. More than this qualifying at once is a data
        # problem (a stale or half-written prediction sweep), not a signal.
        if len(firing) > MAX_EXITS_PER_RUN:
            self.logger.error(
                f"ABORT model exits: {len(firing)} positions qualify, over "
                f"MAX_EXITS_PER_RUN={MAX_EXITS_PER_RUN}. That is a prediction-feed bug, "
                f"not a signal. No exit orders placed. Symbols: {[f[0] for f in firing]}")
            return []

        if dry_run:
            self.logger.info("--dry-run-exits: no model-exit orders placed.")
            return []

        sold = []
        for sym, qty, reason, detail, days in firing:
            try:
                contract = ibi.Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)

                # Cancel resting bracket legs first, exactly as close_aged_positions
                # does: a live stop would otherwise race this exit and double-sell.
                cancelled = 0
                for tr in self.ib.openTrades():
                    if (tr.contract.symbol.upper() == sym
                            and tr.orderStatus.status not in ('Filled', 'Cancelled',
                                                              'ApiCancelled', 'Inactive')):
                        self.ib.cancelOrder(tr.order)
                        cancelled += 1
                if cancelled:
                    self.logger.info(f"[{sym}] cancelled {cancelled} resting order(s) "
                                     f"before the model exit.")
                    await asyncio.sleep(1.5)

                exit_order = build_timeout_exit_order(abs(int(qty)),
                                                      use_moc=self.exp_moc_exit)
                self.ib.placeOrder(contract, exit_order)
                self.logger.info(f"[{sym}] MODEL EXIT ({reason}) {qty} sh @ "
                                 f"{'MOC (closing auction)' if self.exp_moc_exit else 'MKT/Adaptive'} "
                                 f"({detail}, held {days}d).")
                sold.append(sym)
            except Exception as e:
                self.logger.error(f"[{sym}] model exit FAILED ({e}) - position left open "
                                  f"with its bracket possibly cancelled. CHECK THIS MANUALLY.")
                self.logger.error(traceback.format_exc())

        if sold:
            await asyncio.sleep(8)
        return sold

    # ── Runner exits: EOD stop repeg + scale-out (bracket_config.USE_RUNNER_EXITS) ──

    def _resting_stops(self, sym: str) -> list:
        """Live (unfilled, uncancelled) SELL STP / STP LMT trades for a symbol."""
        out = []
        for tr in self.ib.openTrades():
            if (tr.contract.symbol.upper() == sym.upper()
                    and tr.orderStatus.status not in ('Filled', 'Cancelled',
                                                      'ApiCancelled', 'Inactive')
                    and tr.order.action == 'SELL'
                    and tr.order.orderType in ('STP', 'STP LMT')):
                out.append(tr)
        return out

    def _fresh_stop_order(self, size: int, stop_price: float) -> 'ibi.Order':
        """A standalone GTC protective stop (no parent, no OCA - runner mode has no TP
        sibling). STP LMT with the validated floor when --exp-stop-limit is on."""
        return build_hard_stop_order(
            order_id=self.ib.client.getReqId(), size=abs(int(size)),
            stop_price=stop_price, parent_id=0, oca_group='',
            half_spread=None, use_stop_limit=self.exp_stop_limit, logger=self.logger)

    def _committed_sell_qty(self, sym: str) -> int:
        """Total working SELL quantity across ALL order types (STP, MOC, MKT, LMT, ...)
        for a symbol. Defense in depth for the repair branch: `_resting_stops` only
        sees STP/STP LMT, so it can't tell "stop is missing" apart from "shares are
        already fully committed to a same-day MOC/MKT exit"."""
        total = 0
        for tr in self.ib.openTrades():
            if (tr.contract.symbol.upper() == sym.upper()
                    and tr.orderStatus.status not in ('Filled', 'Cancelled',
                                                      'ApiCancelled', 'Inactive')
                    and tr.order.action == 'SELL'):
                total += abs(int(tr.order.totalQuantity))
        return total

    async def repeg_stops(self, dry_run: bool = False, skip: set = None) -> int:
        """Raise every resting stop to prior close * (1 - TRAIL_EOD_PCT), never lower.

        This is the EOD ratchet the slot sim validated. It is deliberately NOT an
        exchange TRAIL order: the 1-min referee measured intraday-HWM trailing at
        -0.57 pp/signal, 0/9 books. A position with NO resting stop gets a fresh one
        at the repeg level (protection repair). Returns the number of stops moved.

        `skip` is the set of symbols age-exited earlier THIS run (via a same-day MOC/
        MKT sell): those are not "missing a stop", they are already fully committed
        to exit and must not get a stop added on top, or IBKR reads it as a short."""
        if not (RUNNER_MODE and TRAIL_EOD_PCT > 0) or not self.current_positions:
            return 0
        skip = skip or set()
        moved = 0
        for sym, qty in sorted(self.current_positions.items()):
            sym = sym.upper()
            if not qty or qty <= 0:
                continue
            if sym in skip:
                self.logger.info(f"[{sym}] repeg skipped: age-exited earlier this run.")
                continue
            try:
                contract = ibi.Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)
                [ticker] = await self.ib.reqTickersAsync(contract)
                prior_close = float(ticker.close) if ticker.close else None
                if not prior_close or prior_close <= 0:
                    self.logger.warning(f"[{sym}] repeg skipped: no prior close from IBKR.")
                    continue
                newlvl = round(prior_close * (1 - TRAIL_EOD_PCT / 100.0), 2)
                stops = self._resting_stops(sym)
                if not stops:
                    committed = self._committed_sell_qty(sym)
                    if committed >= qty:
                        self.logger.warning(f"[{sym}] repair skipped: {committed} sh already "
                                            f"committed to a working SELL (>= {qty} held) - "
                                            f"not adding a stop on top of it.")
                        continue
                    self.logger.warning(f"[{sym}] NO resting stop found - placing fresh "
                                        f"GTC stop {qty} sh @ ${newlvl:.2f} (repair).")
                    if dry_run:
                        moved += 1
                        continue
                    trade = self.ib.placeOrder(contract, self._fresh_stop_order(qty, newlvl))
                    await asyncio.sleep(1.0)
                    if trade.orderStatus.status in ('Cancelled', 'ApiCancelled', 'Inactive'):
                        self.logger.error(f"[{sym}] repair stop REJECTED by IBKR (status="
                                          f"{trade.orderStatus.status}) - position has NO "
                                          f"protective stop. CHECK THIS MANUALLY.")
                        continue
                    moved += 1
                    continue
                tr = stops[0]
                cur = float(tr.order.auxPrice or 0)
                if newlvl <= cur + 0.005:
                    continue
                stop_qty = abs(int(tr.order.totalQuantity))
                self.logger.info(f"[{sym}] REPEG stop ${cur:.2f} -> ${newlvl:.2f} "
                                 f"(close ${prior_close:.2f} - {TRAIL_EOD_PCT}%), "
                                 f"{stop_qty} sh.")
                if dry_run:
                    moved += 1
                    continue
                self.ib.cancelOrder(tr.order)
                await asyncio.sleep(1.0)
                if tr.orderStatus.status not in ('Cancelled', 'ApiCancelled', 'Inactive'):
                    self.logger.warning(f"[{sym}] repeg cancel not confirmed - leaving the "
                                        f"old stop in place rather than risking two sells.")
                    continue
                self.ib.placeOrder(contract, self._fresh_stop_order(stop_qty, newlvl))
                moved += 1
            except Exception as e:
                self.logger.error(f"[{sym}] stop repeg FAILED: {e}")
                self.logger.error(traceback.format_exc())
        if moved:
            self.logger.info(f"Repegged/repaired {moved} stop(s).")
        return moved

    async def scale_out_winners(self, dry_run: bool = False, skip: set = None) -> list:
        """Sell (1 - RUNNER_KEEP_FRAC) of any position up >= RUNNER_TRIG_PCT from
        avgCost; the kept runner rides the repegged stop on the RUNNER_MAX_HOLD_DAYS
        clock. One-shot per position, tracked via the ledger's Scaled flag.

        `skip` is the set of symbols age-exited earlier THIS run: already fully
        committed to exit today, so they must not also be scaled out."""
        if not RUNNER_MODE or not self.current_positions:
            return []
        skip = skip or set()
        pos_cost = {p.contract.symbol.upper(): float(p.avgCost)
                    for p in self.ib.positions() if p.position and p.position > 0}
        scaled_now = []
        for sym, qty in sorted(self.current_positions.items()):
            sym = sym.upper()
            if not qty or qty <= 0 or sym in getattr(self, '_scaled', set()):
                continue
            if sym in skip:
                self.logger.info(f"[{sym}] scale-out skipped: age-exited earlier this run.")
                continue
            cost = pos_cost.get(sym)
            if not cost or cost <= 0:
                continue
            try:
                contract = ibi.Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)
                [ticker] = await self.ib.reqTickersAsync(contract)
                px = ticker.marketPrice()
                if not px or px != px:            # nan guard
                    px = float(ticker.last or ticker.close or 0)
                if not px or px <= 0:
                    continue
                if px < cost * (1 + RUNNER_TRIG_PCT / 100.0):
                    continue
                qty_i = abs(int(qty))
                keep = max(1, int(round(qty_i * RUNNER_KEEP_FRAC)))
                sell_n = qty_i - keep
                if sell_n < 1:
                    continue
                self.logger.info(f"[{sym}] SCALE-OUT: +{(px / cost - 1) * 100:.1f}% vs cost "
                                 f"${cost:.2f} - selling {sell_n}, keeping {keep} as runner "
                                 f"(clock -> {RUNNER_MAX_HOLD_DAYS}d).")
                if dry_run:
                    scaled_now.append(sym)
                    continue
                # 1. cancel the resting stop (full-size) before any sell can race it
                for tr in self._resting_stops(sym):
                    self.ib.cancelOrder(tr.order)
                await asyncio.sleep(1.5)
                if self._resting_stops(sym):
                    self.logger.warning(f"[{sym}] stop cancel not confirmed - skipping the "
                                        f"scale-out this run rather than double-selling.")
                    continue
                # 2. sell the banked fraction, MKT/Adaptive (DAY)
                self.ib.placeOrder(contract, build_timeout_exit_order(sell_n, use_moc=False))
                # 3. protect the runner: fresh GTC stop at the current repeg level
                prior_close = float(ticker.close) if ticker.close else px
                runner_stop = round(prior_close * (1 - TRAIL_EOD_PCT / 100.0), 2)
                self.ib.placeOrder(contract, self._fresh_stop_order(keep, runner_stop))
                self._scaled.add(sym)
                scaled_now.append(sym)
            except Exception as e:
                self.logger.error(f"[{sym}] scale-out FAILED: {e}")
                self.logger.error(traceback.format_exc())
        if scaled_now and not dry_run:
            ledger = self._load_ledger()
            self._scaled |= set(scaled_now)
            self._save_ledger({**ledger})
            await asyncio.sleep(5)
        return scaled_now

    # ── Take-profit repair ─────────────────────────────────────────────────────

    async def ensure_take_profits(self, dry_run: bool = False):
        """Re-attach a missing take-profit leg to any long position that has a resting stop.

        Guards, because a stray SELL on a flat book opens a SHORT:
          - long positions only, quantity > 0
          - skip if a resting SELL LMT already exists for the symbol
          - REQUIRE an existing resting stop that carries an ocaGroup, and join that
            group, so TP and stop still cancel each other. No stop or no OCA group =>
            skip and log, never place a naked sell.
          - quantity is the stop's quantity, not the position, so the two legs match
        """
        if RUNNER_MODE:
            self.logger.info("Take-profit repair skipped: runner mode places no TP legs.")
            return
        if not self.current_positions:
            return

        pos_cost = {}
        for p in self.ib.positions():
            if p.position and p.position > 0:
                # avgCost is per share for stocks
                pos_cost[p.contract.symbol.upper()] = float(p.avgCost)

        by_symbol = {}
        for tr in self.ib.openTrades():
            if tr.orderStatus.status in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
                continue
            by_symbol.setdefault(tr.contract.symbol.upper(), []).append(tr)

        for sym, qty in self.current_positions.items():
            sym = sym.upper()
            if not qty or qty <= 0:
                continue
            trades = by_symbol.get(sym, [])
            has_tp = any(t.order.action == 'SELL' and t.order.orderType == 'LMT'
                         for t in trades)
            if has_tp:
                continue
            stops = [t for t in trades
                     if t.order.action == 'SELL' and t.order.orderType in ('STP', 'STP LMT')]
            if not stops:
                self.logger.warning(f"[{sym}] has NO take-profit and NO resting stop. Not "
                                    f"placing a naked sell - the max-hold exit is its only "
                                    f"protection. Investigate the bracket for this name.")
                continue
            stop_tr = stops[0]
            oca = getattr(stop_tr.order, 'ocaGroup', '') or ''
            if not oca:
                self.logger.warning(f"[{sym}] missing take-profit and its stop has no OCA "
                                    f"group - refusing to add an unlinked TP (a filled stop "
                                    f"would leave it orphaned and able to open a short).")
                continue
            cost = pos_cost.get(sym)
            if not cost or cost <= 0:
                self.logger.warning(f"[{sym}] missing take-profit but avgCost unavailable - skipped.")
                continue

            tp_price = round(cost * (1 + TAKE_PROFIT_PCT / 100.0), 2)
            tp_qty = abs(int(stop_tr.order.totalQuantity))
            self.logger.info(f"[{sym}] REPAIR: take-profit leg missing - attaching SELL {tp_qty} "
                             f"LMT ${tp_price:.2f} (+{TAKE_PROFIT_PCT}% on avgCost ${cost:.2f}) "
                             f"to OCA '{oca}'.")
            if dry_run:
                continue
            try:
                contract = ibi.Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)
                tp = ibi.Order(
                    action='SELL',
                    totalQuantity=tp_qty,
                    orderType='LMT',
                    lmtPrice=tp_price,
                    tif='GTC',
                    outsideRth=True,
                    ocaGroup=oca,
                    ocaType=1,
                    transmit=True,
                )
                self.ib.placeOrder(contract, tp)
            except Exception as e:
                self.logger.error(f"[{sym}] take-profit repair failed: {e}")

    # ── Trail calculation ──────────────────────────────────────────────────────

    def _calculate_dynamic_trail(self, price, atr_fallback=None):
        atr = atr_fallback if atr_fallback and atr_fallback > 0 else (price * 0.02)
        atr_percent = (atr / price) * 100
        # 2026-06-10: floor raised 1.2% -> 1.5% to MATCH the backtest trail floor (was
        # intentionally tighter; now aligned with the validated params). Cap stays 4.0%
        # (vs the backtest's 5.0%) - still "closer but a touch more restrictive".
        trailing_percent = max(1.5, 0.75 * atr_percent)
        return min(trailing_percent, 4.0)

    # ── Step 3 + 4: Execute with entry filters ─────────────────────────────────

    def _check_book_freshness(self, signals_df, src=None) -> bool:
        """Reject a STALE narrowed book before any orders are placed.

        Returns True if the book is fresh enough to trade, False if the broker should
        ABORT (place no orders). Staleness = trading days between the book's TargetDate
        and today (ET): 0 fine, 1 warn-and-proceed, >= STALE_ABORT_TDAYS loud banner +
        abort. Guards against the 2026-06-26 stale-book incident.
        """
        date_col = next(
            (c for c in ('TargetDate', 'SignalDate', 'CreatedDate', 'LastUpdated')
             if c in signals_df.columns),
            None,
        )
        dates = (pd.to_datetime(signals_df[date_col], errors='coerce').dropna()
                 if date_col else pd.Series([], dtype='datetime64[ns]'))
        if dates.empty:
            self.logger.warning(
                f"STALE CHECK SKIPPED: {os.path.basename(BUY_SIGNALS_FILE)} has no usable "
                f"date column (looked for TargetDate/SignalDate/CreatedDate/LastUpdated) -- "
                f"cannot verify freshness. Proceeding, but confirm the book was built today."
            )
            return True

        book_date   = dates.max().date()
        today_et    = datetime.now(ET).date()
        stale_tdays = int(np.busday_count(book_date, today_et))
        symbols     = signals_df['Symbol'].unique().tolist()

        if stale_tdays <= 0:
            self.logger.info(
                f"Book freshness OK: TargetDate {book_date:%Y-%m-%d} (today {today_et:%Y-%m-%d})."
            )
            return True

        if stale_tdays < STALE_ABORT_TDAYS:   # unreachable while STALE_ABORT_TDAYS == 1
            self.logger.warning(
                f"STALE BOOK ({stale_tdays} trading day old): {src or os.path.basename(BUY_SIGNALS_FILE)} "
                f"is dated {book_date:%Y-%m-%d (%a)} but today is {today_et:%Y-%m-%d (%a)}. "
                f"Proceeding, but verify these are still today's intended names: {symbols}."
            )
            return True

        # >= STALE_ABORT_TDAYS - refuse and scream.
        self._scream_stale_book(book_date, today_et, stale_tdays, symbols, src=src)
        return False

    def _scream_stale_book(self, book_date, today_et, stale_tdays, symbols, src=None):
        """Emit an unmissable ~25-line red banner explaining the stale-book abort.

        ASCII-only on purpose: a Windows cp1252 console raises UnicodeEncodeError on box-
        drawing/emoji glyphs, which would turn a safety abort into a crash.
        """
        bar  = "#" * 64
        yell = "  ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR  "
        lines = [
            "", bar, bar, yell, yell, bar, "",
            "        STALE SIGNAL BOOK  --  REFUSING TO TRADE",
            "",
            f"   {src or os.path.basename(BUY_SIGNALS_FILE)} is {stale_tdays} TRADING DAYS OLD.",
            f"   Book TargetDate : {book_date:%Y-%m-%d (%a)}",
            f"   Today (ET)      : {today_et:%Y-%m-%d (%a)}",
            f"   Stale symbols   : {', '.join(symbols)}",
            "",
            "   The morning narrowing step (7__MacroFilter.py / the trade-signals",
            "   funnel) did NOT refresh the book, so the broker was about to fill a",
            "   days-old shortlist as if it were today's signals.",
            "",
            "   >>>  NO ORDERS PLACED.  <<<",
            f"   Regenerate {src or os.path.basename(BUY_SIGNALS_FILE)} for today, then relaunch the broker.",
            "",
            yell, yell, bar, bar, "",
        ]
        for ln in lines:
            self.logger.error(ln)





    async def execute_batch(self, market_conds: dict):
        """
        Execute pending signals, applying three EDA-derived entry filters:

          Filter 1 (already applied): SPY direction gate in market_conds
          Filter 2: Per-stock open-gap check
                    Gap > +1.5% → skip  (EDA: rest-of-day mean -0.24%, fade effect)
                    Dip -1.5% to -0.5% → log as favorable  (rest-of-day mean +0.24%)
          Filter 3: Parent order uses tif=DAY + outsideRth=False
                    Prevents pre-market fills that bypass all checks above.
                    TP and trail stop remain GTC/outsideRth to catch after-hours exits.

        Order execution uses IBKR Adaptive algo (Normal priority) so the limit
        sits inside the spread and walks out - cutting entry slippage from ~10-15 bps
        (plain LMT at ask) to ~2-5 bps on typical S&P 500 names.
        limit_price is a ceiling/safety cap, not the expected fill price.
        """
        # Read the final narrowed book from _Buy_Signals.parquet (not the legacy
        # Z_signals.parquet that read_signals() points at). Filter to Pending if the
        # column is present; the post-rubric file may omit it.
        if not os.path.exists(BUY_SIGNALS_FILE):
            self.logger.info(f"Signals file {BUY_SIGNALS_FILE} not found.")
            return
        signals_df = pd.read_parquet(BUY_SIGNALS_FILE)

        # ── Fail-safe: only ever trade a NARROWED book ────────────────────────────
        # If the file has no Status column it's the raw backtester ledger, not a
        # post-rubric shortlist - refuse. If it has more than MAX_BOOK pending rows
        # it hasn't been narrowed - refuse. Either way, place NO orders rather than
        # fire at an un-narrowed / wrong file.
        if 'Status' not in signals_df.columns:
            self.logger.error(
                f"ABORT: {BUY_SIGNALS_FILE} has no 'Status' column - looks like the raw "
                f"trading ledger, not a narrowed book. No orders placed. "
                f"Run the morning narrowing step first."
            )
            return
        signals_df = signals_df[signals_df['Status'] == 'Pending']
        if len(signals_df) > MAX_BOOK:
            self.logger.error(
                f"ABORT: {len(signals_df)} pending signals in {BUY_SIGNALS_FILE} exceeds "
                f"MAX_BOOK={MAX_BOOK} - file has not been narrowed. No orders placed."
            )
            return
        if signals_df.empty:
            self.logger.info("No pending signals found in _Buy_Signals.parquet.")
            return

        # ── Fail-safe: reject a STALE book before placing any orders ───────────────
        if not self._check_book_freshness(signals_df):
            return

        symbols = signals_df['Symbol'].unique().tolist()
        self.logger.info(f"Processing Batch: {symbols}")

        contracts = [ibi.Stock(s, 'SMART', 'USD') for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)

        self.logger.info("Requesting Market Snapshots...")
        tickers = await self.ib.reqTickersAsync(*contracts)

        orders_staged = []
        # Cash-aware staging (2026-07-30): the sizer sees the same available_cash for
        # every name in a batch, so on multi-name days the tail brackets exceeded real
        # settled cash and IBKR rejected them without a trace. Statement forensics:
        # all 9 unfilled staged entries (07-14/15/27/28) were the TAIL of their batch;
        # 07-15 lost half the book. This ledger is debited as names stage.
        projected_cash = self.available_cash

        spy_move = market_conds.get('spy_move_pct')
        spy_tag  = f"  [SPY {spy_move:+.2f}%]" if spy_move is not None else ""

        for contract, ticker in zip(contracts, tickers):
            symbol = contract.symbol

            if symbol in self.current_positions:
                self.logger.warning(f"[{symbol}] Skipping: Already hold position.")
                continue

            # ── Price reference: midpoint, not ask ────────────────────────────
            # Using ask as reference overstates entry cost by half the spread
            # before the Adaptive algo even runs.
            bid    = ticker.bid  if (ticker.bid  and ticker.bid  > 0) else None
            ask    = ticker.ask  if (ticker.ask  and ticker.ask  > 0) else None
            spread = (ask - bid) if (bid and ask) else None
            mid    = (bid + ask) / 2 if (bid and ask) else (ticker.last or ask or bid)

            if not mid or mid <= 0:
                self.logger.error(f"[{symbol}] Bad Data: No price available.")
                continue

            spread_bps = (spread / mid * 10_000) if spread else None
            self.logger.info(
                f"[{symbol}] Quotes: bid=${bid} ask=${ask} mid=${mid:.2f}"
                + (f" spread={spread_bps:.1f} bps" if spread_bps else "")
            )

            # ── Filter 2: Per-stock open-gap check ────────────────────────────
            try:
                today_open = float(ticker.open) if ticker.open else None
            except (TypeError, ValueError):
                today_open = None

            if today_open and today_open > 0:
                open_gap_pct = (mid / today_open - 1) * 100

                if open_gap_pct > STOCK_GAP_SKIP:
                    self.logger.warning(
                        f"[{symbol}] SKIP: Gapped up {open_gap_pct:+.1f}% from open "
                        f"(EDA: stocks up >{STOCK_GAP_SKIP:.0f}% at open show "
                        f"mean rest-of-day -0.24% fade - not a good entry)."
                    )
                    continue
                elif STOCK_DIP_GOOD_LO <= open_gap_pct <= STOCK_DIP_GOOD_HI:
                    self.logger.info(
                        f"[{symbol}] FAVORABLE DIP: {open_gap_pct:+.1f}% from open "
                        f"(EDA: dip entries in this range show mean rest-of-day +0.24%)."
                    )
                else:
                    self.logger.info(f"[{symbol}] Open gap: {open_gap_pct:+.1f}% - within normal range.")
            else:
                self.logger.warning(f"[{symbol}] No open price available for gap check - proceeding.")

            # ── Position sizing ────────────────────────────────────────────────
            row            = signals_df[signals_df['Symbol'] == symbol].iloc[0]
            parquet_stop   = row['StopPrice']   if 'StopPrice'   in row and pd.notnull(row['StopPrice'])   else None
            parquet_target = row['TargetPrice'] if 'TargetPrice' in row and pd.notnull(row['TargetPrice']) else None
            parquet_atr    = row['ATR']         if 'ATR'         in row and pd.notnull(row['ATR'])         else None

            current_slots_used = len(self.current_positions) + len(orders_staged)
            size = self.position_sizer.calculate_position_size(
                account_value=self.account_value,
                current_cash=self.available_cash,
                price=mid,
                current_positions=current_slots_used,
                symbol=symbol
            )
            if size <= 0:
                continue

            # ── Price levels - all anchored off mid, not ask ───────────────────
            # cushion = larger of 5 bps or one full spread; this becomes the
            # Adaptive algo ceiling, NOT the expected fill price
            cushion     = max(0.0005 * mid, spread if spread else 0.0)
            limit_price = round(mid + cushion, 2)

            # ── Cash guard: never stage what cannot transmit ───────────────────
            cost = size * limit_price
            if cost > projected_cash:
                fit = int(projected_cash // limit_price)
                if fit >= 1 and fit * limit_price >= 100:   # no dust positions
                    self.logger.warning(
                        f"[{symbol}] CASH GUARD: downsized {size} -> {fit} shares "
                        f"(${cost:,.0f} wanted, ${projected_cash:,.0f} projected cash left).")
                    size = fit
                    cost = fit * limit_price
                else:
                    self.logger.warning(
                        f"[{symbol}] CASH GUARD: skipped - ${cost:,.0f} wanted, only "
                        f"${projected_cash:,.0f} projected cash left. Before this guard "
                        f"the order transmitted anyway and died as a silent IBKR rejection.")
                    continue
            projected_cash -= cost

            if parquet_stop and 0 < parquet_stop < limit_price:
                stop_price = parquet_stop
            else:
                stop_price = limit_price * 0.9825    # 1.75% stop (tighter than 2% backtest default)

            if parquet_target and parquet_target > limit_price:
                take_profit = parquet_target
            else:
                risk        = limit_price - stop_price
                take_profit = limit_price + (risk * 2.0)

            trail_pct = self._calculate_dynamic_trail(limit_price, parquet_atr)

            # ── Build bracket ──────────────────────────────────────────────────
            # Each call to getReqId() is guaranteed to return a fresh incrementing
            # ID - safer than reserving a block of 4 from a single call.
            parent_id    = self.ib.client.getReqId()
            tp_id        = self.ib.client.getReqId()
            stop_id      = self.ib.client.getReqId()
            hard_stop_id = self.ib.client.getReqId()

            oca_group = f"bracket_{symbol}_{parent_id}"

            # Store mid for slippage comparison when fills arrive
            self._signal_mids[symbol] = mid

            # Filter 3: DAY + outsideRth=False - prevents pre-market fills.
            # Adaptive/Normal: sits inside the spread and walks outward.
            # Expected slippage: ~2-5 bps vs ~10-15 bps for a plain LMT at ask.
            parent = ibi.Order(
                orderId=parent_id,
                action='BUY',
                totalQuantity=size,
                orderType='LMT',
                lmtPrice=limit_price,
                tif='DAY',
                outsideRth=False,
                algoStrategy='Adaptive',
                algoParams=[ibi.TagValue('adaptivePriority', 'Normal')],
                transmit=False
            )

            # TP and trail stay GTC + outsideRth to catch after-hours gaps on exit.
            # Explicit OCA group ensures exactly one fills and the other cancels - 
            # without this, a filled TP can leave an orphaned trail that opens a short.
            # RUNNER MODE: no take-profit leg at all - the +RUNNER_TRIG_PCT scale-out
            # in scale_out_winners() is the profit-taking mechanism.
            take_profit_ord = None
            if not RUNNER_MODE:
                take_profit_ord = ibi.Order(
                    orderId=tp_id,
                    action='SELL',
                    totalQuantity=size,
                    orderType='LMT',
                    lmtPrice=take_profit,
                    tif='GTC',
                    outsideRth=True,
                    parentId=parent_id,
                    ocaGroup=oca_group,
                    ocaType=1,
                    transmit=False
                )

            # ── STAGE-2  ──────────────────────

            # Hard stop: fires immediately if price falls HARD_STOP_PCT% from entry.
            # Sits inside the same OCA group - whichever of TP / trail / hard-stop
            # fills first cancels the other two. Tighter than the trailing floor so
            # a stock moving against us on entry cuts out fast instead of letting
            # the trail give it 1.2% of rope.
            hard_stop_price = round(mid * (1 - HARD_STOP_PCT / 100), 2)
            # Default: STP, byte-identical to what has always shipped. Only --exp-stop-limit
            # changes this, to STP LMT with a floor under the trigger. `spread` is the live
            # NBBO width captured above, so half_spread is a real measurement here, not a proxy.
            hard_stop_ord = build_hard_stop_order(
                order_id=hard_stop_id, size=size, stop_price=hard_stop_price,
                parent_id=parent_id, oca_group=oca_group,
                half_spread=(spread / 2.0) if spread else None,
                use_stop_limit=self.exp_stop_limit, logger=self.logger,
            )

            tp_txt = (f"TP: ${take_profit:.2f}" if take_profit_ord is not None
                      else f"TP: NONE (runner mode, scale-out at +{RUNNER_TRIG_PCT:g}%)")
            self.logger.info(
                f"[{symbol}] Staging: Buy {size} @ lmt=${limit_price:.2f} (mid=${mid:.2f}) | "
                f"HardStop: ${hard_stop_price:.2f} ({HARD_STOP_PCT}%) | "
                f"{tp_txt} | trail=OFF | OCA: {oca_group}{spy_tag}"
            )
            legs = [parent] + ([take_profit_ord] if take_profit_ord is not None else []) \
                   + [hard_stop_ord]
            orders_staged.append((contract, legs))
            self._staged_parents.append((contract, parent, symbol))

        if orders_staged:
            if self.dry_run:
                self.logger.warning(
                    f"DRY RUN: {len(orders_staged)} bracket(s) staged and NOT transmitted "
                    f"({[s for _, _, s in self._staged_parents]}). Nothing was sent.")
                self._staged_parents = []
            else:
                self.logger.info(f"Transmitting {len(orders_staged)} brackets to exchange...")
                for contract, orders in orders_staged:
                    for o in orders:
                        self.ib.placeOrder(contract, o)
                self.logger.info("All orders transmitted.")
        else:
            self.logger.info("No valid orders after entry filters.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TRIGGER ENTRY  (--trigger-entry, default OFF)
    # ═══════════════════════════════════════════════════════════════════════════

    # =======================================================================
    # TRIGGER ENTRY  (--trigger-entry, default OFF)
    #
    # TWO ARMING MODES, and the difference is economic, not cosmetic.
    #
    #   virtual (default)  Nothing is sent at 10:00. The trigger table is held in
    #                      memory, streaming quotes are watched, and ONE real order
    #                      goes out the moment a name reaches its own trigger. At
    #                      most `free_slots` orders ever exist at once.
    #
    #   resting            Every eligible name gets a resting BUY LIMIT at 10:00 and
    #                      they sit on the book all day. This is what the backtest
    #                      modelled, because a resting bid is filled AT the trigger
    #                      and therefore EARNS the half-spread.
    #
    # WHAT VIRTUAL COSTS. You cannot rest and react at the same time. Virtual learns
    # the price reached the trigger only AFTER it has, so it pays:
    #   1. poll latency -- up to --trigger-poll seconds of staleness, during which a
    #      fast dip can bounce back through the level, and
    #   2. the spread -- a resting bid collects the half-spread; an order sent on
    #      touch does not.
    # The measured arm (+0.7730%/trade over ten seeds) assumed a resting fill at the
    # trigger price. Neither cost above is in that number, and 9p__PaperTriggerBroker
    # was built to measure exactly this and has not yet done so. Treat virtual-mode
    # results as unvalidated against the backtest until it has.
    #
    # The order sent on touch is a LIMIT AT THE TRIGGER, never a market order, so a
    # touch can never fill worse than the modelled entry price. The tradeoff is the
    # honest one: a dip that touches and rebounds may leave the limit unfilled.
    # =======================================================================

    # ── Rubric veto (2026-08-28) ───────────────────────────────────────────────
    # The morning FilterRubric used to NARROW the pool to 3 names and write them to
    # _Buy_Signals.parquet, which the trigger arm does not read at all. Under reach-24
    # its job changes shape: it stays a judgment layer, but it SHAVES a wide book
    # instead of picking a small one. It marks the worst few names RubricExclude=True
    # in the pool and the broker drops them here, next to the mechanical exclusions.
    #
    # WHY THERE IS A CEILING ON IT. A veto layer that can cut without limit is a
    # narrowing step wearing a different name, and a stale or malfunctioning verdict
    # column that flagged 20 of 24 would arm a 4-name book that looked completely
    # normal in the log. That is the exact silent-degradation shape that hid the
    # 52-week gate for months. So the cut is capped: past the cap the veto is REFUSED
    # rather than trimmed, because a rubric that flagged 20 names is not a rubric whose
    # first 4 flags you should trust either.
    #
    # Absent column = no veto ran = arm the whole mechanically-clean pool. That is the
    # safe direction: the rubric can only ever REMOVE candidates, so its absence costs
    # selectivity, never protection.
    def _apply_rubric_veto(self, pool_df):
        if 'RubricExclude' not in pool_df.columns:
            self.logger.info(
                "TRIGGER: pool carries no RubricExclude column - no discretionary veto "
                "was recorded for today. Arming the full mechanically-clean pool of "
                f"{len(pool_df)}.")
            return pool_df

        flag = pool_df['RubricExclude'].fillna(False).astype(bool)
        n_cut = int(flag.sum())
        if n_cut == 0:
            self.logger.info(f"TRIGGER: rubric veto present, vetoed nothing. "
                             f"{len(pool_df)} candidates stand.")
            return pool_df

        self._last_rubric_cut = n_cut
        if n_cut > self.trigger_rubric_max_cut:
            self._last_rubric_refused = True
            self.logger.error(
                f"TRIGGER: rubric vetoed {n_cut} of {len(pool_df)} names, over the "
                f"ceiling of {self.trigger_rubric_max_cut}. REFUSING THE WHOLE VETO and "
                f"arming the mechanically-clean pool instead. A cut this deep is a "
                f"narrowing, not a shave - check that the RubricExclude column was "
                f"written for TODAY and not carried over from a previous session.")
            for _, r in pool_df[flag].iterrows():
                self.logger.error(f"  IGNORED VETO {r['Symbol']}: "
                                  f"{r.get('RubricReasons', 'no reason recorded')}")
            return pool_df

        for _, r in pool_df[flag].iterrows():
            self.logger.warning(
                f"[{r['Symbol']}] TRIGGER drop: rubric veto - "
                f"{r.get('RubricReasons', 'no reason recorded')}")
        kept = pool_df[~flag]
        self.logger.info(
            f"TRIGGER: rubric veto dropped {n_cut} (ceiling {self.trigger_rubric_max_cut}), "
            f"{len(kept)} remain.")
        return kept

    async def stage_trigger_entries(self, market_conds: dict):
        """Build the trigger table and ARM it. Sends orders only in resting mode."""
        from auxiliary import trigger_entry as TE

        def _log(msg='', level='INFO'):
            (self.logger.warning if level == 'WARN' else self.logger.info)(msg)

        try:
            pool_df, target_date = TE.load_pool(self.trigger_reach, _log)
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"TRIGGER: cannot load pool ({e}). No orders placed.")
            return
        self._trigger_target = target_date
        self._last_rubric_cut = 0
        self._last_rubric_refused = False
        _pool_rows_loaded = len(pool_df)
        _mech_dropped = None

        if not self._check_book_freshness(pool_df, src='Data/0__Signals.parquet'):
            return

        # Mechanical hard exclusions. Trigger mode reads the WIDE pool and therefore
        # never passes through 7__MacroFilter, where the Stage-1 HARD exclusions live
        # (price floor, micro-cap, weekly-vol cliff, stuck price, quarantine). Widening
        # the pool must not also drop the safety gates. 5__NightlyBackTester bakes the
        # rubric's own verdict into the pool as MechExclude, so honour that rather than
        # reimplementing the gates and letting the two drift.
        if 'MechExclude' in pool_df.columns:
            flagged = pool_df[pool_df['MechExclude'].fillna(False).astype(bool)]
            for _, r in flagged.iterrows():
                self.logger.warning(
                    f"[{r['Symbol']}] TRIGGER drop: MechExclude - "
                    f"{r.get('MechReasons', 'no reason recorded')}")
            pool_df = pool_df[~pool_df['MechExclude'].fillna(False).astype(bool)]
            _mech_dropped = len(flagged)
            self.logger.info(
                f"TRIGGER: mechanical filter dropped {len(flagged)}, {len(pool_df)} remain.")
            if pool_df.empty:
                self.logger.error("TRIGGER: every candidate was mechanically excluded. "
                                  "No orders placed.")
                return
        else:
            self.logger.warning(
                "TRIGGER: pool has no MechExclude column, so the Stage-1 hard exclusions "
                "are NOT applied to this book. Run signal_filter.py on the pool.")

        pool_df = self._apply_rubric_veto(pool_df)
        if pool_df is None or pool_df.empty:
            self.logger.error("TRIGGER: nothing survived the rubric veto. No orders placed.")
            return

        plan = TE.tradable(TE.build_triggers(pool_df, self.trigger_k, _log))
        if plan.empty:
            self.logger.error("TRIGGER: no candidate could be priced. No orders placed.")
            return

        self.logger.info(
            f"TRIGGER: {len(plan)} of {len(pool_df)} candidates priced "
            f"(K={self.trigger_k:g}, reach={self.trigger_reach}, target={target_date}, "
            f"arm={self.trigger_arm})")

        free_slots = self.position_sizer.max_positions - len(self.current_positions)
        if _diag.enabled():
            _diag.stamp('9_SuperFastBroker')
            _diag.dump_parquet(f'X_plan_{target_date}', plan)
            _diag.append_jsonl('X_stage', {
                'target_date': str(target_date), 'reach': self.trigger_reach, 'k': self.trigger_k,
                'arm': self.trigger_arm, 'pool_rows': _pool_rows_loaded, 'mech_dropped': _mech_dropped,
                'mech_present': _mech_dropped is not None, 'rubric_cut': self._last_rubric_cut,
                'rubric_refused': self._last_rubric_refused, 'after_veto': int(len(pool_df)),
                'priced': int(len(plan)), 'free_slots': int(free_slots),
                'held_at_start': int(len(self.current_positions)),
                'max_positions': self.position_sizer.max_positions, 'dry_run': self.dry_run,
                'gap_gate': dict(TE.LAST_GAP_CENSUS)})
        _skipped = []
        if free_slots <= 0:
            self.logger.info(
                f"TRIGGER: book already full ({len(self.current_positions)}/"
                f"{self.position_sizer.max_positions}). Nothing to arm.")
            return

        symbols = plan['symbol'].tolist()
        contracts = [ibi.Stock(s, 'SMART', 'USD') for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)
        tickers = await self.ib.reqTickersAsync(*contracts)
        quote = {c.symbol: t for c, t in zip(contracts, tickers)}
        by_symbol = {c.symbol: c for c in contracts}

        spy_move = market_conds.get('spy_move_pct')
        spy_tag  = f"  [SPY {spy_move:+.2f}%]" if spy_move is not None else ""

        for row in plan.itertuples():
            symbol  = row.symbol
            trigger = float(row.trigger)

            if symbol in self.current_positions:
                self.logger.warning(f"[{symbol}] TRIGGER skip: already hold position.")
                _skipped.append({'symbol': symbol, 'reason': 'held'})
                continue

            t = quote.get(symbol)
            bid = t.bid if (t and t.bid and t.bid > 0) else None
            ask = t.ask if (t and t.ask and t.ask > 0) else None
            spread = (ask - bid) if (bid and ask) else None
            mid = (bid + ask) / 2 if (bid and ask) else ((t.last or ask or bid) if t else None)
            spread_bps = (spread / mid * 10_000) if (spread and mid) else None

            if self.trigger_max_spread_bps > 0:
                if spread_bps is None:
                    self.logger.warning(
                        f"[{symbol}] TRIGGER skip: spread guard armed but no two-sided "
                        f"quote available to measure it.")
                    _skipped.append({'symbol': symbol, 'reason': 'no_quote'})
                    continue
                if spread_bps > self.trigger_max_spread_bps:
                    self.logger.warning(
                        f"[{symbol}] TRIGGER skip: spread {spread_bps:.1f} bps exceeds "
                        f"ceiling {self.trigger_max_spread_bps:.1f} bps.")
                    _skipped.append({'symbol': symbol, 'reason': 'spread', 'spread_bps': spread_bps})
                    continue

            # A trigger at or above the market is not a dip. In resting mode it would
            # fill instantly; in virtual mode it would fire on the first poll. Either
            # way it is not this strategy.
            if mid and trigger >= mid:
                self.logger.warning(
                    f"[{symbol}] TRIGGER skip: trigger ${trigger:.2f} is at/above mid "
                    f"${mid:.2f} - that is a market buy, not a dip entry.")
                _skipped.append({'symbol': symbol, 'reason': 'at_mid', 'trigger': trigger, 'mid': mid})
                continue

            depth_txt = f"{row.depth_pct:.2f}%" + (f" [{row.clamped}]" if row.clamped else "")
            self.logger.info(
                f"[{symbol}] TRIGGER rank {int(row.rank):>2}: arm @ ${trigger:.2f} "
                f"(prior close ${row.prior_close:.2f}, depth {depth_txt}, "
                f"beta {row.beta:.2f}/{row.beta_src}, vol {row.vol_20d:.4f})"
                + (f" spread={spread_bps:.1f}bps" if spread_bps else " spread=n/a")
                + spy_tag)

            self._armed.append(dict(
                symbol=symbol, contract=by_symbol[symbol], trigger=trigger,
                rank=int(row.rank), spread=spread,
                depth_pct=float(row.depth_pct), beta=float(row.beta),
                prior_close=float(row.prior_close),
                # ── lifecycle ──────────────────────────────────────────────────
                # ARMED    watched, nothing on the book
                # WORKING  a real limit order is live and unfilled
                # FILLED   confirmed shares in the account
                # DEAD     cancelled, rejected or expired without filling
                state='ARMED', sent=False, order=None, order_id=None,
                stop_order=None, oca=None, sent_at=None, want_qty=0,
                filled_qty=0, fill_price=None, note=''))

        if _diag.enabled():
            _diag.append_jsonl('X_armed', {
                'target_date': str(target_date),
                'armed': [{k: a.get(k) for k in ('symbol', 'rank', 'trigger', 'depth_pct', 'beta', 'prior_close')}
                          | {'spread_bps': (a['spread'] / ((a['trigger'] + a['spread'] / 2) or 1) * 10_000
                                            if a.get('spread') else None)}
                          for a in self._armed],
                'skipped': _skipped})
        # Per-run census for the gap gate: pool_width / skipped_by_gap / armed. Printed
        # on every run, gate on or off, so an OFF run reads n=0 and never looks like a
        # gate that silently did nothing.
        _gc = TE.LAST_GAP_CENSUS
        _short = (len(self._armed) < self.trigger_reach and _gc.get('skipped'))
        self.logger.info(
            f"GAP GATE CENSUS: n={_gc.get('n', 0)} pct={_gc.get('pct')} "
            f"pool_width={_gc.get('pool_width')} skipped_by_gap={len(_gc.get('skipped', []))} "
            f"{_gc.get('skipped', [])} survivors={_gc.get('survivors')} "
            f"armed={len(self._armed)} reach={self.trigger_reach}"
            + ("  <- REACH SHORTFALL caused by the gate; widen BT_SIGNAL_POOL_SIZE" if _short else ""))
        if not self._armed:
            self.logger.info("TRIGGER: nothing armed after guards.")
            return

        # Frozen HERE, before a single order goes out, and never recomputed from
        # self.current_positions again. prepare_account_data() is re-run after every
        # fire so the sizer sees committed cash, which means current_positions grows as
        # our own entries fill - counting it live would double-count each fill against
        # the book cap and disarm the pool early.
        self._held_at_start = len(self.current_positions)

        if self.trigger_arm == 'resting':
            self.logger.info(f"TRIGGER: resting {len(self._armed)} entry order(s) for "
                             f"{free_slots} free slot(s)...")
            for a in list(self._armed):
                self._send_trigger_order(a, a['trigger'], reason='resting')
            self.logger.info("TRIGGER: all resting entries transmitted.")
        else:
            self.logger.info(
                f"TRIGGER: {len(self._armed)} name(s) armed VIRTUALLY for {free_slots} "
                f"free slot(s). NO orders are on the book. One order is sent per name "
                f"only when it reaches its trigger.")
        self._write_monitor_state('armed')

    def _send_trigger_order(self, armed, limit_price, reason=''):
        """Send one entry: LMT at the trigger plus its hard-stop child.

        Moves the record ARMED -> WORKING on success. WORKING is not a position: the
        limit sits at the trigger and may never fill. Only _reconcile_working() may
        promote it to FILLED, and only against shares IBKR actually reports.
        """
        symbol   = armed['symbol']
        contract = armed['contract']
        trigger  = armed['trigger']
        spread   = armed.get('spread')

        size = self.position_sizer.calculate_position_size(
            account_value=self.account_value,
            current_cash=self.available_cash,
            price=limit_price,
            current_positions=len(self.current_positions),
            symbol=symbol,
        )
        if size <= 0:
            self.logger.warning(f"[{symbol}] TRIGGER: sizer returned {size}, not sent.")
            armed['state'] = 'DEAD'
            armed['note']  = f'sizer returned {size}'
            self._write_monitor_state('sizer-zero')
            return False

        cost = size * limit_price
        if cost > self.available_cash:
            fit = int(self.available_cash // limit_price)
            if fit >= 1 and fit * limit_price >= 100:
                self.logger.warning(
                    f"[{symbol}] TRIGGER downsized {size} -> {fit} shares to fit "
                    f"${self.available_cash:,.0f} cash.")
                size = fit
            else:
                self.logger.warning(
                    f"[{symbol}] TRIGGER skip: ${cost:,.0f} wanted, only "
                    f"${self.available_cash:,.0f} cash available.")
                armed['state'] = 'DEAD'
                armed['note']  = f'insufficient cash (${self.available_cash:,.0f})'
                self._write_monitor_state('no-cash')
                return False

        hard_stop_price = round(trigger * (1 - HARD_STOP_PCT / 100), 2)

        if self.dry_run:
            self.logger.warning(
                f"[{symbol}] DRY RUN, nothing transmitted. WOULD BUY {size} @ LMT "
                f"${limit_price:.2f} (trigger ${trigger:.2f}, reason {reason}) with "
                f"HardStop ${hard_stop_price:.2f}.")
            armed['state'] = 'DEAD'
            armed['note']  = f'dry run ({reason})'
            armed['want_qty'] = size
            self._write_monitor_state('dry-run')
            return False

        parent_id    = self.ib.client.getReqId()
        hard_stop_id = self.ib.client.getReqId()
        oca_group    = f"trigger_{symbol}_{parent_id}"

        # Slippage reference is the TRIGGER, not the mid: the question this mode exists
        # to answer is whether you got the price you armed at.
        self._signal_mids[symbol] = trigger

        parent = ibi.Order(
            orderId=parent_id, action='BUY', totalQuantity=size,
            orderType='LMT', lmtPrice=round(limit_price, 2),
            tif='DAY', outsideRth=False, transmit=False,
        )
        # Stop anchors off the TRIGGER, the price this position is being opened at.
        hard_stop_ord = build_hard_stop_order(
            order_id=hard_stop_id, size=size, stop_price=hard_stop_price,
            parent_id=parent_id, oca_group=oca_group,
            half_spread=(spread / 2.0) if spread else None,
            use_stop_limit=self.exp_stop_limit, logger=self.logger,
        )

        self.logger.warning(
            f"[{symbol}] TRIGGER FIRE ({reason}): BUY {size} @ LMT ${limit_price:.2f} "
            f"(trigger ${trigger:.2f}) | HardStop ${hard_stop_price:.2f} | OCA {oca_group}")
        for o in (parent, hard_stop_ord):
            self.ib.placeOrder(contract, o)

        armed.update(state='WORKING', sent=True, order=parent, order_id=parent_id,
                     stop_order=hard_stop_ord, oca=oca_group, want_qty=size,
                     sent_at=datetime.now(ET), note=reason)
        self._trigger_orders.append((contract, parent, symbol))
        self._write_monitor_state('fired')
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # POTENTIAL-POSITION MONITOR
    #
    # Reach-24 arms 24 names for 3 slots, so on any given morning ~21 of them are
    # POTENTIAL positions: watched, priced, sized, and entitled to become real the
    # instant price reaches their trigger. Tracking them is the whole job of this
    # block, and it is the part with NO backtest counterpart, because in the backtest a
    # limit order that touches its level simply fills.
    #
    # THE DEFECT THIS REPLACES. The first watcher counted orders SENT against the book
    # cap:
    #       sent = sum(1 for a in self._armed if a['sent'])
    #       if held_at_start + sent >= max_positions: disarm everything
    # An order sent at the trigger is not a fill. A limit that touches and rebounds
    # leaves you with nothing, and three of those ended the session with the counter
    # reading FULL, 21 names disarmed and an empty book. The cap has to count what IBKR
    # confirms, which means every working order must be reconciled on every poll.
    #
    # SLOT ARITHMETIC, and why it is deliberately conservative:
    #       committed = held_at_start + filled_today + working
    # A WORKING order counts against the cap because it can fill in the next second and
    # crediting the slot back early is how you end up over-booked on the one day the
    # whole market gaps. Positions CLOSED during the session (a stop hit at noon) do
    # NOT credit a slot back: re-entering a name the same day it stopped out is not the
    # arm that was measured, and the safe error here is trading too little.
    # ══════════════════════════════════════════════════════════════════════════

    def _committed(self):
        """Slots consumed: opening book + our confirmed fills + our live orders."""
        filled  = sum(1 for a in self._armed if a['state'] == 'FILLED')
        working = sum(1 for a in self._armed if a['state'] == 'WORKING')
        return self._held_at_start + filled + working, filled, working

    def _write_monitor_state(self, event=''):
        """Crash breadcrumb. Best-effort: never let bookkeeping kill a trading loop."""
        try:
            os.makedirs(os.path.dirname(TRIGGER_MONITOR_FILE), exist_ok=True)
            committed, filled, working = self._committed()
            payload = {
                'written': datetime.now(ET).isoformat(),
                'event': event,
                'arm_mode': self.trigger_arm,
                'reach': self.trigger_reach,
                'dry_run': self.dry_run,
                'held_at_start': self._held_at_start,
                'max_positions': self.position_sizer.max_positions,
                'committed': committed, 'filled': filled, 'working': working,
                'names': [
                    {k: (v.isoformat() if isinstance(v, datetime) else v)
                     for k, v in a.items()
                     if k in ('symbol', 'rank', 'state', 'trigger', 'depth_pct', 'beta',
                              'want_qty', 'filled_qty', 'fill_price', 'order_id',
                              'sent_at', 'note')}
                    for a in self._armed],
                'fills': self._fill_log,
            }
            # History, one line per state change (the json file is overwritten each time).
            _diag.append_jsonl('X_monitor', dict(payload, target_date=str(getattr(self, '_trigger_target', ''))))
            with open(TRIGGER_MONITOR_FILE, 'w') as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"TRIGGER monitor state not written ({e}). Trading continues.")

    def _reconcile_working(self):
        """Promote / retire every WORKING record against what IBKR reports.

        This is the only place a record becomes FILLED. Returns the list of records
        that newly filled on this pass so the caller can do the post-fill work once.
        """
        working = [a for a in self._armed if a['state'] == 'WORKING']
        if not working:
            return []

        by_id = {t.order.orderId: t for t in self.ib.trades()}
        newly_filled = []
        for a in working:
            tr = by_id.get(a['order_id'])
            if tr is None:
                # Not in the trade list at all. Do NOT assume it died: an order the API
                # has not echoed back yet looks identical to one that never existed, and
                # freeing the slot on a hunch is how a name gets bought twice.
                continue
            st  = tr.orderStatus
            fq  = int(st.filled or 0)
            rem = float(st.remaining or 0)
            a['filled_qty'] = fq
            if fq > 0 and st.avgFillPrice:
                a['fill_price'] = float(st.avgFillPrice)

            if fq > 0 and rem <= 0:
                a['state'] = 'FILLED'
                slip = ((a['fill_price'] - a['trigger']) / a['trigger'] * 1e4
                        if a['fill_price'] else 0.0)
                self.logger.warning(
                    f"[{a['symbol']}] TRIGGER FILLED {fq} @ ${a['fill_price']:.2f} "
                    f"(armed at ${a['trigger']:.2f}, {slip:+.1f} bps vs trigger).")
                self._fill_log.append(dict(
                    symbol=a['symbol'], rank=a['rank'], qty=fq,
                    trigger=a['trigger'], fill_price=a['fill_price'],
                    slip_bps=round(slip, 1), note=a.get('note', ''),
                    at=datetime.now(ET).isoformat()))
                newly_filled.append(a)
            elif st.status in ('Cancelled', 'ApiCancelled', 'Inactive') and fq <= 0:
                a['state'] = 'DEAD'
                a['note']  = f'order {st.status}'
                self.logger.warning(
                    f"[{a['symbol']}] TRIGGER entry {st.status} with no fill - slot "
                    f"released, name retired for today.")
            elif fq > 0 and rem > 0:
                # Partial. It still holds its slot and stays WORKING; the stop child is
                # sized to the full order, which IBKR reduces as the parent fills.
                self.logger.info(
                    f"[{a['symbol']}] TRIGGER partial: {fq}/{a['want_qty']} filled, "
                    f"{rem:g} still working.")
        return newly_filled

    async def _post_fill_checks(self, filled_records):
        """Everything a newly-real position needs, done NOW and not at the cutoff.

        The session runs to 15:45. Before this, a name that filled at 10:30 waited five
        hours for _post_trade_maintenance to stamp its entry date, and a taskkill in
        between lost the stamp entirely - which sets the max-hold clock to whenever the
        position is next noticed, holding it past MAX_HOLD_DAYS.

        The stop check is the important half. The hard stop is transmitted as a CHILD of
        the entry, so if IBKR rejects the child (bad price, contract rules) the parent
        can still fill and leave a NAKED long with no protection until tomorrow's run.
        That is the worst outcome this system can produce, so it is checked on every
        fill and repaired if missing.
        """
        if not filled_records:
            return
        try:
            await self.prepare_account_data()
        except Exception as e:
            self.logger.error(f"TRIGGER: account refresh after fill failed: {e}")

        # Stamp the max-hold clock immediately.
        try:
            self._reconcile_ledger()
        except Exception as e:
            self.logger.error(f"TRIGGER: ledger stamp after fill failed: {e}. The "
                              f"max-hold clock for these names may be wrong - CHECK IT.")

        live_stops = {}
        for tr in self.ib.openTrades():
            if tr.orderStatus.status in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
                continue
            if tr.order.action == 'SELL' and tr.order.orderType in ('STP', 'STP LMT'):
                live_stops.setdefault(tr.contract.symbol.upper(), []).append(tr)

        for a in filled_records:
            sym = a['symbol'].upper()
            if live_stops.get(sym):
                self.logger.info(f"[{sym}] protection confirmed: "
                                 f"{len(live_stops[sym])} resting stop order(s).")
                continue
            held = abs(int(self.current_positions.get(a['symbol'], 0)))
            if held <= 0:
                self.logger.warning(
                    f"[{sym}] filled but IBKR reports no position yet - stop check "
                    f"deferred to the next poll.")
                a['state'] = 'WORKING'   # re-check rather than assume
                continue
            self.logger.error(
                f"[{sym}] NAKED POSITION: {held} shares filled and NO resting stop. The "
                f"bracket child did not survive. Placing a standalone hard stop now.")
            try:
                stop_px = round(a['trigger'] * (1 - HARD_STOP_PCT / 100), 2)
                repair = ibi.Order(
                    orderId=self.ib.client.getReqId(), action='SELL',
                    totalQuantity=held, orderType='STP', auxPrice=stop_px,
                    tif='GTC', outsideRth=True, transmit=True)
                self.ib.placeOrder(a['contract'], repair)
                self.logger.warning(
                    f"[{sym}] stop repaired: SELL {held} STP ${stop_px:.2f} "
                    f"(-{HARD_STOP_PCT}% off the ${a['trigger']:.2f} trigger). "
                    f"NOTE: standalone, no OCA group - verify it in TWS.")
            except Exception as e:
                self.logger.error(
                    f"[{sym}] STOP REPAIR FAILED: {e}. THIS POSITION IS UNPROTECTED. "
                    f"Place a stop by hand immediately.")

    def _log_watch_table(self, quotes):
        """Heartbeat: what is being watched and how far it is from firing."""
        committed, filled, working = self._committed()
        rows = []
        for a in sorted(self._armed, key=lambda x: x['rank']):
            t = quotes.get(a['symbol'])
            ask = t.ask if (t and t.ask and t.ask > 0) else None
            ref = ask or (t.last if (t and t.last and t.last > 0) else None)
            dist = ((ref / a['trigger'] - 1) * 100) if ref else None
            rows.append(f"  {a['rank']:>2} {a['symbol']:<7} {a['state']:<8} "
                        f"trig ${a['trigger']:>9.2f}  now "
                        + (f"${ref:>9.2f}" if ref else f"{'n/a':>10}")
                        + (f"  {dist:+6.2f}% away" if dist is not None else "")
                        + (f"  [{a['note']}]" if a['note'] else ""))
        self.logger.info(
            f"TRIGGER watch: {committed}/{self.position_sizer.max_positions} committed "
            f"({self._held_at_start} pre-held, {filled} filled today, {working} working)\n"
            + "\n".join(rows))

    async def watch_virtual_triggers(self):
        """Hold the session and monitor every potential position until the cutoff.

        Touch test is "can I buy at or below my trigger right now", i.e. ask <= trigger,
        with a traded print at or below the trigger accepted as well. That is deliberately
        stricter than the backtest, which fills whenever the day's LOW reaches the level:
        a low can be printed on a trade that was never available to you.

        GAP-THROUGH. A name already below its trigger on the first poll is reported and
        taken. A resting bid would have been filled at the open, at a BETTER price than
        the trigger, so refusing it would deviate from the modelled arm - but it is also
        where adverse selection lives, so it is logged distinctly rather than silently.
        """
        if not self._armed:
            return

        cutoff_h, cutoff_m = [int(x) for x in self.trigger_cutoff.split(':')]
        et = ZoneInfo("America/New_York")
        max_positions = self.position_sizer.max_positions

        # Streaming, not snapshots: snapshot=False and regulatorySnapshot=False carry no
        # per-request charge. Line budget was measured at 100 for this account.
        watched = {}
        for a in self._armed:
            try:
                watched[a['symbol']] = self.ib.reqMktData(a['contract'], '', False, False)
            except Exception as e:
                self.logger.error(f"[{a['symbol']}] TRIGGER: market data failed: {e}")
                a['state'] = 'DEAD'
                a['note']  = 'no market data'
        await asyncio.sleep(self.trigger_md_settle)

        self.logger.info(
            f"TRIGGER watcher: {len(watched)} name(s) armed, streaming until "
            f"{self.trigger_cutoff} ET, poll {self.trigger_poll:g}s. "
            f"Book cap {max_positions} ({self._held_at_start} held). "
            f"Unfill timeout: "
            + (f"{self.trigger_unfill_min:g} min" if self.trigger_unfill_min > 0
               else "off (orders rest to the cutoff, the measured arm)"))

        first_pass = True
        last_status = datetime.now(et)
        while True:
            now = datetime.now(et)
            if (now.hour, now.minute) >= (cutoff_h, cutoff_m):
                self.logger.info(f"TRIGGER watcher: cutoff {self.trigger_cutoff} reached.")
                break

            # 1. Reconcile first, ALWAYS. Slot arithmetic downstream is only as good as
            #    this pass, and a fill that lands between polls must be seen before any
            #    new order is considered.
            newly = self._reconcile_working()
            if newly:
                await self._post_fill_checks(newly)
                self._write_monitor_state('filled')

            # 2. Unfill timeout (off by default). Cancels a working order that has sat
            #    unfilled too long and returns the name to ARMED so it can fire again on
            #    a later touch. Off = the order rests to the cutoff, which is the arm
            #    that was actually backtested.
            if self.trigger_unfill_min > 0:
                for a in [x for x in self._armed if x['state'] == 'WORKING']:
                    if not a['sent_at'] or a['filled_qty'] > 0:
                        continue
                    age_min = (now - a['sent_at']).total_seconds() / 60.0
                    if age_min < self.trigger_unfill_min:
                        continue
                    self.logger.warning(
                        f"[{a['symbol']}] TRIGGER unfilled after {age_min:.1f} min - "
                        f"cancelling and re-arming.")
                    try:
                        self.ib.cancelOrder(a['order'])
                    except Exception as e:
                        self.logger.error(f"[{a['symbol']}] cancel failed: {e}")
                        continue
                    a.update(state='ARMED', sent=False, order=None, order_id=None,
                             sent_at=None, note='re-armed after unfill timeout')
                    self._write_monitor_state('re-armed')

            committed, n_filled, n_working = self._committed()

            # 3. Stop conditions. "Book full" now means filled + working, and it only
            #    ENDS the session when there is nothing left that could still change:
            #    with orders still working we keep watching, because one of them may be
            #    cancelled or rejected and hand its slot back.
            if committed >= max_positions:
                if n_working == 0:
                    unsent = [a['symbol'] for a in self._armed if a['state'] == 'ARMED']
                    self.logger.warning(
                        f"TRIGGER watcher: book FULL at {committed}/{max_positions} on "
                        f"confirmed fills. Disarming {len(unsent)} unfired name(s): {unsent}")
                    for a in self._armed:
                        if a['state'] == 'ARMED':
                            a['state'] = 'DEAD'
                            a['note']  = 'book full'
                    break
            else:
                # 4. Fire. Only ARMED records, only while a slot is genuinely free.
                for a in self._armed:
                    if a['state'] != 'ARMED':
                        continue
                    if self._committed()[0] >= max_positions:
                        break
                    t = watched.get(a['symbol'])
                    if t is None:
                        continue
                    ask  = t.ask  if (t.ask  and t.ask  > 0) else None
                    last = t.last if (t.last and t.last > 0) else None
                    trig = a['trigger']
                    hit = (ask is not None and ask <= trig) or (last is not None and last <= trig)
                    if not hit:
                        continue
                    reason = 'gap-through at arm' if first_pass else 'touch'
                    if first_pass:
                        self.logger.warning(
                            f"[{a['symbol']}] already at/below trigger on the FIRST poll "
                            f"(ask={ask} last={last} vs trigger {trig:.2f}). A resting bid "
                            f"would have filled at the open. Taking it, flagged as gap-through.")
                    self._send_trigger_order(a, trig, reason=reason)
                    # Re-read the account so the next sizing sees the committed cash.
                    try:
                        await self.prepare_account_data()
                    except Exception as e:
                        self.logger.error(f"TRIGGER watcher: account refresh failed: {e}")

            # 5. Nothing left that can change state? Stop early rather than idling to
            #    15:45 holding market-data lines for names that can never fire.
            if not any(a['state'] in ('ARMED', 'WORKING') for a in self._armed):
                self.logger.info("TRIGGER watcher: every armed name is resolved.")
                break

            if (self.trigger_status_min > 0
                    and (now - last_status).total_seconds() >= self.trigger_status_min * 60):
                self._log_watch_table(watched)
                last_status = now

            first_pass = False
            await asyncio.sleep(self.trigger_poll)

        # Cancel anything still working at the cutoff: a DAY limit would expire anyway,
        # but an explicit cancel makes the close deterministic and logs it.
        for a in [x for x in self._armed if x['state'] == 'WORKING' and x['filled_qty'] <= 0]:
            self.logger.info(f"[{a['symbol']}] cancelling unfilled entry at cutoff.")
            try:
                self.ib.cancelOrder(a['order'])
                a['state'] = 'DEAD'
                a['note']  = 'unfilled at cutoff'
            except Exception as e:
                self.logger.error(f"[{a['symbol']}] cutoff cancel failed: {e}")

        # Final reconcile: a fill can land in the same second as the cutoff.
        try:
            newly = self._reconcile_working()
            if newly:
                await self._post_fill_checks(newly)
        except Exception as e:
            self.logger.error(f"TRIGGER watcher: final reconcile failed: {e}")

        for a in self._armed:
            try:
                self.ib.cancelMktData(a['contract'])
            except Exception:
                pass

        self._log_watch_table(watched)
        self._write_monitor_state('session-end')
        filled = [a['symbol'] for a in self._armed if a['state'] == 'FILLED']
        if _diag.enabled():
            _diag.dump_json(f"X_session_{getattr(self, '_trigger_target', 'unknown')}", {
                'target_date': str(getattr(self, '_trigger_target', '')), 'armed': len(self._armed),
                'filled': len(filled), 'filled_symbols': filled, 'fill_log': self._fill_log,
                'states': [{k: a.get(k) for k in ('symbol', 'rank', 'state', 'trigger', 'want_qty',
                                                  'filled_qty', 'fill_price', 'note')} for a in self._armed],
                'cutoff': self.trigger_cutoff, 'poll_s': self.trigger_poll,
                'unfill_min': self.trigger_unfill_min, 'held_at_start': self._held_at_start,
                'max_positions': max_positions, 'dry_run': self.dry_run})
        self.logger.info(
            f"TRIGGER watcher: done. FILLED {len(filled)}/{len(self._armed)} armed: {filled}")
        if self._fill_log:
            avg = sum(f['slip_bps'] for f in self._fill_log) / len(self._fill_log)
            self.logger.info(
                f"TRIGGER slippage vs armed trigger: {avg:+.1f} bps mean over "
                f"{len(self._fill_log)} fill(s). Positive = paid more than the modelled "
                f"entry; this is the virtual-vs-resting cost the backtest does not carry.")

    async def supervise_trigger_entries(self):
        """Resting mode only: cancel the surplus the moment the book fills.

        THIS HAS NO BACKTEST COUNTERPART. The backtest caps the book because surplus
        orders are margin-rejected one at a time as cash drains. Live, a market-wide dip
        can touch many triggers in the same second and IBKR will fill everything it can
        fund before any cancel lands, so without this the worst day for the strategy is
        also the day it takes its largest book. Virtual mode does not need it: it never
        has more than one order per slot on the book in the first place.
        """
        if not self._trigger_orders:
            return

        cutoff_h, cutoff_m = [int(x) for x in self.trigger_cutoff.split(':')]
        et = ZoneInfo("America/New_York")
        max_positions = self.position_sizer.max_positions
        held_at_start = len(self.current_positions)

        self.logger.info(
            f"TRIGGER supervisor: watching {len(self._trigger_orders)} resting entries "
            f"until {self.trigger_cutoff} ET. Book cap {max_positions} "
            f"({held_at_start} already held).")

        cancelled_for_cap = False
        while True:
            now = datetime.now(et)
            if (now.hour, now.minute) >= (cutoff_h, cutoff_m):
                self.logger.info(f"TRIGGER supervisor: cutoff {self.trigger_cutoff} reached.")
                break

            open_by_id = {t.order.orderId: t for t in self.ib.openTrades()}
            resting, filled = [], []
            for contract, order, symbol in self._trigger_orders:
                tr = open_by_id.get(order.orderId)
                if tr is None:
                    continue
                st = tr.orderStatus
                if st.status in ('Filled',) or (st.filled and st.filled > 0):
                    filled.append(symbol)
                if st.remaining and st.remaining > 0 and st.status not in ('Cancelled', 'Inactive'):
                    resting.append((contract, order, symbol))

            book = held_at_start + len(set(filled))
            if book >= max_positions and resting:
                self.logger.warning(
                    f"TRIGGER supervisor: book FULL at {book}/{max_positions} "
                    f"(filled {sorted(set(filled))}). Cancelling {len(resting)} "
                    f"remaining resting entries: {[s for _, _, s in resting]}")
                for _, order, symbol in resting:
                    try:
                        self.ib.cancelOrder(order)
                    except Exception as e:
                        self.logger.error(f"[{symbol}] TRIGGER cancel failed: {e}")
                cancelled_for_cap = True
                break
            if not resting:
                self.logger.info("TRIGGER supervisor: no resting entries left.")
                break

            await asyncio.sleep(15)

        if not cancelled_for_cap:
            open_by_id = {t.order.orderId: t for t in self.ib.openTrades()}
            leftover = [(o, s) for _, o, s in self._trigger_orders
                        if o.orderId in open_by_id
                        and (open_by_id[o.orderId].orderStatus.remaining or 0) > 0]
            if leftover:
                self.logger.info(
                    f"TRIGGER supervisor: cancelling {len(leftover)} unfilled entries "
                    f"at cutoff: {[s for _, s in leftover]}")
                for order, symbol in leftover:
                    try:
                        self.ib.cancelOrder(order)
                    except Exception as e:
                        self.logger.error(f"[{symbol}] TRIGGER cancel failed: {e}")

        await asyncio.sleep(2)
        self.logger.info("TRIGGER supervisor: done.")

    async def sweep_stranded_stops(self, dry_run=False):
        """The give-up leg, as coarse as a once-a-day process can make it.

        A STP LMT that triggered but was never lifted leaves you STILL LONG with no
        protection - the one failure mode that makes stop-limits dangerous. In the sim
        that case rode a name to -30.3% before a 3-bar give-up removed it entirely.

        Live, nothing watches overnight, so this runs at the next 10:00 and closes any
        position that is holding a TRIGGERED-but-unfilled stop-limit. Detection is
        deliberately conservative: a live 'STP LMT' order on a symbol we still hold,
        whose trigger is at or above the current price (so it must have fired) and which
        has filled nothing. Anything ambiguous is logged and left alone - a false
        positive here liquidates a healthy position.

        Only meaningful alongside --exp-stop-limit; harmless otherwise (plain STP orders
        never reach the stranded state because they fill as market orders on trigger).
        """
        stranded = []
        try:
            held = {p.contract.symbol.upper(): p.position
                    for p in self.ib.positions() if p.position and p.position > 0}
            for tr in self.ib.openTrades():
                sym = tr.contract.symbol.upper()
                o, st = tr.order, tr.orderStatus
                if o.orderType != 'STP LMT' or o.action != 'SELL':
                    continue
                if sym not in held:
                    continue
                if (st.filled or 0) > 0:            # partially/fully filled -> not stranded
                    continue
                tick = self.ib.reqMktData(tr.contract, '', True, False)
                await asyncio.sleep(1.2)
                last = tick.last or tick.close or tick.marketPrice()
                self.ib.cancelMktData(tr.contract)
                if not last or last != last or last <= 0:
                    self.logger.warning(f"[{sym}] STP LMT stranded-check: no price, skipping.")
                    continue
                if last > (o.auxPrice or 0):        # price is back above the trigger
                    continue                        # the stop has not fired; leave it be
                stranded.append((sym, tr, float(last), int(held[sym])))
        except Exception as e:
            self.logger.error(f"stranded-stop sweep failed to scan: {e}")
            return []

        for sym, tr, last, qty in stranded:
            self.logger.warning(
                f"[{sym}] STRANDED STOP-LIMIT: trigger ${tr.order.auxPrice:.2f} fired, "
                f"floor ${tr.order.lmtPrice:.2f} never lifted, last ${last:.2f}, "
                f"still long {qty} sh. {'WOULD GIVE UP -> MKT' if dry_run else 'GIVING UP -> MKT'}.")
            if dry_run:
                continue
            try:
                self.ib.cancelOrder(tr.order)
                await asyncio.sleep(1.5)
                self.ib.placeOrder(tr.contract,
                                   build_timeout_exit_order(qty, use_moc=False))
            except Exception as e:
                self.logger.error(f"[{sym}] give-up order FAILED: {e}. CHECK MANUALLY.")
        if not stranded:
            self.logger.info("Stranded stop-limit sweep: none found.")
        return [s for s, *_ in stranded]

    # ── Post-trade maintenance ─────────────────────────────────────────────────

    async def _post_trade_maintenance(self):
        """Stamp the max-hold clock for newly filled names and repair missing TP legs.

        Runs on EVERY path out of run() that got as far as connecting - including the
        SPY-abort path - because the ledger must stay in sync with reality even on days
        when no entry is placed.
        """
        try:
            await self.prepare_account_data()
            self._reconcile_ledger()          # stamps today's fills, drops closed names
            if not self.no_exits:
                await self.ensure_take_profits(dry_run=self.dry_run_exits)
                # Give-up leg for the experimental stop-limit. No-op on a plain STP book.
                if self.exp_stop_limit:
                    await self.sweep_stranded_stops(dry_run=self.dry_run_exits)
        except Exception as e:
            self.logger.error(f"Post-trade maintenance failed: {e}")
            self.logger.error(traceback.format_exc())

    # ── Main entry point ───────────────────────────────────────────────────────

    def _on_ib_error(self, reqId, errorCode, errorString, contract=None):
        """TWS error/status callback -> log. Codes 200/201/203 are order killers."""
        sym = f" [{getattr(contract, 'symbol', '')}]" if contract is not None else ""
        if errorCode in (200, 201, 202, 203, 321, 461):
            self.logger.error(f"IB ORDER EVENT {errorCode} (reqId {reqId}):{sym} {errorString}")
        else:
            self.logger.info(f"IB event {errorCode} (reqId {reqId}):{sym} {errorString}")

    async def escalate_unfilled_entries(self):
        """Raise the Adaptive ceiling to the prevailing ask for any entry still
        unfilled `exp_entry_escalate_min` minutes after transmission.

        Measured basis (account-fill replay, 2026-07-30, 202 timestamped real fills):
        entries completed by 10:05 matched the 1-min engine's modeled fill to the
        basis point (mean -2.8 bps); the 1-in-3 that kept resting cost +55 bps on
        average, and the error grows with lateness (r=0.47). The engine prices a
        3-bar walk to marketable; unbounded patience is the one live entry behavior
        it does not model. Chasing is deliberate and unconditional, matching the
        engine's chase rule: a non-fill dropped silently launders adverse selection
        into fake alpha, and the live system still wants the position.

        Only lmtPrice changes, on the same orderId: bracket children, OCA group and
        the algo stay untouched; the ceiling is never lowered.
        """
        if not self._staged_parents:
            return
        self.logger.info(f"Entry escalation armed: t+{self.exp_entry_escalate_min:g} min "
                         f"for {len(self._staged_parents)} entry order(s).")
        await asyncio.sleep(self.exp_entry_escalate_min * 60.0)

        open_by_id = {t.order.orderId: t for t in self.ib.openTrades()}
        live = [(c, o, s) for (c, o, s) in self._staged_parents if o.orderId in open_by_id]
        if not live:
            self.logger.info("Entry escalation: all entries filled or closed. Nothing to do.")
            return
        tickers = await self.ib.reqTickersAsync(*[c for c, _, _ in live])
        for (contract, order, symbol), ticker in zip(live, tickers):
            status = open_by_id[order.orderId].orderStatus
            if not status.remaining or status.remaining <= 0:
                continue
            bid = ticker.bid if (ticker.bid and ticker.bid > 0) else None
            ask = ticker.ask if (ticker.ask and ticker.ask > 0) else None
            ref = ask or ticker.last or ticker.close
            if not ref or ref <= 0:
                self.logger.warning(f"[{symbol}] Escalation: no usable quote, order left as-is.")
                continue
            spread  = (ask - bid) if (ask and bid) else None
            cushion = max(0.0005 * ref, spread if spread else 0.0)
            new_limit = round(ref + cushion, 2)
            if new_limit <= order.lmtPrice:
                self.logger.info(
                    f"[{symbol}] Escalation: ceiling ${order.lmtPrice:.2f} already at/above "
                    f"ask+cushion ${new_limit:.2f} - algo still working, not touched.")
                continue
            old = order.lmtPrice
            order.lmtPrice = new_limit
            self.ib.placeOrder(contract, order)
            self.logger.warning(
                f"[{symbol}] ESCALATED entry after {self.exp_entry_escalate_min:g} min: "
                f"ceiling ${old:.2f} -> ${new_limit:.2f} "
                f"({(new_limit / old - 1) * 1e4:+.1f} bps), {status.remaining:g} share(s) unfilled.")

    async def run(self):
        try:
            # Connect early so we're ready at entry time (avoids connection
            # latency eating into the entry window)
            await self.connect()

            # Step 1: Wait until 10:00 ET
            if not self.skip_wait:
                await self.wait_until_ready()
            else:
                self.logger.info("--skip-wait flag set: executing immediately.")

            # Refresh account state RIGHT before entry, not at script start
            await self.prepare_account_data()

            # Step 1.5: MAX-HOLD EXITS - deliberately BEFORE the SPY gate and before the
            # entry batch. Before the SPY gate because reducing risk must never be
            # conditional on market conditions: on a red day we still want the aged
            # position out. Before the entry batch so the freed slots recycle the same
            # morning, which is where the edge actually comes from.
            if not self.no_exits:
                sold = await self.close_aged_positions(dry_run=self.dry_run_exits)
                if sold:
                    self.logger.info(f"Max-hold sold {len(sold)} name(s): {sold}. "
                                     f"Refreshing account before sizing new entries.")
                    await self.prepare_account_data()
                # Step 1.55: MODEL EXITS (--model-exits, default OFF). Skips anything
                # the age exit already flattened. Placed before repeg/scale so a name
                # being sold today is not repegged or scaled first, and before the
                # entry batch so its slot recycles this morning.
                if self.model_exits:
                    msold = await self.close_model_exits(dry_run=self.dry_run_exits,
                                                         skip={x.upper() for x in sold})
                    if msold:
                        sold = list(sold) + msold
                        self.logger.info(f"Model exits sold {len(msold)} name(s): {msold}. "
                                         f"Refreshing account before sizing new entries.")
                        await self.prepare_account_data()
                else:
                    self.logger.info("Model exits DISABLED (pass --model-exits to enable). "
                                     "This is the shipped behaviour.")

                # Step 1.6 (runner mode only, no-ops otherwise): ratchet every resting
                # stop up to prior close - TRAIL_EOD_PCT, then bank winners at
                # +RUNNER_TRIG_PCT and leave the runners on the longer clock. Ordered
                # after the age exits so a position being flattened today is not
                # repegged or scaled first.
                if RUNNER_MODE:
                    self.logger.info(f"Runner mode: {BRACKET.describe()}")
                    aged_syms = {s.upper() for s in sold}
                    await self.repeg_stops(dry_run=self.dry_run_exits, skip=aged_syms)
                    scaled = await self.scale_out_winners(dry_run=self.dry_run_exits,
                                                          skip=aged_syms)
                    if scaled:
                        self.logger.info(f"Scaled out {len(scaled)} name(s): {scaled}. "
                                         f"Refreshing account before sizing new entries.")
                        await self.prepare_account_data()
            else:
                self.logger.warning("--no-exits: max-hold exit DISABLED for this run.")

            # Step 2: SPY gate - abort if market too red
            market_conds = await self.check_market_conditions()
            if not market_conds['market_ok']:
                self.logger.warning("Market conditions gate FAILED - no entries today.")
                await self._post_trade_maintenance()
                return

            # Steps 3 + 4: execute with gap filter and correct order flags
            start_time = datetime.now()
            if self.trigger_entry:
                await self.stage_trigger_entries(market_conds)
            else:
                await self.execute_batch(market_conds)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Batch Execution Time: {duration:.4f}s")

            # Keep alive briefly to ensure transmission confirmation
            await asyncio.sleep(10)

            if self.trigger_entry:
                # Holds the session either way. Virtual watches quotes and sends one
                # order per touch; resting cancels the surplus once the book fills.
                if self.trigger_arm == 'virtual':
                    await self.watch_virtual_triggers()
                else:
                    await self.supervise_trigger_entries()
            elif self.exp_entry_escalate_min > 0:
                # Step 4.5: bounded patience on the entry orders (latency fix, see
                # escalate_unfilled_entries docstring). No-op unless the flag is set.
                await self.escalate_unfilled_entries()

            # Step 5: stamp entry dates for anything newly filled (starts its max-hold
            # clock) and re-attach any take-profit leg that failed to land.
            await self._post_trade_maintenance()

        except Exception as e:
            self.logger.error(f"Runtime Error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='IBKR Smart Entry Executor - waits until 10:00 ET, '
                    'checks SPY direction, filters per-stock gaps.'
    )
    parser.add_argument(
        '--port', type=int, default=7496,
        help='IBKR TWS/Gateway port (7496=live, 7497=paper)'
    )
    parser.add_argument(
        '--entry-time', type=str, default='10:00',
        help='Entry time ET in HH:MM format (default: 10:00)'
    )
    parser.add_argument(
        '--spy-abort', type=float, default=SPY_ABORT_THRESHOLD,
        help=f'Abort all trades if SPY move %% <= this value (default: {SPY_ABORT_THRESHOLD})'
    )
    parser.add_argument(
        '--skip-wait', action='store_true',
        help='Skip the time wait entirely (useful for manual runs / testing)'
    )
    parser.add_argument(
        '--no-exits', action='store_true',
        help='KILL SWITCH: disable the max-hold exit and take-profit repair entirely.'
    )
    parser.add_argument(
        '--dry-run-exits', action='store_true',
        help='Report which positions WOULD be age-exited / TP-repaired, place no orders. '
             'Entries still run normally.'
    )
    parser.add_argument(
        '--exp-stop-limit', action='store_true',
        help='EXPERIMENTAL (default OFF): send the hard stop as STP LMT with a floor '
             f'{STOP_LIMIT_HALF_SPREADS:g} half-spreads under the trigger instead of STP. '
             'Also enables the next-morning stranded-stop give-up sweep. '
             'Evidence: analysis_output/EXEC_ORDER_TYPE_REPORT.md'
    )
    parser.add_argument(
        '--exp-moc-exit', action='store_true',
        help='EXPERIMENTAL (default OFF): route max-hold/timeout exits to the closing '
             'auction (MOC) instead of MKT/Adaptive. Changes WHEN you exit, not just how.'
    )
    parser.add_argument(
        '--exp-entry-escalate-min', type=float, default=0.0, metavar='MIN',
        help='EXPERIMENTAL (default 0 = OFF): minutes to let the Adaptive entry work '
             'before raising its ceiling to the prevailing ask. Account-fill replay '
             '2026-07-30: fills done by 10:05 matched the 1-min sim to the basis point; '
             'the 1-in-3 that rested longer cost +55 bps average. 3 matches the sim\'s '
             'modeled 3-bar walk. Evidence: analysis_output/account_fill_replay/'
    )
    parser.add_argument(
        '--trigger-entry', dest='trigger_entry', action='store_true',
        help='DEFAULT ON since the 2026-08-28 changeover; accepted so existing scripts '
             'keep working. Replaces the marketable Adaptive entry with BUY LIMITs at a '
             'per-name dip depth, armed across --trigger-reach candidates from the WIDE '
             'pool so that ~3 fill. Evidence: MORNING_2026-08-26.md section 3-4. The '
             'return edge is +0.037pp at t 0.53 (5 of 10 seeds) and is NOT a reason to '
             'run this; the drawdown difference (33.25%% -> 17.8-23.9%%, zero seed '
             'overlap, on ~3/4 of the capital) is.'
    )
    parser.add_argument(
        '--no-trigger-entry', dest='trigger_entry', action='store_false',
        help='ROLLBACK: restore the marketable-Adaptive entry that shipped before '
             '2026-08-28. With this flag the entry path is byte-identical to the old '
             'one, reads the NARROWED _Buy_Signals.parquet, and no trigger code runs.'
    )
    parser.set_defaults(trigger_entry=True)
    parser.add_argument(
        '--model-exits', action='store_true',
        help='Enable the UpProbability decay exits (momentum + prob_drop) that the '
             'backtest runs and this broker never has. Full slot sim, 4 paired seeds: '
             'annualised volatility +6.91 (t +3.63, 0/4 seeds) and win rate +9.7pp '
             '(t -7.58, 0/4) in its favour, but RETURN IS A NULL (t -0.35) and without '
             'it mean PnL/trade is +0.99pp HIGHER on 5.4pp more capital deployed. This '
             'is a volatility and win-rate lever, not a return lever. Default OFF.'
    )
    parser.add_argument(
        '--trigger-k', type=float, default=TRIGGER_K, metavar='K',
        help=f'Trigger depth multiplier: depth = K * beta * vol_20d, clamped to '
             f'[0.5%%, 15%%] (default: {TRIGGER_K:g}, the measured arm).'
    )
    parser.add_argument(
        '--trigger-reach', type=int, default=TRIGGER_REACH, metavar='N',
        help=f'How many pool names to rest orders on (default: {TRIGGER_REACH}). Reach '
             f'is inside the noise band (24 vs 15 = 0.23 sd, 24 vs 30 = 0.49), so this '
             f'is a convenience knob, not an optimum. A pool shorter than N logs a '
             f'REACH SHORTFALL rather than silently becoming a smaller arm.'
    )
    parser.add_argument(
        '--trigger-max-spread-bps', type=float, default=TRIGGER_MAX_SPREAD_BPS,
        metavar='BPS',
        help='Refuse to rest on a name whose quoted spread exceeds this, in bps '
             '(default 0 = OFF, matching the measured arm). The hazard is real but '
             'unmeasured: 2026-08-25 armed DRUG at a 214.3 bps median spread against '
             'an edge of roughly 43 bps.'
    )
    parser.add_argument(
        '--trigger-arm', type=str, default=TRIGGER_ARM, choices=['virtual', 'resting'],
        help="How the trigger is armed. 'virtual' (default) sends NOTHING at 10:00: it "
             "watches streaming quotes and sends one real LMT-at-trigger the moment a "
             "name reaches its level, so at most free_slots orders exist at once. "
             "'resting' rests a live limit on every eligible name for the whole session "
             "- that is what the backtest measured, because a resting bid fills AT the "
             "trigger and earns the half-spread. Virtual pays poll latency and the "
             "spread instead, neither of which is in the +0.7730%%/trade figure."
    )
    parser.add_argument(
        '--trigger-poll', type=float, default=TRIGGER_POLL, metavar='SEC',
        help=f'Virtual mode only: seconds between quote polls (default {TRIGGER_POLL:g}). '
             f'This is the window in which a dip can touch and rebound unseen.'
    )
    parser.add_argument(
        '--trigger-cutoff', type=str, default='15:45', metavar='HH:MM',
        help='ET time to cancel any still-unfilled resting entry (default: 15:45). The '
             'backtest cancels an unfilled trigger at the next bar; a bid left resting '
             'overnight is a different strategy than the one that was measured.'
    )
    parser.add_argument(
        '--trigger-unfill-min', type=float, default=TRIGGER_UNFILL_MIN, metavar='MIN',
        help=f'Minutes a sent-but-unfilled entry may rest before it is cancelled and its '
             f'name re-armed for a later touch (default: {TRIGGER_UNFILL_MIN:g} = never, '
             f'which IS the measured arm - the backtested trigger rests all session). '
             f'Anything above 0 is a deliberate deviation.'
    )
    parser.add_argument(
        '--trigger-status-min', type=float, default=TRIGGER_STATUS_MIN, metavar='MIN',
        help=f'Minutes between watch-table heartbeats showing every potential position '
             f'and its distance from firing (default: {TRIGGER_STATUS_MIN:g}; 0 = log '
             f'only on state changes).'
    )
    parser.add_argument(
        '--trigger-rubric-max-cut', type=int, default=TRIGGER_RUBRIC_MAX_CUT, metavar='N',
        help=f'Most names the morning rubric may veto from the pool (default: '
             f'{TRIGGER_RUBRIC_MAX_CUT}). The rubric SHAVES a wide book now; a veto '
             f'deeper than this is treated as a malfunction and refused whole, because '
             f'a silently narrowed book looks exactly like a normal day in the log.'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='FULL dry run: place NO orders of any kind, entries included. Loads the '
             'pool, applies both filters, prices every trigger, arms the monitor and '
             'reports what WOULD fire. This is the flag to use for a structural check - '
             '--no-exits gates only exits and --dry-run-exits says outright that '
             'entries still run normally, so neither is safe against a fresh pool.'
    )
    args = parser.parse_args()

    entry_h, entry_m = [int(x) for x in args.entry_time.split(':')]

    if args.trigger_entry and args.exp_entry_escalate_min > 0:
        parser.error(
            'trigger entry and --exp-entry-escalate-min are mutually exclusive. '
            'Escalation raises a resting entry to the prevailing ask, which converts '
            'the trigger back into the marketable entry it was meant to replace. '
            'Trigger entry is ON BY DEFAULT since 2026-08-28, so if you want escalation '
            'you are asking for the old entry path: pass --no-trigger-entry as well.')

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    executor = AsyncFastExecutor(
        port=args.port,
        entry_hour=entry_h,
        entry_minute=entry_m,
        spy_abort=args.spy_abort,
        skip_wait=args.skip_wait,
        no_exits=args.no_exits,
        model_exits=args.model_exits,
        dry_run_exits=args.dry_run_exits,
        exp_stop_limit=args.exp_stop_limit,
        exp_moc_exit=args.exp_moc_exit,
        exp_entry_escalate_min=args.exp_entry_escalate_min,
        trigger_entry=args.trigger_entry,
        trigger_k=args.trigger_k,
        trigger_reach=args.trigger_reach,
        trigger_max_spread_bps=args.trigger_max_spread_bps,
        trigger_cutoff=args.trigger_cutoff,
        trigger_arm=args.trigger_arm,
        trigger_poll=args.trigger_poll,
        trigger_unfill_min=args.trigger_unfill_min,
        trigger_status_min=args.trigger_status_min,
        trigger_rubric_max_cut=args.trigger_rubric_max_cut,
        dry_run=args.dry_run,
    )
    try:
        asyncio.run(executor.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass



