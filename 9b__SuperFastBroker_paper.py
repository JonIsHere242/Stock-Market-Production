#!/usr/bin/env python
"""
9b__SuperFastBroker_paper.py — PAPER-ONLY test broker for the reworked exit stack.

Derived from 9_SuperFastBroker.py after the 2026-07-20 exit investigation. This file
exists to validate NEW ORDER MECHANICS on the paper account (port 7497) before any
live change. It refuses to connect to anything but paper: the port is pinned to 7497
and, after connecting, it verifies the managed account id starts with 'D' (IBKR paper
accounts are DU/DF...; live accounts start with 'U'). Either check failing aborts.

What is different from the live broker (9_SuperFastBroker.py):

  1. HARD STOP WIDENED 1.9% -> 5.0%. The only exit result that survived the
     out-of-sample split: stop-width arms rank identically in both time halves
     (rank corr +1.000, monotonic in width), and the 1.9% stop sits ON the median
     winner's adverse excursion (-1.82%..-1.87%), killing ~half of eventual winners.
  2. MAX-HOLD EXIT ADDED — a time-conditioned MKT SELL leg (IBKR order condition,
     held server-side) that fires at 15:50 ET after position_timeout=5 trading days,
     matching the backtest's timeout exit (Util.py STRATEGY_PARAMS). Live had NO
     max-hold at all (INSP sat 13 trading days), which breaks slot recycling.
  3. TP LEG VERIFICATION — live showed 8 stops / 0 take-profits despite the code
     building the TP correctly; suspected orderId collision. Here only the parent
     gets a pre-assigned id; every child is submitted with orderId=0 so ib_insync
     assigns a fresh id AT PLACEMENT. After transmitting, verify_open_brackets()
     re-reads open orders and screams if any symbol is missing its TP / stop /
     max-hold leg. That check is the whole point of the paper test.
  4. OPTIONAL BREAKEVEN ARM (--breakeven-arm-pct, default OFF) — IBKR native
     adjustable order: when price touches entry*(1+arm%), the server rewrites the
     5% stop to a breakeven STP. Server-side, survives TWS being closed. OFF by
     default because trailing-style policies were REFUTED for returns (H1-only);
     included so the adjustable-order mechanics can be tested.
  5. NO TRAILING STOP. trail=OFF was already the live state; the trailing policy
     failed the time-split audit, so the trail-calculation path is gone entirely.
  6. --test-symbols mode: place tiny throwaway brackets (default 1 share) on named
     tickers, bypassing the signals file / freshness / SPY / gap gates, so order
     mechanics can be exercised on paper any market day without a fresh book.
  7. Slippage log goes to Data/paper_slippage_log.csv (never contaminates the live
     calibration CSV).

Entry logic (10:00 wait, SPY gate, gap filter, Adaptive LMT parent, narrowed-book
and staleness fail-safes) is unchanged from the live broker.

Typical test runs:
    python 9b__SuperFastBroker_paper.py --test-symbols AAPL,MSFT --skip-wait
    python 9b__SuperFastBroker_paper.py --skip-wait --ignore-stale   (real book, paper fills)
"""
import argparse
import asyncio
import csv
import os
import sys
import traceback
from datetime import datetime

from zoneinfo import ZoneInfo

import ib_insync as ibi
import numpy as np
import pandas as pd

from Util import get_logger, PositionSizer

# ── Entry filter thresholds — same EDA-derived values as the live broker ──────
ENTRY_HOUR          = 10
ENTRY_MINUTE        = 0
SPY_ABORT_THRESHOLD = -0.5
SPY_WARN_THRESHOLD  =  0.0
SPY_STRONG_THRESHOLD=  0.5
STOCK_GAP_SKIP      =  1.5
STOCK_DIP_GOOD_LO   = -1.5
STOCK_DIP_GOOD_HI   = -0.5

# ── NEW exit-policy parameters (2026-07-20 investigation) ─────────────────────
HARD_STOP_PCT       = 5.0     # was 1.9 live. 5% ranked above 1.9% in BOTH time halves
                              # (paired +0.144pp/trade, t=+1.85); 1.9% sits on the median
                              # winner's dip and cuts 47-50% of eventual winners.
MAX_HOLD_TDAYS      = 5       # matches backtest position_timeout=5 (Util.STRATEGY_PARAMS).
MAX_HOLD_EXIT_TIME  = "15:50:00"   # ET; exits before the close auction, regular hours.
BREAKEVEN_ARM_PCT   = 0.0     # 0 = OFF. >0 arms an IBKR adjustable order: stop rewrites
                              # to breakeven once price touches entry*(1+arm/100).
TP_FALLBACK_RR      = 2.0     # TP fallback when the parquet has no TargetPrice:
                              # entry + RR * stop-distance.

PAPER_PORT = 7497             # THE ONLY PORT THIS FILE WILL EVER USE.

SLIPPAGE_LOG = os.path.join(os.path.dirname(__file__), "Data", "paper_slippage_log.csv")

BUY_SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "_Buy_Signals.parquet")
MAX_BOOK = 12
STALE_WARN_TDAYS  = 1
STALE_ABORT_TDAYS = 1

ET = ZoneInfo('America/New_York')


class AsyncPaperExecutor:
    """Paper-account executor for the v2 exit stack. Never touches live."""

    def __init__(self, client_id=98,   # live broker uses 99 — never collide
                 entry_hour=ENTRY_HOUR, entry_minute=ENTRY_MINUTE,
                 spy_abort=SPY_ABORT_THRESHOLD, skip_wait=False,
                 stop_pct=HARD_STOP_PCT, max_hold_tdays=MAX_HOLD_TDAYS,
                 breakeven_arm_pct=BREAKEVEN_ARM_PCT, tp_fallback_rr=TP_FALLBACK_RR,
                 test_symbols=None, test_size=1, ignore_stale=False):
        self.ib           = ibi.IB()
        self.host         = '127.0.0.1'
        self.port         = PAPER_PORT
        self.client_id    = client_id
        self.entry_hour   = entry_hour
        self.entry_minute = entry_minute
        self.spy_abort    = spy_abort
        self.skip_wait    = skip_wait
        self.logger       = get_logger("PaperExecutor")

        self.stop_pct          = stop_pct
        self.max_hold_tdays    = max_hold_tdays
        self.breakeven_arm_pct = breakeven_arm_pct
        self.tp_fallback_rr    = tp_fallback_rr
        self.test_symbols      = test_symbols or []
        self.test_size         = test_size
        self.ignore_stale      = ignore_stale

        self.position_sizer = PositionSizer(
            cash_buffer_pct=10.0,
            max_positions=10
        )

        self.account_value     = 0.0
        self.available_cash    = 0.0
        self.current_positions = {}

        self._signal_mids: dict[str, float] = {}
        # symbol -> {'oca': str, 'parent_id': int} for post-placement verification
        self._staged_brackets: dict[str, dict] = {}

    # ── Connection — paper-account hard lock ──────────────────────────────────

    async def connect(self):
        try:
            self.logger.info(f"Connecting to IBKR PAPER on port {self.port}...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.ib.execDetailsEvent += self._on_fill
        except Exception as e:
            self.logger.critical(f"Could not connect: {e}")
            sys.exit(1)

        # Second lock: even on 7497, refuse anything that isn't a paper account.
        # IBKR paper account ids start with 'D' (DU/DF...); live ids start with 'U'.
        accounts = self.ib.managedAccounts()
        bad = [a for a in accounts if not a.upper().startswith('D')]
        if not accounts or bad:
            self.logger.critical(
                f"ACCOUNT LOCK FAILED: managed accounts {accounts} do not all look like "
                f"paper accounts (must start with 'D'). Refusing to run — this file is "
                f"paper-only by design. Disconnecting."
            )
            self.ib.disconnect()
            sys.exit(1)
        self.logger.info(f"Connected to PAPER account(s): {accounts}")

    # ── Fill handler — logs fill vs mid-at-signal (paper CSV, separate file) ──

    def _on_fill(self, _trade, fill):
        symbol   = fill.contract.symbol
        fill_px  = fill.execution.price
        side     = fill.execution.side      # 'BOT' or 'SLD'
        shares   = fill.execution.shares
        mid_ref  = self._signal_mids.get(symbol)
        slippage = (fill_px - mid_ref) if (mid_ref and side == 'BOT') else None
        slip_bps = (slippage / mid_ref * 10_000) if slippage is not None else None
        now_str  = datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')

        msg = f"[{symbol}] FILL: {side} {shares} @ ${fill_px:.2f}"
        if mid_ref:
            msg += f" | mid-ref: ${mid_ref:.2f}"
        if slip_bps is not None:
            msg += f" | slippage: {slip_bps:+.1f} bps"
        self.logger.info(msg)

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
        now_et    = datetime.now(ET)
        target_et = now_et.replace(
            hour=self.entry_hour, minute=self.entry_minute,
            second=0, microsecond=0
        )
        wait_secs = (target_et - now_et).total_seconds()

        if wait_secs > 0:
            self.logger.info(
                f"Waiting {wait_secs:.0f}s until {target_et.strftime('%H:%M')} ET"
            )
            await asyncio.sleep(wait_secs)
        else:
            self.logger.info(
                f"Already past {target_et.strftime('%H:%M')} ET — proceeding immediately."
            )

    # ── Step 2: SPY market-condition gate (unchanged from live) ───────────────

    async def check_market_conditions(self) -> dict:
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
                self.logger.warning("SPY market data unavailable — proceeding without SPY filter.")
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
                    f"(threshold {self.spy_abort:+.1f}%). No orders placed."
                )
            elif spy_move < SPY_WARN_THRESHOLD:
                self.logger.warning(
                    f"SPY CAUTION: SPY {spy_move:+.2f}% — market slightly red. Proceeding."
                )
            elif result['spy_strong']:
                self.logger.info(f"SPY STRONG: {spy_move:+.2f}% from open.")
            else:
                self.logger.info(f"SPY OK: {spy_move:+.2f}% from open.")

        except Exception as e:
            self.logger.warning(f"SPY check failed ({e}) — proceeding without filter.")

        return result

    # ── Account snapshot ──────────────────────────────────────────────────────

    async def prepare_account_data(self):
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

    # ── Book freshness (unchanged; --ignore-stale bypass for paper testing) ───

    def _check_book_freshness(self, signals_df) -> bool:
        date_col = next(
            (c for c in ('TargetDate', 'SignalDate', 'CreatedDate', 'LastUpdated')
             if c in signals_df.columns),
            None,
        )
        dates = (pd.to_datetime(signals_df[date_col], errors='coerce').dropna()
                 if date_col else pd.Series([], dtype='datetime64[ns]'))
        if dates.empty:
            self.logger.warning(
                "STALE CHECK SKIPPED: no usable date column in the book — proceeding."
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

        if self.ignore_stale:
            self.logger.warning(
                f"STALE BOOK ({stale_tdays} trading days old) — proceeding anyway because "
                f"--ignore-stale is set and this is the PAPER account. Symbols: {symbols}."
            )
            return True

        if stale_tdays < STALE_ABORT_TDAYS:
            self.logger.warning(
                f"STALE BOOK ({stale_tdays} trading day old): dated {book_date:%Y-%m-%d} "
                f"vs today {today_et:%Y-%m-%d}. Proceeding: {symbols}."
            )
            return True

        self.logger.error(
            f"STALE SIGNAL BOOK — REFUSING TO TRADE. {os.path.basename(BUY_SIGNALS_FILE)} "
            f"is {stale_tdays} trading days old (TargetDate {book_date:%Y-%m-%d}). "
            f"NO ORDERS PLACED. Use --ignore-stale to override on paper."
        )
        return False

    # ── v2 bracket builder ────────────────────────────────────────────────────

    def _max_hold_condition_time(self) -> str:
        """Datetime string for the max-hold TimeCondition: 15:50 ET on the
        max_hold_tdays-th trading day after today (weekends skipped; matches how
        the backtest counts days_held in trading days)."""
        today_et  = datetime.now(ET).date()
        exit_date = np.busday_offset(today_et, self.max_hold_tdays, roll='forward')
        exit_date = pd.Timestamp(exit_date).date()
        return f"{exit_date:%Y%m%d} {MAX_HOLD_EXIT_TIME} US/Eastern"

    def _build_bracket(self, symbol, mid, size, limit_price,
                       parquet_target=None, parquet_stop=None):
        """Build the v2 order stack: Adaptive LMT parent + OCA(TP LMT, wide STP,
        time-conditioned MKT max-hold). Returns the ordered list to place.

        OrderId handling — THE FIX UNDER TEST: only the parent gets a pre-assigned
        id (children must reference it via parentId before placement). Every child
        is created with orderId=0 so ib_insync assigns a fresh id at placeOrder()
        time. The live broker pre-assigned all four ids up front; the suspected
        failure mode is the hard stop landing on the TP's id slot and replacing it
        (live: 8 stops, 0 TPs, every stop id = parent_id+1).
        """
        parent_id = self.ib.client.getReqId()
        oca_group = f"pbracket_{symbol}_{parent_id}"

        # Wide hard stop — flat stop_pct below entry ref. The parquet StopPrice
        # (ATR-based, ~2%) is deliberately NOT used: the whole point of v2 is the
        # wider stop. Logged for comparison only.
        hard_stop_price = round(mid * (1 - self.stop_pct / 100), 2)
        if parquet_stop and parquet_stop > 0:
            self.logger.info(
                f"[{symbol}] parquet StopPrice ${parquet_stop:.2f} ignored — "
                f"v2 uses flat {self.stop_pct}% stop (${hard_stop_price:.2f})."
            )

        # Take-profit: parquet target if sane, else RR-multiple of stop distance.
        if parquet_target and parquet_target > limit_price:
            take_profit = round(float(parquet_target), 2)
        else:
            risk        = limit_price - hard_stop_price
            take_profit = round(limit_price + risk * self.tp_fallback_rr, 2)

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

        take_profit_ord = ibi.Order(
            orderId=0,                    # assigned fresh at placeOrder()
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

        hard_stop_ord = ibi.Order(
            orderId=0,
            action='SELL',
            totalQuantity=size,
            orderType='STP',
            auxPrice=hard_stop_price,
            tif='GTC',
            outsideRth=True,
            parentId=parent_id,
            ocaGroup=oca_group,
            ocaType=1,
            transmit=False
        )

        # Optional IBKR adjustable order: once price touches entry*(1+arm%), the
        # SERVER rewrites this stop to a breakeven STP. Survives TWS being closed.
        arm_note = "off"
        if self.breakeven_arm_pct > 0:
            trigger = round(mid * (1 + self.breakeven_arm_pct / 100), 2)
            hard_stop_ord.triggerPrice      = trigger
            hard_stop_ord.adjustedOrderType = 'STP'
            hard_stop_ord.adjustedStopPrice = round(mid, 2)
            arm_note = f"breakeven @ +{self.breakeven_arm_pct}% (trigger ${trigger:.2f})"

        # Max-hold exit — the leg live never had. A MKT SELL held behind an IBKR
        # TimeCondition: submits at 15:50 ET on trading day +max_hold_tdays. Same
        # OCA group, so whichever of TP / stop / max-hold goes first cancels the
        # rest. Server-side; matches the backtest position_timeout exit.
        cond_time = self._max_hold_condition_time()
        max_hold_ord = ibi.Order(
            orderId=0,
            action='SELL',
            totalQuantity=size,
            orderType='MKT',
            tif='GTC',
            parentId=parent_id,
            ocaGroup=oca_group,
            ocaType=1,
            conditions=[ibi.TimeCondition(conjunction='a', isMore=True, time=cond_time)],
            conditionsIgnoreRth=False,
            conditionsCancelOrder=False,   # condition SUBMITS the order, not cancels
            transmit=True                  # last order transmits the whole bracket
        )

        self.logger.info(
            f"[{symbol}] Staging v2 bracket: BUY {size} @ lmt=${limit_price:.2f} "
            f"(mid=${mid:.2f}) | Stop: ${hard_stop_price:.2f} (-{self.stop_pct}%) | "
            f"TP: ${take_profit:.2f} | MaxHold: MKT @ {cond_time} | "
            f"BreakevenArm: {arm_note} | OCA: {oca_group}"
        )

        self._staged_brackets[symbol] = {'oca': oca_group, 'parent_id': parent_id}
        return [parent, take_profit_ord, hard_stop_ord, max_hold_ord]

    # ── Post-placement verification — the actual paper test ───────────────────

    async def verify_open_brackets(self):
        """Re-read open orders and confirm every staged symbol has all four legs.

        This is the direct probe for the live failure (8 stops / 0 TPs): if the
        orderId-collision theory is right, the OLD construction would show a
        missing TP here; the new construction must show LMT SELL + STP SELL +
        conditional MKT SELL alive for every symbol.
        """
        if not self._staged_brackets:
            return

        await asyncio.sleep(5)   # let TWS register/acknowledge the brackets
        trades = self.ib.openTrades()

        all_ok = True
        for symbol, meta in self._staged_brackets.items():
            legs = [t for t in trades
                    if t.contract.symbol == symbol
                    and (t.order.ocaGroup == meta['oca']
                         or t.order.orderId == meta['parent_id'])]

            parent = [t for t in legs if t.order.action == 'BUY']
            tp     = [t for t in legs if t.order.action == 'SELL' and t.order.orderType == 'LMT']
            stp    = [t for t in legs if t.order.action == 'SELL' and t.order.orderType == 'STP']
            mkt    = [t for t in legs if t.order.action == 'SELL' and t.order.orderType == 'MKT']

            def _fmt(ts):
                return ", ".join(
                    f"#{t.order.orderId} {t.order.orderType} {t.orderStatus.status}"
                    for t in ts) or "MISSING"

            self.logger.info(
                f"[{symbol}] VERIFY: parent[{_fmt(parent)}] tp[{_fmt(tp)}] "
                f"stop[{_fmt(stp)}] maxhold[{_fmt(mkt)}]"
            )

            missing = [name for name, ts in
                       (('PARENT', parent), ('TAKE-PROFIT', tp),
                        ('HARD-STOP', stp), ('MAX-HOLD', mkt)) if not ts]
            # A filled parent legitimately leaves openTrades; ignore parent-missing
            # if the other three legs are alive.
            if missing == ['PARENT'] and tp and stp and mkt:
                missing = []
            if missing:
                all_ok = False
                self.logger.error(
                    f"[{symbol}] BRACKET LEG(S) MISSING: {missing} — this is the "
                    f"live-broker failure mode reproducing. Do NOT promote this "
                    f"construction. OCA={meta['oca']} parent_id={meta['parent_id']}"
                )

        if all_ok:
            self.logger.info(
                "VERIFY PASSED: every staged symbol has TP + hard-stop + max-hold "
                "legs registered. Order construction is safe to consider for live."
            )
        else:
            self.logger.error("VERIFY FAILED: at least one bracket is incomplete (see above).")

    # ── Shared per-symbol staging (quotes → filters → sizing → bracket) ───────

    async def _stage_symbol(self, contract, ticker, size_override=None,
                            parquet_row=None, apply_gap_filter=True,
                            slots_used=0):
        symbol = contract.symbol

        if symbol in self.current_positions:
            self.logger.warning(f"[{symbol}] Skipping: Already hold position.")
            return None

        bid    = ticker.bid  if (ticker.bid  and ticker.bid  > 0) else None
        ask    = ticker.ask  if (ticker.ask  and ticker.ask  > 0) else None
        spread = (ask - bid) if (bid and ask) else None
        mid    = (bid + ask) / 2 if (bid and ask) else (ticker.last or ask or bid)

        if not mid or mid <= 0:
            self.logger.error(f"[{symbol}] Bad Data: No price available.")
            return None

        spread_bps = (spread / mid * 10_000) if spread else None
        self.logger.info(
            f"[{symbol}] Quotes: bid=${bid} ask=${ask} mid=${mid:.2f}"
            + (f" spread={spread_bps:.1f} bps" if spread_bps else "")
        )

        if apply_gap_filter:
            try:
                today_open = float(ticker.open) if ticker.open else None
            except (TypeError, ValueError):
                today_open = None

            if today_open and today_open > 0:
                open_gap_pct = (mid / today_open - 1) * 100
                if open_gap_pct > STOCK_GAP_SKIP:
                    self.logger.warning(
                        f"[{symbol}] SKIP: Gapped up {open_gap_pct:+.1f}% from open."
                    )
                    return None
                elif STOCK_DIP_GOOD_LO <= open_gap_pct <= STOCK_DIP_GOOD_HI:
                    self.logger.info(f"[{symbol}] FAVORABLE DIP: {open_gap_pct:+.1f}% from open.")
                else:
                    self.logger.info(f"[{symbol}] Open gap: {open_gap_pct:+.1f}% — normal range.")
            else:
                self.logger.warning(f"[{symbol}] No open price for gap check — proceeding.")

        parquet_stop = parquet_target = None
        if parquet_row is not None:
            parquet_stop   = parquet_row['StopPrice']   if 'StopPrice'   in parquet_row and pd.notnull(parquet_row['StopPrice'])   else None
            parquet_target = parquet_row['TargetPrice'] if 'TargetPrice' in parquet_row and pd.notnull(parquet_row['TargetPrice']) else None

        if size_override is not None:
            size = size_override
        else:
            size = self.position_sizer.calculate_position_size(
                account_value=self.account_value,
                current_cash=self.available_cash,
                price=mid,
                current_positions=slots_used,
                symbol=symbol
            )
        if size <= 0:
            return None

        cushion     = max(0.0005 * mid, spread if spread else 0.0)
        limit_price = round(mid + cushion, 2)

        self._signal_mids[symbol] = mid
        orders = self._build_bracket(symbol, mid, size, limit_price,
                                     parquet_target=parquet_target,
                                     parquet_stop=parquet_stop)
        return (contract, orders)

    # ── Step 3 + 4: Execute (signals-file path) ───────────────────────────────

    async def execute_batch(self, market_conds: dict):
        if not os.path.exists(BUY_SIGNALS_FILE):
            self.logger.info(f"Signals file {BUY_SIGNALS_FILE} not found.")
            return
        signals_df = pd.read_parquet(BUY_SIGNALS_FILE)

        if 'Status' not in signals_df.columns:
            self.logger.error(
                f"ABORT: {BUY_SIGNALS_FILE} has no 'Status' column — looks like the raw "
                f"ledger, not a narrowed book. No orders placed."
            )
            return
        signals_df = signals_df[signals_df['Status'] == 'Pending']
        if len(signals_df) > MAX_BOOK:
            self.logger.error(
                f"ABORT: {len(signals_df)} pending signals exceeds MAX_BOOK={MAX_BOOK} — "
                f"book not narrowed. No orders placed."
            )
            return
        if signals_df.empty:
            self.logger.info("No pending signals found in _Buy_Signals.parquet.")
            return

        if not self._check_book_freshness(signals_df):
            return

        symbols = signals_df['Symbol'].unique().tolist()
        self.logger.info(f"Processing Batch (PAPER): {symbols}")

        contracts = [ibi.Stock(s, 'SMART', 'USD') for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)

        self.logger.info("Requesting Market Snapshots...")
        tickers = await self.ib.reqTickersAsync(*contracts)

        orders_staged = []
        for contract, ticker in zip(contracts, tickers):
            row = signals_df[signals_df['Symbol'] == contract.symbol].iloc[0]
            staged = await self._stage_symbol(
                contract, ticker,
                parquet_row=row,
                apply_gap_filter=True,
                slots_used=len(self.current_positions) + len(orders_staged),
            )
            if staged:
                orders_staged.append(staged)

        await self._transmit(orders_staged)

    # ── Test-symbols path: tiny brackets, no book/gates — mechanics only ──────

    async def execute_test_batch(self):
        symbols = [s.strip().upper() for s in self.test_symbols if s.strip()]
        self.logger.warning(
            f"TEST MODE: placing {self.test_size}-share v2 brackets on {symbols}. "
            f"Signals file, freshness, SPY and gap gates ALL BYPASSED (paper only)."
        )

        contracts = [ibi.Stock(s, 'SMART', 'USD') for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)
        tickers = await self.ib.reqTickersAsync(*contracts)

        orders_staged = []
        for contract, ticker in zip(contracts, tickers):
            staged = await self._stage_symbol(
                contract, ticker,
                size_override=self.test_size,
                apply_gap_filter=False,
            )
            if staged:
                orders_staged.append(staged)

        await self._transmit(orders_staged)

    async def _transmit(self, orders_staged):
        if not orders_staged:
            self.logger.info("No valid orders after entry filters.")
            return
        self.logger.info(f"Transmitting {len(orders_staged)} v2 brackets to PAPER...")
        for contract, orders in orders_staged:
            for o in orders:
                trade = self.ib.placeOrder(contract, o)
                self.logger.info(
                    f"[{contract.symbol}] placed #{trade.order.orderId} "
                    f"{trade.order.action} {trade.order.orderType}"
                    + (f" parentId={trade.order.parentId}" if trade.order.parentId else "")
                )
        self.logger.info("All orders transmitted.")
        await self.verify_open_brackets()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self):
        try:
            await self.connect()

            if self.test_symbols:
                # Mechanics test: no waiting, no SPY gate — just place + verify.
                await self.prepare_account_data()
                await self.execute_test_batch()
                await asyncio.sleep(10)
                return

            if not self.skip_wait:
                await self.wait_until_ready()
            else:
                self.logger.info("--skip-wait flag set: executing immediately.")

            await self.prepare_account_data()

            market_conds = await self.check_market_conditions()
            if not market_conds['market_ok']:
                self.logger.warning("Market conditions gate FAILED — no orders placed today.")
                return

            start_time = datetime.now()
            await self.execute_batch(market_conds)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Batch Execution Time: {duration:.4f}s")

            await asyncio.sleep(10)

        except Exception as e:
            self.logger.error(f"Runtime Error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PAPER-ONLY v2 exit-stack broker: wide stop, verified TP leg, '
                    'time-conditioned max-hold exit. Port pinned to 7497.'
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
        '--stop-pct', type=float, default=HARD_STOP_PCT,
        help=f'Hard stop %% below entry (default: {HARD_STOP_PCT}; live legacy was 1.9)'
    )
    parser.add_argument(
        '--max-hold-days', type=int, default=MAX_HOLD_TDAYS,
        help=f'Trading days before the conditioned MKT max-hold exit fires '
             f'(default: {MAX_HOLD_TDAYS}, matches backtest position_timeout)'
    )
    parser.add_argument(
        '--breakeven-arm-pct', type=float, default=BREAKEVEN_ARM_PCT,
        help='If > 0, IBKR adjustable order moves the stop to breakeven once price '
             'is up this %% (default: 0 = off; trailing-style policies are refuted, '
             'this exists to test adjustable-order mechanics)'
    )
    parser.add_argument(
        '--tp-fallback-rr', type=float, default=TP_FALLBACK_RR,
        help=f'TP = entry + RR * stop-distance when the book has no TargetPrice '
             f'(default: {TP_FALLBACK_RR})'
    )
    parser.add_argument(
        '--test-symbols', type=str, default='',
        help='Comma-separated tickers: place tiny test brackets on these, bypassing '
             'the signals file and all entry gates (paper mechanics test)'
    )
    parser.add_argument(
        '--test-size', type=int, default=1,
        help='Share count per test-symbol bracket (default: 1)'
    )
    parser.add_argument(
        '--ignore-stale', action='store_true',
        help='Trade a stale book anyway (PAPER ONLY convenience for testing)'
    )
    args = parser.parse_args()

    entry_h, entry_m = [int(x) for x in args.entry_time.split(':')]

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    executor = AsyncPaperExecutor(
        entry_hour=entry_h,
        entry_minute=entry_m,
        spy_abort=args.spy_abort,
        skip_wait=args.skip_wait,
        stop_pct=args.stop_pct,
        max_hold_tdays=args.max_hold_days,
        breakeven_arm_pct=args.breakeven_arm_pct,
        tp_fallback_rr=args.tp_fallback_rr,
        test_symbols=[s for s in args.test_symbols.split(',') if s.strip()],
        test_size=args.test_size,
        ignore_stale=args.ignore_stale,
    )
    try:
        asyncio.run(executor.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
