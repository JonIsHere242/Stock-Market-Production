#!/usr/bin/env python
import argparse
import asyncio
import sys
import traceback
from datetime import datetime

from zoneinfo import ZoneInfo

import ib_insync as ibi
import numpy as np
import pandas as pd

from Util import read_signals, get_logger, PositionSizer

# ── Entry filter thresholds — derived from EDA_IntradayEntry.ipynb ────────────
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
STOCK_GAP_SKIP      =  1.5    # Skip stock if it gapped up > 1.5% from open
STOCK_DIP_GOOD_LO   = -1.5    # Favorable dip-entry band lower bound
STOCK_DIP_GOOD_HI   = -0.5    # Favorable dip-entry band upper bound

ET = ZoneInfo('America/New_York')


class AsyncFastExecutor:

    # Live port = 7496  |  Paper port = 7497
    def __init__(self, host='127.0.0.1', port=7496, client_id=99,
                 entry_hour=ENTRY_HOUR, entry_minute=ENTRY_MINUTE,
                 spy_abort=SPY_ABORT_THRESHOLD, skip_wait=False):
        self.ib           = ibi.IB()
        self.host         = host
        self.port         = port
        self.client_id    = client_id
        self.entry_hour   = entry_hour
        self.entry_minute = entry_minute
        self.spy_abort    = spy_abort
        self.skip_wait    = skip_wait
        self.logger       = get_logger("FastExecutor")

        self.position_sizer = PositionSizer(
            cash_buffer_pct=10.0,
            max_positions=5
        )

        self.account_value    = 0.0
        self.available_cash   = 0.0
        self.current_positions = {}

    # ── Connection ─────────────────────────────────────────────────────────────

    async def connect(self):
        try:
            self.logger.info("Connecting to IB Gateway/TWS...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.logger.info("Connected.")
        except Exception as e:
            self.logger.critical(f"Could not connect: {e}")
            sys.exit(1)

    # ── Step 1: Wait until entry time ─────────────────────────────────────────

    async def wait_until_ready(self):
        """
        Sleep until entry_hour:entry_minute ET, then return.
        If already past that time, returns immediately.

        Why 10:00?  EDA shows opening-minute noise causes the worst fill prices.
        Every extra minute you wait improves Sharpe — 9:30→10:00 is +256%.
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
                f"({now_et.strftime('%H:%M:%S')}) — proceeding immediately."
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
                    f"(threshold {self.spy_abort:+.1f}%). "
                    f"EDA: Sharpe={-6.50:.2f} on days this weak — no orders placed."
                )
            elif spy_move < SPY_WARN_THRESHOLD:
                self.logger.warning(
                    f"SPY CAUTION: SPY {spy_move:+.2f}% — market slightly red. "
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
            self.logger.warning(f"SPY check failed ({e}) — proceeding without filter.")

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

    # ── Trail calculation ──────────────────────────────────────────────────────

    def _calculate_dynamic_trail(self, price, atr_fallback=None):
        atr = atr_fallback if atr_fallback and atr_fallback > 0 else (price * 0.02)
        atr_percent      = (atr / price) * 100
        # Optimized from parameter sweep (was: 1.5 + 0.75 * max(0, atr_pct - 2.0), cap 5.0)
        trailing_percent = 0.5 + 0.75 * max(0, atr_percent - 2.0)
        return min(trailing_percent, 5.0)

    # ── Step 3 + 4: Execute with entry filters ─────────────────────────────────

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
        """
        signals_df = read_signals(status_filter="Pending")
        if signals_df.empty:
            self.logger.info("No pending signals found in parquet.")
            return

        symbols = signals_df['Symbol'].unique().tolist()
        self.logger.info(f"Processing Batch: {symbols}")

        contracts = [ibi.Stock(s, 'SMART', 'USD') for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)

        self.logger.info("Requesting Market Snapshots...")
        tickers = await self.ib.reqTickersAsync(*contracts)

        orders_staged = []
        next_valid_id = self.ib.client.getReqId()
        self.logger.info(f"Starting Order ID Sequence from: {next_valid_id}")

        spy_move = market_conds.get('spy_move_pct')
        spy_tag  = f"  [SPY {spy_move:+.2f}%]" if spy_move is not None else ""

        for contract, ticker in zip(contracts, tickers):
            symbol = contract.symbol

            if symbol in self.current_positions:
                self.logger.warning(f"[{symbol}] Skipping: Already hold position.")
                continue

            market_price = (ticker.ask  if (ticker.ask  and ticker.ask  > 0)
                            else ticker.last)
            if not market_price or market_price <= 0:
                self.logger.error(f"[{symbol}] Bad Data: No Ask/Last price available.")
                continue

            # ── Filter 2: Per-stock open-gap check ────────────────────────────
            try:
                today_open = float(ticker.open) if ticker.open else None
            except (TypeError, ValueError):
                today_open = None

            if today_open and today_open > 0:
                open_gap_pct = (market_price / today_open - 1) * 100

                if open_gap_pct > STOCK_GAP_SKIP:
                    self.logger.warning(
                        f"[{symbol}] SKIP: Gapped up {open_gap_pct:+.1f}% from open "
                        f"(EDA: stocks up >{STOCK_GAP_SKIP:.0f}% at open show "
                        f"mean rest-of-day -0.24% fade — not a good entry)."
                    )
                    continue
                elif STOCK_DIP_GOOD_LO <= open_gap_pct <= STOCK_DIP_GOOD_HI:
                    self.logger.info(
                        f"[{symbol}] FAVORABLE DIP: {open_gap_pct:+.1f}% from open "
                        f"(EDA: dip entries in this range show mean rest-of-day +0.24%)."
                    )
                else:
                    self.logger.info(f"[{symbol}] Open gap: {open_gap_pct:+.1f}% — within normal range.")
            else:
                self.logger.warning(f"[{symbol}] No open price available for gap check — proceeding.")

            # ── Position sizing ────────────────────────────────────────────────
            row            = signals_df[signals_df['Symbol'] == symbol].iloc[0]
            parquet_stop   = row['StopPrice']   if 'StopPrice'   in row and pd.notnull(row['StopPrice'])   else None
            parquet_target = row['TargetPrice'] if 'TargetPrice' in row and pd.notnull(row['TargetPrice']) else None
            parquet_atr    = row['ATR']         if 'ATR'         in row and pd.notnull(row['ATR'])         else None

            current_slots_used = len(self.current_positions) + len(orders_staged)
            size = self.position_sizer.calculate_position_size(
                account_value=self.account_value,
                current_cash=self.available_cash,
                price=market_price,
                current_positions=current_slots_used,
                symbol=symbol
            )
            if size <= 0:
                continue

            # ── Price levels ───────────────────────────────────────────────────
            limit_price = round(market_price * 1.001, 2)

            if parquet_stop and 0 < parquet_stop < limit_price:
                stop_price = parquet_stop
            else:
                stop_price = limit_price * 0.98      # 2% stop loss

            if parquet_target and parquet_target > limit_price:
                take_profit = parquet_target
            else:
                risk        = limit_price - stop_price
                take_profit = limit_price + (risk * 2.0)

            trail_pct = self._calculate_dynamic_trail(limit_price, parquet_atr)

            # ── Build bracket ──────────────────────────────────────────────────
            parent_id = next_valid_id
            tp_id     = next_valid_id + 1
            stop_id   = next_valid_id + 2
            next_valid_id += 3

            # Filter 3: DAY + outsideRth=False — prevents pre-market fills.
            # The whole point of waiting until 10:00 is to see market conditions
            # first; a GTC+outsideRth order would fill before any check runs.
            parent = ibi.Order(
                orderId=parent_id,
                action='BUY',
                totalQuantity=size,
                orderType='LMT',
                lmtPrice=limit_price,
                tif='DAY',           # was GTC — signals are for today, not overnight
                outsideRth=False,    # was True  — no pre-market fills
                transmit=False
            )

            # TP and trail stay GTC + outsideRth to catch after-hours gaps on exit
            take_profit_ord = ibi.Order(
                orderId=tp_id,
                action='SELL',
                totalQuantity=size,
                orderType='LMT',
                lmtPrice=take_profit,
                tif='GTC',
                outsideRth=True,
                parentId=parent_id,
                transmit=False
            )

            trail_stop_ord = ibi.Order(
                orderId=stop_id,
                action='SELL',
                totalQuantity=size,
                orderType='TRAIL',
                trailingPercent=trail_pct,
                tif='GTC',
                outsideRth=True,
                parentId=parent_id,
                transmit=True        # Last order transmits the entire bracket
            )

            self.logger.info(
                f"[{symbol}] Staging: Buy {size} @ ${limit_price:.2f} | "
                f"Stop: ${stop_price:.2f} | TP: ${take_profit:.2f} | "
                f"Trail: {trail_pct:.2f}%{spy_tag}"
            )
            orders_staged.append((contract, [parent, take_profit_ord, trail_stop_ord]))

        if orders_staged:
            self.logger.info(f"Transmitting {len(orders_staged)} brackets to exchange...")
            for contract, orders in orders_staged:
                for o in orders:
                    self.ib.placeOrder(contract, o)
            self.logger.info("All orders transmitted.")
        else:
            self.logger.info("No valid orders after entry filters.")

    # ── Main entry point ───────────────────────────────────────────────────────

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

            # Step 2: SPY gate — abort if market too red
            market_conds = await self.check_market_conditions()
            if not market_conds['market_ok']:
                self.logger.warning("Market conditions gate FAILED — no orders placed today.")
                return

            # Steps 3 + 4: execute with gap filter and correct order flags
            start_time = datetime.now()
            await self.execute_batch(market_conds)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Batch Execution Time: {duration:.4f}s")

            # Keep alive briefly to ensure transmission confirmation
            await asyncio.sleep(2)

        except Exception as e:
            self.logger.error(f"Runtime Error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='IBKR Smart Entry Executor — waits until 10:00 ET, '
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
    args = parser.parse_args()

    entry_h, entry_m = [int(x) for x in args.entry_time.split(':')]

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    executor = AsyncFastExecutor(
        port=args.port,
        entry_hour=entry_h,
        entry_minute=entry_m,
        spy_abort=args.spy_abort,
        skip_wait=args.skip_wait,
    )
    try:
        asyncio.run(executor.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass


