#!/usr/bin/env python
import asyncio
import sys
import traceback
from datetime import datetime
import ib_insync as ibi
import pandas as pd
import numpy as np

from Util import read_signals, get_logger, PositionSizer

class AsyncFastExecutor:

    # Live port = 7496
    # Paper port = 7497
    def __init__(self, host='127.0.0.1', port=7496, client_id=99):
        self.ib = ibi.IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.logger = get_logger("FastExecutor")
        
        # Initialize Position Sizer
        self.position_sizer = PositionSizer(
            cash_buffer_pct=10.0,
            max_positions=5
        )
        
        # State Cache
        self.account_value = 0.0
        self.available_cash = 0.0
        self.current_positions = {}

    async def connect(self):
        """Async connection to IB"""
        try:
            self.logger.info("Connecting to IB Gateway/TWS...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.logger.info("Connected.")
            
        except Exception as e:
            self.logger.critical(f"Could not connect: {e}")
            sys.exit(1)

    async def prepare_account_data(self):
        """
        Fetches account summary ONE time to be used for all calculations.
        """
        self.logger.info("Snapshotting Account Data...")
        
        # 1. Get Positions
        positions = self.ib.positions()
        self.current_positions = {p.contract.symbol: p.position for p in positions if p.position != 0}
        
        # 2. Get Cash/NAV
        tags = self.ib.accountValues()
        for tag in tags:
            if tag.tag == 'NetLiquidation':
                self.account_value = float(tag.value)
            elif tag.tag == 'AvailableFunds':
                self.available_cash = float(tag.value)
                
        self.logger.info(f"Account Ready. NAV: ${self.account_value:,.0f} | Cash: ${self.available_cash:,.0f} | Open Pos: {len(self.current_positions)}")

    def _calculate_dynamic_trail(self, price, atr_fallback=None):

        atr = atr_fallback if atr_fallback and atr_fallback > 0 else (price * 0.02)
        atr_percent = (atr / price) * 100
        # Optimized from parameter sweep (was: 1.5 + 0.75 * max(0, atr_pct - 2.0), cap 5.0)
        trailing_percent = 0.5 + 0.75 * max(0, atr_percent - 2.0)
        return min(trailing_percent, 5.0)



    
    async def execute_batch(self):

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
    
        # Reserve 3 IDs per bracket: parent + 2 children
        next_valid_id = self.ib.client.getReqId() 
        self.logger.info(f"Starting Order ID Sequence from: {next_valid_id}")
    
        for contract, ticker in zip(contracts, tickers):
            symbol = contract.symbol
            
            if symbol in self.current_positions:
                self.logger.warning(f"[{symbol}] Skipping: Already hold position.")
                continue
            
            market_price = ticker.ask if (ticker.ask and ticker.ask > 0) else ticker.last
            if not market_price or market_price <= 0:
                self.logger.error(f"[{symbol}] Bad Data: No Ask/Last price available.")
                continue
            
            row = signals_df[signals_df['Symbol'] == symbol].iloc[0]
            parquet_stop = row['StopPrice'] if 'StopPrice' in row else None
            parquet_target = row['TargetPrice'] if 'TargetPrice' in row else None
            parquet_atr = row['ATR'] if 'ATR' in row and pd.notnull(row['ATR']) else None
    
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
            
            limit_price = round(market_price * 1.001, 2)
            
            if parquet_stop and 0 < parquet_stop < limit_price:
                stop_price = parquet_stop
            else:
                stop_price = limit_price * 0.98  # 2% stop loss (was 5%)
            
            if parquet_target and parquet_target > limit_price:
                take_profit = parquet_target
            else:
                risk = limit_price - stop_price
                take_profit = limit_price + (risk * 2.0)
                    
            trail_pct = self._calculate_dynamic_trail(limit_price, parquet_atr)
    
            # Assign IDs explicitly for all 3 orders
            parent_id = next_valid_id
            tp_id = next_valid_id + 1
            stop_id = next_valid_id + 2
            next_valid_id += 3  # Reserve 3 IDs per bracket
            
            parent = ibi.Order(
                orderId=parent_id,
                action='BUY',
                totalQuantity=size,
                orderType='LMT',
                lmtPrice=limit_price,
                tif='GTC',
                outsideRth=True,
                transmit=False 
            )
            
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
                transmit=True  # Last order transmits the entire bracket
            )
            
            bracket_orders = [parent, take_profit_ord, trail_stop_ord]
            
            self.logger.info(f"[{symbol}] Staging: Buy {size} @ ${limit_price} | Trail: {trail_pct:.2f}% | TP: ${take_profit}")
            orders_staged.append((contract, bracket_orders))
    
        if orders_staged:
            self.logger.info(f"Transmitting {len(orders_staged)} brackets to exchange...")
            for contract, orders in orders_staged:
                for o in orders:
                    self.ib.placeOrder(contract, o)
                    
            self.logger.info("All orders transmitted.")
        else:
            self.logger.info("No valid orders generated from batch.")

    async def run(self):
        """Main Entry Point"""
        try:
            await self.connect()
            await self.prepare_account_data()

            start_time = datetime.now()
            await self.execute_batch()
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Batch Execution Time: {duration:.4f} seconds")
            
            # Keep alive briefly to ensure transmission confirmation
            await asyncio.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Runtime Error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    executor = AsyncFastExecutor()
    try:
        asyncio.run(executor.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass