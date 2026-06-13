#!/usr/bin/env python
import random
import sys
import time
import uuid
import os
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import backtrader_contrib as bt
import ib_insync as ibi
import pandas as pd
import numpy as np
import exchange_calendars as ec
import pprint

from Util import *

# Debug mode toggle
DEBUG_MODE = True
nyse = ec.get_calendar('XNYS')
DEFAULT_HOST = '127.0.0.1'
#7496 live 7497 paper
DEFAULT_PAPER_PORT = 7496
DEFAULT_LIVE_PORT = 7496
MAX_RECONNECT_ATTEMPTS = 3
CONNECTION_TIMEOUT = 20.0
SIGNALS_FILE = 'Data/Production/LiveTradingData/pending_signals.parquet'
POSITIONS_FILE = 'Data/Production/LiveTradingData/active_positions.parquet'
COMPLETED_TRADES_FILE = 'Data/Production/LiveTradingData/completed_trades.parquet'



def should_use_production_files():
    """Check if we should use the new production files"""
    all_exist = (
        os.path.exists(SIGNALS_FILE) and
        os.path.exists(POSITIONS_FILE)
    )
    
    if all_exist:
        # Check if signals file contains data for today's date
        try:
            signals_df = pd.read_parquet(SIGNALS_FILE)
            if not signals_df.empty:
                tomorrow = get_next_trading_day(datetime.now().date())
                for_tomorrow = signals_df['TargetDate'].dt.date == tomorrow
                return for_tomorrow.any()
        except Exception as e:
            logger.error(f"Error reading signals file: {str(e)}")
    
    return False


def get_open_positions(ib):
    """Get current open positions from IB."""
    try:
        positions = ib.positions()
        positions_list = []
        for position in positions:
            if position.position != 0:
                contract = position.contract
                positions_list.append(contract.symbol)
                logger.info(f'Position found: {contract.symbol}, {position.position}')
        
        if not positions_list:
            logger.info('No open positions found')
        
        return positions_list
    except Exception as e:
        logger.error(f'Error fetching positions: {e}')
        return []











class StockSniperLive(bt.Strategy):
    params = STRATEGY_PARAMS

    def __init__(self):
        logger.info("Initializing StockSniperLive strategy - IB BRACKET ONLY")
        
        self.bar_counter = 0
        self.start_time = datetime.now()
        self._ib = self._get_ib_connection()
        
        self.position_sizer = PositionSizer(
            cash_buffer_pct=10.0,
            max_positions=self.p.max_positions
        )
        
        self.data_ready = {d: False for d in self.datas}
        self.debug_mode = True
        self.bar_counter = 0
        # Only track direct IB bracket orders
        self.order_attempted = set()
        self.ib_bracket_orders = {}
        
        self._check_trading_mode()
        logger.info("Strategy initialized with IB bracket orders only")

    def _cleanup_order_tracking(self, order_id, symbol, allow_retry=False):
        """Clean up order tracking dictionaries"""
        try:
            if order_id in self.ib_bracket_orders:
                del self.ib_bracket_orders[order_id]
                logger.info(f"[CLEANUP] {symbol} - Removed bracket order {order_id} from tracking")
                
            if allow_retry and symbol in self.order_attempted:
                self.order_attempted.remove(symbol)
                logger.info(f"[CLEANUP] {symbol} - Allowing retry by removing from attempted set")
                
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] {symbol} - Error cleaning up order tracking: {e}")

    def _check_trading_mode(self):
        """Check if we're in paper trading mode"""
        try:
            if self._ib:
                if hasattr(self._ib, 'client') and hasattr(self._ib.client, 'port'):
                    port = self._ib.client.port
                    if port == 7496:
                        logger.info("[TRADING MODE] Connected to paper trading (port 7497)")
                        self.paper_trading = True
                    else:
                        logger.info(f"[TRADING MODE] Connected to live trading (port {port})")
                        self.paper_trading = False
                else:
                    logger.warning("[TRADING MODE] Cannot determine trading mode")
                    self.paper_trading = True
        except Exception as e:
            logger.error(f"[TRADING MODE] Error checking trading mode: {e}")
            self.paper_trading = True

    def _get_ib_connection(self):
        if hasattr(self.broker, 'ib'):
            return self.broker.ib
            
        for var_name in ['ib', 'store']:
            if var_name in globals():
                obj = globals()[var_name]
                if hasattr(obj, 'ib'):
                    return obj.ib
                elif hasattr(obj, 'isConnected'):
                    return obj
                    
        logger.warning("No IB connection found - position sync will be limited")
        return None

    def _is_raw_feed(self, data):
        return hasattr(data, '_name') and '_5sec' in data._name

    def _get_tradeable_feed(self, symbol):
        """Get the original data feed (with tradecontract) for a given symbol."""
        target_name = f"{symbol}_5sec"
        for d in self.datas:
            if hasattr(d, '_name') and d._name == target_name:
                if hasattr(d, 'tradecontract'):
                    logger.info(f"[TRADEABLE FEED] Found: {d._name}")
                    return d
        logger.error(f"[TRADEABLE FEED] No tradeable feed found for {symbol}")
        return None

    def prenext(self):
        self.bar_counter += 1
        logger.info(f"=== PRENEXT BAR #{self.bar_counter} ===")
        
        for data in self.datas:
            if not self._is_raw_feed(data):
                bar_count = len(data)
                if bar_count > 0 and not self.data_ready.get(data, False):
                    logger.info(f"[READY] {data._name} - {bar_count} bars ready")
                    self.data_ready[data] = True



    def next(self):
        self.bar_counter += 1
        current_date = self.datetime.date(0)
        current_time = self.datetime.datetime(0)
        logger.info(f"=== NEXT BAR #{self.bar_counter} - {current_date} {current_time.time()} ===")

        # Mark any new data as ready
        for data in self.datas:
            if not self._is_raw_feed(data):
                bar_count = len(data)
                if bar_count > 0 and not self.data_ready.get(data, False):
                    logger.info(f"[READY] {data._name} - {bar_count} bars ready")
                    self.data_ready[data] = True

        # SAFETY CHECK: Only trade after market open with confirmed data
        market_open_time = current_time.replace(hour=9, minute=30, second=0)
        if current_time < market_open_time:
            logger.info(f"[PRE-MARKET] Waiting for market open at 9:30 AM")
            return

        # Process each symbol
        for data in self.datas:
            if not self.data_ready.get(data, False) or self._is_raw_feed(data):
                continue
            
            symbol = data._name
            logger.info(f"[PROCESSING] {symbol}")

            # Get the tradeable feed
            original_data = self._get_tradeable_feed(symbol)
            if not original_data:
                logger.error(f"[ERROR] {symbol} - No tradeable feed found")
                continue
            
            # SAFETY: Ensure we have enough bars for trend analysis
            if len(original_data) < 5:
                logger.info(f"[INSUFFICIENT DATA] {symbol} - Only {len(original_data)} bars, waiting for more")
                continue
            
            # Check current position
            try:
                current_position = self.getposition(original_data)
                current_size = current_position.size
                logger.info(f"[POSITION] {symbol} - Current size: {current_size}")
            except Exception as e:
                logger.error(f"[POSITION ERROR] {symbol} - {e}")
                continue
            
            # If we don't own the stock, check if we should buy
            if current_size == 0:
                if symbol in self.order_attempted:
                    logger.info(f"[ALREADY ATTEMPTED] {symbol} - Already tried to place order this session")
                    continue

                if self._has_buy_signal(symbol):
                    # CRITICAL: Wait for uptrend confirmation
                    #if not self._confirm_uptrend(original_data, symbol):
                    #    logger.info(f"[WAITING] {symbol} - No uptrend confirmed yet")
                    #    continue
                    
                    position_size = self._calculate_position_size(original_data)
                    if position_size > 0:
                        logger.info(f"[BUY SIGNAL] {symbol} - Position size calculated: {position_size}")
                        current_price = original_data.close[0]
                        logger.info(f"[ENTRY PRICE] {symbol} - ${current_price}")

                        if self._ib:
                            logger.info(f"[BRACKET IB] {symbol} - Placing IB bracket order")
                            success = self._place_bracket_via_ib_direct(symbol, original_data, position_size, current_price)

                            if success:
                                logger.info(f"[BRACKET SUCCESS] {symbol} - IB bracket order placed successfully")
                                self.order_attempted.add(symbol)
                            else:
                                logger.error(f"[BRACKET FAILED] {symbol} - IB bracket order failed")
                        else:
                            logger.error(f"[NO IB CONNECTION] {symbol} - Cannot place orders without IB connection")
                    else:
                        logger.warning(f"[INSUFFICIENT CAPITAL] {symbol} - Cannot calculate position size")
                else:
                    logger.info(f"[NO SIGNAL] {symbol} - No buy signal")
            else:
                logger.info(f"[ALREADY OWN] {symbol} - Already own {current_size} shares")

        logger.info(f"[CYCLE COMPLETE] Bar #{self.bar_counter} processing completed")
        self._check_ib_bracket_status()

        if self.bar_counter >= 10:
            logger.info(f"[STOPPING STRATEGY] Bar counter reached {self.bar_counter} - stopping strategy")
            self.stop()









    def _has_buy_signal(self, symbol):
        """Check if we have a buy signal for this symbol"""
        try:
            signals_df = read_signals(status_filter="Pending")
            if signals_df.empty:
                return False
            today = datetime.now().date()
            tomorrow = get_next_trading_day(today)
            symbol_signals = signals_df[
                (signals_df['Symbol'] == symbol) & 
                (signals_df['TargetDate'].dt.date.isin([today, tomorrow]))
            ]
            return not symbol_signals.empty
        except Exception as e:
            logger.error(f"Error checking buy signal for {symbol}: {e}")
            return False

    def _calculate_position_size(self, data):
        """Calculate position size using the PositionSizer"""
        try:
            return self.position_sizer.calculate_position_size(
                account_value=self.broker.getvalue(),
                current_cash=self.broker.getcash(),
                price=data.close[0],
                current_positions=0,
                symbol=data._name.replace('_5sec', '')
            )
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0








    def _place_bracket_via_ib_direct(self, symbol, data, position_size, current_price):
        """Place bracket order using real-time market ask price"""
        try:
            contract = ibi.Stock(symbol, 'SMART', 'USD')
            # Get real-time market data from IB
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract, '', False, False)
            self._ib.sleep(0.5)
            # Use real-time ask price if available, otherwise fall back to current_price
            if ticker.ask and ticker.ask > 0:
                market_ask = ticker.ask
                logger.info(f"[MARKET DATA] {symbol} - Real-time ask: ${market_ask}, Bar close was: ${current_price}")
            else:
                market_ask = current_price
                logger.warning(f"[MARKET DATA] {symbol} - No real-time ask available, using bar close: ${current_price}")
            # Cancel the market data subscription
            self._ib.cancelMktData(contract)
            # Use market ask + small offset for limit price
            limit_price = round(market_ask * 1.001, 2)
            stop_price = self._get_stop_price(symbol, market_ask)
            take_profit_price = self._get_take_profit_price(symbol, market_ask, stop_price)
            atr = self.inds[data]['atr'][0] if hasattr(self, 'inds') else market_ask * 0.02
            # Dynamic trailing percentage based on volatility
            atr_percent = (atr / market_ask) * 100
            trailing_percent = 1.5 + 0.75 * max(0, atr_percent - 2.0)
            trailing_percent = min(trailing_percent, 5.0)
            logger.info(f"[BRACKET IB] {symbol} - Creating GTC bracket: Buy @ ${limit_price} (ask + 0.1%), "
                       f"Trailing Stop @ {trailing_percent}%, Profit @ ${take_profit_price}")
            # Pre-fetch order IDs
            parent_id = self._ib.client.getReqId()
            stop_id = self._ib.client.getReqId()
            profit_id = self._ib.client.getReqId()
            # Parent order
            parent_order = ibi.Order(
                orderId=parent_id,
                action='BUY',
                totalQuantity=position_size,
                orderType='LMT',
                lmtPrice=limit_price,
                tif='GTC',
                outsideRth=True,
                transmit=False
            )
            # Trailing stop child order
            trailing_stop_order = ibi.Order(
                orderId=stop_id,
                action='SELL',
                totalQuantity=position_size,
                orderType='TRAIL',
                trailingPercent=trailing_percent,
                parentId=parent_id,
                tif='GTC',
                outsideRth=True,
                transmit=False
            )
            # Take profit child order
            profit_order = ibi.Order(
                orderId=profit_id,
                action='SELL',
                totalQuantity=position_size,
                orderType='LMT',
                lmtPrice=take_profit_price,
                parentId=parent_id,
                tif='GTC',
                outsideRth=True,
                transmit=False
            )
            # Place orders
            parent_trade = self._ib.placeOrder(contract, parent_order)
            trailing_stop_trade = self._ib.placeOrder(contract, trailing_stop_order)
            profit_trade = self._ib.placeOrder(contract, profit_order)
            # Transmit parent to activate bracket
            parent_order.transmit = True
            self._ib.placeOrder(contract, parent_order)
            return all([parent_trade, trailing_stop_trade, profit_trade])
        except Exception as e:
            logger.error(f"[BRACKET ERROR] {symbol} - {e}")
            return False






    def _get_stop_price(self, symbol, entry_price):
        """Get stop price from signals data"""
        try:
            signals_df = read_signals(status_filter="Pending")
            symbol_signal = signals_df[signals_df['Symbol'] == symbol]
            
            if symbol_signal.empty:
                signals_df = read_signals(status_filter="Active")
                symbol_signal = signals_df[signals_df['Symbol'] == symbol]
            
            if symbol_signal.empty:
                signals_df = read_signals()
                symbol_signal = signals_df[signals_df['Symbol'] == symbol]
            
            if not symbol_signal.empty:
                stop_price = symbol_signal.iloc[0]['StopPrice']
                logger.info(f"[STOP PRICE] {symbol} - Using stop price from signal: ${stop_price}")
                
                if stop_price > 0 and stop_price < entry_price:
                    return stop_price
                else:
                    logger.warning(f"[STOP PRICE] {symbol} - Invalid stop price from signal: ${stop_price}")
            
            # Fallback to 5% below entry price
            fallback_stop = entry_price * 0.95
            logger.warning(f"[STOP PRICE] {symbol} - Using 5% fallback stop: ${fallback_stop}")
            return fallback_stop
            
        except Exception as e:
            logger.error(f"Error getting stop price for {symbol}: {e}")
            return entry_price * 0.95

    def _get_take_profit_price(self, symbol, entry_price, stop_price):
        """Calculate take profit price using 2:1 reward:risk ratio"""
        try:
            signals_df = read_signals(status_filter="Pending")
            symbol_signal = signals_df[signals_df['Symbol'] == symbol]
            
            if not symbol_signal.empty and 'TargetPrice' in symbol_signal.columns:
                target_price = symbol_signal.iloc[0]['TargetPrice']
                if target_price > entry_price:
                    logger.info(f"[TAKE PROFIT] {symbol} - Using target price from signal: ${target_price}")
                    return target_price
           
            # Calculate 2:1 reward:risk ratio
            risk = entry_price - stop_price
            reward = risk * 2.0
            take_profit = entry_price + reward
            
            logger.info(f"[TAKE PROFIT] {symbol} - Using 2:1 ratio: ${take_profit} (Risk: ${risk}, Reward: ${reward})")
            return take_profit
            
        except Exception as e:
            logger.error(f"[TAKE PROFIT ERROR] {symbol} - Error calculating take profit: {e}")
            return entry_price * 1.05

    def _check_ib_bracket_status(self):
        """Check status of direct IB bracket orders"""
        if not self.ib_bracket_orders:
            return
            
        try:
            for order_id, bracket_info in list(self.ib_bracket_orders.items()):
                symbol = bracket_info['symbol']
                parent_trade = bracket_info['parent_trade']
                
                if parent_trade.orderStatus.status == 'Filled':
                    logger.info(f"[BRACKET FILLED] {symbol} - Parent buy order filled, bracket is now active")
                    
                    try:
                        mark_signal_as_active(symbol, parent_trade.orderStatus.avgFillPrice, bracket_info['size'])
                        logger.info(f"[BRACKET SIGNAL] {symbol} - Marked signal as Active")
                    except Exception as e:
                        logger.error(f"[BRACKET SIGNAL ERROR] {symbol} - {e}")
                    
                    del self.ib_bracket_orders[order_id]
                    
                elif parent_trade.orderStatus.status in ['Cancelled', 'Rejected']:
                    logger.warning(f"[BRACKET FAILED] {symbol} - Parent order {parent_trade.orderStatus.status}")
                    if symbol in self.order_attempted:
                        self.order_attempted.remove(symbol)
                    del self.ib_bracket_orders[order_id]
                    
        except Exception as e:
            logger.error(f"[BRACKET STATUS ERROR] Error checking bracket status: {e}")

    def notify_order(self, order):
        """Called when order status changes - Simplified for bracket orders only"""
        try:
            symbol = order.data._name.replace('_5sec', '') if hasattr(order.data, '_name') else 'Unknown'
            
            if order.status in [order.Submitted]:
                logger.info(f"[ORDER SUBMITTED] {symbol} - ID: {order.ref}")
                
            elif order.status in [order.Accepted]:
                logger.info(f"[ORDER ACCEPTED] {symbol} - ID: {order.ref}")
                
            elif order.status in [order.Completed]:
                logger.info(f"[ORDER COMPLETED] {symbol} - ID: {order.ref}")
                
            elif order.status in [order.Canceled]:
                logger.warning(f"[ORDER CANCELED] {symbol} - ID: {order.ref}")
                    
            elif order.status in [order.Margin, order.Rejected]:
                logger.error(f"[ORDER REJECTED] {symbol} - ID: {order.ref}")
                
        except Exception as e:
            logger.error(f"Error in notify_order: {e}")

    def notify_trade(self, trade):
        """Called when trade is opened/closed"""
        try:
            symbol = trade.data._name.replace('_5sec', '') if hasattr(trade.data, '_name') else 'Unknown'
            
            if trade.isclosed:
                logger.info(f"[TRADE CLOSED] {symbol} - P&L: ${trade.pnl:.2f}")
            elif trade.isopen:
                logger.info(f"[TRADE OPENED] {symbol} - Size: {trade.size}, Price: ${trade.price}")
                
        except Exception as e:
            logger.error(f"Error in notify_trade: {e}")

    def stop(self):
        logger.info("Strategy stopping")
        
        if self.ib_bracket_orders:
            logger.warning(f"Strategy stopping with {len(self.ib_bracket_orders)} pending bracket orders")
            
        portfolio_value = self.broker.getvalue()
        logger.info(f"Final portfolio value: ${portfolio_value:.2f}")
        super().stop()





































def finalize_positions_sync(ib, trading_data_path='_Live_trades.parquet'):
    """
    Fetch IB's official open positions and update local data to reflect any mismatches.
    """
    try:
        # 1. Get real open positions from IB
        ib_positions = ib.positions()
        real_positions = {}  # e.g., { 'AAPL': 100, 'TSLA': 50, ... }
        for pos in ib_positions:
            if pos.position != 0:
                symbol = pos.contract.symbol
                size = pos.position
                real_positions[symbol] = size
        
        df = pd.read_parquet(trading_data_path)

        required_columns = {'Symbol', 'IsCurrentlyBought'}
        if not required_columns.issubset(df.columns):
            logger.warning("DataFrame missing required columns (Symbol, IsCurrentlyBought).")
        
        # Convert local positions to a dictionary for easy comparison
        local_positions = (
            df[df['IsCurrentlyBought'] == True]
            .set_index('Symbol')['PositionSize']
            .to_dict()
        )  # e.g., { 'AAPL': 100, 'TSLA': 50 }

        # 3. Reconcile differences
        # Case A: Symbol in IB but not in local -> Mark as bought
        for symbol, real_size in real_positions.items():
            if symbol not in local_positions:
                logger.info(f"{symbol} is open in IB but not marked locally; updating local data.")
                new_data = {
                    'Symbol': symbol,
                    'IsCurrentlyBought': True,
                    'PositionSize': real_size,
                    'LastBuySignalDate': pd.Timestamp.now(),
                    'LastBuySignalPrice': 0,  # Adjust as needed
                }
                # Append or update the DataFrame
                if symbol not in df['Symbol'].values:
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                else:
                    for col, val in new_data.items():
                        df.loc[df['Symbol'] == symbol, col] = val
            else:
                local_size = local_positions[symbol]
                if local_size != real_size:
                    logger.info(
                        f"Mismatch for {symbol}: local size={local_size}, IB size={real_size}. Correcting local data."
                    )
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = real_size
        
        # Case B: Symbol in local but not in IB -> Mark as closed
        for symbol, local_size in local_positions.items():
            if symbol not in real_positions:
                logger.info(f"{symbol} is locally open but not in IB; marking as closed in local data.")
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp.now()

        # 4. Save the updated DataFrame
        df.to_parquet(trading_data_path, index=False)
        logger.info("Final data sync completed successfully. Local data now matches IB reality.")
    
    except Exception as e:
        logger.error(f"Error in finalize_positions_sync: {e}")
        logger.error(traceback.format_exc())


def disconnect_ib_safely(store, ib, debug=True):
    if debug:
        dprint("Starting IB disconnection process")
    
    # First disconnect the IB instance if it exists and is connected
    if ib:
        try:
            if ib.isConnected():
                if debug:
                    dprint("Disconnecting IB instance")
                ib.disconnect()
                if debug:
                    dprint("IB disconnect() called")
        except Exception as e:
            if debug:
                dprint(f"Error during IB disconnect: {type(e).__name__}: {e}")
    
    # Then stop the store if it exists
    if store:
        try:
            if debug:
                dprint("Stopping IBStore")
            store.stop()
            if debug:
                dprint("IBStore stop() called")
        except Exception as e:
            if debug:
                dprint(f"Error stopping IBStore: {type(e).__name__}: {e}")
    
    return True


def start(manual_override=False):
    """
    Main entry point for the trading session with proper data feed configuration.
    Reads signals from the consolidated 0__Signals.parquet file.
    
    Args:
        manual_override (bool): Whether to bypass market open checks
    """

    from Util import read_signals, update_signal_status, mark_signal_as_active
    
    start_time = datetime.now()
    logger.info(f"Starting trading session at {start_time.isoformat()}")
 
    cerebro = bt.Cerebro()
    
    # Create connection with proper client ID management
    store, ib = create_ib_connection(
        port=7496,
        max_attempts=3,
        timeout=5.0
    )
    
    if not store or not ib or not ib.isConnected():
        logger.error("Could not establish connection to IB. Exiting.")
        return

    logger.info("IB connection established successfully")
    
    # Make store and ib available globally for reference by strategy
    globals()['store'] = store
    globals()['ib'] = ib
    globals()['cerebro'] = cerebro
    
    try:
        # Check market status if not overridden
        if not manual_override:
            market_status = wait_for_market_open(manual_override=manual_override)
            if not market_status:
                logger.info("Market not open. Exiting.")
                disconnect_ib_safely(store, ib)
                return
        else:
            logger.info("Manual override enabled - skipping market check")
        
        # Get open positions from IB
        open_positions = get_open_positions(ib)
        logger.info(f"Found {len(open_positions)} open positions in IB")
        
        # Get pending signals from our consolidated file
        signals_df = read_signals(status_filter="Pending")
        
        if not signals_df.empty:
            logger.info(f"Found {len(signals_df)} pending signals")

        # If no pending signals file exists, log and exit
        if signals_df.empty:
            logger.info("No pending signals found. Exiting.")
            disconnect_ib_safely(store, ib)
            return
            
        # Also get ACTIVE positions from our signals file
        active_positions_df = read_signals(status_filter="Active")
        logger.info(f"Found {len(active_positions_df)} active positions in signals file")
        
        # Combine symbols from positions and signals
        all_symbols = set(open_positions)
        
        # Add symbols from pending signals - DATE PROCESSING
        today = datetime.now().date()
        
        # Get the next trading day if needed
        try:
            from Util import get_next_trading_day
            tomorrow = get_next_trading_day(today)
        except Exception as e:
            logger.warning(f"Error getting next trading day: {e}")
            # Simple fallback if function not available
            tomorrow = today + timedelta(days=1)
            
        # Filter signals for today or tomorrow
        try:
            # Step-by-step filtering
            target_date_column = signals_df['TargetDate']
            target_date_dates = target_date_column.dt.date
            filter_list = [today, tomorrow]
            mask = target_date_dates.isin(filter_list)
            valid_signals = signals_df[mask]
            
        except Exception as e:
            logger.warning(f"Error in date filtering: {e}. Using manual filtering.")
            
            # Manual filtering fallback
            valid_signals_list = []
            for idx, row in signals_df.iterrows():
                row_target_date = row['TargetDate'].date() if hasattr(row['TargetDate'], 'date') else row['TargetDate']
                if row_target_date in [today, tomorrow]:
                    valid_signals_list.append(row)
            
            if valid_signals_list:
                valid_signals = pd.DataFrame(valid_signals_list)
            else:
                valid_signals = pd.DataFrame()
        
        signal_symbols = valid_signals['Symbol'].tolist()
        logger.info(f"Found {len(signal_symbols)} valid signals for today/tomorrow")
        
        # Add to all_symbols
        all_symbols.update(signal_symbols)
        
        # Add symbols from active positions in our file
        active_symbols = active_positions_df['Symbol'].tolist()
        all_symbols.update(active_symbols)
        
        if not all_symbols:
            logger.info("No symbols to trade. Exiting.")
            disconnect_ib_safely(store, ib)
            return
        
        logger.info(f"Will process {len(all_symbols)} total symbols: {', '.join(all_symbols)}")
        
        # Add data feeds for each symbol with proper configuration
        for symbol in all_symbols:
            try:
                # Create contract
                contract = ibi.Stock(symbol, 'SMART', 'USD')
                logger.info(f"Creating data feed for {symbol}")
                
                # Configure and add the data feed using RTBars (5-second bars)
                data = store.getdata(
                    dataname=symbol,
                    contract=contract,
                    sectype='STK',  # Explicit stock type
                    exchange='SMART',
                    currency='USD',
                    rtbar=True,  # Use 5-second RealTimeBars
                    timeframe=bt.TimeFrame.Seconds,  # Must be Seconds for rtbar
                    compression=5,  # 5-second bars (minimum for rtbar)
                    useRTH=False,  # Regular trading hours only
                    qcheck=1.0,  # Check for new bars every second
                    backfill_start=False,  # Don't request historical backfill
                    backfill=False,
                    reconnect=True,
                    live=True,
                )
                
                # Set a name for the data feed for easier identification
                data._name = f"{symbol}_5sec"
                
                # Add the 5-second data to cerebro
                cerebro.adddata(data)
                logger.info(f"Added 5-second data feed for {symbol}")
                
                # Resample to 1-minute bars
                minute_data = cerebro.resampledata(
                    data,
                    timeframe=bt.TimeFrame.Minutes,
                    compression=1,
                    bar2edge=True,  # Align to minute boundaries
                    boundoff=0.5  # Allow 0.5 compression units to complete bar
                )
                
                # Set a clear name for the resampled data
                minute_data._name = symbol  # Just use the symbol name for the 1-min data
                
                logger.info(f"Successfully resampled {symbol} to 1-minute bars")
                
            except Exception as e:
                logger.error(f"Error adding data feed for {symbol}: {e}")
                logger.error(traceback.format_exc())
        
        # Set up the broker and strategy
        broker = store.getbroker()
        cerebro.setbroker(broker)
        
        # Keep a reference to the original IB connection in the broker for easier access
        # This may help with position synchronization
        if not hasattr(broker, 'ib'):
            try:
                broker.ib = ib
                logger.info("Added direct ib reference to broker")
            except Exception as e:
                logger.warning(f"Could not add ib reference to broker: {e}")
        
        # Add the strategy
        cerebro.addstrategy(StockSniperLive)
        logger.info("Added StockSniperLive strategy")
        
        # Run the strategy with a shorter max runtime
        logger.info("Starting strategy execution")
        cerebro.run()
        logger.info("Strategy execution completed")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

    finally:
        # Proper cleanup in finally block
        try:
            # Final sync if available
            if ib and ib.isConnected():
                logger.info("Performing final position synchronization")
                finalize_positions_sync(ib)
                
            # Always disconnect safely
            if store or ib:
                logger.info("Disconnecting from IB...")
                success = disconnect_ib_safely(store, ib)
                logger.info(f"Disconnection {'successful' if success else 'may have had issues'}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        # Clean up global references
        for var in ['store', 'ib', 'cerebro']:
            if var in globals():
                globals().pop(var, None)
            
        # Log total runtime
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"Session completed. Total duration: {total_duration:.2f} seconds")







if __name__ == "__main__":
    # Set up logger using imported function
    ##print the time 
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {time}")

    logger = get_logger(script_name="LiveTrading")
    logger.info('Logger initialized')
    logger.info('========== Starting StockSniper Live Trading Session ==========')
    
    # Check market status using imported function
    manual_override = True  # Set to True for testing outside market hours
    if not is_nyse_open(manual_override=manual_override) and not manual_override:
        logger.error("Market is closed. Use manual_override=True to force execution.")
        sys.exit(1)
        
    # Start the trading system
    start(manual_override=manual_override)


