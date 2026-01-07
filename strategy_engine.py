"""
ORB (Opening Range Breakout) Strategy Engine
==========================================

Python implementation of 15-minute Opening Range Breakout strategy with:
- Opening range identification (6:30-6:45 AM Pacific)
- 50-EMA trend filter
- 2:1 reward-to-risk ratio
- Single trade per day limit
- State persistence across restarts
"""

import json
import datetime
from collections import deque
from typing import Dict, Any, Optional, Tuple
import pytz
from alpaca_trade_api.rest import REST
import os

class ORBStrategy:
    """
    Opening Range Breakout Strategy Implementation
    
    Features:
    - 15-minute opening range (6:30-6:45 AM Pacific)
    - 50-period EMA trend filter
    - Dynamic stop-loss and take-profit based on range size
    - Single trade per day logic
    - State persistence for restart recovery
    """
    
    def __init__(self, api: REST, session_name: str, tz: str = "America/Los_Angeles", 
                 symbol: str = "TSLA", log_trade_func=None):
        """
        Initialize the ORB strategy
        
        Args:
            api: Alpaca REST API client
            session_name: Session identifier for logging
            tz: Timezone for opening range calculation
            symbol: Trading symbol
            log_trade_func: Function to log trades to CSV
        """
        self.api = api
        self.session_name = session_name
        self.symbol = symbol
        self.timezone = pytz.timezone(tz)
        self.log_trade_func = log_trade_func
        
        # Strategy state
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        self.entry_price = None
        self.tp = None
        self.sl = None
        self.current_date = None
        self.position_id = None
        
        # EMA calculation
        self.close_prices = deque(maxlen=50)
        self.ema50 = None
        self.ema_alpha = 2.0 / (50 + 1)
        
        # State file
        self.state_file = "state.json"
        
        # Load existing state
        self.load_state()
        
        # Verify state matches actual Alpaca positions
        self.verify_position_state()
        
        # Load historical data for EMA initialization
        self.load_history()
        
        print(f"ORB Strategy initialized for {symbol} ({tz})")
    
    def reset_for_new_day(self):
        """Reset strategy state for a new trading day"""
        # Check if we have an open position before resetting
        has_open_position = (self.entry_price is not None and 
                            self.tp is not None and 
                            self.sl is not None)
        
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        
        # IMPORTANT: Only reset tp/sl/entry if we DON'T have an open position
        # If we have a position, preserve the original stop loss and target!
        if not has_open_position:
            self.entry_price = None
            self.tp = None
            self.sl = None
            self.position_id = None
            print(f"✅ Reset all levels for new day (no open position)")
        else:
            # Keep the position details from entry day
            print(f"⚠️ Open position detected - preserving TP=${self.tp:.2f} and SL=${self.sl:.2f} from entry")
            print(f"   Entry: ${self.entry_price:.2f} | Position will use ORIGINAL levels, not today's OR")
        
        current_date = datetime.datetime.now(self.timezone).date()
        self.current_date = current_date
        
        self.save_state()
        print(f"Reset opening range for new day: {current_date}")
    
    def load_history(self):
        """Load recent historical data to prime EMA calculation"""
        try:
            from alpaca_trade_api.rest import TimeFrame
            
            # Get last 100 bars to properly initialize EMA
            end_time = datetime.datetime.now(pytz.UTC)
            start_time = end_time - datetime.timedelta(days=30)
            
            bars = self.api.get_bars(
                self.symbol,
                TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=100
            ).df
            
            if not bars.empty:
                # Initialize close prices deque with recent data
                for close in bars['close'].tail(50):
                    self.close_prices.append(float(close))
                
                # Calculate initial EMA
                if len(self.close_prices) > 0:
                    self.ema50 = sum(self.close_prices) / len(self.close_prices)
                    
                print(f"Loaded {len(self.close_prices)} historical bars, EMA50: {self.ema50:.2f}")
            
        except Exception as e:
            print(f"Could not load history: {e}")
            self.ema50 = None
    
    def save_state(self):
        """Save current strategy state to JSON file"""
        state = {
            'opening_high': self.opening_high,
            'opening_low': self.opening_low,
            'range_set': self.range_set,
            'trade_taken': self.trade_taken,
            'entry_price': self.entry_price,
            'tp': self.tp,
            'sl': self.sl,
            'current_date': self.current_date.isoformat() if self.current_date else None,
            'position_id': self.position_id,
            'ema50': self.ema50,
            'symbol': self.symbol
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def load_state(self):
        """Load strategy state from JSON file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.opening_high = state.get('opening_high')
                self.opening_low = state.get('opening_low')
                self.range_set = state.get('range_set', False)
                self.trade_taken = state.get('trade_taken', False)
                self.entry_price = state.get('entry_price')
                self.tp = state.get('tp')
                self.sl = state.get('sl')
                self.position_id = state.get('position_id')
                self.ema50 = state.get('ema50')
                
                if state.get('current_date'):
                    self.current_date = datetime.datetime.fromisoformat(state['current_date']).date()
                
                print(f"Loaded strategy state from {self.state_file}")
        
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def verify_position_state(self):
        """Verify strategy state matches actual Alpaca positions"""
        try:
            positions = self.api.list_positions()
            tsla_position = next((pos for pos in positions if pos.symbol == self.symbol), None)
            
            if tsla_position and float(tsla_position.qty) > 0:
                # We have an actual position in Alpaca
                if not self.trade_taken:
                    print(f"⚠️ Found open {self.symbol} position in Alpaca but trade_taken=False!")
                    print(f"   Syncing state: Setting trade_taken=True")
                    self.trade_taken = True
                    
                    # If we have TP/SL in state, keep them. Otherwise, we can't auto-set without OR
                    if self.tp and self.sl:
                        print(f"   Using existing TP=${self.tp:.2f} and SL=${self.sl:.2f} from state")
                    else:
                        print(f"   ⚠️ No TP/SL in state - manual exit required")
                    
                    self.save_state()
            else:
                # No actual position in Alpaca
                if self.trade_taken:
                    print(f"⚠️ trade_taken=True but no {self.symbol} position in Alpaca!")
                    print(f"   Syncing state: Setting trade_taken=False")
                    self.trade_taken = False
                    self.entry_price = None
                    self.save_state()
                    
        except Exception as e:
            print(f"⚠️ Error verifying position state: {e}")
    
    async def on_bar(self, bar: Dict[str, Any]):
        """
        Process a completed 5-minute bar
        
        Args:
            bar: Dict with keys: time, open, high, low, close, volume
        """
        try:
            bar_time = bar['time']
            if isinstance(bar_time, str):
                bar_time = datetime.datetime.fromisoformat(bar_time.replace('Z', '+00:00'))
            
            # Convert to local timezone
            local_time = bar_time.astimezone(self.timezone)
            current_date = local_time.date()
            
            # Check for new day
            if self.current_date is None or current_date != self.current_date:
                self.reset_for_new_day()
            
            # Update EMA
            close_price = float(bar['close'])
            self.update_ema(close_price)
            
            # Opening range logic (6:30-6:45 AM Pacific)
            if local_time.time() >= datetime.time(6, 30) and local_time.time() < datetime.time(6, 45):
                self.update_opening_range(float(bar['high']), float(bar['low']))
            
            # Set range after opening period
            elif local_time.time() >= datetime.time(6, 45) and not self.range_set:
                if self.opening_high is not None and self.opening_low is not None:
                    self.range_set = True
                    print(f"Opening range set: {self.opening_low:.2f} - {self.opening_high:.2f}")
            
            # Get real-time Yahoo Finance price for decision making
            yahoo_price = None
            try:
                from backup_data_provider import get_tsla_price_with_fallback
                yahoo_price = get_tsla_price_with_fallback()
            except Exception:
                yahoo_price = close_price  # Fallback to bar price
            
            # Use Yahoo Finance price for trading decisions
            decision_price = yahoo_price if yahoo_price else close_price
            
            # Entry logic
            if (self.range_set and not self.trade_taken and 
                self.opening_high is not None and self.ema50 is not None):
                
                # Long entry condition - use Yahoo Finance price
                if decision_price > self.opening_high and decision_price > self.ema50:
                    print(f"📊 Entry signal: Yahoo price ${decision_price:.2f} > OR high ${self.opening_high:.2f} > EMA50 ${self.ema50:.2f}")
                    self.enter_long(decision_price, local_time)
            
            # Exit logic (if in position)
            if self.trade_taken and self.entry_price is not None:
                if (decision_price >= self.tp or decision_price <= self.sl):
                    reason = f"TP hit at ${decision_price:.2f}" if decision_price >= self.tp else f"SL hit at ${decision_price:.2f}"
                    self.exit_position(decision_price, local_time, reason)
            
            # End of day exit (3:50 PM Pacific to allow for execution)
            if (self.trade_taken and self.entry_price is not None and 
                local_time.time() >= datetime.time(15, 50)):
                self.exit_position(decision_price, local_time, "End of day")
            
            self.save_state()
            
        except Exception as e:
            print(f"❌ Error processing bar: {e}")
    
    def update_ema(self, close_price: float):
        """Update 50-period EMA"""
        self.close_prices.append(close_price)
        
        if self.ema50 is None:
            # Initialize with SMA
            if len(self.close_prices) >= 10:  # Minimum data points
                self.ema50 = sum(list(self.close_prices)[-10:]) / 10
        else:
            # Update EMA
            self.ema50 = (close_price * self.ema_alpha) + (self.ema50 * (1 - self.ema_alpha))
    
    def update_opening_range(self, high: float, low: float):
        """Update opening range high and low"""
        if self.opening_high is None:
            self.opening_high = high
            self.opening_low = low
        else:
            self.opening_high = max(self.opening_high, high)
            self.opening_low = min(self.opening_low, low)
    
    def enter_long(self, price: float, timestamp: datetime.datetime):
        """Execute long entry"""
        try:
            range_size = self.opening_high - self.opening_low
            self.tp = self.opening_high + (2 * range_size)  # 2:1 reward/risk
            self.sl = self.opening_low
            
            # Submit buy order
            order_id, execution_price = self.submit_buy(price)
            
            if order_id:
                self.trade_taken = True
                self.entry_price = execution_price
                self.position_id = f"{self.symbol}_{timestamp.strftime('%Y%m%d')}"
                
                print(f"🟢 LONG ENTRY: {self.symbol} @ ${execution_price:.2f}")
                print(f"   TP: ${self.tp:.2f} | SL: ${self.sl:.2f} | Range: ${range_size:.2f}")
                
        except Exception as e:
            print(f"❌ Error entering long position: {e}")
    
    def exit_position(self, price: float, timestamp: datetime.datetime, reason: str):
        """Exit current position"""
        try:
            order_id, execution_price = self.submit_sell(price)
            
            if order_id:
                pnl = execution_price - self.entry_price if self.entry_price else 0
                print(f"🔴 EXIT: {self.symbol} @ ${execution_price:.2f} | PnL: ${pnl:.2f} | Reason: {reason}")
                
                # Reset position state but keep trade_taken=True for the day
                self.entry_price = None
                self.tp = None
                self.sl = None
                
        except Exception as e:
            print(f"❌ Error exiting position: {e}")
    
    def submit_buy(self, price: float) -> Tuple[Optional[str], Optional[float]]:
        """
        Submit buy order via Alpaca API using Yahoo Finance price validation
        
        Returns:
            Tuple of (order_id, execution_price)
        """
        try:
            # Check if we already have a position - prevent buying again
            try:
                existing_position = self.api.get_position(self.symbol)
                if existing_position and float(existing_position.qty) > 0:
                    print(f"⚠️ Already holding {existing_position.qty} shares of {self.symbol} - skipping new buy")
                    return None, None
            except Exception:
                # No position exists, proceed with buy
                pass
            
            # Get real-time price from Yahoo Finance for validation
            from backup_data_provider import get_tsla_price_with_fallback
            current_yahoo_price = get_tsla_price_with_fallback()
            
            # Use Yahoo Finance price for order execution (more accurate)
            execution_price = current_yahoo_price
            print(f"📊 Using Yahoo Finance price for order: ${execution_price:.2f} (vs bar price: ${price:.2f})")
            
            # Get account info for position sizing
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Use 95% of buying power for safety
            max_shares = round((buying_power * 0.95) / execution_price, 6)
            
            if max_shares <= 0:
                print("❌ Insufficient buying power")
                return None, None
            
            # Check if market is open
            clock = self.api.get_clock()
            order_type = "market" if clock.is_open else "limit"
            
            if order_type == "market":
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=max_shares,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
            else:
                # Use Yahoo Finance price for limit orders
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=max_shares,
                    side="buy",
                    type="limit",
                    limit_price=execution_price,
                    time_in_force="gtc",
                    extended_hours=True
                )
            
            # Log trade if function provided
            if self.log_trade_func:
                account_after = self.api.get_account()
                self.log_trade_func(
                    self.session_name, self.symbol, "buy", max_shares, execution_price,
                    max_shares * execution_price, buying_power, float(account_after.buying_power),
                    f"ORB Entry (Yahoo: ${execution_price:.2f}) - Order ID: {order.id}"
                )
            
            return order.id, execution_price
            
        except Exception as e:
            print(f"❌ Error submitting buy order: {e}")
            return None, None
    
    def submit_sell(self, price: float) -> Tuple[Optional[str], Optional[float]]:
        """
        Submit sell order via Alpaca API using Yahoo Finance price validation
        
        Returns:
            Tuple of (order_id, execution_price)
        """
        try:
            # Get real-time price from Yahoo Finance for validation
            from backup_data_provider import get_tsla_price_with_fallback
            current_yahoo_price = get_tsla_price_with_fallback()
            
            # Use Yahoo Finance price for order execution (more accurate)
            execution_price = current_yahoo_price
            print(f"📊 Using Yahoo Finance price for sell order: ${execution_price:.2f} (vs bar price: ${price:.2f})")
            
            # Get current position
            position = self.api.get_position(self.symbol)
            shares = float(position.qty)
            
            if shares <= 0:
                print("❌ No shares to sell")
                return None, None
            
            # Check if market is open
            clock = self.api.get_clock()
            order_type = "market" if clock.is_open else "limit"
            
            if order_type == "market":
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=shares,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
            else:
                # Use Yahoo Finance price for limit orders
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=shares,
                    side="sell",
                    type="limit",
                    limit_price=execution_price,
                    time_in_force="gtc",
                    extended_hours=True
                )
            
            # Log trade if function provided
            if self.log_trade_func:
                account = self.api.get_account()
                account_after = self.api.get_account()
                self.log_trade_func(
                    self.session_name, self.symbol, "sell", shares, execution_price,
                    shares * execution_price, float(account.buying_power), float(account_after.buying_power),
                    f"ORB Exit (Yahoo: ${execution_price:.2f}) - Order ID: {order.id}"
                )
            
            return order.id, execution_price
            
        except Exception as e:
            print(f"❌ Error submitting sell order: {e}")
            return None, None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            'symbol': self.symbol,
            'current_date': self.current_date.isoformat() if self.current_date else None,
            'opening_high': self.opening_high,
            'opening_low': self.opening_low,
            'range_set': self.range_set,
            'trade_taken': self.trade_taken,
            'in_position': self.entry_price is not None,
            'entry_price': self.entry_price,
            'take_profit': self.tp,
            'stop_loss': self.sl,
            'ema50': self.ema50
        }
