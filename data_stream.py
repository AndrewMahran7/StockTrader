"""
Market Data Stream for ORB Strategy
===================================

Handles real-time market data via Alpaca websockets with REST polling fallback.
Includes market hours detection and automatic data feed management.
"""

import asyncio
import datetime
import threading
import time
from typing import Dict, Any, Callable, Optional
import pytz
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream
import json

class MarketClock:
    """
    Market hours and timing utilities using Alpaca API
    """
    
    def __init__(self, api: REST):
        self.api = api
        self.eastern = pytz.timezone('US/Eastern')
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"⚠️ Error checking market status: {e}")
            return False
    
    def get_market_hours(self) -> Dict[str, datetime.datetime]:
        """Get today's market open and close times"""
        try:
            calendar = self.api.get_calendar(
                start=datetime.date.today(),
                end=datetime.date.today()
            )
            
            if calendar:
                day = calendar[0]
                return {
                    'open': day.open.astimezone(self.eastern),
                    'close': day.close.astimezone(self.eastern)
                }
        except Exception as e:
            print(f"⚠️ Error getting market hours: {e}")
        
        # Default market hours if API fails
        today = datetime.date.today()
        return {
            'open': self.eastern.localize(datetime.datetime.combine(today, datetime.time(9, 30))),
            'close': self.eastern.localize(datetime.datetime.combine(today, datetime.time(16, 0)))
        }
    
    def minutes_until_open(self) -> int:
        """Get minutes until market opens"""
        try:
            clock = self.api.get_clock()
            if clock.is_open:
                return 0
            
            next_open = clock.next_open.astimezone(self.eastern)
            now = datetime.datetime.now(self.eastern)
            delta = next_open - now
            return max(0, int(delta.total_seconds() / 60))
            
        except Exception as e:
            print(f"⚠️ Error calculating time to open: {e}")
            return 60  # Default to 1 hour
    
    def minutes_until_close(self) -> int:
        """Get minutes until market closes"""
        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                return 0
            
            market_close = clock.next_close.astimezone(self.eastern)
            now = datetime.datetime.now(self.eastern)
            delta = market_close - now
            return max(0, int(delta.total_seconds() / 60))
            
        except Exception as e:
            print(f"⚠️ Error calculating time to close: {e}")
            return 30  # Default to 30 minutes

class DataStream:
    """
    Real-time market data stream with websocket and REST polling fallback
    """
    
    def __init__(self, api: REST, symbol: str, on_bar_callback: Callable[[Dict[str, Any]], None], 
                 use_streaming: bool = True, polling_seconds: int = 60):
        """
        Initialize data stream
        
        Args:
            api: Alpaca REST API client
            symbol: Trading symbol to stream
            on_bar_callback: Function to call when new bar is received
            use_streaming: Whether to use websocket streaming
            polling_seconds: Polling interval for REST fallback (60-300s)
        """
        self.api = api
        self.symbol = symbol
        self.on_bar_callback = on_bar_callback
        self.use_streaming = use_streaming
        self.polling_seconds = max(60, min(300, polling_seconds))  # Clamp to 60-300s
        
        self.market_clock = MarketClock(api)
        self.running = False
        self.stream = None
        self.polling_thread = None
        
        # Track last bar to avoid duplicates
        self.last_bar_time = None
        
        print(f"📡 DataStream initialized for {symbol}")
        print(f"   Mode: {'Websocket' if use_streaming else 'REST Polling'}")
        print(f"   Polling interval: {polling_seconds}s")
    
    def start(self):
        """Start the data stream"""
        if self.running:
            return
        
        self.running = True
        
        if self.use_streaming:
            self._start_websocket_stream()
        else:
            self._start_polling()
    
    def stop(self):
        """Stop the data stream"""
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop()
            except:
                pass
        
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
    
    def _start_websocket_stream(self):
        """Start websocket streaming in background thread"""
        def run_stream():
            try:
                self.stream = Stream(
                    self.api._key_id,
                    self.api._secret_key,
                    base_url=self.api._base_url
                )
                
                @self.stream.on_bar(self.symbol)
                async def on_minute_bar(bar):
                    """Handle incoming minute bar from websocket"""
                    try:
                        bar_dict = {
                            'time': bar.timestamp.isoformat(),
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume)
                        }
                        
                        # Check for duplicate bars
                        if self.last_bar_time != bar.timestamp:
                            self.last_bar_time = bar.timestamp
                            self.on_bar_callback(bar_dict)
                        
                    except Exception as e:
                        print(f"❌ Error processing websocket bar: {e}")
                
                # Subscribe to minute bars
                self.stream.subscribe_bars(self.symbol, 'minute')
                
                print(f"🚀 Websocket stream started for {self.symbol}")
                self.stream.run()
                
            except Exception as e:
                print(f"❌ Websocket stream failed: {e}")
                print("🔄 Falling back to REST polling...")
                self.use_streaming = False
                self._start_polling()
        
        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()
    
    def _start_polling(self):
        """Start REST API polling in background thread"""
        def poll_bars():
            print(f"🔄 REST polling started for {self.symbol} (every {self.polling_seconds}s)")
            
            while self.running:
                try:
                    # Only poll during market hours or shortly after
                    if not self._should_poll():
                        time.sleep(60)  # Check market status every minute
                        continue
                    
                    # Get the latest bar
                    end_time = datetime.datetime.now(pytz.UTC)
                    start_time = end_time - datetime.timedelta(minutes=5)  # Get last 5 minutes
                    
                    bars = self.api.get_bars(
                        self.symbol,
                        TimeFrame.Minute,
                        start=start_time.isoformat(),
                        end=end_time.isoformat(),
                        limit=5
                    )
                    
                    if hasattr(bars, 'df') and not bars.df.empty:
                        # Get the most recent complete bar
                        latest_bar = bars.df.iloc[-1]
                        bar_time = latest_bar.name.to_pydatetime()
                        
                        # Check if this is a new bar
                        if self.last_bar_time != bar_time:
                            self.last_bar_time = bar_time
                            
                            bar_dict = {
                                'time': bar_time.isoformat(),
                                'open': float(latest_bar['open']),
                                'high': float(latest_bar['high']),
                                'low': float(latest_bar['low']),
                                'close': float(latest_bar['close']),
                                'volume': int(latest_bar['volume'])
                            }
                            
                            self.on_bar_callback(bar_dict)
                    
                    time.sleep(self.polling_seconds)
                    
                except Exception as e:
                    print(f"❌ Error in polling loop: {e}")
                    time.sleep(min(300, self.polling_seconds * 2))  # Back off on error
        
        self.polling_thread = threading.Thread(target=poll_bars, daemon=True)
        self.polling_thread.start()
    
    def _should_poll(self) -> bool:
        """Determine if we should poll for data based on market hours"""
        try:
            # Always poll during market hours
            if self.market_clock.is_market_open():
                return True
            
            # Poll for 30 minutes after market close to catch final bars
            hours = self.market_clock.get_market_hours()
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            
            if now > hours['close'] and (now - hours['close']).total_seconds() < 1800:  # 30 minutes
                return True
            
            # Poll 15 minutes before market open
            minutes_to_open = self.market_clock.minutes_until_open()
            return minutes_to_open <= 15 and minutes_to_open > 0
            
        except Exception:
            return False

class BacktestDataProvider:
    """
    Provides historical data for backtesting the ORB strategy
    """
    
    def __init__(self, api: REST):
        self.api = api
    
    def get_historical_bars(self, symbol: str, start_date: str, end_date: str) -> list:
        """
        Get historical minute bars for backtesting
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of bar dictionaries
        """
        try:
            print(f"📊 Fetching historical data for {symbol}: {start_date} to {end_date}")
            
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Minute,
                start=start_date,
                end=end_date,
                asof=None,
                feed='iex'  # Use IEX feed for historical data
            )
            
            if hasattr(bars, 'df') and not bars.df.empty:
                bar_list = []
                df = bars.df
                
                for timestamp, row in df.iterrows():
                    bar_dict = {
                        'time': timestamp.isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume'])
                    }
                    bar_list.append(bar_dict)
                
                print(f"✅ Retrieved {len(bar_list)} historical bars")
                return bar_list
            
            else:
                print("❌ No historical data received")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return []
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                     on_bar_callback: Callable[[Dict[str, Any]], None]):
        """
        Run a backtest by feeding historical bars to the strategy
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format  
            end_date: End date in YYYY-MM-DD format
            on_bar_callback: Strategy callback function
        """
        print(f"🎯 Starting backtest: {symbol} from {start_date} to {end_date}")
        
        bars = self.get_historical_bars(symbol, start_date, end_date)
        
        if not bars:
            print("❌ No data for backtest")
            return
        
        processed_bars = 0
        for bar in bars:
            try:
                on_bar_callback(bar)
                processed_bars += 1
                
                # Progress indicator
                if processed_bars % 1000 == 0:
                    print(f"📈 Processed {processed_bars:,} bars...")
                    
            except Exception as e:
                print(f"❌ Error processing bar in backtest: {e}")
                continue
        
        print(f"✅ Backtest completed: {processed_bars:,} bars processed")
