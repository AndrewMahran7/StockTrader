"""
Market Data Provider - Yahoo Finance
=====================================

This module provides market data using Yahoo Finance (yfinance).
If yfinance fails, returns None so the UI can display the error.
"""

import yfinance as yf
import datetime
import pytz
from typing import Dict, List, Optional
import time
import logging


class MarketDataProvider:
    """Market data provider using Yahoo Finance"""
    
    def __init__(self):
        self.last_request_time = {}
        self.min_request_interval = 2.0
        self.error_backoff = {}
        self.api_failure_times = {}
        # Track last error for UI display
        self.last_error = None
        self.last_error_time = None
        
    def _rate_limit(self, source: str):
        """Rate limiting with exponential backoff for errors"""
        current_time = time.time()
        
        base_interval = self.min_request_interval
        error_count = self.error_backoff.get(source, 0)
        
        if error_count > 0:
            backoff_time = min(base_interval * (2 ** error_count), 60.0)
        else:
            backoff_time = base_interval
        
        if source in self.last_request_time:
            time_since_last = current_time - self.last_request_time[source]
            if time_since_last < backoff_time:
                sleep_time = backoff_time - time_since_last
                print(f"🕐 Rate limiting {source}: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.last_request_time[source] = time.time()
    
    def _record_error(self, source: str, error_msg: str):
        """Record an error for exponential backoff and UI display"""
        self.error_backoff[source] = self.error_backoff.get(source, 0) + 1
        self.last_error = error_msg
        self.last_error_time = datetime.datetime.now(pytz.UTC).isoformat()
        print(f"⚠️ Error count for {source}: {self.error_backoff[source]}")
    
    def _record_success(self, source: str):
        """Record success and reset error count"""
        if source in self.error_backoff:
            del self.error_backoff[source]
        if source in self.api_failure_times:
            del self.api_failure_times[source]
        self.last_error = None
        self.last_error_time = None
    
    def _record_api_failure(self, source: str, error_msg: str):
        """Record an API failure for longer backoff (1 hour)"""
        self.api_failure_times[source] = time.time()
        self.last_error = error_msg
        self.last_error_time = datetime.datetime.now(pytz.UTC).isoformat()
        print(f"🚫 API failure recorded for {source} - will retry after backoff")
    
    def _is_api_in_backoff(self, source: str) -> bool:
        """Check if API is in 1-hour backoff period"""
        if source not in self.api_failure_times:
            return False
        
        failure_time = self.api_failure_times[source]
        current_time = time.time()
        backoff_duration = 3600
        
        if current_time - failure_time < backoff_duration:
            remaining_minutes = (backoff_duration - (current_time - failure_time)) / 60
            print(f"⏰ {source} API in backoff - {remaining_minutes:.1f} minutes remaining")
            return True
        else:
            del self.api_failure_times[source]
            print(f"✅ {source} API backoff period expired - ready to retry")
            return False
    
    def get_last_error(self) -> Optional[Dict]:
        """Get the last error for UI display"""
        if self.last_error:
            return {
                'error': self.last_error,
                'timestamp': self.last_error_time,
                'source': 'yfinance'
            }
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance. Returns None on failure."""
        if self._is_api_in_backoff('yfinance'):
            return None
        
        try:
            self._rate_limit('yfinance')
            ticker = yf.Ticker(symbol)
            
            # Try ticker.info first
            try:
                info = ticker.info
                if not info or len(info) == 0:
                    raise Exception("Empty ticker info")
                
                price_fields = ['regularMarketPrice', 'currentPrice', 'price', 'previousClose']
                for field in price_fields:
                    if field in info and info[field]:
                        price = float(info[field])
                        if 50 <= price <= 2000:
                            print(f"📊 Yahoo Finance: {symbol} = ${price:.2f}")
                            self._record_success('yfinance')
                            return price
            except Exception as info_error:
                if "429" in str(info_error):
                    self._record_api_failure('yfinance', f"Yahoo Finance rate limited: {info_error}")
                    return None
                elif "Expecting value" in str(info_error):
                    self._record_api_failure('yfinance', f"Yahoo Finance API returned empty response")
                    return None
                else:
                    print(f"⚠️ Yahoo Finance info error: {info_error}")
                    self._record_error('yfinance', str(info_error))
            
            # Fallback: get from recent history
            try:
                hist = ticker.history(period='5d', interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if 50 <= price <= 2000:
                        print(f"📊 Yahoo Finance (history): {symbol} = ${price:.2f}")
                        self._record_success('yfinance')
                        return price
                else:
                    print(f"⚠️ Yahoo Finance history data is empty for {symbol}")
            except Exception as hist_error:
                if "Expecting value" in str(hist_error):
                    self._record_api_failure('yfinance', f"Yahoo Finance history API empty response")
                    return None
                else:
                    self._record_error('yfinance', str(hist_error))
                
        except Exception as e:
            if "429" in str(e):
                self._record_api_failure('yfinance', f"Yahoo Finance rate limited")
            elif "Expecting value" in str(e):
                self._record_api_failure('yfinance', f"Yahoo Finance API service issue")
            else:
                self._record_error('yfinance', str(e))
            
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical OHLCV data from Yahoo Finance"""
        if self._is_api_in_backoff('yfinance_history'):
            return []
            
        try:
            self._rate_limit('yfinance_history')
            ticker = yf.Ticker(symbol)
            
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist.empty:
                self._record_error('yfinance_history', f"No historical data returned for {symbol}")
                return []
            
            price_data = []
            for date_idx, row in hist.iterrows():
                price_data.append({
                    'date': date_idx.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            print(f"📊 Yahoo Finance: Retrieved {len(price_data)} days of {symbol} data")
            self._record_success('yfinance_history')
            return price_data
            
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                self._record_api_failure('yfinance_history', str(e))
            else:
                self._record_error('yfinance_history', str(e))
            return []
    
    def get_intraday_data(self, symbol: str, days: int = 5, interval: str = '5m') -> List[Dict]:
        """Get intraday data from Yahoo Finance"""
        if self._is_api_in_backoff('yfinance_intraday'):
            return []
            
        try:
            self._rate_limit('yfinance_intraday')
            ticker = yf.Ticker(symbol)
            
            hist = ticker.history(period=f"{days}d", interval=interval)
            
            if hist.empty:
                self._record_error('yfinance_intraday', f"No intraday data returned for {symbol}")
                return []
            
            bars = []
            for timestamp, row in hist.iterrows():
                et_time = timestamp.tz_convert('US/Eastern')
                bars.append({
                    'time': et_time.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            print(f"📊 Yahoo Finance: Retrieved {len(bars)} {interval} bars for {symbol}")
            self._record_success('yfinance_intraday')
            return bars
            
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                self._record_api_failure('yfinance_intraday', str(e))
            else:
                self._record_error('yfinance_intraday', str(e))
            return []
    
    def create_bar_from_price(self, symbol: str, price: float) -> Dict:
        """Create a minute bar from current price"""
        now = datetime.datetime.now(pytz.timezone('US/Eastern'))
        return {
            'time': now.isoformat(),
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0
        }
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        now_et = datetime.datetime.now(pytz.timezone('US/Eastern'))
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now_et.weekday() < 5
        is_market_hours = market_open <= now_et <= market_close
        
        return {
            'is_open': is_weekday and is_market_hours,
            'current_time': now_et.isoformat(),
            'market_open': market_open.isoformat(),
            'market_close': market_close.isoformat()
        }


# Global provider instance
data_provider = MarketDataProvider()

# Keep backward-compatible name for data_stream.py import
backup_provider = data_provider


def get_tsla_price_with_fallback() -> Optional[float]:
    """
    Get TSLA price from Yahoo Finance, then Alpaca as fallback.
    Returns None if all sources fail (no hardcoded fallback prices).
    """
    from data_cache import get_cached_tsla_price, update_tsla_price_cache, is_market_open
    
    # 1. Try Yahoo Finance (primary source)
    price = data_provider.get_current_price("TSLA")
    if price is not None:
        update_tsla_price_cache(price)
        return price
    
    # 2. Try Alpaca as fallback
    try:
        import alpaca_trade_api as tradeapi
        import os
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and secret_key:
            api = tradeapi.REST(api_key, secret_key, base_url=os.getenv('ALPACA_BASE_URL'))
            quote = api.get_latest_quote('TSLA')
            if quote and hasattr(quote, 'bid') and quote.bid > 0:
                price = float(quote.bid)
                if 50 <= price <= 2000:
                    print(f"📊 Alpaca TSLA price: ${price:.2f}")
                    update_tsla_price_cache(price)
                    data_provider.last_error = None
                    data_provider.last_error_time = None
                    return price
    except Exception as e:
        print(f"⚠️ Alpaca API error: {e}")
    
    # 3. Try cache (stale data is better than nothing for display)
    cached_price = get_cached_tsla_price()
    if cached_price is not None:
        print(f"📋 Using cached TSLA price: ${cached_price:.2f}")
        return cached_price
    
    # 4. Try any cached price even if expired
    if not is_market_open():
        try:
            from data_cache import tsla_cache
            cache_data = tsla_cache.load_cache()
            cached = cache_data.get('current_price')
            if cached and 50 <= cached <= 2000:
                return cached
        except Exception:
            pass
    
    # 5. All sources failed - return None (UI will show error)
    print("❌ All data sources failed - no price available")
    data_provider._record_error('all_sources', "All data sources failed: Yahoo Finance and Alpaca both unavailable")
    return None


def get_data_source_error() -> Optional[Dict]:
    """Get current data source error for UI display"""
    return data_provider.get_last_error()


def get_tsla_historical_data(days: int = 5) -> List[Dict]:
    """Get TSLA historical data for charting with caching"""
    from data_cache import get_cached_tsla_daily_data, update_tsla_daily_cache
    
    cached_data = get_cached_tsla_daily_data()
    if cached_data:
        print(f"📋 Using cached TSLA daily data: {len(cached_data)} days")
        return cached_data
    
    print(f"🔍 Fetching {days} days of TSLA historical data...")
    historical_data = data_provider.get_historical_data("TSLA", days)
    
    if historical_data:
        update_tsla_daily_cache(historical_data)
    
    return historical_data


def get_tsla_intraday_bars(days: int = 1, interval: str = '5m') -> List[Dict]:
    """Get TSLA intraday bars for ORB strategy"""
    return data_provider.get_intraday_data("TSLA", days, interval)


def get_tsla_minute_candlesticks(limit: int = 100) -> List[Dict]:
    """Get recent TSLA 1-minute candlesticks for table display"""
    try:
        bars = data_provider.get_intraday_data("TSLA", days=1)
        bars.sort(key=lambda x: x['time'], reverse=True)
        return bars[:limit]
    except Exception as e:
        print(f"Error getting minute candlesticks: {e}")
        return []


if __name__ == "__main__":
    print("Testing Yahoo Finance data provider...")
    
    provider = MarketDataProvider()
    
    print("\n1. Testing current price...")
    tsla_price = provider.get_current_price("TSLA")
    if tsla_price:
        print(f"Current TSLA price: ${tsla_price}")
    else:
        print(f"Failed to get price. Error: {provider.get_last_error()}")
    
    print("\n2. Testing historical data...")
    historical = provider.get_historical_data("TSLA", 7)
    print(f"Historical data points: {len(historical)}")
    
    print("\n3. Testing market status...")
    status = provider.get_market_status()
    print(f"Market status: {status}")
    
    print("\n✅ Testing complete!")