"""
Backup Data Provider - Free Market Data Sources
==============================================

This module provides backup data sources when Alpaca SIP data is unavailable.
Uses Yahoo Finance and web scraping for free, real-time market data.
"""

import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import pytz
from typing import Dict, List, Optional, Tuple
import time
import logging

class BackupDataProvider:
    """Backup data provider using free sources when Alpaca fails"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        # Set session timeout and retry settings
        self.session.timeout = 15
        self.last_request_time = {}
        self.min_request_interval = 2.0  # 2 seconds between requests (slower to avoid rate limits)
        self.error_backoff = {}  # Track consecutive errors for exponential backoff
        self.api_failure_times = {}  # Track when APIs fail to implement longer backoffs
        
    def _rate_limit(self, source: str):
        """Enhanced rate limiting with exponential backoff for errors"""
        current_time = time.time()
        
        # Calculate wait time (base interval + exponential backoff for errors)
        base_interval = self.min_request_interval
        error_count = self.error_backoff.get(source, 0)
        
        # Exponential backoff: 2s, 4s, 8s, 16s, max 60s
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
    
    def _record_error(self, source: str):
        """Record an error for exponential backoff"""
        self.error_backoff[source] = self.error_backoff.get(source, 0) + 1
        print(f"⚠️ Error count for {source}: {self.error_backoff[source]}")
    
    def _record_success(self, source: str):
        """Record success and reset error count"""
        if source in self.error_backoff:
            del self.error_backoff[source]
        # Also reset API failure time on success
        if source in self.api_failure_times:
            del self.api_failure_times[source]
    
    def _record_api_failure(self, source: str):
        """Record an API failure for longer backoff (1 hour)"""
        self.api_failure_times[source] = time.time()
        print(f"🚫 API failure recorded for {source} - will use fallbacks for 1 hour")
    
    def _is_api_in_backoff(self, source: str) -> bool:
        """Check if API is in 1-hour backoff period"""
        if source not in self.api_failure_times:
            return False
        
        failure_time = self.api_failure_times[source]
        current_time = time.time()
        backoff_duration = 3600  # 1 hour in seconds
        
        if current_time - failure_time < backoff_duration:
            remaining_minutes = (backoff_duration - (current_time - failure_time)) / 60
            print(f"⏰ {source} API in backoff - {remaining_minutes:.1f} minutes remaining")
            return True
        else:
            # Backoff period expired, remove the failure record
            del self.api_failure_times[source]
            print(f"✅ {source} API backoff period expired - ready to retry")
            return False
    
    def get_current_price_yfinance(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance with better error handling"""
        
        # Check if Yahoo Finance API is in 1-hour backoff period
        if self._is_api_in_backoff('yfinance'):
            return None
        
        try:
            self._rate_limit('yfinance')
            ticker = yf.Ticker(symbol)
            
            # Try to get basic info first (faster)
            try:
                info = ticker.info
                # Check if we got valid data (not empty)
                if not info or len(info) == 0:
                    print(f"⚠️ Yahoo Finance returned empty data for {symbol}")
                    raise Exception("Empty ticker info")
                
                # Try multiple price fields in order of preference
                price_fields = ['regularMarketPrice', 'currentPrice', 'price', 'previousClose']
                for field in price_fields:
                    if field in info and info[field]:
                        price = float(info[field])
                        if 50 <= price <= 2000:  # Reasonable TSLA price range
                            print(f"📊 Yahoo Finance: {symbol} = ${price:.2f}")
                            self._record_success('yfinance')
                            return price
            except Exception as info_error:
                if "429" in str(info_error):
                    print(f"⚠️ Yahoo Finance rate limited, backing off...")
                    self._record_api_failure('yfinance')  # Use longer backoff for rate limits
                    return None
                elif "Expecting value" in str(info_error):
                    print(f"⚠️ Yahoo Finance API returned empty response for {symbol}")
                    self._record_api_failure('yfinance')  # API service issue - use longer backoff
                    return None
                else:
                    print(f"⚠️ Yahoo Finance info error: {info_error}")
                    self._record_error('yfinance')  # Regular error - use exponential backoff
            
            # Fallback: get from recent history (last trading day)
            try:
                hist = ticker.history(period='5d', interval='1d')  # Get last 5 days
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if 50 <= price <= 2000:  # Reasonable TSLA price range
                        print(f"📊 Yahoo Finance (history): {symbol} = ${price:.2f}")
                        self._record_success('yfinance')
                        return price
                else:
                    print(f"⚠️ Yahoo Finance history data is empty for {symbol}")
            except Exception as hist_error:
                if "Expecting value" in str(hist_error):
                    print(f"⚠️ Yahoo Finance history API returned empty response for {symbol}")
                    self._record_api_failure('yfinance')  # API service issue - use longer backoff
                    return None
                else:
                    print(f"⚠️ Yahoo Finance history error: {hist_error}")
                    self._record_error('yfinance')  # Regular error - use exponential backoff
                
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Yahoo Finance rate limited, backing off...")
                self._record_api_failure('yfinance')  # Rate limit - use longer backoff
            elif "Expecting value" in str(e):
                print(f"⚠️ Yahoo Finance API service issue - empty JSON response for {symbol}")
                self._record_api_failure('yfinance')  # API service issue - use longer backoff
            else:
                print(f"⚠️ Yahoo Finance error: {e}")
                self._record_error('yfinance')  # Regular error - use exponential backoff
            
        return None
    
    def get_current_price_webscrape(self, symbol: str) -> Optional[float]:
        """Get current price via web scraping multiple financial websites"""
        scrapers = [
            self._scrape_stockscan,
            self._scrape_yahoo_finance,
            self._scrape_google_finance,
            self._scrape_marketwatch
        ]
        
        for scraper in scrapers:
            try:
                self._rate_limit('webscraping')
                price = scraper(symbol)
                if price and self._validate_price(price, symbol):
                    self._record_success('webscraping')
                    return price
            except Exception as e:
                print(f"⚠️ Web scraping error with {scraper.__name__}: {e}")
                continue
        
        self._record_error('webscraping')
        return None
    
    def _validate_price(self, price: float, symbol: str) -> bool:
        """Validate if the scraped price is reasonable for the given symbol"""
        if symbol == "TSLA":
            # TSLA typically trades between $50-$2000
            return 50.0 <= price <= 2000.0
        else:
            # General validation for other stocks
            return 0.01 <= price <= 50000.0
    
    def _scrape_stockscan(self, symbol: str) -> Optional[float]:
        """Scrape stockscan.io for stock price - PRIMARY FALLBACK SOURCE"""
        try:
            url = f"https://stockscan.io/stocks/{symbol}"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # StockScan.io price selectors (based on observed HTML structure)
            price_selectors = [
                # Main price display near the top
                'div:contains("NASDAQ:") + *',  # Look for price near NASDAQ: TSLA
                # Try to find price in various containers
                'span:contains("$")',
                'div[class*="price"]',
                # Look for numeric values that could be the price
                'div:contains("450")',  # Current approximate TSLA price range
                'span:contains("450")',
                # Fallback selectors
                '*[class*="stock-price"]',
                '*[data-price]',
                'h1:contains("$")',
                'h2:contains("$")',
                'h3:contains("$")'
            ]
            
            for selector in price_selectors:
                try:
                    # Find elements that might contain the price
                    elements = soup.select(selector) if ':contains(' not in selector else soup.find_all(string=lambda text: text and '$' in str(text))
                    
                    if ':contains(' in selector:
                        # Handle text-based selectors
                        for text in elements:
                            if text and '$' in str(text):
                                # Extract price from text like "$450.07"
                                import re
                                price_match = re.search(r'\$?([0-9,]+\.?[0-9]*)', str(text))
                                if price_match:
                                    price_text = price_match.group(1).replace(',', '')
                                    try:
                                        price = float(price_text)
                                        if self._validate_price(price, symbol):
                                            print(f"📊 StockScan.io (scraped): {symbol} = ${price:.2f}")
                                            return price
                                    except ValueError:
                                        continue
                    else:
                        # Handle CSS selector elements
                        for element in elements:
                            if element:
                                price_text = element.get_text(strip=True)
                                # Clean and extract price
                                import re
                                price_match = re.search(r'\$?([0-9,]+\.?[0-9]*)', price_text)
                                if price_match:
                                    clean_price = price_match.group(1).replace(',', '')
                                    try:
                                        price = float(clean_price)
                                        if self._validate_price(price, symbol):
                                            print(f"📊 StockScan.io (scraped): {symbol} = ${price:.2f}")
                                            return price
                                    except ValueError:
                                        continue
                except Exception as selector_error:
                    continue
            
            # Alternative approach: look for the specific price pattern in page text
            page_text = soup.get_text()
            import re
            
            # Look for price patterns like "$450.07" in the page
            price_patterns = [
                # Most specific patterns first
                r'NASDAQ:\s*' + symbol + r'\s+([0-9,]+\.?[0-9]*)',  # "NASDAQ: TSLA 450.07"
                rf'{symbol}\s+([0-9,]+\.?[0-9]*)',  # "TSLA 450.07"
                r'([0-9]{3}\.[0-9]{2})\s*![^0-9]',  # "450.07 !" (price followed by icon)
                r'\$([0-9]{3}\.[0-9]{2})',  # "$450.07" 
                # Broader TSLA price range patterns
                r'([4-5][0-9]{2}\.[0-9]{2})',  # Any price in 400-599 range
                r'([2-9][0-9]{2}\.[0-9]{2})',  # Broader range 200-999
            ]
            
            print(f"🔍 Searching StockScan.io page text for {symbol} price patterns...")
            
            for i, pattern in enumerate(price_patterns):
                try:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    print(f"  Pattern {i+1}: Found {len(matches)} matches")
                    
                    for match in matches:
                        try:
                            # Clean the match (remove $ and commas if present)
                            clean_price = str(match).replace('$', '').replace(',', '').strip()
                            price = float(clean_price)
                            
                            if self._validate_price(price, symbol):
                                print(f"📊 StockScan.io (pattern {i+1}): {symbol} = ${price:.2f}")
                                return price
                            else:
                                print(f"  Rejected price ${price:.2f} (outside valid range)")
                        except (ValueError, AttributeError) as e:
                            print(f"  Failed to parse match '{match}': {e}")
                            continue
                except Exception as pattern_error:
                    print(f"  Pattern {i+1} error: {pattern_error}")
                    continue
                        
        except Exception as e:
            print(f"⚠️ StockScan.io scraping error: {e}")
        
        return None
    
    def _scrape_yahoo_finance(self, symbol: str) -> Optional[float]:
        """Scrape Yahoo Finance stock page"""
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for the current price
            price_selectors = [
                f'[data-symbol="{symbol}"][data-field="regularMarketPrice"]',
                'fin-streamer[data-field="regularMarketPrice"]',
                '[data-reactid*="YFINANCE:{}"]'.format(symbol),
                'span[class*="Trsdu(0.3s)"]',
                'fin-streamer[class*="Fw(b)"]'
            ]
            
            for selector in price_selectors:
                try:
                    price_element = soup.select_one(selector)
                    if price_element:
                        price_text = price_element.get_text(strip=True)
                        # Remove commas and extract numeric value
                        price_text = price_text.replace(',', '').replace('$', '')
                        price = float(price_text)
                        print(f"📊 Yahoo Finance (scraped): {symbol} = ${price:.2f}")
                        return price
                except (ValueError, AttributeError):
                    continue
                    
        except Exception as e:
            print(f"⚠️ Yahoo Finance scraping error: {e}")
        
        return None
    
    def _scrape_google_finance(self, symbol: str) -> Optional[float]:
        """Scrape Google Finance for stock price"""
        try:
            url = f"https://www.google.com/finance/quote/{symbol}:NASDAQ"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Google Finance price selectors
            price_selectors = [
                'div[class*="YMlKec fxKbKc"]',
                'div[class*="YMlKec"]',
                'span[class*="IsqQVc NprOob XcVN5d"]',
                'div[data-last-price]'
            ]
            
            for selector in price_selectors:
                try:
                    price_element = soup.select_one(selector)
                    if price_element:
                        price_text = price_element.get_text(strip=True)
                        # Extract price from text like "$446.66" or "446.66"
                        price_text = price_text.replace('$', '').replace(',', '').split()[0]
                        price = float(price_text)
                        print(f"📊 Google Finance (scraped): {symbol} = ${price:.2f}")
                        return price
                except (ValueError, AttributeError, IndexError):
                    continue
                    
        except Exception as e:
            print(f"⚠️ Google Finance scraping error: {e}")
        
        return None
    
    def _scrape_marketwatch(self, symbol: str) -> Optional[float]:
        """Scrape MarketWatch for stock price"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # MarketWatch price selectors
            price_selectors = [
                'h2[class*="intraday__price"] bg-quote',
                'bg-quote[class*="value"]',
                'span[class*="value"]',
                'h3[class*="intraday__price"]'
            ]
            
            for selector in price_selectors:
                try:
                    price_element = soup.select_one(selector)
                    if price_element:
                        price_text = price_element.get_text(strip=True)
                        # Clean up price text
                        price_text = price_text.replace('$', '').replace(',', '')
                        price = float(price_text)
                        print(f"📊 MarketWatch (scraped): {symbol} = ${price:.2f}")
                        return price
                except (ValueError, AttributeError):
                    continue
                    
        except Exception as e:
            print(f"⚠️ MarketWatch scraping error: {e}")
        
        return None
    
    def get_historical_data_yfinance(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical OHLCV data from Yahoo Finance"""
        
        # Check if Yahoo Finance API is in 1-hour backoff period
        if self._is_api_in_backoff('yfinance_history'):
            return []
            
        try:
            self._rate_limit('yfinance_history')
            ticker = yf.Ticker(symbol)
            
            # Get last N days of data
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist.empty:
                return []
            
            # Convert to our format
            price_data = []
            for date, row in hist.iterrows():
                price_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            print(f"📊 Yahoo Finance: Retrieved {len(price_data)} days of {symbol} data")
            return price_data
            
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                print(f"⚠️ Yahoo Finance historical data API issue: {e}")
                self._record_api_failure('yfinance_history')  # API issue - use longer backoff
            else:
                print(f"⚠️ Yahoo Finance historical data error: {e}")
                self._record_error('yfinance_history')  # Regular error
            return []
    
    def get_intraday_data_yfinance(self, symbol: str, days: int = 5, interval: str = '5m') -> List[Dict]:
        """Get intraday data from Yahoo Finance with configurable intervals"""
        
        # Check if Yahoo Finance API is in 1-hour backoff period
        if self._is_api_in_backoff('yfinance_intraday'):
            return []
            
        try:
            self._rate_limit('yfinance_intraday')
            ticker = yf.Ticker(symbol)
            
            # Get intraday data (last few days) with specified interval
            # Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            hist = ticker.history(period=f"{days}d", interval=interval)
            
            if hist.empty:
                return []
            
            # Convert to bar format
            bars = []
            for timestamp, row in hist.iterrows():
                # Convert to Eastern time for consistency
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
            return bars
            
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                print(f"⚠️ Yahoo Finance intraday API issue: {e}")
                self._record_api_failure('yfinance_intraday')  # API issue - use longer backoff
            else:
                print(f"⚠️ Yahoo Finance intraday data error: {e}")
                self._record_error('yfinance_intraday')  # Regular error
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with fallback sources"""
        print(f"🔍 Getting backup price data for {symbol}...")
        
        # Try Yahoo Finance API first (most reliable)
        price = self.get_current_price_yfinance(symbol)
        if price:
            return price
        
        print(f"⚠️ Yahoo Finance API failed, trying web scraping...")
        
        # Fallback to web scraping multiple sources
        price = self.get_current_price_webscrape(symbol)
        if price:
            return price
            
        print(f"❌ Could not get price for {symbol} from any backup sources")
        return None
    
    def create_bar_from_price(self, symbol: str, price: float) -> Dict:
        """Create a minute bar from current price (for real-time updates)"""
        now = datetime.datetime.now(pytz.timezone('US/Eastern'))
        
        return {
            'time': now.isoformat(),
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0  # Unknown volume
        }
    
    def get_market_status(self) -> Dict:
        """Get market status from Yahoo Finance"""
        try:
            # Use SPY as a proxy for market status
            ticker = yf.Ticker("SPY")
            info = ticker.info
            
            # Check if market is open
            now_et = datetime.datetime.now(pytz.timezone('US/Eastern'))
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_weekday = now_et.weekday() < 5  # Monday=0, Sunday=6
            is_market_hours = market_open <= now_et <= market_close
            
            return {
                'is_open': is_weekday and is_market_hours,
                'current_time': now_et.isoformat(),
                'market_open': market_open.isoformat(),
                'market_close': market_close.isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ Market status error: {e}")
            return {'is_open': False, 'error': str(e)}


# Global backup provider instance
backup_provider = BackupDataProvider()


def get_tsla_price_with_fallback() -> float:
    """
    Get TSLA price with Yahoo Finance first (most accurate), then Alpaca and other fallbacks
    Returns the most recent price available
    """
    from data_cache import get_cached_tsla_price, update_tsla_price_cache, is_market_open
    
    # 1. First try Yahoo Finance and web scraping (most accurate during market hours)
    if is_market_open():
        print("🔍 Market is open - trying backup sources for fresh TSLA data...")
        price = backup_provider.get_current_price("TSLA")
        
        if price is not None:
            update_tsla_price_cache(price)  # Cache the good price
            return price
    
    # 2. Try Alpaca if Yahoo Finance fails or market is closed
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
                if 50 <= price <= 2000:  # Validate reasonable TSLA price
                    print(f"📊 Fallback Alpaca TSLA price: ${price:.2f}")
                    update_tsla_price_cache(price)  # Cache the good price
                    return price
    except Exception as e:
        print(f"⚠️ Alpaca API error: {e}")
    
    # 3. If both Yahoo and Alpaca fail, fall back to cached data
    print("⚠️ All live sources failed - falling back to cached data")
    
    # 3. Check cache if market is closed or backup sources failed
    cached_price = get_cached_tsla_price()
    if cached_price is not None:
        print(f"📋 Using cached TSLA price: ${cached_price:.2f}")
        return cached_price
    
    # 4. Market closed fallback - try any cached price but validate it
    if not is_market_open():
        print("🕐 Market closed - checking for any valid cached price")
        try:
            from data_cache import tsla_cache
            cache_data = tsla_cache.load_cache()
            cached = cache_data.get('current_price')
            if cached and 50 <= cached <= 2000:  # Validate cached price
                return cached
        except Exception:
            pass
    
    # 5. Final fallback - use reasonable recent price
    print("⚠️ Using final fallback TSLA price")
    return 450.07  # Updated to match recent stockscan.io price


def get_tsla_historical_data(days: int = 5) -> List[Dict]:
    """Get TSLA historical data for charting with caching"""
    from data_cache import get_cached_tsla_daily_data, update_tsla_daily_cache, is_market_open
    
    # Check cache first
    cached_data = get_cached_tsla_daily_data()
    if cached_data:
        print(f"📋 Using cached TSLA daily data: {len(cached_data)} days")
        return cached_data
    
    print(f"🔍 Fetching {days} days of TSLA historical data...")
    
    # Fetch fresh data (reduced days for efficiency)
    historical_data = backup_provider.get_historical_data_yfinance("TSLA", days)
    
    # Update cache
    if historical_data:
        update_tsla_daily_cache(historical_data)
    
    return historical_data


def get_tsla_intraday_bars(days: int = 1, interval: str = '5m') -> List[Dict]:
    """Get TSLA intraday bars for ORB strategy (default 5-minute intervals)"""
    return backup_provider.get_intraday_data_yfinance("TSLA", days, interval)


def get_tsla_minute_candlesticks(limit: int = 100) -> List[Dict]:
    """Get recent TSLA 1-minute candlesticks for table display"""
    try:
        # Get recent intraday data
        bars = backup_provider.get_intraday_data_yfinance("TSLA", days=1)
        
        # Sort by time (newest first) and limit results
        bars.sort(key=lambda x: x['time'], reverse=True)
        
        # Return limited number of most recent bars
        return bars[:limit]
        
    except Exception as e:
        print(f"Error getting minute candlesticks: {e}")
        return []


if __name__ == "__main__":
    # Test the backup data provider
    print("Testing backup data provider...")
    
    provider = BackupDataProvider()
    
    # Test current price (includes web scraping fallback)
    print("\n1. Testing current price with all fallbacks...")
    tsla_price = provider.get_current_price("TSLA")
    print(f"Current TSLA price: ${tsla_price}")
    
    # Test web scraping directly
    print("\n2. Testing web scraping fallback directly...")
    scraped_price = provider.get_current_price_webscrape("TSLA")
    print(f"Web scraped TSLA price: ${scraped_price}")
    
    # Test historical data
    print("\n3. Testing historical data...")
    historical = provider.get_historical_data_yfinance("TSLA", 7)
    print(f"Historical data points: {len(historical)}")
    
    # Test market status
    print("\n4. Testing market status...")
    status = provider.get_market_status()
    print(f"Market status: {status}")
    
    print("\n✅ Testing complete!")