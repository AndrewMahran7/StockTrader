"""
Simple Data Cache for TSLA Price Data
===================================

Reduces API calls by caching recent price data locally.
Only fetches new data when needed.
"""

import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import pytz


class TSLADataCache:
    """Simple file-based cache for TSLA price data"""
    
    def __init__(self, cache_file: str = "tsla_price_cache.json"):
        self.cache_file = cache_file
        self.eastern_tz = pytz.timezone('US/Eastern')
    
    def load_cache(self) -> Dict:
        """Load cached data from file"""
        if not os.path.exists(self.cache_file):
            return {
                'last_updated': None,
                'current_price': None,
                'daily_data': [],
                'last_trading_day': None
            }
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {
                'last_updated': None,
                'current_price': None,
                'daily_data': [],
                'last_trading_day': None
            }
    
    def save_cache(self, cache_data: Dict):
        """Save cache data to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now_et = datetime.now(self.eastern_tz)
        
        # Debug: Print current time info
        print(f"🕐 Current ET time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"📅 Weekday: {now_et.weekday()} (0=Mon, 6=Sun)")
        
        # Check if it's a weekday
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            print(f"🚫 Market closed - Weekend")
            return False
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = market_open <= now_et <= market_close
        if is_open:
            print(f"✅ Market OPEN - {now_et.strftime('%H:%M')} is between 09:30-16:00")
        else:
            print(f"🚫 Market CLOSED - {now_et.strftime('%H:%M')} is outside 09:30-16:00")
        
        return is_open
    
    def should_update_price(self, cache_data: Dict) -> bool:
        """Determine if we need to update the current price"""
        if not cache_data.get('last_updated'):
            return True
        
        try:
            last_updated = datetime.fromisoformat(cache_data['last_updated'])
            now = datetime.now(self.eastern_tz)
            
            # Since Alpaca provides live data, be much more conservative with backup sources
            if self.is_market_open():
                # Only update backup data every 10 minutes during market hours (Alpaca handles real-time)
                return (now - last_updated).total_seconds() > 600  # 10 minutes
            
            # Update if cache is older than 8 hours outside market hours
            return (now - last_updated).total_seconds() > 28800  # 8 hours
            
        except Exception:
            return True
    
    def should_update_daily_data(self, cache_data: Dict) -> bool:
        """Determine if we need to update daily historical data"""
        if not cache_data.get('daily_data'):
            return True
        
        try:
            # Check if we have today's data
            today = date.today().isoformat()
            dates = [item['date'] for item in cache_data['daily_data']]
            
            # If we don't have today's data and market is open, update
            if today not in dates and self.is_market_open():
                return True
            
            # If cache is older than 1 day, update
            if cache_data.get('last_trading_day') != today:
                return True
                
            return False
            
        except Exception:
            return True
    
    def get_cached_price(self) -> Optional[float]:
        """Get cached current price if still valid"""
        cache_data = self.load_cache()
        
        # Validate cached price is reasonable for TSLA (reject web scraping errors)
        current_price = cache_data.get('current_price')
        if current_price and (current_price < 50 or current_price > 2000):
            print(f"⚠️ Cached price ${current_price:.2f} seems wrong, forcing refresh")
            return None
        
        if not self.should_update_price(cache_data):
            return current_price
        
        return None
    
    def update_current_price(self, price: float):
        """Update cached current price"""
        cache_data = self.load_cache()
        cache_data['current_price'] = price
        cache_data['last_updated'] = datetime.now(self.eastern_tz).isoformat()
        self.save_cache(cache_data)
    
    def get_cached_daily_data(self) -> List[Dict]:
        """Get cached daily data if still valid"""
        cache_data = self.load_cache()
        
        if not self.should_update_daily_data(cache_data):
            return cache_data.get('daily_data', [])
        
        return []
    
    def update_daily_data(self, daily_data: List[Dict]):
        """Update cached daily data"""
        cache_data = self.load_cache()
        cache_data['daily_data'] = daily_data
        cache_data['last_trading_day'] = date.today().isoformat()
        cache_data['last_updated'] = datetime.now(self.eastern_tz).isoformat()
        self.save_cache(cache_data)
    
    def add_today_data(self, price_data: Dict):
        """Add or update today's price data"""
        cache_data = self.load_cache()
        daily_data = cache_data.get('daily_data', [])
        
        today = date.today().isoformat()
        
        # Find if today's data already exists
        today_index = None
        for i, data in enumerate(daily_data):
            if data['date'] == today:
                today_index = i
                break
        
        if today_index is not None:
            # Update existing data
            daily_data[today_index] = price_data
        else:
            # Add new data
            daily_data.append(price_data)
        
        # Keep only last 10 days to prevent cache from growing too large
        daily_data = sorted(daily_data, key=lambda x: x['date'])[-10:]
        
        cache_data['daily_data'] = daily_data
        self.save_cache(cache_data)


# Global cache instance
tsla_cache = TSLADataCache()


def get_cached_tsla_price() -> Optional[float]:
    """Get TSLA price from cache if valid, None if needs update"""
    return tsla_cache.get_cached_price()


def update_tsla_price_cache(price: float):
    """Update TSLA price in cache"""
    tsla_cache.update_current_price(price)


def get_cached_tsla_daily_data() -> List[Dict]:
    """Get TSLA daily data from cache if valid, empty list if needs update"""
    return tsla_cache.get_cached_daily_data()


def update_tsla_daily_cache(daily_data: List[Dict]):
    """Update TSLA daily data in cache"""
    tsla_cache.update_daily_data(daily_data)


def is_market_open() -> bool:
    """Check if market is currently open"""
    return tsla_cache.is_market_open()


if __name__ == "__main__":
    # Test cache functionality
    cache = TSLADataCache()
    
    print(f"Market open: {cache.is_market_open()}")
    
    # Test price cache
    cached_price = cache.get_cached_price()
    print(f"Cached price: {cached_price}")
    
    if cached_price is None:
        print("Price cache needs update")
    
    # Test daily data cache
    cached_daily = cache.get_cached_daily_data()
    print(f"Cached daily data points: {len(cached_daily)}")