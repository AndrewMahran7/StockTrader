from flask import Flask, request, jsonify, render_template
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, date, time, timedelta
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
import threading
import numpy as np
from scipy import stats
import pytz
import logging

# Import our config modules
from secure_config import Config, create_env_file

# Import our strategy components
from strategy_engine import ORBStrategy
from data_stream import DataStream, BacktestDataProvider
from backup_data_provider import get_tsla_price_with_fallback, get_tsla_historical_data

app = Flask(__name__)

# Price cache to ensure consistency across endpoints during refresh
price_cache = {'price': None, 'timestamp': None}
PRICE_CACHE_SECONDS = 2  # Cache price for 2 seconds

# Initialize security configuration
create_env_file()
config_errors = Config.validate_config()
if config_errors:
    print("Configuration errors:")
    for error in config_errors:
        print(f"  - {error}")
    print("Please check your .env file and environment variables.")
    exit(1)

app.secret_key = Config.SECRET_KEY

def get_cached_tsla_price():
    """Get TSLA price with short-term caching to ensure consistency across endpoints"""
    global price_cache
    now = datetime.now()
    
    # Check if cache is valid (within cache period)
    if price_cache['price'] is not None and price_cache['timestamp'] is not None:
        age = (now - price_cache['timestamp']).total_seconds()
        if age < PRICE_CACHE_SECONDS:
            return price_cache['price']
    
    # Cache expired or empty, fetch new price
    price = get_tsla_price_with_fallback()
    if price:
        price_cache['price'] = price
        price_cache['timestamp'] = now
        return price
    
    # Fallback if fetch fails but cache exists
    if price_cache['price'] is not None:
        return price_cache['price']
    
    return 395.94  # Final fallback

# Flask-Login removed for local development

# Rate limiting removed for local development

# Security headers removed for local development

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_security.log'),
        logging.StreamHandler()
    ]
)

# Filter out SSL handshake errors from werkzeug
class SSLErrorFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # Filter out SSL/TLS handshake errors more comprehensively
        ssl_patterns = [
            'Bad request version',
            'Bad HTTP/0.9 request type', 
            'Bad request syntax',
            '\\x16\\x03\\x01',  # SSL handshake signature
            'code 400',
            '\\x03\\x03',
            'localhost\\x00',
            '\\x16\\x03',
            'http/1.1\\x00'
        ]
        
        # Check if message contains any SSL-related patterns
        for pattern in ssl_patterns:
            if pattern in message:
                return False
        
        # Also filter messages that are mostly non-printable characters (SSL data)
        if len([c for c in message if ord(c) < 32 or ord(c) > 126]) > len(message) * 0.3:
            return False
            
        return True

# Apply filter to werkzeug logger
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(SSLErrorFilter())

# Configuration file path (legacy support)
CONFIG_FILE = "sessions_config.json"

def load_sessions_config():
    """Load session configurations - now from secure config"""
    try:
        # Use secure configuration instead of JSON file
        return Config.get_alpaca_config()
    except Exception as e:
        print(f"Error loading secure configuration: {e}")
        return {"sessions": {}, "default_session": "main"}

def initialize_sessions():
    """Initialize all API sessions from configuration"""
    config = load_sessions_config()
    sessions = {}
    
    for session_id, session_config in config["sessions"].items():
        if session_config.get("enabled", True):
            try:
                api = REST(
                    session_config["api_key"], 
                    session_config["api_secret"], 
                    session_config["base_url"]
                )
                sessions[session_id] = {
                    "api": api,
                    "config": session_config
                }
                print(f"Initialized session '{session_id}' ({session_config['name']})")
            except Exception as e:
                print(f"Failed to initialize session '{session_id}': {e}")
        else:
            print(f"⏸️ Session '{session_id}' is disabled")
    
    return sessions, config.get("default_session", "session1")

# Initialize all sessions
SESSIONS, DEFAULT_SESSION = initialize_sessions()

# Strategy and data stream instances
strategy_engine = None
data_stream = None
strategy_thread_started = False

def load_strategy_config():
    """Load strategy configuration from sessions config"""
    config = load_sessions_config()
    return config.get("strategy", {
        "enabled": True,
        "symbol": "TSLA",
        "polling_seconds": 60,
        "use_streaming": False  # Disabled due to websocket compatibility issues
    })

def check_account_limits(api):
    """Check Alpaca account limits and status"""
    try:
        account = api.get_account()
        print(f"📊 Account Status: {account.status}")
        print(f"   Account Type: {'Paper' if 'paper' in str(api._base_url) else 'Live'}")
        
        # Get current positions to verify connection
        positions = api.list_positions()
        print(f"   Current Positions: {len(positions)}")
        
    except Exception as e:
        print(f"⚠️ Error checking account: {e}")

def initialize_strategy():
    """Initialize the ORB strategy engine and data stream"""
    global strategy_engine, data_stream, strategy_thread_started
    
    if strategy_thread_started:
        return
    
    try:
        strategy_config = load_strategy_config()
        
        if not strategy_config.get("enabled", True):
            print("⏸️ Strategy is disabled in configuration")
            return
        
        # Get API for default session
        if DEFAULT_SESSION not in SESSIONS:
            print("No valid session for strategy")
            return
            
        api = SESSIONS[DEFAULT_SESSION]["api"]
        session_name = DEFAULT_SESSION
        
        # Check account limits and status
        check_account_limits(api)
        
        # Initialize strategy engine
        strategy_engine = ORBStrategy(
            api=api,
            session_name=session_name,
            symbol=strategy_config.get("symbol", "TSLA"),
            log_trade_func=log_trade
        )
        
        # Initialize data stream with websockets disabled (compatibility issue)
        data_stream = DataStream(
            api=api,
            symbol=strategy_config.get("symbol", "TSLA"),
            on_bar_callback=strategy_engine.on_bar,
            use_streaming=False,  # Force disable websockets
            polling_seconds=strategy_config.get("polling_seconds", 60)
        )
        
        # Start data stream in background thread
        def start_stream():
            try:
                data_stream.start()
            except Exception as e:
                print(f"Error starting data stream: {e}")
        
        stream_thread = threading.Thread(target=start_stream, daemon=True)
        stream_thread.start()
        
        strategy_thread_started = True
        print(f"ORB Strategy initialized and running for {strategy_config.get('symbol', 'TSLA')}")
        
    except Exception as e:
        print(f"Error initializing strategy: {e}")

def ensure_analytics_directory():
    """Create analytics directory if it doesn't exist"""
    os.makedirs("analytics", exist_ok=True)

def compute_analytics(session_id=None):
    """Compute comprehensive trading analytics"""
    try:
        ensure_analytics_directory()
        
        if session_id is None:
            session_id = DEFAULT_SESSION
            
        log_file = get_log_file_path(session_id)
        
        if not os.path.exists(log_file):
            print("No trades log found for analytics")
            return
            
        # Read trades data
        df = pd.read_csv(log_file)
        
        if df.empty:
            print("No trades found for analytics")
            return
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate total deposits (initial capital + any additional deposits)
        initial_capital = 1000.0  # Default starting capital
        deposits = df[df['action'] == 'deposit']
        if not deposits.empty:
            total_deposits = deposits['price'].sum()  # 'price' field holds deposit amount
            initial_capital = total_deposits
            print(f"💰 Total deposits: ${total_deposits:.2f}")
        
        # Filter for completed trades (both entry and exit)
        # Handle multiple buys that get sold together (averaging up/down)
        completed_trades = []
        open_positions = {}
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            action = row['action']
            
            if action == 'buy':
                # Track each buy separately (can have multiple buys for same ticker)
                if ticker not in open_positions:
                    open_positions[ticker] = []
                    
                open_positions[ticker].append({
                    'entry_time': row['timestamp'],
                    'entry_price': row['price'],
                    'quantity': row['quantity'],
                    'side': 'long'
                })
                
            elif action == 'sell' and ticker in open_positions:
                # Closing position(s) - match sells with buys using FIFO
                remaining_sell_qty = row['quantity']
                exit_price = row['price']
                exit_time = row['timestamp']
                
                while remaining_sell_qty > 0 and open_positions[ticker]:
                    entry = open_positions[ticker].pop(0)  # FIFO - first buy out first
                    
                    # Calculate quantity for this trade (handle partial fills)
                    trade_qty = min(entry['quantity'], remaining_sell_qty)
                    
                    # Calculate PNL for long position
                    pnl = (exit_price - entry['entry_price']) * trade_qty
                    pnl_pct = (pnl / (entry['entry_price'] * trade_qty)) * 100
                    
                    trade_days = (exit_time - entry['entry_time']).days
                    
                    completed_trades.append({
                        'ticker': ticker,
                        'entry_time': entry['entry_time'],
                        'exit_time': exit_time,
                        'entry_price': entry['entry_price'],
                        'exit_price': exit_price,
                        'quantity': trade_qty,
                        'side': entry['side'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'days_held': trade_days,
                        'mfe': 0,  # TODO: Calculate from minute data
                        'mae': 0   # TODO: Calculate from minute data
                    })
                    
                    print(f"✅ Completed trade: {ticker} - Entry: ${entry['entry_price']:.2f} ({trade_qty:.4f} shares), Exit: ${exit_price:.2f}, P&L: ${pnl:.2f}")
                    
                    remaining_sell_qty -= trade_qty
                    
                    # If we used only part of the buy, put the remainder back
                    if entry['quantity'] > trade_qty:
                        entry['quantity'] -= trade_qty
                        open_positions[ticker].insert(0, entry)
                        break
                
                # Clean up empty position lists
                if ticker in open_positions and not open_positions[ticker]:
                    del open_positions[ticker]
        
        if not completed_trades:
            print("⚠️ No completed round-trip trades found (need both buy AND sell)")
            # Still save empty analytics
            summary = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'start_date': date.today().isoformat(),
                'end_date': date.today().isoformat(),
                'symbol': 'TSLA'
            }
            with open('analytics/summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            return
            
        trades_df = pd.DataFrame(completed_trades)
        
        # Calculate key metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate equity curve
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        # initial_capital already calculated from deposits above
        trades_df['equity'] = initial_capital + trades_df['cumulative_pnl']
        
        # Calculate drawdown
        peak = trades_df['equity'].expanding().max()
        drawdown = (trades_df['equity'] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Calculate Sharpe ratio (assuming 252 trading days per year)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = (returns.mean() * np.sqrt(252)) / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Buy and hold comparison (use first trade symbol)
        symbol = trades_df.iloc[0]['ticker']
        start_date = trades_df.iloc[0]['entry_time'].date()
        end_date = trades_df.iloc[-1]['exit_time'].date()
        
        buyhold_equity = calculate_buyhold_performance(symbol, start_date, end_date, initial_capital)
        
        # Prepare summary metrics
        summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'symbol': symbol,
            'initial_capital': initial_capital
        }
        
        # Save analytics files
        with open('analytics/summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed trades
        trades_df.to_csv('analytics/trades_detailed.csv', index=False)
        
        # Save equity curve
        equity_data = {
            'dates': trades_df['exit_time'].dt.strftime('%Y-%m-%d').tolist(),
            'strategy_equity': trades_df['equity'].tolist(),
            'buyhold_equity': buyhold_equity,
            'drawdown': drawdown.tolist()
        }
        
        with open('analytics/equity_curve.json', 'w') as f:
            json.dump(equity_data, f, indent=2)
            
        print(f"Analytics computed: {total_trades} trades, {win_rate:.1f}% win rate, ${total_pnl:.2f} P&L")
        
    except Exception as e:
        print(f"Error computing analytics: {e}")

def calculate_buyhold_performance(symbol, start_date, end_date, initial_capital):
    """Calculate buy and hold performance for comparison"""
    try:
        # Get the default session API
        api = SESSIONS[DEFAULT_SESSION]["api"]
        
        # Get historical data
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_date.isoformat(),
            end=end_date.isoformat()
        )
        
        if hasattr(bars, 'df') and not bars.df.empty:
            prices = bars.df['close']
            initial_price = prices.iloc[0]
            shares = initial_capital / initial_price
            
            # Calculate equity for each day
            equity_values = (prices * shares).tolist()
            
            print(f"Buy & Hold calculation for {symbol}:")
            print(f"   Initial price: ${initial_price:.2f}")
            print(f"   Final price: ${prices.iloc[-1]:.2f}")
            print(f"   Shares bought: {shares:.4f}")
            print(f"   Initial value: ${initial_capital:.2f}")
            print(f"   Final value: ${equity_values[-1]:.2f}")
            print(f"   Buy & Hold return: {((equity_values[-1] - initial_capital) / initial_capital * 100):+.2f}%")
            
            return equity_values
        
        # Fallback calculation using known TSLA prices
        elif symbol == "TSLA":
            print("Using fallback TSLA prices for buy & hold calculation")
            initial_price = 341.50  # TSLA price on Aug 13
            current_price = 395.94  # Current TSLA price
            shares = initial_capital / initial_price
            final_value = shares * current_price
            
            print(f"   Initial price: ${initial_price:.2f}")
            print(f"   Current price: ${current_price:.2f}")
            print(f"   Shares bought: {shares:.4f}")
            print(f"   Initial value: ${initial_capital:.2f}")
            print(f"   Final value: ${final_value:.2f}")
            print(f"   Buy & Hold return: {((final_value - initial_capital) / initial_capital * 100):+.2f}%")
            
            return [initial_capital, final_value]
        
        return [initial_capital] * 2  # Default fallback
        
    except Exception as e:
        print(f"Error calculating buy-hold performance: {e}")
        # Use TSLA fallback if API fails
        if symbol == "TSLA":
            initial_price = 341.50
            current_price = 395.94
            shares = initial_capital / initial_price
            final_value = shares * current_price
            return [initial_capital, final_value]
        return [initial_capital] * 2

# Dynamic log file management
TRADES_LOG_HEADERS = [
    "timestamp", "ticker", "action", "quantity", "price", "total_value", 
    "buying_power_before", "buying_power_after", "portfolio_value", "notes", "session",
    # Enhanced columns for strategy tracking
    "side_enter", "side_exit", "entry_time", "exit_time", "entry_price", "exit_price",
    "pnl", "run_up", "drawdown", "mfe", "mae", "strategy_name", "day_id", "position_id"
]

def get_log_file_path(session_id):
    """Get the log file path for a specific session"""
    return f"trades_log_{session_id}.csv"

def get_all_log_files():
    """Get all log file paths for existing sessions"""
    return [get_log_file_path(session_id) for session_id in SESSIONS.keys()]

def initialize_trades_log():
    """Initialize the trades log CSV files with headers if they don't exist"""
    for session_id in SESSIONS.keys():
        log_file = get_log_file_path(session_id)
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(TRADES_LOG_HEADERS)
            print(f"Created log file: {log_file}")

def log_trade(session_name, ticker, action, quantity, price, total_value, buying_power_before, buying_power_after, notes=""):
    """Log a trade to the appropriate CSV file"""
    try:
        # Get the API object for this session
        api = SESSIONS[session_name]["api"]
        session_config = SESSIONS[session_name]["config"]
        
        # Get current portfolio value
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get the log file for this session
        log_file = get_log_file_path(session_name)
        
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, ticker, action, quantity, price, total_value,
                buying_power_before, buying_power_after, portfolio_value, notes, session_config["name"]
            ])
        print(f"Trade logged: {action} {quantity} {ticker} at ${price} on {session_config['name']} session")
    except Exception as e:
        print(f"Error logging trade: {e}")

def generate_trading_chart(session_id=None):
    """Generate a chart of trading activity and portfolio performance"""
    try:
        # Use the default session if none specified
        if session_id is None:
            session_id = DEFAULT_SESSION
            
        log_file = get_log_file_path(session_id)
        chart_title = f"{SESSIONS[session_id]['config']['name']} Trading Activity Dashboard"
        
        # Read the log file
        if not os.path.exists(log_file):
            print("No trades log file found")
            return
            
        df = pd.read_csv(log_file)
        if df.empty:
            print("No trades to chart")
            return
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(chart_title, fontsize=16)
        
        # Plot 1: Portfolio Value Over Time
        ax1.plot(df['timestamp'], df['portfolio_value'], 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Trade Actions (Long-Only Strategy)
        buy_trades = df[df['action'] == 'buy']
        sell_trades = df[df['action'] == 'sell']
        
        if not buy_trades.empty:
            ax2.scatter(buy_trades['timestamp'], buy_trades['price'], 
                       c='green', marker='^', s=buy_trades['quantity']*5, 
                       alpha=0.7, label='Buy Orders')
                       
        if not sell_trades.empty:
            ax2.scatter(sell_trades['timestamp'], sell_trades['price'], 
                       c='red', marker='v', s=sell_trades['quantity']*5, 
                       alpha=0.7, label='Sell Orders')
        
        ax2.set_title('Trade Executions (Size = Quantity)')
        ax2.set_ylabel('Stock Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Buying Power Over Time
        ax3.plot(df['timestamp'], df['buying_power_after'], 'orange', linewidth=2, marker='s', markersize=3)
        ax3.set_title('Buying Power Over Time')
        ax3.set_ylabel('Buying Power ($)')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        chart_filename = f"trading_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trading chart saved as: {chart_filename}")
        
        # Print summary statistics (Long-Only Strategy)
        total_trades = len(df)
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        
        if total_trades > 1:
            initial_value = df.iloc[0]['portfolio_value']
            final_value = df.iloc[-1]['portfolio_value']
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            print(f"\n=== Trading Summary ===")
            print(f"Total Trades: {total_trades}")
            print(f"Buy Orders: {buy_count}")
            print(f"Sell Orders: {sell_count}")
            print(f"Initial Portfolio Value: ${initial_value:,.2f}")
            print(f"Current Portfolio Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return:+.2f}%")
        
    except Exception as e:
        print(f"Error generating chart: {e}")

# Initialize trades log on startup
initialize_trades_log()

# Authentication removed for local development

@app.route('/')
def index():
    """Redirect root to dashboard"""
    return render_template('dashboard.html')

# Flask Routes
@app.route('/dashboard')
def dashboard():
    """Serve the main dashboard HTML page"""
    return render_template('dashboard.html')

@app.route('/metrics')
def get_metrics():
    """Get overall strategy KPIs with current TSLA price"""
    try:
        # Get current TSLA price using cached/backup sources (efficient)
        from data_cache import is_market_open
        
        # Use cached price to ensure consistency across dashboard refresh
        current_tsla_price = get_cached_tsla_price()
            
        # Use Yahoo Finance/backup price as it's more accurate than Alpaca free tier
        print(f"� Using Yahoo Finance TSLA price: ${current_tsla_price:.2f}")
        
        # Note: Alpaca free tier quotes can be delayed, so we prioritize Yahoo Finance
        if not is_market_open():
            print("🕐 Market closed - using cached/backup price data")
            
        ensure_analytics_directory()
        summary_file = 'analytics/summary.json'
        
        if not os.path.exists(summary_file):
            # Try to compute analytics first
            compute_analytics()
            
        # Get first trade price from trades log for TSLA change calculation
        first_trade_price = 467.1375  # Default from first trade
        try:
            log_file = get_log_file_path(DEFAULT_SESSION)
            if os.path.exists(log_file):
                trades_df = pd.read_csv(log_file)
                if not trades_df.empty:
                    # Get the first buy trade's price
                    first_buy = trades_df[trades_df['action'] == 'buy'].iloc[0]
                    first_trade_price = float(first_buy['price'])
        except Exception as e:
            logging.warning(f"Could not read first trade price: {e}")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                metrics = json.load(f)
                
            # Add TSLA price info
            metrics['current_tsla_price'] = current_tsla_price
            metrics['tsla_change_percent'] = ((current_tsla_price - first_trade_price) / first_trade_price) * 100
            
            # Calculate percentage gain from initial capital (including deposits)
            initial_capital = metrics.get('initial_capital', 1000.0)
            metrics['total_pnl_percent'] = (metrics.get('total_pnl', 0) / initial_capital) * 100
            
            # Calculate unrealized P&L from current open position
            unrealized_pnl = 0
            try:
                log_file = get_log_file_path(DEFAULT_SESSION)
                if os.path.exists(log_file):
                    trades_df = pd.read_csv(log_file)
                    if not trades_df.empty:
                        # Check if last trade was a buy (indicating open position)
                        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                        trades_df = trades_df.sort_values('timestamp')
                        last_trade = trades_df.iloc[-1]
                        
                        if last_trade['action'] == 'buy':
                            # We have an open position
                            entry_price = float(last_trade['price'])
                            quantity = float(last_trade['quantity'])
                            unrealized_pnl = (current_tsla_price - entry_price) * quantity
                            print(f"💼 Unrealized P&L: ${unrealized_pnl:.2f} ({quantity:.4f} shares @ ${entry_price:.2f} → ${current_tsla_price:.2f})")
            except Exception as e:
                logging.warning(f"Could not calculate unrealized P&L: {e}")
            
            # Calculate current account balance (initial capital + realized P&L + unrealized P&L)
            realized_pnl = metrics.get('total_pnl', 0)
            total_pnl_with_unrealized = realized_pnl + unrealized_pnl
            metrics['account_balance'] = initial_capital + total_pnl_with_unrealized
            metrics['unrealized_pnl'] = unrealized_pnl
            metrics['realized_pnl'] = realized_pnl
            
            # Update percentage to include unrealized gains
            metrics['total_pnl_percent'] = (total_pnl_with_unrealized / initial_capital) * 100
            
            return jsonify(metrics)
        else:
            tsla_change_percent = ((current_tsla_price - first_trade_price) / first_trade_price) * 100
            initial_capital = 1000.0
            return jsonify({
                'error': 'No analytics data available',
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'current_tsla_price': current_tsla_price,
                'tsla_change_percent': tsla_change_percent,
                'account_balance': initial_capital
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/equity')
def get_equity():
    """Get equity curve data for strategy vs buy & hold"""
    try:
        ensure_analytics_directory()
        equity_file = 'analytics/equity_curve.json'
        
        if not os.path.exists(equity_file):
            compute_analytics()
            
        if os.path.exists(equity_file):
            with open(equity_file, 'r') as f:
                equity_data = json.load(f)
            return jsonify(equity_data)
        else:
            return jsonify({
                'error': 'No equity data available',
                'dates': [],
                'strategy_equity': [],
                'buyhold_equity': [],
                'drawdown': []
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/trades_table')
def get_trades_table():
    """Get detailed trades data for table display"""
    try:
        ensure_analytics_directory()
        trades_file = 'analytics/trades_detailed.csv'
        
        if not os.path.exists(trades_file):
            compute_analytics()
            
        if os.path.exists(trades_file):
            df = pd.read_csv(trades_file)
            trades_data = df.to_dict('records')
            return jsonify({'trades': trades_data})
        else:
            return jsonify({'error': 'No trades data available', 'trades': []})
            
    except Exception as e:
        return jsonify({'error': str(e), 'trades': []})

@app.route('/candlestick_table')
def get_candlestick_table():
    """Get 5-minute candlestick data for Yahoo Finance-style table"""
    try:
        from backup_data_provider import get_tsla_minute_candlesticks
        from data_cache import is_market_open
        
        # Get recent 5-minute candlesticks (limit to 50 for table display)
        candlesticks = get_tsla_minute_candlesticks(limit=50)
        
        # Add market status info
        market_status = {
            'is_open': is_market_open(),
            'last_updated': datetime.now().isoformat(),
            'data_source': 'yahoo_finance'
        }
        
        return jsonify({
            'candlesticks': candlesticks,
            'market_status': market_status,
            'total_bars': len(candlesticks)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'candlesticks': [],
            'market_status': {'is_open': False, 'error': str(e)}
        })

@app.route('/position_status')
def get_position_status():
    """Get current TSLA position and strategy status with detailed levels"""
    try:
        if DEFAULT_SESSION not in SESSIONS:
            return jsonify({'error': 'No valid API session'})
            
        api = SESSIONS[DEFAULT_SESSION]["api"]
        
        # Get current TSLA price (cached for consistency)
        current_price = get_cached_tsla_price()
        
        # Initialize response data
        response_data = {
            'current_price': current_price,
            'has_position': False,
            'position_details': {},
            'opening_range': {},
            'entry_signals': {},
            'strategy_status': 'No Strategy Active',
            'last_updated': datetime.now().isoformat()
        }
        
        # Check trades log to see if strategy is active
        log_status = check_strategy_active_from_log()
        
        # Get current positions from Alpaca
        try:
            positions = api.list_positions()
            tsla_position = next((pos for pos in positions if pos.symbol == 'TSLA'), None)
            
            if tsla_position:
                shares = float(tsla_position.qty)
                entry_price = float(getattr(tsla_position, 'avg_entry_price', 0) or 0)
                current_value = float(getattr(tsla_position, 'market_value', 0) or 0)
                unrealized_pnl = float(getattr(tsla_position, 'unrealized_pl', 0) or 0)
                
                # Only track long positions in this strategy
                if shares > 0:
                    response_data['has_position'] = True
                    unrealized_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0.0
                    response_data['position_details'] = {
                        'shares': float(shares),
                        'side': 'Long',
                        'entry_price': float(entry_price),
                        'current_value': float(current_value if current_value else shares * current_price),
                        'unrealized_pnl': float(unrealized_pnl if unrealized_pnl else (current_price - entry_price) * shares),
                        'unrealized_pnl_percent': float(unrealized_pct),
                        'target_price': 0.0,
                        'stop_loss': 0.0,
                        'distance_to_target': 0.0,
                        'distance_to_stop': 0.0,
                        'risk_reward_ratio': 0.0,
                        'entry_time': getattr(tsla_position, 'avg_entry_time', None),
                        'source': 'alpaca'
                    }
            elif log_status['active']:
                # Alpaca may not show position yet, but trades log indicates we have one
                response_data['has_position'] = True
                qty = float(log_status.get('quantity', 0) or 0)
                entry = float(log_status.get('entry_price', 0) or 0)
                unrealized_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0.0
                response_data['position_details'] = {
                    'shares': float(qty),
                    'side': 'Long',
                    'entry_price': float(entry),
                    'current_value': float(qty * current_price),
                    'unrealized_pnl': float((current_price - entry) * qty),
                    'unrealized_pnl_percent': float(unrealized_pct),
                    'target_price': 0.0,
                    'stop_loss': 0.0,
                    'distance_to_target': 0.0,
                    'distance_to_stop': 0.0,
                    'risk_reward_ratio': 0.0,
                    'entry_time': log_status.get('entry_time'),
                    'source': 'trades_log'
                }
                print(f"📊 Position detected from trades log: {qty} shares @ ${entry:.2f}")
                
        except Exception as pos_error:
            print(f"Error getting positions: {pos_error}")
            # If Alpaca fails but log shows active position, use log data
            if log_status['active']:
                response_data['has_position'] = True
                qty = float(log_status.get('quantity', 0) or 0)
                entry = float(log_status.get('entry_price', 0) or 0)
                unrealized_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0.0
                response_data['position_details'] = {
                    'shares': float(qty),
                    'side': 'Long',
                    'entry_price': float(entry),
                    'current_value': float(qty * current_price),
                    'unrealized_pnl': float((current_price - entry) * qty),
                    'unrealized_pnl_percent': float(unrealized_pct),
                    'target_price': 0.0,
                    'stop_loss': 0.0,
                    'distance_to_target': 0.0,
                    'distance_to_stop': 0.0,
                    'risk_reward_ratio': 0.0,
                    'entry_time': log_status.get('entry_time'),
                    'source': 'trades_log'
                }
                print(f"📊 Position from trades log (Alpaca unavailable): {qty} shares @ ${entry:.2f}")
        
        # Calculate actual opening range from historical data
        or_high = None
        or_low = None
        or_range = 0
        
        # Try to get opening range from strategy engine first
        if strategy_engine:
            current_state = strategy_engine.get_status()
            if current_state.get('opening_high') and current_state.get('opening_low'):
                or_high = current_state['opening_high']
                or_low = current_state['opening_low']
        
        # If strategy engine doesn't have it, calculate from intraday data
        if not or_high or not or_low:
            try:
                from backup_data_provider import get_tsla_intraday_bars
                from data_cache import is_market_open
                import pytz
                
                # Get today's 5-minute bars for ORB calculation
                intraday_bars = get_tsla_intraday_bars(days=1, interval='5m')
                
                if intraday_bars and len(intraday_bars) > 0:
                    # Convert to ET timezone and find first 15 minutes of trading (9:30-9:45 AM ET)
                    # With 5-minute bars: 9:30-9:35, 9:35-9:40, 9:40-9:45 (3 bars total)
                    et_tz = pytz.timezone('US/Eastern')
                    today = date.today()
                    
                    # Market opens at 9:30 AM ET
                    market_open = et_tz.localize(datetime.combine(today, time(9, 30)))
                    market_open_15min = market_open + timedelta(minutes=15)  # 9:45 AM ET
                    
                    # Filter bars for first 15 minutes
                    opening_bars = []
                    for bar in intraday_bars:
                        try:
                            # Parse bar time
                            bar_time_str = bar['time']
                            if 'Z' in bar_time_str:
                                bar_time_str = bar_time_str.replace('Z', '+00:00')
                            
                            bar_time = datetime.fromisoformat(bar_time_str)
                            
                            # Convert to ET if needed
                            if bar_time.tzinfo is None:
                                bar_time = et_tz.localize(bar_time)
                            else:
                                bar_time = bar_time.astimezone(et_tz)
                            
                            # Check if bar is within first 15 minutes
                            if market_open <= bar_time <= market_open_15min:
                                opening_bars.append(bar)
                                
                        except Exception as bar_error:
                            continue
                    
                    # Calculate opening range from first 15 minutes
                    if opening_bars:
                        highs = [bar['high'] for bar in opening_bars]
                        lows = [bar['low'] for bar in opening_bars]
                        or_high = max(highs)
                        or_low = min(lows)
                        print(f"📊 Calculated OR from {len(opening_bars)} 5-minute bars: High=${or_high:.2f}, Low=${or_low:.2f}")
                    else:
                        print("⚠️ No bars found for first 15 minutes of trading")
                        
            except Exception as or_error:
                print(f"⚠️ Error calculating opening range: {or_error}")
        
        # Set opening range data
        if or_high and or_low:
            or_range = or_high - or_low
            response_data['opening_range'] = {
                'high': or_high,
                'low': or_low,
                'range': or_range,
                'range_percent': (or_range / or_low) * 100,
                'current_date': date.today().isoformat(),
                'calculated_from': 'historical_data' if not (strategy_engine and strategy_engine.get_status().get('opening_high')) else 'strategy_engine'
            }
        else:
            response_data['opening_range'] = {
                'high': None,
                'low': None,
                'range': 0,
                'range_percent': 0,
                'current_date': date.today().isoformat(),
                'error': 'Could not determine opening range'
            }
        
        # Calculate targets and stops based on opening range OR strategy engine state
        if or_high and or_low:
            if response_data['has_position']:
                side = response_data['position_details']['side']
                if side == 'Long':
                    # Try to get stored TP/SL from strategy engine first (preserves entry day levels)
                    if strategy_engine and strategy_engine.tp and strategy_engine.sl:
                        response_data['position_details']['target_price'] = strategy_engine.tp
                        response_data['position_details']['stop_loss'] = strategy_engine.sl
                        print(f"📊 Using stored levels from entry: TP=${strategy_engine.tp:.2f}, SL=${strategy_engine.sl:.2f}")
                    else:
                        # Fallback to current OR (shouldn't happen if strategy is working correctly)
                        response_data['position_details']['target_price'] = or_high + (or_range * 2)
                        response_data['position_details']['stop_loss'] = or_low  # No cushion!
                    
                    # Calculate distances to targets (only for long positions)
                    target_distance = abs(response_data['position_details']['target_price'] - current_price)
                    stop_distance = abs(current_price - response_data['position_details']['stop_loss'])
                    
                    response_data['position_details']['distance_to_target'] = target_distance
                    response_data['position_details']['distance_to_stop'] = stop_distance
                    response_data['position_details']['risk_reward_ratio'] = target_distance / stop_distance if stop_distance > 0 else 0
            
            else:
                # No position - show long entry triggers only
                response_data['entry_signals'] = {
                    'long_trigger': or_high,
                    'distance_to_long_trigger': or_high - current_price,
                    'long_target': or_high + (or_range * 2),
                    'long_stop': or_low  # Stop at opening low, no cushion
                }
        
        # Strategy status - prioritize trades log information
        if response_data['has_position']:
            side = response_data['position_details'].get('side', 'Unknown')
            entry_price = response_data['position_details'].get('entry_price', 0)
            response_data['strategy_status'] = f"In Position ({side}) @ ${entry_price:.2f}"
        elif strategy_engine:
            current_state = strategy_engine.get_status()
            if current_state.get('strategy_active', False):
                response_data['strategy_status'] = 'Active - Waiting for Breakout'
            else:
                response_data['strategy_status'] = 'Strategy Inactive'
        else:
            response_data['strategy_status'] = 'Strategy Inactive'
        
        # Add log status for debugging
        response_data['log_status'] = log_status
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e), 'current_price': 0})

@app.route('/chart-data')
def get_chart_data():
    """Get TSLA price data with ORB strategy levels for chart visualization"""
    try:
        # Get API for data retrieval
        if DEFAULT_SESSION not in SESSIONS:
            return jsonify({'error': 'No valid API session'})
            
        api = SESSIONS[DEFAULT_SESSION]["api"]
        
        chart_data = {
            'price_data': [],
            'orb_levels': [],
            'trades': [],
            'current_levels': {}
        }
        
        # Import caching utilities and intraday data
        from data_cache import is_market_open
        from backup_data_provider import get_tsla_intraday_bars
        
        market_open = is_market_open()
        
        try:
            # Get 5-minute intraday candlesticks for ORB visualization (TODAY ONLY)
            print("🔍 Fetching 5-minute TSLA candlesticks for ORB chart...")
            intraday_bars = get_tsla_intraday_bars(days=1, interval='5m')  # Get today's 5-minute data
            
            if intraday_bars and len(intraday_bars) > 0:
                # Filter to only show today's bars
                import pytz
                et_tz = pytz.timezone('US/Eastern')
                today = date.today()
                
                # Convert intraday bars to chart format (TODAY ONLY)
                chart_data['price_data'] = []
                for bar in intraday_bars:
                    # Parse the timestamp properly
                    bar_time = datetime.fromisoformat(bar['time'].replace('Z', '+00:00'))
                    
                    # Only include bars from today
                    if bar_time.tzinfo is None:
                        bar_time_et = et_tz.localize(bar_time)
                    else:
                        bar_time_et = bar_time.astimezone(et_tz)
                    
                    bar_date = bar_time_et.date()
                    
                    if bar_date == today:
                        chart_data['price_data'].append({
                            'time': bar_time.isoformat(),
                            'open': bar['open'],
                            'high': bar['high'], 
                            'low': bar['low'],
                            'close': bar['close'],
                            'volume': bar['volume']
                        })
                
                print(f"✅ Got {len(chart_data['price_data'])} 5-minute bars for ORB chart")
            else:
                # Fallback to current price if no intraday data
                print("⚠️ No intraday data available, using current price")
                current_price = get_cached_tsla_price()
                now = datetime.now()
                
                chart_data['price_data'] = [{
                    'time': now.isoformat(),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0
                }]
            
        except Exception as intraday_error:
            print(f"❌ Intraday data failed: {intraday_error}")
            
            # Fallback to current price
            try:
                current_price = get_cached_tsla_price()
                now = datetime.now()
                
                chart_data['price_data'] = [{
                    'time': now.isoformat(),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0
                }]
                print(f"✅ Using fallback current price: ${current_price:.2f}")
                
            except Exception as final_error:
                print(f"❌ All data sources failed: {final_error}")
                return jsonify({
                    'error': 'Could not fetch price data from any source',
                    'price_data': [],
                    'orb_levels': [],
                    'trades': [],
                    'current_levels': {}
                })
        
        # Calculate opening range from data (same logic as position_status)
        or_high = None
        or_low = None
        or_range = 0
        
        # Try to get opening range from strategy engine first
        if strategy_engine:
            current_state = strategy_engine.get_status()
            if current_state.get('opening_high') and current_state.get('opening_low'):
                or_high = current_state['opening_high']
                or_low = current_state['opening_low']
        
        # If strategy engine doesn't have it, calculate from intraday data
        if not or_high or not or_low:
            try:
                import pytz
                
                # Use the intraday_bars we already fetched above (5-minute bars)
                if intraday_bars and len(intraday_bars) > 0:
                    et_tz = pytz.timezone('US/Eastern')
                    today = date.today()
                    
                    # Market opens at 9:30 AM ET, first 15 minutes covered by 3 five-minute bars
                    market_open = et_tz.localize(datetime.combine(today, time(9, 30)))
                    market_open_15min = market_open + timedelta(minutes=15)  # 9:45 AM ET
                    
                    # Filter bars for first 15 minutes (typically 3 five-minute bars)
                    opening_bars = []
                    for bar in intraday_bars:
                        try:
                            bar_time_str = bar['time']
                            if 'Z' in bar_time_str:
                                bar_time_str = bar_time_str.replace('Z', '+00:00')
                            
                            bar_time = datetime.fromisoformat(bar_time_str)
                            
                            if bar_time.tzinfo is None:
                                bar_time = et_tz.localize(bar_time)
                            else:
                                bar_time = bar_time.astimezone(et_tz)
                            
                            if market_open <= bar_time <= market_open_15min:
                                opening_bars.append(bar)
                                
                        except Exception:
                            continue
                    
                    # Calculate opening range from first 15 minutes
                    if opening_bars:
                        highs = [bar['high'] for bar in opening_bars]
                        lows = [bar['low'] for bar in opening_bars]
                        or_high = max(highs)
                        or_low = min(lows)
                        print(f"📊 Chart: Calculated OR from {len(opening_bars)} 5-minute bars: High=${or_high:.2f}, Low=${or_low:.2f}")
                        
            except Exception as or_error:
                print(f"⚠️ Chart: Error calculating opening range: {or_error}")
        
        # Only proceed with ORB calculations if we have valid opening range
        if or_high and or_low:
            or_range = or_high - or_low
            
            # Calculate ORB strategy levels (long-only)
            # If we have an open position, use strategy engine's stored TP/SL from entry day
            if strategy_engine and strategy_engine.tp and strategy_engine.sl:
                target_high = strategy_engine.tp
                stop_high = strategy_engine.sl
                breakout_high = or_high  # Current day's breakout level for reference
                print(f"📊 Chart using stored entry levels: TP=${target_high:.2f}, SL=${stop_high:.2f}")
            else:
                # No position - calculate from current OR
                breakout_high = or_high
                target_high = or_high + (or_range * 2)  # 2:1 reward-to-risk
                stop_high = or_low  # Stop at opening low, no cushion
            
            # Get current TSLA price (cached for consistency)
            current_price = get_cached_tsla_price()
            
            # Determine position side and current holdings
            position_side = None
            position_info = {
                'has_position': False,
                'shares': 0,
                'entry_price': 0,
                'current_value': 0,
                'unrealized_pnl': 0,
                'sell_target': 0,
                'stop_loss': 0
            }
            
            # Check current positions via Alpaca API
            try:
                positions = api.list_positions()
                tsla_position = next((pos for pos in positions if pos.symbol == 'TSLA'), None)
                
                if tsla_position:
                    position_info['has_position'] = True
                    position_info['shares'] = float(tsla_position.qty)
                    position_info['entry_price'] = float(getattr(tsla_position, 'avg_entry_price', 0) or 0)
                    position_info['current_value'] = float(getattr(tsla_position, 'market_value', 0) or 0)
                    position_info['unrealized_pnl'] = float(getattr(tsla_position, 'unrealized_pl', 0) or 0)
                    
                    # Only support long positions in this ORB strategy
                    if position_info['shares'] > 0:
                        position_side = 'Long'
                        # Use strategy engine's stored values if available (preserves entry day levels)
                        if strategy_engine and strategy_engine.tp and strategy_engine.sl:
                            position_info['sell_target'] = strategy_engine.tp
                            position_info['stop_loss'] = strategy_engine.sl
                        else:
                            position_info['sell_target'] = target_high
                            position_info['stop_loss'] = stop_high
                
            except Exception as pos_error:
                print(f"Error getting positions: {pos_error}")
                # Fallback to strategy engine state (long positions only)
                if current_state.get('in_position') and current_state.get('entry_price'):
                    entry_price = current_state['entry_price']
                    if entry_price > or_high:
                        position_side = 'Long'
                        position_info['sell_target'] = target_high
                        position_info['stop_loss'] = stop_high
            
            # Calculate entry triggers if no position (long-only strategy)
            entry_triggers = {
                'long_trigger': or_high,  # Buy when price breaks above OR high
                'current_price': current_price,
                'distance_to_long': or_high - current_price
            }
            
            chart_data['current_levels'] = {
                'or_high': or_high,
                'or_low': or_low,
                'or_range': or_range,
                'breakout_high': breakout_high,
                'target_high': target_high,
                'stop_high': stop_high,
                'current_date': current_state.get('current_date'),
                'position_side': position_side,
                'position_info': position_info,
                'entry_triggers': entry_triggers,
                'current_price': current_price
            }
        
        # Get recent trades for chart annotations
        try:
            log_file = get_log_file_path(DEFAULT_SESSION)
            if os.path.exists(log_file):
                df = pd.read_csv(log_file)
                if not df.empty:
                    # Filter trades to only today's trades to match chart data
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    today = date.today()
                    today_trades = df[df['timestamp'].dt.date == today]
                    
                    if not today_trades.empty:
                        for _, trade in today_trades.iterrows():
                            # Use ISO format timestamp to match candlestick time format
                            trade_time = trade['timestamp'].isoformat()
                            
                            chart_data['trades'].append({
                                'date': trade_time,  # Use full ISO timestamp for alignment
                                'time': trade_time,
                                'action': trade['action'],
                                'price': float(trade['price']),
                                'quantity': float(trade['quantity']),
                                'ticker': trade['ticker']
                            })
        except Exception as e:
            print(f"Could not load recent trades: {e}")
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e), 'price_data': [], 'orb_levels': [], 'trades': []})

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run historical backtest"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'TSLA')
        start_date = data.get('start', '2025-01-01')
        end_date = data.get('end', date.today().isoformat())
        
        print(f"Starting backtest: {symbol} from {start_date} to {end_date}")
        
        # Get API for backtesting
        if DEFAULT_SESSION not in SESSIONS:
            return jsonify({'error': 'No valid API session for backtest'})
            
        api = SESSIONS[DEFAULT_SESSION]["api"]
        
        # Create backtest data provider
        backtest_provider = BacktestDataProvider(api)
        
        # Create strategy for backtest (without live trading)
        backtest_strategy = ORBStrategy(
            api=api,
            session_name=f"backtest_{symbol}",
            symbol=symbol,
            log_trade_func=None  # Don't log backtest trades to main CSV
        )
        
        # Run backtest
        backtest_provider.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            on_bar_callback=backtest_strategy.on_bar
        )
        
        # Compute analytics for backtest results
        # Note: This would need a separate analytics computation for backtest data
        # For now, we'll recompute main analytics
        compute_analytics()
        
        return jsonify({
            'success': True,
            'message': f'Backtest completed for {symbol}',
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'analytics_updated': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle webhook requests for trade execution - REQUIRES API KEY"""
    data = request.json or {}
    
    print(f"Received alert: {data}")

    ticker = data.get("ticker")
    action = data.get("action")
    price = data.get("price")
    
    # Since we only have one session, use it directly (ignore session parameter)
    session = DEFAULT_SESSION

    if not ticker or not action:
        return jsonify({"error": "Invalid data"}), 400

    # Get the API object for the main session
    api = SESSIONS[session]["api"]
    session_config = SESSIONS[session]["config"]

    try:
        if action == "buy":
            # --- Read desired params (all optional) ---
            clock = api.get_clock()
            limit_px = data.get("limit_price") or data.get("price")  # accept TradingView's "price" as limit
            order_type = (data.get("type") or ("market" if clock.is_open else "limit")).lower()
            tif = ("day" if order_type == "market" else str(data.get("time_in_force", "gtc")).lower())
            ext_hours = bool(data.get("extended_hours", not clock.is_open))  # allow after-hours for limit
            # UPDATED: Support fractional quantities
            qty_req = float(data.get("qty", 0)) if data.get("qty") else 0

            # --- Account / sizing ---
            account = api.get_account()
            bp_before = float(account.buying_power)
            last_trade = api.get_latest_trade(ticker)
            px_for_size = float(limit_px) if (order_type == "limit" and limit_px) else float(last_trade.price)
            # UPDATED: Calculate fractional shares (round to 6 decimal places for precision)
            max_shares = round(bp_before * 0.99 / px_for_size, 6)
            shares = min(max_shares, qty_req) if qty_req > 0 else max_shares
            if shares <= 0:
                return jsonify({"error": "Insufficient buying power"}), 400

            # --- Route order correctly ---
            if order_type == "market":
                # market orders: only during RTH and must be DAY
                if not clock.is_open:
                    return jsonify({"error": "Market is closed. Use a LIMIT order with extended_hours=true"}), 400
                order = api.submit_order(symbol=ticker, qty=shares, side="buy",
                                        type="market", time_in_force="day")
            else:
                if limit_px is None:
                    return jsonify({"error": "limit_price required for limit orders"}), 400
                order = api.submit_order(symbol=ticker, qty=shares, side="buy",
                                        type="limit", limit_price=float(limit_px),
                                        time_in_force=tif, extended_hours=ext_hours)

            # Log the trade after successful order submission
            account_after = api.get_account()
            bp_after = float(account_after.buying_power)
            total_value = shares * px_for_size
            log_trade(session, ticker, "buy", shares, px_for_size, total_value,
                      bp_before, bp_after, f"Order ID: {order.id}")
            print(f"Bought {shares} shares of {ticker} at ~${px_for_size:.2f}")

                
        elif action == "sell":
            account = api.get_account()
            bp_before = float(account.buying_power)

            try:
                position = api.get_position(ticker)
            except Exception:
                return jsonify({"error": f"No position in {ticker}"}), 400

            # UPDATED: Support fractional positions
            have = float(position.qty)
            if have <= 0:
                return jsonify({"error": "No shares to sell"}), 400

            clock = api.get_clock()
            limit_px = data.get("limit_price") or data.get("price")
            order_type = (data.get("type") or ("market" if clock.is_open else "limit")).lower()
            tif = ("day" if order_type == "market" else str(data.get("time_in_force", "gtc")).lower())
            ext_hours = bool(data.get("extended_hours", not clock.is_open))

            if order_type == "market":
                if not clock.is_open:
                    return jsonify({"error": "Market is closed. Use LIMIT with extended_hours=true"}), 400
                order = api.submit_order(symbol=ticker, qty=have, side="sell",
                                        type="market", time_in_force="day")
            else:
                if limit_px is None:
                    return jsonify({"error": "limit_price required for limit orders"}), 400
                order = api.submit_order(symbol=ticker, qty=have, side="sell",
                                        type="limit", limit_price=float(limit_px),
                                        time_in_force=tif, extended_hours=ext_hours)

            # Log the trade after successful order submission
            account_after = api.get_account()
            bp_after = float(account_after.buying_power)
            sell_price = float(limit_px) if limit_px else float(api.get_latest_trade(ticker).price)
            total_value = have * sell_price
            log_trade(session, ticker, "sell", have, sell_price, total_value,
                      bp_before, bp_after, f"Order ID: {order.id}")
            print(f"Sold {have} shares of {ticker} at ~${sell_price:.2f}")



        else:
            return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        print(f"Error placing order: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "success", "time": str(datetime.now())})

@app.route("/chart", methods=["GET"])
def create_chart():
    """Endpoint to generate and view trading charts"""
    try:
        # Since we only have one session, always generate chart for it
        generate_trading_chart(DEFAULT_SESSION)
        return jsonify({"status": "Chart generated successfully", "time": str(datetime.now())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/trades", methods=["GET"])
def get_trades():
    """Endpoint to view all trades"""
    try:
        # Since we only have one session, get its log file
        log_file = get_log_file_path(DEFAULT_SESSION)
        
        all_trades = []
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            if not df.empty:
                all_trades = df.to_dict('records')
                # Sort by timestamp
                all_trades.sort(key=lambda x: x['timestamp'])
        
        return jsonify({"trades": all_trades, "count": len(all_trades)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """Endpoint to list all available sessions"""
    try:
        session_info = {}
        for session_id, session_data in SESSIONS.items():
            config = session_data["config"]
            session_info[session_id] = {
                "name": config["name"],
                "base_url": config["base_url"],
                "enabled": config.get("enabled", True)
            }
        
        return jsonify({
            "sessions": session_info, 
            "default_session": DEFAULT_SESSION,
            "total_sessions": len(SESSIONS),
            "time": str(datetime.now())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reload", methods=["POST"])
def reload_sessions():
    """Endpoint to reload sessions from configuration file"""
    try:
        global SESSIONS, DEFAULT_SESSION
        SESSIONS, DEFAULT_SESSION = initialize_sessions()
        initialize_trades_log()
        
        return jsonify({
            "status": "Sessions reloaded successfully", 
            "sessions": list(SESSIONS.keys()),
            "default_session": DEFAULT_SESSION,
            "time": str(datetime.now())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    """Endpoint to check the status of the trading session"""
    try:
        session_info = SESSIONS[DEFAULT_SESSION]
        api = session_info["api"]
        account = api.get_account()
        config = session_info["config"]
        
        status = {
            "name": config["name"],
            "status": "connected",
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "market_value": float(account.long_market_value) if account.long_market_value else 0.0,
            "day_trade_count": getattr(account, 'day_trade_count', 0),
            "pattern_day_trader": getattr(account, 'pattern_day_trader', False)
        }
        
        return jsonify({"session": status, "time": str(datetime.now())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_deposit", methods=["POST"])
def add_deposit():
    """Add a deposit transaction to the trades log"""
    try:
        data = request.get_json()
        amount = float(data.get('amount', 0))
        
        if amount <= 0:
            return jsonify({'success': False, 'error': 'Invalid deposit amount'}), 400
        
        # Get the log file for the default session
        log_file = get_log_file_path(DEFAULT_SESSION)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add deposit as a special transaction type
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,           # timestamp
                'CASH',             # ticker
                'deposit',          # action
                0,                  # quantity
                amount,             # price (stores deposit amount)
                amount,             # total_value
                0,                  # buying_power_before
                0,                  # buying_power_after
                0,                  # portfolio_value
                f'Deposit of ${amount:.2f}',  # notes
                SESSIONS[DEFAULT_SESSION]['config']['name'],  # session
                '', '', '', '', '', '', '', '', '', '', '', '', '', ''  # empty fields
            ])
        
        print(f"💰 Deposit logged: ${amount:.2f} at {timestamp}")
        
        # Recompute analytics to include the new deposit
        compute_analytics()
        
        return jsonify({
            'success': True,
            'message': f'Deposit of ${amount:.2f} added successfully',
            'amount': amount,
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"Error adding deposit: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def check_strategy_active_from_log(session_id=None):
    """Check if strategy is currently active by examining the last trade in the log"""
    try:
        if session_id is None:
            session_id = DEFAULT_SESSION
            
        log_file = get_log_file_path(session_id)
        
        if not os.path.exists(log_file):
            return {'active': False, 'reason': 'No trades log found'}
            
        # Read trades data
        df = pd.read_csv(log_file)
        
        if df.empty:
            return {'active': False, 'reason': 'No trades recorded'}
            
        # Get the last trade
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        last_trade = df.iloc[-1]
        
        # If last trade is a buy, strategy is active at that price
        if last_trade['action'].lower() == 'buy':
            return {
                'active': True,
                'entry_price': float(last_trade['price']),
                'quantity': float(last_trade['quantity']),
                'entry_time': last_trade['timestamp'],
                'ticker': last_trade['ticker']
            }
        else:
            # Last trade was a sell, so no active position
            return {
                'active': False,
                'reason': 'Last trade was a sell order',
                'last_exit_time': last_trade['timestamp']
            }
            
    except Exception as e:
        logging.error(f"Error checking strategy status from log: {e}")
        return {'active': False, 'reason': f'Error: {str(e)}'}

def get_deployment_config():
    """Get configuration based on deployment environment"""
    # Detect AWS environment
    is_aws = any([
        os.environ.get('AWS_EXECUTION_ENV'),
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME'),
        os.environ.get('ECS_CONTAINER_METADATA_URI'),
        os.path.exists('/opt/aws')
    ])
    
    # Detect Docker environment
    is_docker = os.path.exists('/.dockerenv')
    
    if is_aws or is_docker:
        # Cloud/Container deployment
        return {
            'host': '0.0.0.0',  # Listen on all interfaces
            'port': int(os.environ.get('PORT', 5000)),
            'debug': False,
            'threaded': True,
            'use_reloader': False
        }
    else:
        # Local development
        return {
            'host': 'localhost',
            'port': 5000,
            'debug': False,
            'use_reloader': False
        }

def setup_logging():
    """Setup production-ready logging"""
    log_level = getattr(logging, Config.LOG_LEVEL)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging with rotation for production
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_security.log'),
            logging.StreamHandler()
        ]
    )
    
    # Apply SSL error filter
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.addFilter(SSLErrorFilter())

if __name__ == "__main__":
    # Setup logging first
    setup_logging()
    
    # Initialize the ORB strategy on startup
    print("Starting Flask server with ORB strategy...")
    print(f"Environment: {'AWS/Cloud' if any([os.environ.get('AWS_EXECUTION_ENV'), os.path.exists('/.dockerenv')]) else 'Local'}")
    
    initialize_strategy()
    
    # Compute initial analytics
    try:
        compute_analytics()
        print("Initial analytics computed")
    except Exception as e:
        print(f"Could not compute initial analytics: {e}")
    
    # Add cleanup function for proper connection management
    import atexit
    
    def cleanup_connections():
        """Cleanup function to properly close connections"""
        global data_stream
        if data_stream:
            print("🔄 Cleaning up data stream connections...")
            data_stream.stop()
    
    atexit.register(cleanup_connections)
    
    # Add health check endpoint before starting
    @app.route('/health')
    def health_check():
        """Health check endpoint for load balancers and monitoring"""
        try:
            # Check if strategy is initialized
            strategy_status = "running" if strategy_engine else "not_initialized"
            
            # Check if data stream is active
            stream_status = "active" if data_stream and hasattr(data_stream, 'is_connected') else "inactive"
            
            return jsonify({
                'status': 'healthy',
                'strategy': strategy_status,
                'data_stream': stream_status,
                'timestamp': datetime.now().isoformat()
            }), 200
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    # Get deployment configuration
    config = get_deployment_config()
    print(f"🚀 Starting ORB Trading Bot in {config.get('environment', 'production')} mode")
    print(f"📊 Dashboard will be available at: http://{config['host']}:{config['port']}")
    print(f"❤️  Health check endpoint: http://{config['host']}:{config['port']}/health")
    
    # Start Flask application
    app.run(**config)

"""
ORB (Opening Range Breakout) Trading System
==========================================

A comprehensive algorithmic trading system that implements 15-minute Opening Range Breakout 
strategy with real-time execution, web dashboard, and performance analytics.

## Quick Start:

1. Install dependencies:
   pip install -r requirements.txt

2. Configure your Alpaca API keys in sessions_config.json

3. Run the system:
   python webhook_server.py

4. View the dashboard:
   http://localhost:5000/dashboard

## API Endpoints:

- GET /dashboard         - Web dashboard with charts and analytics  
- GET /status           - Account and system status
- GET /metrics          - Strategy KPIs (P&L, win rate, Sharpe, etc.)
- GET /equity           - Equity curve data (strategy vs buy & hold)
- GET /trades_table     - Detailed trades with MFE/MAE
- GET /trades           - All trades log data
- GET /chart            - Generate trading activity chart
- POST /backtest        - Run historical backtest

## Backtest Example:

curl -X POST http://localhost:5000/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TSLA", "start": "2025-01-01", "end": "2025-09-12"}'

## Strategy Features:

- 15-minute Opening Range (6:30-6:45 AM Pacific)
- 50-EMA trend filter
- 2:1 reward-to-risk ratio
- Single trade per day limit
- Real-time websocket data feed with REST fallback
- Comprehensive performance analytics
- State persistence across restarts

The system runs the strategy automatically in the background while serving
the web interface and API endpoints.
"""
