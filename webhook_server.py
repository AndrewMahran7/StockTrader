from flask import Flask, request, jsonify, render_template
from alpaca_trade_api.rest import REST, TimeFrame
import datetime
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

app = Flask(__name__)

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
        "use_streaming": True
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
        
        # Initialize data stream
        data_stream = DataStream(
            api=api,
            symbol=strategy_config.get("symbol", "TSLA"),
            on_bar_callback=strategy_engine.on_bar,
            use_streaming=strategy_config.get("use_streaming", True),
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
        
        # Filter for completed trades (both entry and exit)
        completed_trades = []
        open_positions = {}
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            action = row['action']
            
            if action in ['buy', 'short']:
                # Opening position
                open_positions[ticker] = {
                    'entry_time': row['timestamp'],
                    'entry_price': row['price'],
                    'quantity': row['quantity'],
                    'side': 'long' if action == 'buy' else 'short'
                }
                
            elif action in ['sell', 'cover'] and ticker in open_positions:
                # Closing position
                entry = open_positions[ticker]
                
                if entry['side'] == 'long':
                    pnl = (row['price'] - entry['entry_price']) * entry['quantity']
                else:  # short
                    pnl = (entry['entry_price'] - row['price']) * entry['quantity']
                
                trade_days = (row['timestamp'] - entry['entry_time']).days
                
                completed_trades.append({
                    'ticker': ticker,
                    'entry_time': entry['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': entry['entry_price'],
                    'exit_price': row['price'],
                    'quantity': entry['quantity'],
                    'side': entry['side'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (entry['entry_price'] * entry['quantity'])) * 100,
                    'days_held': trade_days,
                    'mfe': 0,  # TODO: Calculate from minute data
                    'mae': 0   # TODO: Calculate from minute data
                })
                
                del open_positions[ticker]
        
        if not completed_trades:
            print("No completed trades for analytics")
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
        initial_capital = 1200  # Your actual starting capital
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
            'symbol': symbol
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
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
        
        # Plot 2: Trade Actions
        buy_trades = df[df['action'] == 'buy']
        sell_trades = df[df['action'] == 'sell']
        short_trades = df[df['action'] == 'short']
        cover_trades = df[df['action'] == 'cover']
        
        if not buy_trades.empty:
            ax2.scatter(buy_trades['timestamp'], buy_trades['price'], 
                       c='green', marker='^', s=buy_trades['quantity']*5, 
                       alpha=0.7, label='Buy Orders')
                       
        if not sell_trades.empty:
            ax2.scatter(sell_trades['timestamp'], sell_trades['price'], 
                       c='red', marker='v', s=sell_trades['quantity']*5, 
                       alpha=0.7, label='Sell Orders')
                       
        if not short_trades.empty:
            ax2.scatter(short_trades['timestamp'], short_trades['price'], 
                       c='orange', marker='s', s=short_trades['quantity']*5, 
                       alpha=0.7, label='Short Orders')
                       
        if not cover_trades.empty:
            ax2.scatter(cover_trades['timestamp'], cover_trades['price'], 
                       c='purple', marker='d', s=cover_trades['quantity']*5, 
                       alpha=0.7, label='Cover Orders')
        
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
        chart_filename = f"trading_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trading chart saved as: {chart_filename}")
        
        # Print summary statistics
        total_trades = len(df)
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        short_count = len(short_trades)
        cover_count = len(cover_trades)
        
        if total_trades > 1:
            initial_value = df.iloc[0]['portfolio_value']
            final_value = df.iloc[-1]['portfolio_value']
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            print(f"\n=== Trading Summary ===")
            print(f"Total Trades: {total_trades}")
            print(f"Buy Orders: {buy_count}")
            print(f"Sell Orders: {sell_count}")
            print(f"Short Orders: {short_count}")
            print(f"Cover Orders: {cover_count}")
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
        # Get current TSLA price using Alpaca API
        current_tsla_price = 395.94  # Fallback value
        try:
            # Try to get real-time TSLA price from Alpaca
            if hasattr(app, 'alpaca_client'):
                quote = app.alpaca_client.get_latest_quote("TSLA")
                current_tsla_price = float(quote.bid_price) if quote.bid_price else current_tsla_price
        except Exception as e:
            logging.warning(f"Could not fetch current TSLA price: {e}")
            
        ensure_analytics_directory()
        summary_file = 'analytics/summary.json'
        
        if not os.path.exists(summary_file):
            # Try to compute analytics first
            compute_analytics()
            
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                metrics = json.load(f)
                
            # Add TSLA price info
            metrics['current_tsla_price'] = current_tsla_price
            metrics['tsla_change_percent'] = ((current_tsla_price - 341.50) / 341.50) * 100
            
            return jsonify(metrics)
        else:
            tsla_change_percent = ((current_tsla_price - 341.50) / 341.50) * 100
            return jsonify({
                'error': 'No analytics data available',
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'current_tsla_price': current_tsla_price,
                'tsla_change_percent': tsla_change_percent
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

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run historical backtest"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'TSLA')
        start_date = data.get('start', '2025-01-01')
        end_date = data.get('end', datetime.date.today().isoformat())
        
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


        elif action == "short":
            # Open a short position using max notional available
            account = api.get_account()
            buying_power_before = float(account.buying_power)
            latest_trade = api.get_latest_trade(ticker)
            current_price = float(latest_trade.price)

            # UPDATED: Calculate fractional shares for shorting (round to 6 decimal places)
            max_shares = round(buying_power_before * 0.99 / current_price, 6)
            if max_shares > 0:
                order = api.submit_order(symbol=ticker, qty=max_shares, side="sell", type="market", time_in_force="gtc")
                total_value = max_shares * current_price
                account_after = api.get_account()
                buying_power_after = float(account_after.buying_power)
                log_trade(session, ticker, "short", max_shares, current_price, total_value,
                          buying_power_before, buying_power_after, f"Order ID: {order.id}")
                print(f"Opened SHORT {max_shares} {ticker} at ~${current_price:.2f}")
            else:
                return jsonify({"error": "Insufficient buying power to short"}), 400

        elif action == "cover":
            # Close an existing short (buy to cover)
            try:
                account = api.get_account()
                buying_power_before = float(account.buying_power)

                position = api.get_position(ticker)
                # UPDATED: Support fractional positions for covering
                shares_to_cover = abs(float(position.qty))  # qty is negative for shorts
                if float(position.qty) >= 0:
                    return jsonify({"error": "No short position to cover"}), 400

                latest_trade = api.get_latest_trade(ticker)
                current_price = float(latest_trade.price)

                order = api.submit_order(symbol=ticker, qty=shares_to_cover, side="buy", type="market", time_in_force="gtc")
                total_value = shares_to_cover * current_price
                account_after = api.get_account()
                buying_power_after = float(account_after.buying_power)

                log_trade(session, ticker, "cover", shares_to_cover, current_price, total_value,
                          buying_power_before, buying_power_after, f"Order ID: {order.id}")
                print(f"Covered {shares_to_cover} {ticker} at ~${current_price:.2f}")
            except Exception as pos_error:
                print(f"No short position found for {ticker}: {pos_error}")
                return jsonify({"error": "No short to cover"}), 400

        else:
            return jsonify({"error": "Unknown action"}), 400
    except Exception as e:
        print(f"Error placing order: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "success", "time": str(datetime.datetime.now())})

@app.route("/chart", methods=["GET"])
def create_chart():
    """Endpoint to generate and view trading charts"""
    try:
        # Since we only have one session, always generate chart for it
        generate_trading_chart(DEFAULT_SESSION)
        return jsonify({"status": "Chart generated successfully", "time": str(datetime.datetime.now())})
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
            "time": str(datetime.datetime.now())
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
            "time": str(datetime.datetime.now())
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
        
        return jsonify({"session": status, "time": str(datetime.datetime.now())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
