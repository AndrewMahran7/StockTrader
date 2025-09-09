from flask import Flask, request, jsonify
from alpaca_trade_api.rest import REST, TimeFrame
import datetime
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json

app = Flask(__name__)

# Configuration file path
CONFIG_FILE = "sessions_config.json"

def load_sessions_config():
    """Load session configurations from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file {CONFIG_FILE} not found!")
        return {"sessions": {}, "default_session": "session1"}
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        return {"sessions": {}, "default_session": "session1"}

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
                print(f"✅ Initialized session '{session_id}' ({session_config['name']})")
            except Exception as e:
                print(f"❌ Failed to initialize session '{session_id}': {e}")
        else:
            print(f"⏸️ Session '{session_id}' is disabled")
    
    return sessions, config.get("default_session", "session1")

# Initialize all sessions
SESSIONS, DEFAULT_SESSION = initialize_sessions()

# Dynamic log file management
TRADES_LOG_HEADERS = ["timestamp", "ticker", "action", "quantity", "price", "total_value", "buying_power_before", "buying_power_after", "portfolio_value", "notes", "session"]

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
            print(f"📄 Created log file: {log_file}")

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

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
