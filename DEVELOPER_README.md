# 🛠️ Developer Guide - Algorithmic Trading System

This guide provides detailed setup instructions for developers who want to run, modify, or contribute to this algorithmic trading system.

## 🚀 Quick Setup

### 1. Prerequisites

- **Python 3.8+** (tested on 3.9-3.12)
- **Git** for version control
- **Alpaca Markets Account** (paper or live trading)
- **Code Editor** (VS Code recommended)

### 2. Installation Steps

```powershell
# Clone the repository
git clone <your-repo-url>
cd Trading

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import alpaca_trade_api, flask, pandas, plotly; print('All dependencies installed')"
```

### 3. Configuration

#### API Keys Setup
1. **Create Alpaca account** at [alpaca.markets](https://alpaca.markets)
2. **Get API credentials** from the dashboard
3. **Update sessions_config.json**:

```json
{
  "sessions": {
    "main": {
      "name": "Paper_Trading",
      "api_key": "YOUR_ALPACA_API_KEY",
      "api_secret": "YOUR_ALPACA_SECRET_KEY",
      "base_url": "https://paper-api.alpaca.markets",
      "enabled": true
    }
  },
  "default_session": "main"
}
```

⚠️ **Important**: Use paper trading URL initially: `https://paper-api.alpaca.markets`

### 4. Running the System

```powershell
# Start the Flask server
python webhook_server.py

# Expected output:
# * Running on http://127.0.0.1:5000
# * Debug mode: on
# Strategy engine initialized
# Data stream started
```

### 5. Access Dashboard

Open your browser and navigate to: `http://localhost:5000/dashboard`

## 📁 Project Architecture

### Core Components

```
webhook_server.py       # Main Flask app (routing, analytics, API)
├── /dashboard         # Web dashboard endpoint  
├── /metrics          # Performance metrics API
├── /equity           # Equity curve data API
├── /trades_table     # Recent trades API
└── /backtest         # Historical backtest API

strategy_engine.py      # ORB strategy implementation
├── ORBStrategy()      # Main strategy class
├── calculate_ema()    # EMA calculation
├── execute_trade()    # Trade execution logic
└── should_trade()     # Entry condition checks

data_stream.py         # Real-time data & market clock
├── DataStream()       # WebSocket/REST data provider
├── MarketClock()      # Trading hours management
└── BacktestDataProvider() # Historical data for testing
```

### Data Flow

```
Market Data → DataStream → ORBStrategy → Trade Execution
     ↓              ↓           ↓             ↓
WebSocket/REST → Processing → Decision → Alpaca API
     ↓              ↓           ↓             ↓
 Live Prices → Range Calc → Entry/Exit → Order Fill
     ↓              ↓           ↓             ↓
  Dashboard ← Analytics ← CSV Log ← Trade Result
```

## 🔧 Development Workflow

### Local Development

```powershell
# Development mode with auto-reload
python webhook_server.py

# Test individual components
python -c "from strategy_engine import ORBStrategy; print('Strategy OK')"
python -c "from data_stream import DataStream; print('DataStream OK')"
```

### Testing

```powershell
# Run strategy with historical data
python -c "
from data_stream import BacktestDataProvider
from strategy_engine import ORBStrategy
import datetime

# Test with specific date
test_date = datetime.date(2024, 8, 26)
provider = BacktestDataProvider('TSLA', test_date)
strategy = ORBStrategy('TSLA', paper_trading=True)
print('Backtest ready')
"
```

### Debugging

1. **Check logs**: All trades logged to `trades_log_main.csv`
2. **Monitor terminal**: Real-time output shows strategy decisions
3. **Dashboard**: Visual confirmation of system status
4. **Analytics**: JSON files in `analytics/` folder

## 📊 Key Files Explained

### `webhook_server.py` (Main Application)
- **Flask routes** for web interface and API
- **Analytics computation** (equity curve, performance metrics)
- **Trade logging** with enhanced CSV schema
- **Error handling** and logging

### `strategy_engine.py` (Trading Logic)
- **ORBStrategy class** implementing the 15-minute opening range breakout
- **EMA calculation** for trend filtering
- **Trade execution** with proper risk management
- **State persistence** across restarts

### `data_stream.py` (Data Management)
- **MarketClock** for trading hours detection
- **DataStream** with WebSocket/REST fallback
- **BacktestDataProvider** for historical testing

### `templates/dashboard.html` (Frontend)
- **Plotly.js charts** for interactive visualizations
- **Real-time updates** via JavaScript fetch
- **Professional UI** with dark theme

## 🛡️ Safety & Testing

### Paper Trading First
Always start with paper trading:
```json
"base_url": "https://paper-api.alpaca.markets"
```

### Risk Controls
- **Single trade per day** limit
- **Position sizing** based on available capital  
- **Stop loss** at opening range boundary
- **Market hours** validation

### Monitoring Tools
- **Dashboard**: Real-time performance tracking
- **CSV logs**: Detailed trade history
- **JSON analytics**: Computed metrics
- **Terminal output**: Debug information

## 🔍 Troubleshooting

### Common Issues

1. **Module not found**
   ```powershell
   pip install -r requirements.txt
   ```

2. **API connection failed**
   - Check API keys in `sessions_config.json`
   - Verify internet connection
   - Confirm Alpaca account status

3. **Dashboard not loading**
   - Ensure Flask server is running
   - Check `http://localhost:5000/dashboard`
   - Look for error messages in terminal

4. **No trades executing**
   - Verify market hours (6:30-6:45 AM Pacific for range setup)
   - Check if paper trading mode is enabled
   - Review strategy conditions in terminal output

### Debug Commands

```powershell
# Test API connection
python -c "
from alpaca_trade_api.rest import REST
import json
with open('sessions_config.json') as f:
    config = json.load(f)
    session = config['sessions']['main']
api = REST(session['api_key'], session['api_secret'], session['base_url'])
print('Account:', api.get_account().status)
"

# Test data stream
python -c "
from data_stream import DataStream
stream = DataStream('TSLA')
print('Data stream initialized')
"

# Test strategy engine
python -c "
from strategy_engine import ORBStrategy
strategy = ORBStrategy('TSLA', paper_trading=True)
print('Strategy initialized')
"
```

## 📝 Code Style & Contribution

### Development Standards
- **Python PEP 8** style guide
- **Type hints** where applicable
- **Docstrings** for functions and classes
- **Error handling** with try/catch blocks
- **Logging** for debugging and monitoring

### Making Changes
1. **Create feature branch**: `git checkout -b feature-name`
2. **Test thoroughly** in paper trading mode
3. **Update documentation** if needed
4. **Submit pull request** with detailed description

## 🚨 Production Deployment

### Before Going Live
1. ✅ **Test extensively** with paper trading
2. ✅ **Verify risk controls** are working
3. ✅ **Monitor for several days** in paper mode
4. ✅ **Start small** with limited capital
5. ✅ **Set up monitoring** and alerts

### Live Trading Setup
```json
{
  "sessions": {
    "main": {
      "name": "Live_Trading",
      "api_key": "YOUR_LIVE_API_KEY",
      "api_secret": "YOUR_LIVE_SECRET_KEY", 
      "base_url": "https://api.alpaca.markets",
      "enabled": true
    }
  },
  "default_session": "main"
}
```

## 📞 Support & Resources

- **Alpaca API Docs**: [alpaca.markets/docs](https://alpaca.markets/docs)
- **Flask Documentation**: [flask.palletsprojects.com](https://flask.palletsprojects.com)
- **Plotly.js Docs**: [plotly.com/javascript](https://plotly.com/javascript)
- **Pandas Guide**: [pandas.pydata.org](https://pandas.pydata.org)

---

🔧 **Happy Trading & Coding!** Remember: Always test in paper trading first, and never risk more than you can afford to lose.
