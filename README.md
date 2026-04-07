# 🚀 Algorithmic Trading System (Raspberry Pi)

An automated trading bot implementing a **15-minute Opening Range Breakout (ORB)** strategy with live execution through Alpaca Markets, designed to run 24/7 on a Raspberry Pi via systemd.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │◄───┤  Flask Server   │───▶│  Alpaca Markets │
│   (Real-time)   │    │ (Strategy Engine)│    │   Live Trading  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Data Stream &  │
                       │   Analytics     │
                       └─────────────────┘
```

## 📁 Project Structure

```
├── webhook_server.py          # Main entry point (Flask + strategy)
├── strategy_engine.py         # ORB strategy implementation
├── data_stream.py             # Real-time data streaming & market clock
├── backup_data_provider.py    # Yahoo Finance / web scraping fallback
├── data_cache.py              # Price caching to reduce API calls
├── secure_config.py           # Environment variable management
├── auth_security.py           # Authentication & rate limiting
├── .env                       # API keys and config (create from .env.template)
├── .env.template              # Template for .env file
├── state.json                 # Strategy state persistence
├── tsla_price_cache.json      # Price cache
├── requirements.txt           # Python dependencies
├── templates/
│   ├── dashboard.html         # Web dashboard
│   └── login.html             # Login page
├── analytics/                 # Generated analytics data
├── logs/                      # Application logs
└── restart-bot.sh             # Manual restart script
```

## 🚀 Raspberry Pi Setup

### Prerequisites

- Raspberry Pi (3B+ or newer recommended) running Raspberry Pi OS
- Python 3.8+
- Alpaca Markets account with API keys

ssh amahran@100.64.180.1

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libxml2-dev libxslt-dev libfreetype6-dev libpng-dev
```

### 2. Clone and set up

```bash
cd /home/pi
git clone <your-repo-url> trading
cd trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.template .env
nano .env  # Add your Alpaca API keys
```

Required variables:
- `ALPACA_API_KEY` — Your Alpaca API key
- `ALPACA_SECRET_KEY` — Your Alpaca secret key
- `SECRET_KEY` — Random string for Flask sessions

### 4. Test run

```bash
source venv/bin/activate
python3 webhook_server.py
```

Visit `http://<pi-ip>:5000/dashboard` from any device on your network.

### 5. Set up systemd (auto-start on boot)

Create `/etc/systemd/system/trading-bot.service`:

```ini
[Unit]
Description=ORB Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/trading
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/pi/trading/venv/bin/python3 /home/pi/trading/webhook_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot   # Verify it's running
```

View logs:

```bash
journalctl -u trading-bot -f
```

## 📊 Strategy: Opening Range Breakout

- **Opening range**: 6:30–6:45 AM Pacific (first 15 minutes)
- **Entry**: Breakout above range high when price > 50-EMA
- **Stop loss**: Opposite end of opening range
- **Take profit**: 2× range size (2:1 R/R)
- **Limit**: One trade per day
- **Exit**: End-of-day close at 3:50 PM Pacific

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/dashboard` | GET | Web dashboard |
| `/metrics` | GET | Strategy KPIs |
| `/equity` | GET | Equity curve data |
| `/trades_table` | GET | Detailed trades |
| `/status` | GET | Account & system status |
| `/health` | GET | Health check |

## 🔒 Security

- API key authentication for webhook endpoints
- IP-based rate limiting and blocking
- Login-protected dashboard
- Environment variables for secrets (never hardcoded)

## 🚨 Disclaimer

This system is for educational purposes. Live trading involves substantial risk of loss. Always test in paper trading first.
