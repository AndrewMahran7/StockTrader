# 🛡️ Secure Production Deployment Guide

This guide shows you how to securely deploy your ORB trading system with enterprise-grade security features.

## 🚀 Quick Security Setup

### 1. Install Security Dependencies

```bash
pip install flask-login flask-limiter flask-talisman python-dotenv bcrypt
```

### 2. Create Environment File

Copy `.env.template` to `.env` and update with your credentials:

```bash
cp .env.template .env
```

Edit `.env`:
```bash
# Security Configuration
SECRET_KEY=your-super-secret-flask-key-change-this-to-random-string
ADMIN_USERNAME=admin
ADMIN_PASSWORD=YourSecurePassword123!

# Alpaca Trading API Configuration
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_SECRET_KEY=your-alpaca-secret-key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# API Security - Generate random key for webhooks
WEBHOOK_API_KEY=your-random-32-character-api-key

# Production Settings
FLASK_ENV=production
ENABLE_HTTPS_REDIRECT=true
ENABLE_SECURITY_HEADERS=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_PER_HOUR=100
```

### 3. Generate Secure Keys

```python
# Run this to generate secure keys
import secrets
print("SECRET_KEY:", secrets.token_hex(32))
print("WEBHOOK_API_KEY:", secrets.token_urlsafe(32))
```

## 🔐 Security Features Implemented

### **1. User Authentication**
- **Login required** for dashboard access
- **Session management** with secure cookies
- **Failed login tracking** and IP blocking
- **Automatic logout** after inactivity

### **2. API Key Protection**
- **Webhook endpoints** require API key in header: `X-API-Key`
- **Separate credentials** for web users vs API access
- **Invalid key attempts** are logged and monitored

### **3. Rate Limiting**
- **10 requests per minute** (configurable)
- **100 requests per hour** (configurable)  
- **Per-IP tracking** prevents abuse
- **Automatic blocking** of excessive requests

### **4. HTTPS & Security Headers**
- **Forced HTTPS** redirects (production)
- **HSTS headers** prevent downgrade attacks
- **CSP headers** prevent XSS attacks
- **Referrer policy** protection

### **5. Security Monitoring**
- **Failed login tracking** with IP blocking
- **Suspicious activity logging** 
- **Trade execution audit trail**
- **Security event alerting**

## 🌐 Hosting Options

### **Option 1: Cloud VPS (Recommended)**

**DigitalOcean/Linode/Vultr:**
```bash
# 1. Create Ubuntu 22.04 VPS ($5-10/month)
# 2. SSH into server
ssh root@your-server-ip

# 3. Install Python and dependencies
apt update && apt upgrade -y
apt install python3 python3-pip nginx certbot python3-certbot-nginx -y

# 4. Clone your code
git clone <your-repo>
cd Trading

# 5. Install dependencies
pip3 install -r requirements.txt

# 6. Set up environment
cp .env.template .env
nano .env  # Add your credentials

# 7. Create systemd service
sudo nano /etc/systemd/system/trading.service
```

**Service file (`/etc/systemd/system/trading.service`):**
```ini
[Unit]
Description=ORB Trading System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/ubuntu/Trading
Environment=FLASK_ENV=production
ExecStart=/usr/bin/python3 webhook_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start service:**
```bash
sudo systemctl enable trading
sudo systemctl start trading
sudo systemctl status trading
```

### **Option 2: Railway/Heroku (Easy)**

**Railway (Recommended):**
1. Connect GitHub repo to Railway
2. Add environment variables in Railway dashboard
3. Deploy automatically

**Heroku:**
1. Create `Procfile`: `web: python webhook_server.py`
2. Set environment variables in Heroku settings
3. Deploy via Git

### **Option 3: AWS/Google Cloud (Advanced)**

Use containerized deployment with Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "webhook_server.py"]
```

## 🔒 SSL/HTTPS Setup

### **With Nginx (VPS hosting):**

```nginx
# /etc/nginx/sites-available/trading
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    
    # Hide sensitive files
    location ~ /\.env {
        deny all;
        return 404;
    }
    
    location ~ \.json$ {
        deny all;
        return 404;
    }
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable and get SSL:**
```bash
sudo ln -s /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

## 📱 Access Your Secure System

### **Web Dashboard:**
- URL: `https://your-domain.com/dashboard`
- Login: Username/password from `.env` file
- Features: All trading analytics with authentication

### **API Access (for external systems):**
```bash
# All API requests need the API key header
curl -X POST https://your-domain.com/webhook \
  -H "X-API-Key: your-webhook-api-key" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "action": "buy"}'
```

### **Monitoring Endpoints:**
- **Metrics**: `GET /metrics` (requires login or API key)
- **Trades**: `GET /trades_table` (requires login or API key)
- **Equity**: `GET /equity` (requires login or API key)

## ⚠️ Security Checklist

Before going live, verify:

- [ ] ✅ **Environment variables** set (no hardcoded credentials)
- [ ] ✅ **Strong passwords** for admin account
- [ ] ✅ **HTTPS enabled** (SSL certificate installed)
- [ ] ✅ **Rate limiting** configured and working
- [ ] ✅ **API keys** different from default values
- [ ] ✅ **Firewall rules** restrict unnecessary ports
- [ ] ✅ **Log monitoring** set up for security events
- [ ] ✅ **Backup strategy** for trade data
- [ ] ✅ **Paper trading** tested first
- [ ] ✅ **Session timeouts** configured appropriately

## 🚨 Security Incident Response

If you detect unauthorized access:

1. **Immediately change** all API keys and passwords
2. **Check logs** in `trading_security.log` for incident details
3. **Revoke Alpaca API keys** in your Alpaca dashboard
4. **Block suspicious IPs** in your server firewall
5. **Review trade history** for unauthorized transactions
6. **Update security measures** based on attack vector

## 📞 Production Support

**Before live trading:**
- Test extensively in paper trading mode
- Monitor logs for any errors or security alerts
- Set up automated alerts for trade executions
- Have a rollback plan if issues occur

**Live trading considerations:**
- Start with small position sizes
- Monitor system 24/7 during market hours
- Have manual override procedures ready
- Keep emergency contact info for brokers

---

**🔒 Remember: Your money's security depends on following these practices exactly. Never skip security steps!**
