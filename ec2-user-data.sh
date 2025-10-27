#!/bin/bash

# EC2 User Data Script - Runs on instance startup
# This script installs Docker and deploys the trading bot

yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install -y git

# Create trading directory
mkdir -p /home/ec2-user/trading
cd /home/ec2-user/trading

# Clone or copy trading bot code (you'll need to upload this manually or use S3)
# For now, create a placeholder structure
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/data

ENV FLASK_APP=webhook_server.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "--log-level", "info", "webhook_server:app"]
EOF

# Create docker-compose file
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  trading-bot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./analytics:/app/analytics
    restart: unless-stopped
    container_name: trading-bot
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
EOF

# Create placeholder requirements.txt
cat > requirements.txt << 'EOF'
Flask==2.3.3
Flask-CORS==4.0.0
alpaca-trade-api==3.1.1
websocket-client==1.6.3
pandas==2.1.1
numpy==1.25.2
matplotlib==3.7.2
scipy==1.11.3
plotly==5.17.0
pytz==2023.3.post1
python-dateutil==2.8.2
python-dotenv==1.0.0
gunicorn==21.2.0
requests==2.31.0
websockets==11.0.3
psutil==5.9.5
EOF

# Create placeholder .env file (user needs to update this)
cat > .env << 'EOF'
# Update these with your actual Alpaca API credentials
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
SECRET_KEY=your_secret_key_here
LOG_LEVEL=INFO
TRADING_SYMBOL=TSLA
POLLING_SECONDS=60
USE_STREAMING=true
AWS_REGION=us-east-1
FLASK_ENV=production
EOF

# Set proper ownership
chown -R ec2-user:ec2-user /home/ec2-user/trading

# Create startup script
cat > /home/ec2-user/start-trading-bot.sh << 'EOF'
#!/bin/bash
cd /home/ec2-user/trading
docker-compose down
docker-compose up --build -d
docker logs -f trading-bot
EOF

chmod +x /home/ec2-user/start-trading-bot.sh
chown ec2-user:ec2-user /home/ec2-user/start-trading-bot.sh

# Create a systemd service to auto-start the trading bot
cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=ORB Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/trading
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable trading-bot

echo "EC2 instance setup complete. Trading bot will start automatically."
echo "Remember to upload your code and update the .env file with your Alpaca API credentials."