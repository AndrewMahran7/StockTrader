# ORB Trading Bot - AWS Deployment Guide

This guide will help you deploy your ORB trading bot to AWS EC2 for 24/7 operation.

## 📋 Prerequisites

1. **AWS Account** with billing set up
2. **AWS CLI** installed and configured
3. **Your Alpaca API credentials** (API key and secret)
4. **Basic command line knowledge**

## 🚀 Quick Deployment

### Step 1: Prepare Your Environment

1. **Install AWS CLI** (if not already installed):
   ```bash
   # Windows (using PowerShell as admin)
   Invoke-WebRequest -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" -OutFile "AWSCLIV2.msi"
   Start-Process msiexec.exe -Wait -ArgumentList '/I AWSCLIV2.msi /quiet'
   ```

2. **Configure AWS CLI**:
   ```bash
   aws configure
   ```
   Enter your AWS Access Key ID, Secret Access Key, and preferred region (e.g., `us-east-1`).

### Step 2: Deploy to AWS

1. **Make the deployment script executable** (if on Linux/Mac):
   ```bash
   chmod +x deploy-aws.sh
   ```

2. **Run the deployment script**:
   ```bash
   # Linux/Mac
   ./deploy-aws.sh
   
   # Windows (use Git Bash or WSL)
   bash deploy-aws.sh
   ```

3. **Wait for deployment** - This takes about 5-10 minutes.

### Step 3: Upload Your Trading Code

After the script completes, you'll get an IP address. Upload your code:

```bash
# Upload all your trading files
scp -i trading-bot-key.pem -r *.py ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/

# Upload your .env file with real Alpaca credentials
scp -i trading-bot-key.pem .env ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/
```

### Step 4: Start Your Trading Bot

```bash
# SSH into your instance
ssh -i trading-bot-key.pem ec2-user@YOUR_IP_ADDRESS

# Start the trading bot
cd trading
docker-compose up --build -d

# View logs
docker logs -f trading-bot
```

## 🔐 Security Configuration

### Private Dashboard Access (Recommended)

To keep your dashboard private, set up SSH tunneling:

```bash
# Create SSH tunnel from your local machine
ssh -i trading-bot-key.pem -L 5000:localhost:5000 ec2-user@YOUR_IP_ADDRESS

# Now access dashboard at: http://localhost:5000
```

### Update Security Group (Optional)

Remove public access to port 5000:
```bash
aws ec2 revoke-security-group-ingress \
    --group-name trading-bot-sg \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0
```

## 📊 Monitoring and Management

### Check Bot Status
```bash
ssh -i trading-bot-key.pem ec2-user@YOUR_IP_ADDRESS
docker ps                    # Check if container is running
docker logs trading-bot      # View recent logs
docker stats trading-bot     # Monitor resource usage
```

### View Trading Analytics
```bash
# Download analytics files to your local machine
scp -i trading-bot-key.pem -r ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/analytics/ ./
```

### Restart the Bot
```bash
ssh -i trading-bot-key.pem ec2-user@YOUR_IP_ADDRESS
cd trading
docker-compose restart
```

## 💰 Cost Optimization

### Expected Costs (US East 1)
- **t3.small instance**: ~$15-20/month
- **EBS storage**: ~$2-5/month
- **Data transfer**: ~$1-3/month
- **Total**: ~$18-28/month

### Cost-Saving Tips
1. **Use Spot Instances** for ~70% savings (but may be interrupted)
2. **Stop instance during weekends** when markets are closed
3. **Use t3.micro** if your strategy doesn't need much CPU

### Auto-Stop During Weekends
```bash
# Create a cron job to stop Friday at market close
0 21 * * 5 /usr/local/bin/docker-compose -f /home/ec2-user/trading/docker-compose.yml stop

# Start Monday morning before market open
0 8 * * 1 /usr/local/bin/docker-compose -f /home/ec2-user/trading/docker-compose.yml start
```

## 🔧 Troubleshooting

### Bot Not Starting
```bash
# Check container logs
docker logs trading-bot

# Check if .env file has correct credentials
cat /home/ec2-user/trading/.env

# Rebuild container
docker-compose down
docker-compose up --build -d
```

### Can't Access Dashboard
```bash
# Check if port 5000 is open
sudo netstat -tlnp | grep :5000

# Check security group allows port 5000
aws ec2 describe-security-groups --group-names trading-bot-sg
```

### High CPU Usage
```bash
# Monitor resource usage
docker stats trading-bot
htop

# Consider upgrading to t3.medium if needed
```

## 🔄 Updates and Maintenance

### Update Your Code
```bash
# Upload new code
scp -i trading-bot-key.pem *.py ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/

# SSH and restart
ssh -i trading-bot-key.pem ec2-user@YOUR_IP_ADDRESS
cd trading
docker-compose up --build -d
```

### Backup Your Data
```bash
# Download all analytics and logs
scp -i trading-bot-key.pem -r ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/analytics/ ./backups/
scp -i trading-bot-key.pem -r ec2-user@YOUR_IP_ADDRESS:/home/ec2-user/trading/logs/ ./backups/
```

## 🆘 Support

If you encounter issues:

1. **Check the logs**: `docker logs trading-bot`
2. **Verify credentials**: Make sure your Alpaca API keys are correct
3. **Check AWS costs**: Monitor your AWS billing dashboard
4. **Test locally first**: Always test changes on your local machine

## 🎯 Next Steps

Once deployed:
1. ✅ Monitor your first few trades carefully
2. ✅ Set up billing alerts in AWS
3. ✅ Consider additional risk management rules
4. ✅ Plan for strategy backtesting and optimization

Your ORB trading bot is now running 24/7 in the cloud! 🚀