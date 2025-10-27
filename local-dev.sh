#!/bin/bash

# Local Development Helper Script
# Use this to test your deployment locally before AWS

echo "🔧 Setting up local development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install it first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating template..."
    python3 -c "
from secure_config import create_env_file
create_env_file()
"
    echo "📝 Please update the .env file with your Alpaca API credentials before continuing."
    echo "   Edit: .env"
    read -p "Press Enter when you've updated the .env file..."
fi

echo "🐳 Building and starting containers..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if the service is healthy
echo "🔍 Checking service health..."
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Trading bot is healthy and running!"
    echo ""
    echo "📊 Dashboard: http://localhost:5000"
    echo "❤️  Health check: http://localhost:5000/health"
    echo ""
    echo "📜 To view logs:"
    echo "   docker logs -f trading-bot"
    echo ""
    echo "🛑 To stop:"
    echo "   docker-compose down"
else
    echo "⚠️  Service might not be ready yet. Check logs:"
    echo "   docker logs trading-bot"
fi

echo ""
echo "🎯 Ready for AWS deployment!"
echo "   When you're satisfied with local testing, run: ./deploy-aws.sh"