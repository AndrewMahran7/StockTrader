#!/bin/bash

# Quick restart script to test the trading bot
echo "🔄 Restarting ORB Trading Bot..."

# Kill existing instances
pkill -f "python webhook_server.py" 2>/dev/null || true

# Wait a moment
sleep 2

echo "🚀 Starting ORB Trading Bot with backup data..."

# Start in background and show output
python webhook_server.py &
FLASK_PID=$!

echo "📊 Flask server started with PID: $FLASK_PID"
echo "🌐 Dashboard: http://localhost:5000"
echo "❤️  Health: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for user interrupt
trap 'echo "Stopping..."; kill $FLASK_PID 2>/dev/null; exit 0' INT

# Show live output
wait $FLASK_PID