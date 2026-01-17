#!/bin/bash

# Zeppelin Tactical OS - Startup Script

echo "🚀 Initializing Zeppelin Tactical OS..."

# Check if port 8050 is free
if lsof -Pi :8050 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8050 is busy. Attempting to clear..."
    lsof -ti:8050 | xargs kill -9 2>/dev/null
    sleep 2
fi

# Start Gunicorn
echo "⚡ Starting Gunicorn Server..."
exec gunicorn --bind 0.0.0.0:8050 --workers 1 --threads 2 realtime_dashboard:server
