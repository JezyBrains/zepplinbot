#!/bin/bash

# ZEPPELIN PRO - SAFE START UTILITY
# This script ensures a clean server launch by clearing existing processes on port 8050.

PORT=8050
SCRIPT="realtime_dashboard.py"

echo "--------------------------------------------------"
echo "🚀 INITIATING SAFE START: Tactical OS Dashboard"
echo "--------------------------------------------------"

# Find and kill process on port 8050
PID=$(lsof -ti:$PORT)
if [ ! -z "$PID" ]; then
    echo "⚠️  Port $PORT is occupied by PID: $PID. Terminating..."
    kill -9 $PID
    sleep 1
    echo "✅ Port $PORT cleared."
else
    echo "✅ Port $PORT is available."
fi

# Final check for any hung python processes with the script name
HUNG_PIDS=$(ps aux | grep "$SCRIPT" | grep -v grep | awk '{print $2}')
if [ ! -z "$HUNG_PIDS" ]; then
    echo "⚠️  Found hung script processes: $HUNG_PIDS. Killing..."
    kill -9 $HUNG_PIDS
    sleep 1
fi

echo "🛰️  Starting Dashboard Engine..."
echo "🌐 Dashboard will be available at http://localhost:$PORT"

# Start the server in the background
python3 $SCRIPT &

# Monitor the first few lines of output to verify load
sleep 2
if ps -p $! > /dev/null; then
   echo "🔗 SERVER ONLINE. PID: $!"
else
   echo "❌ SERVER FAILED TO START. Check the logs."
   exit 1
fi

echo "--------------------------------------------------"
