#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# stop_r1_server.sh — Stop a running DeepSeek-R1 server.
# ==============================================================================

R1_PORT=5002
R1_PID_FILE="/tmp/r1_server_${R1_PORT}.pid"

echo "Stopping DeepSeek-R1 server..."

if [ ! -f "$R1_PID_FILE" ]; then
    echo "No PID file found at $R1_PID_FILE"
    if nc -z localhost $R1_PORT 2>/dev/null; then
        echo "Warning: Port $R1_PORT is in use but no PID file exists."
        if command -v lsof &> /dev/null; then
            PROCS=$(lsof -i :$R1_PORT -t)
            if [ -n "$PROCS" ]; then
                echo "Found processes: $PROCS"
                read -p "Kill these processes? [y/N] " -n 1 -r; echo
                [[ $REPLY =~ ^[Yy]$ ]] && for pid in $PROCS; do kill -9 $pid; done
            fi
        fi
    else
        echo "No process is listening on port $R1_PORT"
    fi
    exit 0
fi

PID=$(cat $R1_PID_FILE)
echo "Found DeepSeek-R1 server with PID: $PID"

if kill -0 $PID 2>/dev/null; then
    echo "Sending SIGTERM to PID $PID"
    kill -15 $PID
    for i in {1..30}; do
        kill -0 $PID 2>/dev/null || { echo "Server shut down successfully."; break; }
        sleep 1
    done
    kill -0 $PID 2>/dev/null && { echo "Force killing..."; kill -9 $PID; sleep 2; }
else
    echo "Process with PID $PID does not exist."
fi

[ -f "$R1_PID_FILE" ] && rm -f "$R1_PID_FILE" && echo "Removed PID file."

nc -z localhost $R1_PORT 2>/dev/null && echo "Warning: Port $R1_PORT still in use." || echo "Port $R1_PORT is now free."
echo "DeepSeek-R1 server stop script completed."
