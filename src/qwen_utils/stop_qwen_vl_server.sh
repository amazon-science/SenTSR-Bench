#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# stop_qwen_vl_server.sh
#
# Script to stop a running Qwen2.5-VL server.
# This script:
# 1. Finds the PID file for the Qwen2.5-VL server
# 2. Sends a SIGTERM signal to gracefully shut down the server
# ==============================================================================

# Qwen2.5-VL server PID file location
QWEN_VL_PORT=5003
QWEN_VL_PID_FILE="/tmp/qwen_vl_server_${QWEN_VL_PORT}.pid"

echo "Stopping Qwen2.5-VL server..."

# Check if PID file exists
if [ ! -f "$QWEN_VL_PID_FILE" ]; then
    echo "No PID file found at $QWEN_VL_PID_FILE"

    # Check if there's a process listening on the Qwen2.5-VL port
    if nc -z localhost $QWEN_VL_PORT 2>/dev/null; then
        echo "Warning: Port $QWEN_VL_PORT is in use but no PID file exists."
        echo "Finding processes using port $QWEN_VL_PORT:"

        # Find and display processes using the port
        if command -v lsof &> /dev/null; then
            PROCS=$(lsof -i :$QWEN_VL_PORT -t)
            if [ -n "$PROCS" ]; then
                echo "Found processes: $PROCS"

                # Ask if we should kill these processes
                read -p "Do you want to kill these processes? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "Killing processes using port $QWEN_VL_PORT"
                    for pid in $PROCS; do
                        echo "Killing process $pid"
                        kill -9 $pid
                    done
                else
                    echo "No processes were killed. Please manually stop the server."
                fi
            else
                echo "No processes found using lsof."
            fi
        else
            echo "lsof command not available, cannot find processes by port."
        fi
    else
        echo "No process is listening on port $QWEN_VL_PORT"
    fi

    exit 0
fi

# Read PID from file
PID=$(cat $QWEN_VL_PID_FILE)
echo "Found Qwen2.5-VL server with PID: $PID"

# Check if the process exists
if kill -0 $PID 2>/dev/null; then
    echo "Sending SIGTERM to PID $PID"
    kill -15 $PID

    # Wait for the process to terminate
    echo "Waiting for server to shut down..."
    for i in {1..30}; do
        if ! kill -0 $PID 2>/dev/null; then
            echo "Server shut down successfully."
            break
        fi
        sleep 1
    done

    # If process still exists, force kill
    if kill -0 $PID 2>/dev/null; then
        echo "Server did not shut down gracefully, sending SIGKILL..."
        kill -9 $PID
        sleep 2
    fi
else
    echo "Process with PID $PID does not exist or is not accessible."
fi

# Remove PID file
if [ -f "$QWEN_VL_PID_FILE" ]; then
    echo "Removing PID file: $QWEN_VL_PID_FILE"
    rm -f "$QWEN_VL_PID_FILE"
fi

# Final check
if nc -z localhost $QWEN_VL_PORT 2>/dev/null; then
    echo "Warning: Port $QWEN_VL_PORT is still in use after stopping the server."
    echo "You might need to manually kill the remaining processes."
else
    echo "Port $QWEN_VL_PORT is now free."
fi

echo "Qwen2.5-VL server stop script completed."
