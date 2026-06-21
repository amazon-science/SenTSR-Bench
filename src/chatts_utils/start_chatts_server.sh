#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# start_chatts_server.sh
#
# Script to start a ChatTS server for faster inference.
# This script:
# 1. Initializes the environment
# 2. Starts the ChatTS server using vLLM
# 3. Checks that the server is running and operational
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/evaluation
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                # project root

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
# ChatTS model paths
CHATTS_MODEL_PATH=""  # Path to ChatTS model checkpoint
CHATTS_PATH=""  # Path to ChatTS directory

# Server configuration
CHATTS_PORT=5000
CHATTS_PID_FILE="/tmp/chatts_server_${CHATTS_PORT}.pid"
export CHATTS_SERVER_PORT="${CHATTS_PORT}"

# Device configuration
CHATTS_DEVICE="4,5,6,7"  # Use 4 GPUs for ChatTS

# Create log directory
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# ChatTS log files
CHATTS_LOG="${LOG_DIR}/chatts_server.$(date +%Y-%m-%d-%H-%M-%S).log"
CHATTS_CONSOLE_LOG="${LOG_DIR}/chatts_console.$(date +%Y-%m-%d-%H-%M-%S).log"

# ── Initialize Conda in this shell ────────────
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh. Do you need to run 'conda init'?"
  exit 1
fi
# ───────────────────────────────────────────────────────────────────────────────

# ===== Start ChatTS Server =====
echo "Starting ChatTS server with chatts-vllm environment..."

# Activate environment for ChatTS
eval "$(conda shell.bash hook)"
conda activate chatts-vllm

# Set environment variables
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
echo "VLLM_ALLOW_INSECURE_SERIALIZATION=$VLLM_ALLOW_INSECURE_SERIALIZATION"

# Don't set CUDA_VISIBLE_DEVICES for ChatTS server, use explicit device selection instead
echo "Using explicit device selection for ChatTS server: cuda:${CHATTS_DEVICE}"

# Check if a server is already running on the port
if nc -z localhost $CHATTS_PORT 2>/dev/null; then
    echo "Warning: Port $CHATTS_PORT is already in use!"
    echo "Another server might be running. Check with: lsof -i :$CHATTS_PORT"
    
    # Ask if we should continue or abort
    read -p "Do you want to continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting server startup."
        exit 1
    fi
fi

# Clear existing PID file if it exists
if [ -f "$CHATTS_PID_FILE" ]; then
    echo "Removing existing PID file: $CHATTS_PID_FILE"
    rm -f "$CHATTS_PID_FILE"
fi

# Copy chatts_server.py to local directory if it doesn't exist
CHATTS_SERVER_SCRIPT="${SCRIPT_DIR}/chatts_server.py"
if [ ! -f "$CHATTS_SERVER_SCRIPT" ]; then
    echo "ERROR: Missing ChatTS server script: $CHATTS_SERVER_SCRIPT"
    echo "Please make sure src/chatts_utils/chatts_server.py exists in this repository."
    exit 1
fi

# Start ChatTS server
echo "Starting ChatTS server with log at ${CHATTS_LOG}"
"$CHATTS_SERVER_SCRIPT" \
    --model_path "${CHATTS_MODEL_PATH}" \
    --chatts_path "${CHATTS_PATH}" \
    --port "${CHATTS_PORT}" \
    --device "${CHATTS_DEVICE}" \
    --context_length 5000 \
    --pid_file "${CHATTS_PID_FILE}" \
    --log_file "${CHATTS_LOG}" \
    --initial_wait 180 \
    > "${CHATTS_CONSOLE_LOG}" 2>&1 &

CHATTS_SERVER_PID=$!
echo "Started ChatTS server process with PID $CHATTS_SERVER_PID"

# Wait briefly to make sure the process starts
sleep 10

# Check if the PID file was created
if [ -f "$CHATTS_PID_FILE" ]; then
    FILE_PID=$(cat $CHATTS_PID_FILE)
    echo "ChatTS server PID file created with PID ${FILE_PID}"
else
    echo "ChatTS server PID file not created yet, writing our tracked PID"
    echo $CHATTS_SERVER_PID > "$CHATTS_PID_FILE"
fi

# Check if the server process is still running
if kill -0 $CHATTS_SERVER_PID 2>/dev/null; then
    echo "ChatTS server process is running"
else
    echo "Error: ChatTS server process exited unexpectedly"
    echo "Check the logs:"
    echo "Console log: $CHATTS_CONSOLE_LOG"
    echo "Server log: $CHATTS_LOG"
    exit 1
fi

# Wait for server initialization - a fixed time instead of relying on the server script's check
echo "Waiting for ChatTS server to initialize (240 seconds)..."
echo "You can monitor the logs with:"
echo "tail -f ${CHATTS_CONSOLE_LOG}"
echo "tail -f ${CHATTS_LOG}"
sleep 240  # 4 minute initial wait

# ===== Test Server Connectivity =====
echo "Testing ChatTS server connectivity..."
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:${CHATTS_PORT}/v1', api_key='dummy-key')
try:
    response = client.models.list()
    print(f'ChatTS models available: {response}')
    print('ChatTS server is operational!')
    exit(0)
except Exception as e:
    print(f'Error testing ChatTS server: {e}')
    exit(1)
"
CHATTS_TEST_EXIT_CODE=$?

if [ $CHATTS_TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: ChatTS server test failed."
    echo "Check the logs:"
    echo "Console log: $CHATTS_CONSOLE_LOG"
    echo "Server log: $CHATTS_LOG"
    exit 1
else
    echo "ChatTS server test passed successfully!"
    echo "The server is running on port $CHATTS_PORT"
    echo "To stop the server later, run: $SCRIPT_DIR/stop_chatts_server.sh"
fi

echo ""
echo "========================================"
echo "ChatTS Server is ready for inference!"
echo "Server URL: http://localhost:$CHATTS_PORT"
echo "========================================"
