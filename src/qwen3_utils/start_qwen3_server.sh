#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# start_qwen3_server.sh
#
# Script to start a Qwen3 server for text-only reasoning inference.
# This script:
# 1. Initializes the environment
# 2. Starts the Qwen3 server using vLLM with a custom chat template
# 3. Checks that the server is running and operational
#
# The custom chat template (simple_chat_template.jinja) passes through
# assistant messages without modifying <think> tags, which is required for
# the continue_final_message injection pattern.
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/src/qwen3_utils
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"               # project root

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
# Qwen3-32B model path (GRLM)
QWEN3_MODEL_PATH=""  # Path to Qwen3-32B checkpoint

# Server configuration
QWEN3_PORT=5001
QWEN3_PID_FILE="/tmp/qwen3_server_${QWEN3_PORT}.pid"
export QWEN3_SERVER_PORT="${QWEN3_PORT}"

# Device configuration
QWEN3_DEVICE="0,1,2,3"
QWEN3_DATA_PARALLEL_SIZE=1
QWEN3_TENSOR_PARALLEL_SIZE=4

# Custom chat template (required for continue_final_message injection)
QWEN3_CHAT_TEMPLATE="${SCRIPT_DIR}/simple_chat_template.jinja"
echo "Using chat template: ${QWEN3_CHAT_TEMPLATE}"
if [ -f "${QWEN3_CHAT_TEMPLATE}" ]; then
    echo "Chat template file exists"
else
    echo "WARNING: Chat template file does not exist!"
fi

# Create log directory
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Qwen3 log files
QWEN3_LOG="${LOG_DIR}/qwen3_server.$(date +%Y-%m-%d-%H-%M-%S).log"
QWEN3_CONSOLE_LOG="${LOG_DIR}/qwen3_console.$(date +%Y-%m-%d-%H-%M-%S).log"

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

# ===== Start Qwen3 Server =====
echo "Starting Qwen3 server with qwen3-vllm environment..."

# Activate environment for Qwen3
eval "$(conda shell.bash hook)"
conda activate qwen3-vllm

# Check if a server is already running on the port
if nc -z localhost $QWEN3_PORT 2>/dev/null; then
    echo "Warning: Port $QWEN3_PORT is already in use!"
    echo "Another server might be running. Check with: lsof -i :$QWEN3_PORT"

    # Ask if we should continue or abort
    read -p "Do you want to continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting server startup."
        exit 1
    fi
fi

# Clear existing PID file if it exists
if [ -f "$QWEN3_PID_FILE" ]; then
    echo "Removing existing PID file: $QWEN3_PID_FILE"
    rm -f "$QWEN3_PID_FILE"
fi

# Make server script executable
QWEN3_SERVER_SCRIPT="${SCRIPT_DIR}/qwen3_server.py"
chmod +x "$QWEN3_SERVER_SCRIPT"

# Start Qwen3 server
echo "Starting Qwen3 server with log at ${QWEN3_LOG}"
"$QWEN3_SERVER_SCRIPT" \
    --model_path "${QWEN3_MODEL_PATH}" \
    --port "${QWEN3_PORT}" \
    --device "${QWEN3_DEVICE}" \
    --data_parallel_size "${QWEN3_DATA_PARALLEL_SIZE}" \
    --tensor_parallel_size "${QWEN3_TENSOR_PARALLEL_SIZE}" \
    --pid_file "${QWEN3_PID_FILE}" \
    --log_file "${QWEN3_LOG}" \
    --chat_template "${QWEN3_CHAT_TEMPLATE}" \
    --initial_wait 180 \
    > "${QWEN3_CONSOLE_LOG}" 2>&1 &

QWEN3_SERVER_PID=$!
echo "Started Qwen3 server process with PID $QWEN3_SERVER_PID"

# Wait briefly to make sure the process starts
sleep 10

# Check if the PID file was created
if [ -f "$QWEN3_PID_FILE" ]; then
    FILE_PID=$(cat $QWEN3_PID_FILE)
    echo "Qwen3 server PID file created with PID ${FILE_PID}"
else
    echo "Qwen3 server PID file not created yet, writing our tracked PID"
    echo $QWEN3_SERVER_PID > "$QWEN3_PID_FILE"
fi

# Check if the server process is still running
if kill -0 $QWEN3_SERVER_PID 2>/dev/null; then
    echo "Qwen3 server process is running"
else
    echo "Error: Qwen3 server process exited unexpectedly"
    echo "Check the logs:"
    echo "Console log: $QWEN3_CONSOLE_LOG"
    echo "Server log: $QWEN3_LOG"
    exit 1
fi

# Wait for server initialization
echo "Waiting for Qwen3 server to initialize (240 seconds)..."
echo "You can monitor the logs with:"
echo "tail -f ${QWEN3_CONSOLE_LOG}"
echo "tail -f ${QWEN3_LOG}"
sleep 240  # 4 minute initial wait

# ===== Test Server Connectivity =====
echo "Testing Qwen3 server connectivity..."
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:${QWEN3_PORT}/v1', api_key='dummy-key')
try:
    response = client.models.list()
    print(f'Qwen3 models available: {response}')
    print('Qwen3 server is operational!')
    exit(0)
except Exception as e:
    print(f'Error testing Qwen3 server: {e}')
    exit(1)
"
QWEN3_TEST_EXIT_CODE=$?

if [ $QWEN3_TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: Qwen3 server test failed."
    echo "Check the logs:"
    echo "Console log: $QWEN3_CONSOLE_LOG"
    echo "Server log: $QWEN3_LOG"
    exit 1
else
    echo "Qwen3 server test passed successfully!"
    echo "The server is running on port $QWEN3_PORT"
    echo "To stop the server later, run: $SCRIPT_DIR/stop_qwen3_server.sh"
fi

echo ""
echo "========================================"
echo "Qwen3 Server is ready for inference!"
echo "Server URL: http://localhost:$QWEN3_PORT"
echo "========================================"
