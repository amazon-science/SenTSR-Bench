#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# start_qwen_vl_server.sh
#
# Script to start a Qwen2.5-VL server for multimodal inference.
# This script:
# 1. Initializes the environment
# 2. Starts the Qwen2.5-VL server using vLLM
# 3. Checks that the server is running and operational
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/src/qwen_utils
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"               # project root

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
# Qwen2.5-VL-3B-Instruct model path (TSLM)
QWEN_VL_MODEL_PATH=""  # Path to Qwen2.5-VL-3B-Instruct checkpoint (or SFT/RL fine-tuned variant)

# Server configuration
QWEN_VL_PORT=5003
QWEN_VL_PID_FILE="/tmp/qwen_vl_server_${QWEN_VL_PORT}.pid"
export QWEN_VL_SERVER_PORT="${QWEN_VL_PORT}"

# Device configuration
QWEN_VL_DEVICE="4,5,6,7"
QWEN_VL_DATA_PARALLEL_SIZE=4
QWEN_VL_TENSOR_PARALLEL_SIZE=1

# Create log directory
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Qwen2.5-VL log files
QWEN_VL_LOG="${LOG_DIR}/qwen_vl_server.$(date +%Y-%m-%d-%H-%M-%S).log"
QWEN_VL_CONSOLE_LOG="${LOG_DIR}/qwen_vl_console.$(date +%Y-%m-%d-%H-%M-%S).log"

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

# ===== Start Qwen2.5-VL Server =====
echo "Starting Qwen2.5-VL server with qwen3-vllm environment..."

# Activate environment for Qwen2.5-VL
eval "$(conda shell.bash hook)"
conda activate qwen3-vllm

# Check if a server is already running on the port
if nc -z localhost $QWEN_VL_PORT 2>/dev/null; then
    echo "Warning: Port $QWEN_VL_PORT is already in use!"
    echo "Another server might be running. Check with: lsof -i :$QWEN_VL_PORT"

    # Ask if we should continue or abort
    read -p "Do you want to continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting server startup."
        exit 1
    fi
fi

# Clear existing PID file if it exists
if [ -f "$QWEN_VL_PID_FILE" ]; then
    echo "Removing existing PID file: $QWEN_VL_PID_FILE"
    rm -f "$QWEN_VL_PID_FILE"
fi

# Make server script executable
QWEN_VL_SERVER_SCRIPT="${SCRIPT_DIR}/qwen_vl_server.py"
chmod +x "$QWEN_VL_SERVER_SCRIPT"

# Start Qwen2.5-VL server
echo "Starting Qwen2.5-VL server with log at ${QWEN_VL_LOG}"
"$QWEN_VL_SERVER_SCRIPT" \
    --model_path "${QWEN_VL_MODEL_PATH}" \
    --port "${QWEN_VL_PORT}" \
    --device "${QWEN_VL_DEVICE}" \
    --data_parallel_size "${QWEN_VL_DATA_PARALLEL_SIZE}" \
    --tensor_parallel_size "${QWEN_VL_TENSOR_PARALLEL_SIZE}" \
    --pid_file "${QWEN_VL_PID_FILE}" \
    --log_file "${QWEN_VL_LOG}" \
    --initial_wait 180 \
    > "${QWEN_VL_CONSOLE_LOG}" 2>&1 &

QWEN_VL_SERVER_PID=$!
echo "Started Qwen2.5-VL server process with PID $QWEN_VL_SERVER_PID"

# Wait briefly to make sure the process starts
sleep 10

# Check if the PID file was created
if [ -f "$QWEN_VL_PID_FILE" ]; then
    FILE_PID=$(cat $QWEN_VL_PID_FILE)
    echo "Qwen2.5-VL server PID file created with PID ${FILE_PID}"
else
    echo "Qwen2.5-VL server PID file not created yet, writing our tracked PID"
    echo $QWEN_VL_SERVER_PID > "$QWEN_VL_PID_FILE"
fi

# Check if the server process is still running
if kill -0 $QWEN_VL_SERVER_PID 2>/dev/null; then
    echo "Qwen2.5-VL server process is running"
else
    echo "Error: Qwen2.5-VL server process exited unexpectedly"
    echo "Check the logs:"
    echo "Console log: $QWEN_VL_CONSOLE_LOG"
    echo "Server log: $QWEN_VL_LOG"
    exit 1
fi

# Wait for server initialization - a fixed time instead of relying on the server script's check
echo "Waiting for Qwen2.5-VL server to initialize (240 seconds)..."
echo "You can monitor the logs with:"
echo "tail -f ${QWEN_VL_CONSOLE_LOG}"
echo "tail -f ${QWEN_VL_LOG}"
sleep 240  # 4 minute initial wait

# ===== Test Server Connectivity =====
echo "Testing Qwen2.5-VL server connectivity..."
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:${QWEN_VL_PORT}/v1', api_key='dummy-key')
try:
    response = client.models.list()
    print(f'Qwen2.5-VL models available: {response}')
    print('Qwen2.5-VL server is operational!')
    exit(0)
except Exception as e:
    print(f'Error testing Qwen2.5-VL server: {e}')
    exit(1)
"
QWEN_VL_TEST_EXIT_CODE=$?

if [ $QWEN_VL_TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: Qwen2.5-VL server test failed."
    echo "Check the logs:"
    echo "Console log: $QWEN_VL_CONSOLE_LOG"
    echo "Server log: $QWEN_VL_LOG"
    exit 1
else
    echo "Qwen2.5-VL server test passed successfully!"
    echo "The server is running on port $QWEN_VL_PORT"
    echo "To stop the server later, run: $SCRIPT_DIR/stop_qwen_vl_server.sh"
fi

echo ""
echo "========================================"
echo "Qwen2.5-VL Server is ready for inference!"
echo "Server URL: http://localhost:$QWEN_VL_PORT"
echo "========================================"
