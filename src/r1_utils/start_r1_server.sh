#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# start_r1_server.sh
#
# Script to start a DeepSeek-R1-Distill-Qwen-32B server for text-only reasoning.
# Uses the same chat template and continue_final_message injection as Qwen3.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"

# ── Configuration ────────────────────────────────────────────────────────────
# DeepSeek-R1-Distill-Qwen-32B model path (GRLM)
R1_MODEL_PATH=""  # Path to DeepSeek-R1-Distill-Qwen-32B checkpoint

R1_PORT=5002
R1_PID_FILE="/tmp/r1_server_${R1_PORT}.pid"
export R1_SERVER_PORT="${R1_PORT}"

R1_DEVICE="0,1,2,3"
R1_DATA_PARALLEL_SIZE=1
R1_TENSOR_PARALLEL_SIZE=4

R1_CHAT_TEMPLATE="${SCRIPT_DIR}/simple_chat_template.jinja"
echo "Using chat template: ${R1_CHAT_TEMPLATE}"
[ -f "${R1_CHAT_TEMPLATE}" ] && echo "Chat template file exists" || echo "WARNING: Chat template file does not exist!"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
R1_LOG="${LOG_DIR}/r1_server.$(date +%Y-%m-%d-%H-%M-%S).log"
R1_CONSOLE_LOG="${LOG_DIR}/r1_console.$(date +%Y-%m-%d-%H-%M-%S).log"

# ── Initialize Conda ────────────
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh."
  exit 1
fi

echo "Starting DeepSeek-R1 server with qwen3-vllm environment..."
eval "$(conda shell.bash hook)"
conda activate qwen3-vllm

if nc -z localhost $R1_PORT 2>/dev/null; then
    echo "Warning: Port $R1_PORT is already in use!"
    read -p "Continue anyway? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

[ -f "$R1_PID_FILE" ] && rm -f "$R1_PID_FILE"

R1_SERVER_SCRIPT="${SCRIPT_DIR}/r1_server.py"
chmod +x "$R1_SERVER_SCRIPT"

echo "Starting DeepSeek-R1 server with log at ${R1_LOG}"
"$R1_SERVER_SCRIPT" \
    --model_path "${R1_MODEL_PATH}" \
    --port "${R1_PORT}" \
    --device "${R1_DEVICE}" \
    --data_parallel_size "${R1_DATA_PARALLEL_SIZE}" \
    --tensor_parallel_size "${R1_TENSOR_PARALLEL_SIZE}" \
    --pid_file "${R1_PID_FILE}" \
    --log_file "${R1_LOG}" \
    --chat_template "${R1_CHAT_TEMPLATE}" \
    --context_length 56320 \
    --initial_wait 180 \
    > "${R1_CONSOLE_LOG}" 2>&1 &

R1_SERVER_PID=$!
echo "Started DeepSeek-R1 server process with PID $R1_SERVER_PID"

sleep 10

if [ -f "$R1_PID_FILE" ]; then
    echo "PID file created with PID $(cat $R1_PID_FILE)"
else
    echo $R1_SERVER_PID > "$R1_PID_FILE"
fi

kill -0 $R1_SERVER_PID 2>/dev/null || { echo "Error: Server exited unexpectedly. Check $R1_CONSOLE_LOG"; exit 1; }

echo "Waiting for DeepSeek-R1 server to initialize (240 seconds)..."
echo "Monitor: tail -f ${R1_CONSOLE_LOG}"
sleep 240

echo "Testing DeepSeek-R1 server connectivity..."
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:${R1_PORT}/v1', api_key='dummy-key')
try:
    response = client.models.list()
    print(f'DeepSeek-R1 models available: {response}')
    print('DeepSeek-R1 server is operational!')
    exit(0)
except Exception as e:
    print(f'Error testing DeepSeek-R1 server: {e}')
    exit(1)
"
[ $? -ne 0 ] && { echo "Error: Server test failed. Check $R1_CONSOLE_LOG"; exit 1; }

echo ""
echo "========================================"
echo "DeepSeek-R1 Server is ready for inference!"
echo "Server URL: http://localhost:$R1_PORT"
echo "To stop: $SCRIPT_DIR/stop_r1_server.sh"
echo "========================================"
