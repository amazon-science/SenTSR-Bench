#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_qwen_inference.sh
#
# Script to run Qwen2.5-VL inference on a dataset.
# This script:
# 1. Checks if Qwen2.5-VL server is running, and starts it if not
# 2. Runs the Qwen2.5-VL inference script on the specified dataset
# 3. Creates output directories and saves results
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/scripts
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                 # project root

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
# Defaults (can be overridden with command line arguments)
DATASET_PATH="${PROJECT_ROOT}/dataset/processed/dataset_a_with_mcq2.json"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation/results"
OUTPUT_NAME="qwen_vl_results.json"

# Server configuration
QWEN_PORT=5003
QWEN_PID_FILE="/tmp/qwen_vl_server_${QWEN_PORT}.pid"
QWEN_SERVER_URL="http://localhost:${QWEN_PORT}"

# Create required directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--dataset PATH] [--output FILENAME] [--output-dir DIR] [--workers N] [--max-samples N]"
      echo ""
      echo "Options:"
      echo "  --dataset PATH      Path to the dataset JSON file"
      echo "  --output FILENAME   Name of the output JSON file (default: qwen_vl_results.json)"
      echo "  --output-dir DIR    Directory to save results (default: ../evaluation/results)"
      echo "  --workers N         Number of parallel workers (default: 4)"
      echo "  --max-samples N     Maximum number of samples to process (default: 200)"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Full paths for output
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"
LOG_PATH="${PROJECT_ROOT}/logs/qwen_inference_$(date +%Y-%m-%d-%H-%M-%S).log"

# Optional parameters for the inference script
OPTIONAL_PARAMS=""
if [[ -v WORKERS ]]; then
  OPTIONAL_PARAMS="${OPTIONAL_PARAMS} --workers ${WORKERS}"
fi

if [[ -v MAX_SAMPLES ]]; then
  OPTIONAL_PARAMS="${OPTIONAL_PARAMS} --max_samples ${MAX_SAMPLES}"
fi

echo "========================================"
echo "Configuration:"
echo "DATASET_PATH = ${DATASET_PATH}"
echo "OUTPUT_PATH  = ${OUTPUT_PATH}"
echo "LOG_PATH     = ${LOG_PATH}"
echo "SERVER_URL   = ${QWEN_SERVER_URL}"
echo "OPTIONAL_PARAMS = ${OPTIONAL_PARAMS}"
echo "========================================"
echo ""

# ── Check if Qwen2.5-VL server is running ────────────────────────────────────
check_server() {
  echo "Checking if Qwen2.5-VL server is running..."

  # Check if PID file exists
  if [ -f "$QWEN_PID_FILE" ]; then
    PID=$(cat "$QWEN_PID_FILE")
    echo "Found PID file with PID: $PID"

    # Check if process is actually running
    if kill -0 "$PID" 2>/dev/null; then
      echo "Qwen2.5-VL server is running with PID: $PID"
      return 0
    else
      echo "PID file exists but process is not running"
      rm -f "$QWEN_PID_FILE"
    fi
  fi

  # Check if port is in use (server might be running without PID file)
  if nc -z localhost "$QWEN_PORT" 2>/dev/null; then
    echo "Port $QWEN_PORT is in use, assuming Qwen2.5-VL server is running"
    return 0
  fi

  echo "Qwen2.5-VL server is not running"
  return 1
}

# ── Start Qwen2.5-VL server if needed ────────────────────────────────────────
if check_server; then
  echo "Using existing Qwen2.5-VL server"
else
  echo "Starting Qwen2.5-VL server..."
  QWEN_UTILS_DIR="${PROJECT_ROOT}/src/qwen_utils"

  if [ ! -x "${QWEN_UTILS_DIR}/start_qwen_vl_server.sh" ]; then
    echo "Error: Qwen2.5-VL server start script not found or not executable: ${QWEN_UTILS_DIR}/start_qwen_vl_server.sh"
    exit 1
  fi

  # Start the server
  echo "Running: ${QWEN_UTILS_DIR}/start_qwen_vl_server.sh"
  "${QWEN_UTILS_DIR}/start_qwen_vl_server.sh"

  # Check if server started successfully
  if ! check_server; then
    echo "Error: Failed to start Qwen2.5-VL server"
    exit 1
  fi
fi

# ── Run Qwen2.5-VL inference script ─────────────────────────────────────────
echo "Running Qwen2.5-VL inference..."
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Log: ${LOG_PATH}"

# Ensure inference script is executable
INFERENCE_SCRIPT="${PROJECT_ROOT}/src/qwen_inference.py"
if [ ! -x "$INFERENCE_SCRIPT" ]; then
  chmod +x "$INFERENCE_SCRIPT"
fi

# Run the inference script
echo "Command: python $INFERENCE_SCRIPT --dataset_path $DATASET_PATH --output_path $OUTPUT_PATH --server_url $QWEN_SERVER_URL $OPTIONAL_PARAMS"
python "$INFERENCE_SCRIPT" \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --server_url "$QWEN_SERVER_URL" \
  $OPTIONAL_PARAMS | tee -a "$LOG_PATH"

# Check if inference completed successfully
if [ $? -eq 0 ]; then
  echo "========================================"
  echo "Qwen2.5-VL inference completed successfully!"
  echo "Results saved to: ${OUTPUT_PATH}"
  echo "Log saved to: ${LOG_PATH}"
  echo "========================================"
else
  echo "========================================"
  echo "Qwen2.5-VL inference failed with an error"
  echo "Check the log for details: ${LOG_PATH}"
  echo "========================================"
  exit 1
fi
