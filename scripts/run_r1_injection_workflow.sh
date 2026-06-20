#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_r1_injection_workflow.sh
#
# A script to run DeepSeek-R1 with Qwen-VL injection pipeline for time
# series analysis. Identical pattern to Qwen3 — DeepSeek-R1-Distill-Qwen-32B
# uses the same tokenizer, API, and continue_final_message injection.
#
# This pipeline:
# 1. Ensures Qwen-VL server is running and generates VL observations
# 2. Ensures DeepSeek-R1 server is running
# 3. Runs R1 with Qwen-VL injection
# 4. Evaluates the results
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "========================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH="${PROJECT_ROOT}/dataset/dataset_a_with_mcq2.json"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation/results"
NUM_RUNS=1
MAX_SAMPLES=200
R1_WORKERS=10
QWEN_VL_WORKERS=4
EVAL_WORKERS=4

# Server configuration
QWEN_VL_PORT=5003
QWEN_VL_SERVER_URL="http://localhost:${QWEN_VL_PORT}"
R1_PORT=5002
R1_SERVER_URL="http://localhost:${R1_PORT}"

STOP_SERVER=false

mkdir -p "${OUTPUT_DIR}"

show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --dataset PATH         Path to the dataset JSON file"
  echo "  --output-dir DIR       Directory to save results (default: ../evaluation/results)"
  echo "  --num-runs N           Number of runs to perform (default: 1)"
  echo "  --r1-workers N         Number of R1 API parallel workers (default: 10)"
  echo "  --qwen-vl-workers N    Number of Qwen-VL parallel workers (default: 4)"
  echo "  --eval-workers N       Number of evaluation parallel workers (default: 4)"
  echo "  --max-samples N        Maximum number of samples to process (default: 200)"
  echo "  --qwen-vl-port N       Port for Qwen-VL server (default: 5003)"
  echo "  --r1-port N            Port for DeepSeek-R1 server (default: 5002)"
  echo "  --stop-server          Stop servers when done"
  echo ""
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)      DATASET_PATH="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --num-runs)     NUM_RUNS="$2"; shift 2 ;;
    --r1-workers)   R1_WORKERS="$2"; shift 2 ;;
    --qwen-vl-workers) QWEN_VL_WORKERS="$2"; shift 2 ;;
    --eval-workers) EVAL_WORKERS="$2"; shift 2 ;;
    --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
    --qwen-vl-port) QWEN_VL_PORT="$2"; QWEN_VL_SERVER_URL="http://localhost:${QWEN_VL_PORT}"; shift 2 ;;
    --r1-port)      R1_PORT="$2"; R1_SERVER_URL="http://localhost:${R1_PORT}"; shift 2 ;;
    --stop-server)  STOP_SERVER=true; shift ;;
    --help)         show_help ;;
    *)              echo "Unknown option: $1"; exit 1 ;;
  esac
done

DATASET_NAME=$(basename "$DATASET_PATH" .json)

mkdir -p "${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG_PATH="${PROJECT_ROOT}/logs/r1_injection_workflow_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_PATH"
echo "Configuration:" | tee -a "$LOG_PATH"
echo "DATASET_PATH        = ${DATASET_PATH}" | tee -a "$LOG_PATH"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}" | tee -a "$LOG_PATH"
echo "NUM_RUNS            = ${NUM_RUNS}" | tee -a "$LOG_PATH"
echo "MAX_SAMPLES         = ${MAX_SAMPLES}" | tee -a "$LOG_PATH"
echo "R1_WORKERS          = ${R1_WORKERS}" | tee -a "$LOG_PATH"
echo "QWEN_VL_WORKERS     = ${QWEN_VL_WORKERS}" | tee -a "$LOG_PATH"
echo "EVAL_WORKERS        = ${EVAL_WORKERS}" | tee -a "$LOG_PATH"
echo "QWEN_VL_SERVER_URL  = ${QWEN_VL_SERVER_URL}" | tee -a "$LOG_PATH"
echo "R1_SERVER_URL       = ${R1_SERVER_URL}" | tee -a "$LOG_PATH"
echo "========================================" | tee -a "$LOG_PATH"
echo "" | tee -a "$LOG_PATH"

# ── Initialize Conda ────────────
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh." | tee -a "$LOG_PATH"
  exit 1
fi

# 1) Verify servers
STARTTIME=$(date +%s)

echo "Checking if Qwen-VL server is running..." | tee -a "$LOG_PATH"
if nc -z localhost $QWEN_VL_PORT 2>/dev/null; then
    echo "Qwen-VL server running on port $QWEN_VL_PORT" | tee -a "$LOG_PATH"
else
    echo "ERROR: Qwen-VL server is not running on port $QWEN_VL_PORT" | tee -a "$LOG_PATH"
    echo "Please start it first with: src/qwen_utils/start_qwen_vl_server.sh" | tee -a "$LOG_PATH"
    exit 1
fi

echo "Checking if DeepSeek-R1 server is running..." | tee -a "$LOG_PATH"
if nc -z localhost $R1_PORT 2>/dev/null; then
    echo "DeepSeek-R1 server running on port $R1_PORT" | tee -a "$LOG_PATH"
else
    echo "ERROR: DeepSeek-R1 server is not running on port $R1_PORT" | tee -a "$LOG_PATH"
    echo "Please start it first with: src/r1_utils/start_r1_server.sh" | tee -a "$LOG_PATH"
    exit 1
fi

# 2) Generate Qwen-VL observations (shared across all runs)
QWEN_VL_OUT_DIR="${OUTPUT_DIR}/qwen-vl-thinking-${DATASET_NAME}"
mkdir -p "$QWEN_VL_OUT_DIR"
QWEN_VL_OUT="${QWEN_VL_OUT_DIR}/generated_answer.json"

if [ -f "$QWEN_VL_OUT" ]; then
    echo "Found existing Qwen-VL observations at $QWEN_VL_OUT" | tee -a "$LOG_PATH"
    echo "Reusing existing observations for injection" | tee -a "$LOG_PATH"
else
    echo "Generating Qwen-VL observations..." | tee -a "$LOG_PATH"
    conda activate evaluation
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    python "${PROJECT_ROOT}/src/qwen_inference.py" \
        --dataset_path "$DATASET_PATH" \
        --output_path "$QWEN_VL_OUT" \
        --server_url "$QWEN_VL_SERVER_URL" \
        --workers $QWEN_VL_WORKERS \
        --max_samples $MAX_SAMPLES | tee -a "$LOG_PATH"

    echo "Qwen-VL observations saved to $QWEN_VL_OUT" | tee -a "$LOG_PATH"
fi

# 3) Run R1 with injection for each run
for RUN_NUM in $(seq 1 $NUM_RUNS); do
    echo "" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"
    echo "Starting DeepSeek-R1 Injection Run #$RUN_NUM of $NUM_RUNS" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"

    RUN_SUFFIX="run${RUN_NUM}"
    INJECTION_OUT_DIR="${OUTPUT_DIR}/r1-injection-${DATASET_NAME}-${RUN_SUFFIX}"
    mkdir -p "$INJECTION_OUT_DIR"
    INJECTION_OUT="${INJECTION_OUT_DIR}/generated_answer.json"

    conda activate qwen3-vllm
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    echo "Running DeepSeek-R1 with Qwen-VL injection (Run #$RUN_NUM)" | tee -a "$LOG_PATH"
    python "${PROJECT_ROOT}/src/qwen3_with_injection.py" \
        --dataset_path "$DATASET_PATH" \
        --injection_path "$QWEN_VL_OUT" \
        --output_path "$INJECTION_OUT" \
        --server_url "$R1_SERVER_URL" \
        --model_name "r1" \
        --workers $R1_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM: Injection results saved to $INJECTION_OUT" | tee -a "$LOG_PATH"

    # 4) Evaluate
    echo "Evaluating injection results (Run #$RUN_NUM)" | tee -a "$LOG_PATH"
    INJECTION_EXP_NAME=$(basename "$INJECTION_OUT_DIR")
    conda activate evaluation
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    python "${PROJECT_ROOT}/evaluation/evaluate_with_sampling.py" \
        --exp "$INJECTION_EXP_NAME" \
        --dataset "$DATASET_PATH" \
        --generated "$INJECTION_OUT" \
        --num_workers $EVAL_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM Complete!" | tee -a "$LOG_PATH"
    echo "  - Injection results: $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "  - Evaluation: ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
done

# 5) Summary
ENDTIME=$(date +%s)
RUNTIME=$((ENDTIME - STARTTIME))
echo "=======================================" | tee -a "$LOG_PATH"
echo "Total runtime: $RUNTIME seconds ($(($RUNTIME / 60)) minutes)" | tee -a "$LOG_PATH"
echo "All $NUM_RUNS Runs Completed Successfully" | tee -a "$LOG_PATH"
echo "=======================================" | tee -a "$LOG_PATH"

# 6) Stop servers if requested
if [ "$STOP_SERVER" = true ]; then
    echo "Stopping servers..." | tee -a "$LOG_PATH"
    "${PROJECT_ROOT}/src/qwen_utils/stop_qwen_vl_server.sh" || true
    "${PROJECT_ROOT}/src/r1_utils/stop_r1_server.sh" || true
    echo "Servers stopped" | tee -a "$LOG_PATH"
else
    echo "" | tee -a "$LOG_PATH"
    echo "NOTE: Servers are still running. Stop them with:" | tee -a "$LOG_PATH"
    echo "  ${PROJECT_ROOT}/src/qwen_utils/stop_qwen_vl_server.sh" | tee -a "$LOG_PATH"
    echo "  ${PROJECT_ROOT}/src/r1_utils/stop_r1_server.sh" | tee -a "$LOG_PATH"
fi

echo "Log saved to: ${LOG_PATH}" | tee -a "$LOG_PATH"
echo "Workflow complete!" | tee -a "$LOG_PATH"
