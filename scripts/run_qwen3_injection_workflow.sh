#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_qwen3_injection_workflow.sh
#
# A script to run the Qwen3 with Qwen-VL injection pipeline for time
# series analysis.
#
# This pipeline:
# 1. Ensures Qwen-VL server is running and generates VL observations
# 2. Ensures Qwen3 server is running
# 3. Runs Qwen3 with Qwen-VL injection (thoughts + answer injected into
#    Qwen3's thinking via continue_final_message)
# 4. Evaluates the results using the evaluation script
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
DATASET_PATH="${PROJECT_ROOT}/dataset/dataset_a_with_mcq2.json"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation/results"
NUM_RUNS=1
MAX_SAMPLES=200
QWEN3_WORKERS=10
QWEN_VL_WORKERS=4
EVAL_WORKERS=4

# Server configuration
QWEN_VL_PORT=5003
QWEN_VL_SERVER_URL="http://localhost:${QWEN_VL_PORT}"
QWEN3_PORT=5001
QWEN3_SERVER_URL="http://localhost:${QWEN3_PORT}"

STOP_SERVER=false

# Create required directories
mkdir -p "${OUTPUT_DIR}"

# Display usage information
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --dataset PATH         Path to the dataset JSON file"
  echo "  --output-dir DIR       Directory to save results (default: ../evaluation/results)"
  echo "  --num-runs N           Number of runs to perform (default: 1)"
  echo "  --qwen3-workers N      Number of Qwen3 API parallel workers (default: 10)"
  echo "  --qwen-vl-workers N    Number of Qwen-VL parallel workers (default: 4)"
  echo "  --eval-workers N       Number of evaluation parallel workers (default: 4)"
  echo "  --max-samples N        Maximum number of samples to process (default: 200)"
  echo "  --qwen-vl-port N       Port for Qwen-VL server (default: 5003)"
  echo "  --qwen3-port N         Port for Qwen3 server (default: 5001)"
  echo "  --stop-server          Stop servers when done"
  echo ""
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --qwen3-workers)
      QWEN3_WORKERS="$2"
      shift 2
      ;;
    --qwen-vl-workers)
      QWEN_VL_WORKERS="$2"
      shift 2
      ;;
    --eval-workers)
      EVAL_WORKERS="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --qwen-vl-port)
      QWEN_VL_PORT="$2"
      QWEN_VL_SERVER_URL="http://localhost:${QWEN_VL_PORT}"
      shift 2
      ;;
    --qwen3-port)
      QWEN3_PORT="$2"
      QWEN3_SERVER_URL="http://localhost:${QWEN3_PORT}"
      shift 2
      ;;
    --stop-server)
      STOP_SERVER=true
      shift
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Extract dataset name without extension and path
DATASET_NAME=$(basename "$DATASET_PATH" .json)

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG_PATH="${PROJECT_ROOT}/logs/qwen3_injection_workflow_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_PATH"
echo "Configuration:" | tee -a "$LOG_PATH"
echo "DATASET_PATH        = ${DATASET_PATH}" | tee -a "$LOG_PATH"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}" | tee -a "$LOG_PATH"
echo "NUM_RUNS            = ${NUM_RUNS}" | tee -a "$LOG_PATH"
echo "MAX_SAMPLES         = ${MAX_SAMPLES}" | tee -a "$LOG_PATH"
echo "QWEN3_WORKERS       = ${QWEN3_WORKERS}" | tee -a "$LOG_PATH"
echo "QWEN_VL_WORKERS     = ${QWEN_VL_WORKERS}" | tee -a "$LOG_PATH"
echo "EVAL_WORKERS        = ${EVAL_WORKERS}" | tee -a "$LOG_PATH"
echo "QWEN_VL_SERVER_URL  = ${QWEN_VL_SERVER_URL}" | tee -a "$LOG_PATH"
echo "QWEN3_SERVER_URL    = ${QWEN3_SERVER_URL}" | tee -a "$LOG_PATH"
echo "LOG_PATH            = ${LOG_PATH}" | tee -a "$LOG_PATH"
echo "========================================" | tee -a "$LOG_PATH"
echo "" | tee -a "$LOG_PATH"

# ── Initialize Conda in this shell ────────────
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Cannot find conda.sh. Do you need to run 'conda init'?" | tee -a "$LOG_PATH"
  exit 1
fi

# ------------------------------------------------------------------------------
# 1) Record start time and verify servers are running
# ------------------------------------------------------------------------------
STARTTIME=$(date +%s)

# Check Qwen-VL server
echo "Checking if Qwen-VL server is running..." | tee -a "$LOG_PATH"
if nc -z localhost $QWEN_VL_PORT 2>/dev/null; then
    echo "Qwen-VL server running on port $QWEN_VL_PORT" | tee -a "$LOG_PATH"
else
    echo "ERROR: Qwen-VL server is not running on port $QWEN_VL_PORT" | tee -a "$LOG_PATH"
    echo "Please start it first with: src/qwen_utils/start_qwen_vl_server.sh" | tee -a "$LOG_PATH"
    exit 1
fi

# Check Qwen3 server
echo "Checking if Qwen3 server is running..." | tee -a "$LOG_PATH"
if nc -z localhost $QWEN3_PORT 2>/dev/null; then
    echo "Qwen3 server running on port $QWEN3_PORT" | tee -a "$LOG_PATH"
else
    echo "ERROR: Qwen3 server is not running on port $QWEN3_PORT" | tee -a "$LOG_PATH"
    echo "Please start it first with: src/qwen3_utils/start_qwen3_server.sh" | tee -a "$LOG_PATH"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2) Generate Qwen-VL observations (shared across all runs)
# ------------------------------------------------------------------------------
QWEN_VL_OUT_DIR="${OUTPUT_DIR}/qwen-vl-thinking-${DATASET_NAME}"
mkdir -p "$QWEN_VL_OUT_DIR"
QWEN_VL_OUT="${QWEN_VL_OUT_DIR}/generated_answer.json"

if [ -f "$QWEN_VL_OUT" ]; then
    echo "Found existing Qwen-VL observations at $QWEN_VL_OUT" | tee -a "$LOG_PATH"
    echo "Reusing existing observations for injection" | tee -a "$LOG_PATH"
else
    echo "Generating Qwen-VL observations..." | tee -a "$LOG_PATH"

    # Activate environment
    conda activate evaluation

    # Set PYTHONPATH
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    python "${PROJECT_ROOT}/src/qwen_inference.py" \
        --dataset_path "$DATASET_PATH" \
        --output_path "$QWEN_VL_OUT" \
        --server_url "$QWEN_VL_SERVER_URL" \
        --workers $QWEN_VL_WORKERS \
        --max_samples $MAX_SAMPLES | tee -a "$LOG_PATH"

    echo "Qwen-VL observations saved to $QWEN_VL_OUT" | tee -a "$LOG_PATH"
fi

# ------------------------------------------------------------------------------
# 3) Run Qwen3 with injection for each run
# ------------------------------------------------------------------------------
for RUN_NUM in $(seq 1 $NUM_RUNS); do
    echo "" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"
    echo "Starting Qwen3 Injection Run #$RUN_NUM of $NUM_RUNS" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    # Define output paths for this run
    RUN_SUFFIX="run${RUN_NUM}"
    INJECTION_OUT_DIR="${OUTPUT_DIR}/qwen3-injection-${DATASET_NAME}-${RUN_SUFFIX}"
    mkdir -p "$INJECTION_OUT_DIR"
    INJECTION_OUT="${INJECTION_OUT_DIR}/generated_answer.json"

    # Activate Qwen3 environment
    echo "Activating qwen3-vllm environment..." | tee -a "$LOG_PATH"
    conda activate qwen3-vllm

    # Set PYTHONPATH
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    # Run Qwen3 with injection
    echo "Running Qwen3 with Qwen-VL injection (Run #$RUN_NUM)" | tee -a "$LOG_PATH"
    echo "    Dataset    : $DATASET_PATH" | tee -a "$LOG_PATH"
    echo "    Injection  : $QWEN_VL_OUT" | tee -a "$LOG_PATH"
    echo "    Output     : $INJECTION_OUT" | tee -a "$LOG_PATH"

    python "${PROJECT_ROOT}/src/qwen3_with_injection.py" \
        --dataset_path "$DATASET_PATH" \
        --injection_path "$QWEN_VL_OUT" \
        --output_path "$INJECTION_OUT" \
        --server_url "$QWEN3_SERVER_URL" \
        --workers $QWEN3_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM: Injection results saved to $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    # --------------------------------------------------------------------------
    # 4) Evaluate injection results
    # --------------------------------------------------------------------------
    echo "Evaluating injection results (Run #$RUN_NUM)" | tee -a "$LOG_PATH"

    INJECTION_EXP_NAME=$(basename "$INJECTION_OUT_DIR")

    # Switch to evaluation environment
    conda activate evaluation

    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    echo "   Evaluating Qwen3 injection results..." | tee -a "$LOG_PATH"
    python "${PROJECT_ROOT}/evaluation/evaluate_with_sampling.py" \
        --exp "$INJECTION_EXP_NAME" \
        --dataset "$DATASET_PATH" \
        --generated "$INJECTION_OUT" \
        --num_workers $EVAL_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM: Evaluation complete" | tee -a "$LOG_PATH"
    echo "   Results in ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM Complete!" | tee -a "$LOG_PATH"
    echo "Results:" | tee -a "$LOG_PATH"
    echo "  - Injection results: $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "  - Evaluation: ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
done

# ------------------------------------------------------------------------------
# 5) Runtime Summary
# ------------------------------------------------------------------------------
ENDTIME=$(date +%s)
RUNTIME=$((ENDTIME - STARTTIME))
echo "=======================================" | tee -a "$LOG_PATH"
echo "Total runtime: $RUNTIME seconds ($(($RUNTIME / 60)) minutes)" | tee -a "$LOG_PATH"
echo "=======================================" | tee -a "$LOG_PATH"

# ------------------------------------------------------------------------------
# 6) Summary of all runs
# ------------------------------------------------------------------------------
echo "========================================" | tee -a "$LOG_PATH"
echo "All $NUM_RUNS Runs Completed Successfully" | tee -a "$LOG_PATH"
echo "========================================" | tee -a "$LOG_PATH"

echo "Summary of result locations:" | tee -a "$LOG_PATH"
echo "  - Qwen-VL observations: $QWEN_VL_OUT" | tee -a "$LOG_PATH"

for RUN_NUM in $(seq 1 $NUM_RUNS); do
    RUN_SUFFIX="run${RUN_NUM}"
    INJECTION_OUT_DIR="${OUTPUT_DIR}/qwen3-injection-${DATASET_NAME}-${RUN_SUFFIX}"
    INJECTION_OUT="${INJECTION_OUT_DIR}/generated_answer.json"
    INJECTION_EXP_NAME=$(basename "$INJECTION_OUT_DIR")

    echo "Run #$RUN_NUM:" | tee -a "$LOG_PATH"
    echo "  - Injection results: $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "  - Evaluation: ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
done

# ------------------------------------------------------------------------------
# 7) Stop servers if requested
# ------------------------------------------------------------------------------
if [ "$STOP_SERVER" = true ]; then
    echo "Stopping servers..." | tee -a "$LOG_PATH"
    "${PROJECT_ROOT}/src/qwen_utils/stop_qwen_vl_server.sh" || true
    "${PROJECT_ROOT}/src/qwen3_utils/stop_qwen3_server.sh" || true
    echo "Servers stopped" | tee -a "$LOG_PATH"
else
    echo "" | tee -a "$LOG_PATH"
    echo "NOTE: Both the Qwen-VL and Qwen3 servers are still running." | tee -a "$LOG_PATH"
    echo "When you are done with all evaluations, stop them using:" | tee -a "$LOG_PATH"
    echo "  ${PROJECT_ROOT}/src/qwen_utils/stop_qwen_vl_server.sh" | tee -a "$LOG_PATH"
    echo "  ${PROJECT_ROOT}/src/qwen3_utils/stop_qwen3_server.sh" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
fi

echo "Log saved to: ${LOG_PATH}" | tee -a "$LOG_PATH"
echo "Workflow complete!" | tee -a "$LOG_PATH"
