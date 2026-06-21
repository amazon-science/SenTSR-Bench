#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_injection_workflow.sh
#
# A script to run the Claude with ChatTS injection pipeline for time
# series analysis.
#
# This pipeline:
# 1. Starts the ChatTS server if not already running
# 2. Runs Claude with ChatTS injection, where ChatTS injects knowledge at
#    the beginning of Claude's thinking process
# 3. Evaluates the results using the evaluation script
# 4. Optionally stops the server when complete
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
CLAUDE_WORKERS=10
CHATTS_WORKERS=10
EVAL_WORKERS=4
CHATTS_PORT=5000
CHATTS_SERVER_URL="http://localhost:${CHATTS_PORT}"
STOP_SERVER=false

# Create required directories
mkdir -p "${OUTPUT_DIR}"

# Display usage information
show_help() {
  echo "Usage: $0 [--dataset PATH] [--output-dir DIR] [--num-runs N] [--workers N] [--max-samples N] [--stop-server]"
  echo ""
  echo "Options:"
  echo "  --dataset PATH       Path to the dataset JSON file"
  echo "  --output-dir DIR     Directory to save results (default: ../evaluation/results)"
  echo "  --num-runs N         Number of runs to perform (default: 1)"
  echo "  --claude-workers N   Number of Claude API parallel workers (default: 10)"
  echo "  --chatts-workers N   Number of ChatTS API parallel workers (default: 10)"
  echo "  --eval-workers N     Number of evaluation parallel workers (default: 4)"
  echo "  --max-samples N      Maximum number of samples to process (default: 200)"
  echo "  --chatts-port N      Port for ChatTS server (default: 5000)"
  echo "  --stop-server        Stop the ChatTS server when done"
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
    --claude-workers)
      CLAUDE_WORKERS="$2"
      shift 2
      ;;
    --chatts-workers)
      CHATTS_WORKERS="$2"
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
    --chatts-port)
      CHATTS_PORT="$2"
      CHATTS_SERVER_URL="http://localhost:${CHATTS_PORT}"
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
LOG_PATH="${PROJECT_ROOT}/logs/injection_workflow_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_PATH"
echo "Configuration:" | tee -a "$LOG_PATH"
echo "DATASET_PATH        = ${DATASET_PATH}" | tee -a "$LOG_PATH"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}" | tee -a "$LOG_PATH"
echo "NUM_RUNS            = ${NUM_RUNS}" | tee -a "$LOG_PATH"
echo "MAX_SAMPLES         = ${MAX_SAMPLES}" | tee -a "$LOG_PATH"
echo "CLAUDE_WORKERS      = ${CLAUDE_WORKERS}" | tee -a "$LOG_PATH"
echo "CHATTS_WORKERS      = ${CHATTS_WORKERS}" | tee -a "$LOG_PATH"
echo "EVAL_WORKERS        = ${EVAL_WORKERS}" | tee -a "$LOG_PATH"
echo "CHATTS_SERVER_URL   = ${CHATTS_SERVER_URL}" | tee -a "$LOG_PATH"
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
# 1) Record start time and verify ChatTS server is running
# ------------------------------------------------------------------------------
# Record start time
STARTTIME=$(date +%s)

echo "Checking if ChatTS server is running..." | tee -a "$LOG_PATH"

if nc -z localhost $CHATTS_PORT 2>/dev/null; then
    echo "ChatTS server running on port $CHATTS_PORT" | tee -a "$LOG_PATH"
else
    echo "ERROR: ChatTS server is not running on port $CHATTS_PORT" | tee -a "$LOG_PATH"
    echo "Please start it first with: src/chatts_utils/start_chatts_server.sh" | tee -a "$LOG_PATH"
    exit 1
fi

# Run the pipeline multiple times with different output paths
for RUN_NUM in $(seq 1 $NUM_RUNS); do
    echo "" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"
    echo "Starting Injection Run #$RUN_NUM of $NUM_RUNS" | tee -a "$LOG_PATH"
    echo "========================================" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    # Define output paths for this run
    RUN_SUFFIX="run${RUN_NUM}"

    # Output paths with run number suffix
    INJECTION_OUT="${OUTPUT_DIR}/claude-injection-${DATASET_NAME}-${RUN_SUFFIX}/generated_answer.json"

    # Create output directory
    mkdir -p "$(dirname "$INJECTION_OUT")"

    # ------------------------------------------------------------------------------
    # 2) Generate ChatTS observations (shared across all runs if cached)
    # ------------------------------------------------------------------------------
    CHATTS_OBS_DIR="${OUTPUT_DIR}/chatts-observations-${DATASET_NAME}"
    mkdir -p "$CHATTS_OBS_DIR"
    CHATTS_OBS="${CHATTS_OBS_DIR}/generated_answer.json"

    if [ -f "$CHATTS_OBS" ]; then
        echo "Found existing ChatTS observations at $CHATTS_OBS" | tee -a "$LOG_PATH"
        echo "Reusing existing observations for injection" | tee -a "$LOG_PATH"
    else
        echo "Generating ChatTS observations..." | tee -a "$LOG_PATH"

        conda activate evaluation
        export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

        python "${PROJECT_ROOT}/src/chatts_injection.py" \
            --dataset_path "$DATASET_PATH" \
            --output_path "$CHATTS_OBS" \
            --server_url "$CHATTS_SERVER_URL" \
            --workers $CHATTS_WORKERS | tee -a "$LOG_PATH"

        echo "ChatTS observations saved to $CHATTS_OBS" | tee -a "$LOG_PATH"
    fi

    # ------------------------------------------------------------------------------
    # 3) Run Claude with ChatTS injection
    # ------------------------------------------------------------------------------
    echo "Running Claude with ChatTS injection (Run #$RUN_NUM)" | tee -a "$LOG_PATH"
    echo "    Dataset    : $DATASET_PATH" | tee -a "$LOG_PATH"
    echo "    Injection  : $CHATTS_OBS" | tee -a "$LOG_PATH"
    echo "    Output     : $INJECTION_OUT" | tee -a "$LOG_PATH"

    conda activate evaluation
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    python "${PROJECT_ROOT}/src/claude_thinking_with_injection.py" \
        --dataset_path "$DATASET_PATH" \
        --injection_path "$CHATTS_OBS" \
        --output_path "$INJECTION_OUT" \
        --workers $CLAUDE_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM: Injection results saved to $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    # ------------------------------------------------------------------------------
    # 4) Evaluate injection results
    # ------------------------------------------------------------------------------
    echo "Evaluating injection results (Run #$RUN_NUM)" | tee -a "$LOG_PATH"

    # Extract experiment name
    INJECTION_EXP_NAME=$(basename "$(dirname "$INJECTION_OUT")")

    # Ensure we're in the evaluation environment
    conda activate evaluation

    # Set PYTHONPATH to include the project root directory
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    # Evaluate injection results
    echo "   Evaluating injection results..." | tee -a "$LOG_PATH"
    python "${PROJECT_ROOT}/evaluation/evaluate_with_sampling.py" \
        --exp "$INJECTION_EXP_NAME" \
        --dataset "$DATASET_PATH" \
        --generated "$INJECTION_OUT" \
        --num_workers $EVAL_WORKERS | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM: Injection evaluation complete" | tee -a "$LOG_PATH"
    echo "   Results in ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"

    echo "Run #$RUN_NUM Complete!" | tee -a "$LOG_PATH"
    echo "Results:" | tee -a "$LOG_PATH"
    echo "  - Injection results: $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "  - Injection evaluation: ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
done

# ------------------------------------------------------------------------------
# 4) Runtime Summary
# ------------------------------------------------------------------------------
ENDTIME=$(date +%s)
RUNTIME=$((ENDTIME - STARTTIME))
echo "=======================================" | tee -a "$LOG_PATH"
echo "Total runtime: $RUNTIME seconds ($(($RUNTIME / 60)) minutes)" | tee -a "$LOG_PATH"
echo "=======================================" | tee -a "$LOG_PATH"

# ------------------------------------------------------------------------------
# 5) Summary of all runs
# ------------------------------------------------------------------------------
echo "========================================" | tee -a "$LOG_PATH"
echo "All $NUM_RUNS Runs Completed Successfully" | tee -a "$LOG_PATH"
echo "========================================" | tee -a "$LOG_PATH"

echo "Summary of result locations:" | tee -a "$LOG_PATH"
for RUN_NUM in $(seq 1 $NUM_RUNS); do
    RUN_SUFFIX="run${RUN_NUM}"

    INJECTION_OUT="${OUTPUT_DIR}/claude-injection-${DATASET_NAME}-${RUN_SUFFIX}/generated_answer.json"
    INJECTION_EXP_NAME=$(basename "$(dirname "$INJECTION_OUT")")

    echo "Run #$RUN_NUM:" | tee -a "$LOG_PATH"
    echo "  - Injection results: $INJECTION_OUT" | tee -a "$LOG_PATH"
    echo "  - Injection evaluation: ${PROJECT_ROOT}/evaluation/exp/$INJECTION_EXP_NAME/" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
done

# ------------------------------------------------------------------------------
# 6) Stop server if requested
# ------------------------------------------------------------------------------
if [ "$STOP_SERVER" = true ]; then
    echo "Stopping ChatTS server..." | tee -a "$LOG_PATH"
    "${PROJECT_ROOT}/src/chatts_utils/stop_chatts_server.sh" || true
    echo "ChatTS server stopped" | tee -a "$LOG_PATH"
else
    echo "" | tee -a "$LOG_PATH"
    echo "NOTE: The ChatTS server is still running." | tee -a "$LOG_PATH"
    echo "When you are done with all evaluations, stop it using:" | tee -a "$LOG_PATH"
    echo "  ${PROJECT_ROOT}/src/chatts_utils/stop_chatts_server.sh" | tee -a "$LOG_PATH"
    echo "" | tee -a "$LOG_PATH"
fi

echo "Log saved to: ${LOG_PATH}" | tee -a "$LOG_PATH"
echo "Workflow complete!" | tee -a "$LOG_PATH"
