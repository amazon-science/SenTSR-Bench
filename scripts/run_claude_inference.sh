#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_claude_inference.sh
#
# Runs Claude inference on a dataset.
# Handles setting up proper directories and configurations.
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
TEXT_ONLY=false
MAX_SAMPLES=200
WORKERS=4

# Create required directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

# Display usage information
show_help() {
  echo "Usage: $0 [--dataset PATH] [--output-dir DIR] [--output-name NAME] [--workers N] [--max-samples N] [--text-only]"
  echo ""
  echo "Options:"
  echo "  --dataset PATH       Path to the dataset JSON file"
  echo "  --output-dir DIR     Directory to save results (default: ../evaluation/results)"
  echo "  --output-name NAME   Name for the output JSON file (default: based on mode)"
  echo "  --workers N          Number of parallel workers (default: 4)"
  echo "  --max-samples N      Maximum number of samples to process (default: 200)"
  echo "  --text-only          Run inference with text-only mode (for VLM)"
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
    --output-name)
      OUTPUT_NAME="$2"
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
    --text-only)
      TEXT_ONLY=true
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

# Determine output filename based on mode and other settings
if [[ -z "${OUTPUT_NAME-}" ]]; then
  BASE_NAME="claude"

  # Add text-only suffix if used
  if [[ "$TEXT_ONLY" == true ]]; then
    BASE_NAME="${BASE_NAME}_text_only"
  fi

  OUTPUT_NAME="${BASE_NAME}_results.json"
fi

# Full paths for output
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"
LOG_PATH="${PROJECT_ROOT}/logs/${BASE_NAME}_$(date +%Y-%m-%d-%H-%M-%S).log"

echo "========================================"
echo "Configuration:"
echo "DATASET_PATH   = ${DATASET_PATH}"
echo "OUTPUT_PATH    = ${OUTPUT_PATH}"
echo "LOG_PATH       = ${LOG_PATH}"
echo "TEXT_ONLY      = ${TEXT_ONLY}"
echo "MAX_SAMPLES    = ${MAX_SAMPLES}"
echo "WORKERS        = ${WORKERS}"
echo "========================================"
echo ""

# ── Run Claude inference script ───────────────────────────────────────────────
# Construct the command
SCRIPT="${PROJECT_ROOT}/src/claude_thinking_inference.py"

COMMAND="python ${SCRIPT} \
  --dataset_path ${DATASET_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_samples ${MAX_SAMPLES} \
  --workers ${WORKERS}"

# Add optional arguments
if [[ "$TEXT_ONLY" == true ]]; then
  COMMAND="${COMMAND} --text_only"
fi

echo "Running command: ${COMMAND}"
eval "${COMMAND}" | tee -a "${LOG_PATH}"

# Check if inference completed successfully
if [ $? -eq 0 ]; then
  echo "========================================"
  echo "Claude inference completed successfully!"
  echo "Results saved to: ${OUTPUT_PATH}"
  echo "Log saved to: ${LOG_PATH}"
  echo "========================================"
else
  echo "========================================"
  echo "Claude inference failed with an error"
  echo "Check the log for details: ${LOG_PATH}"
  echo "========================================"
  exit 1
fi