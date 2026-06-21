#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_stochastic_refinement.sh
#
# Stage 2: Stochastic refinement of synthetic data generators.
# Simplifies Stage 1 models into sampling-based generators that produce
# diverse synthetic time series data.
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/scripts
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                 # project root
SYNTHETIC_DIR="${PROJECT_ROOT}/dataset/synthetic"

echo "========================================"
echo "Stage 2: Stochastic Refinement"
echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "SYNTHETIC_DIR = $SYNTHETIC_DIR"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH="${SYNTHETIC_DIR}/sample_data/qa_benchmark_base_train.json"
RESULTS_DIR="${SYNTHETIC_DIR}/results/iterative_results"
OUTPUT_DIR="${SYNTHETIC_DIR}/results/stochastic_results"
NUM_CLAUDE_CALLS=3
MAX_WORKERS=""
REGION="us-west-2"

# ── Parse command line arguments ─────────────────────────────────────────────
SAMPLE_IDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_claude_calls)
            NUM_CLAUDE_CALLS="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --sample_ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SAMPLE_IDS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Dataset:           $DATASET_PATH"
echo "Stage 1 results:   $RESULTS_DIR"
echo "Output:            $OUTPUT_DIR"
echo "Claude calls:      $NUM_CLAUDE_CALLS"
echo "Region:            $REGION"
if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    echo "Sample IDs:        ${SAMPLE_IDS[*]}"
else
    echo "Sample IDs:        (all available)"
fi
echo "Start time:        $(date)"
echo ""

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ── Build Python command ─────────────────────────────────────────────────────
CMD=(
    python "${SYNTHETIC_DIR}/stochastic_ts_generation.py"
    --dataset_path="${DATASET_PATH}"
    --results_dir="${RESULTS_DIR}"
    --output_dir="${OUTPUT_DIR}"
    --num_claude_calls="${NUM_CLAUDE_CALLS}"
    --region="${REGION}"
)

if [ -n "${MAX_WORKERS}" ]; then
    CMD+=(--max_workers="${MAX_WORKERS}")
fi

if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    CMD+=(--sample_ids "${SAMPLE_IDS[@]}")
fi

# ── Run ──────────────────────────────────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "========================================"
echo "Stage 2 Complete"
echo "Results: ${OUTPUT_DIR}"
echo ""
echo "NEXT STEP: Review the results in ${OUTPUT_DIR} and keep the best"
echo "stochastic function for each sample before running Stage 3."
echo "End time: $(date)"
echo "========================================"
