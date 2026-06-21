#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_iterative_generation.sh
#
# Stage 1: Iterative synthetic data generation using Claude.
# Uses Claude (via AWS Bedrock) to iteratively generate Python code that models
# anomaly patterns from real training data.
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/scripts
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                 # project root
SYNTHETIC_DIR="${PROJECT_ROOT}/dataset/synthetic"

echo "========================================"
echo "Stage 1: Iterative Synthetic Data Generation"
echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "SYNTHETIC_DIR = $SYNTHETIC_DIR"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH="${SYNTHETIC_DIR}/sample_data/qa_benchmark_base_train.json"
OUTPUT_DIR="${SYNTHETIC_DIR}/results/iterative_results"
ITERATIONS=3
MAX_WORKERS=10
REGION="us-west-2"

# ── Parse command line arguments ─────────────────────────────────────────────
SAMPLE_IDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
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

echo "Dataset:      $DATASET_PATH"
echo "Output:       $OUTPUT_DIR"
echo "Iterations:   $ITERATIONS"
echo "Max workers:  $MAX_WORKERS"
echo "Region:       $REGION"
if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    echo "Sample IDs:   ${SAMPLE_IDS[*]}"
else
    echo "Sample IDs:   (all what_happened samples)"
fi
echo "Start time:   $(date)"
echo ""

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ── Build Python command ─────────────────────────────────────────────────────
CMD=(
    python "${SYNTHETIC_DIR}/iterative_ts_generation.py"
    --dataset_path="${DATASET_PATH}"
    --output_dir="${OUTPUT_DIR}"
    --iterations="${ITERATIONS}"
    --max_workers="${MAX_WORKERS}"
    --region="${REGION}"
    --thinking
)

if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    CMD+=(--sample_ids "${SAMPLE_IDS[@]}")
fi

# ── Run ──────────────────────────────────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "========================================"
echo "Stage 1 Complete"
echo "Results: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "========================================"
