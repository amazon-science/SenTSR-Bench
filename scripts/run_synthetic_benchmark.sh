#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_synthetic_benchmark.sh
#
# Stages 3-4: Generate synthetic benchmark dataset.
# Orchestrates the full pipeline from synthetic data generation to filtered
# MCQ benchmark creation.
#
# Usage: ./scripts/run_synthetic_benchmark.sh [SAMPLES_PER_SOURCE]
#   SAMPLES_PER_SOURCE defaults to 100 if not specified.
# ==============================================================================

# ── Script path handling ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # e.g., …/scripts
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                 # project root
SYNTHETIC_DIR="${PROJECT_ROOT}/dataset/synthetic"

echo "========================================"
echo "Stages 3-4: Synthetic Benchmark Generation"
echo "========================================"
echo "SCRIPT_DIR    = $SCRIPT_DIR"
echo "PROJECT_ROOT  = $PROJECT_ROOT"
echo "SYNTHETIC_DIR = $SYNTHETIC_DIR"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
STOCHASTIC_RESULTS_DIR="${SYNTHETIC_DIR}/results/stochastic_results"
TRAINING_DATA_DIR="${SYNTHETIC_DIR}/results/synthetic_training_data"
DATASET_PATH="${SYNTHETIC_DIR}/sample_data/qa_benchmark_base_train.json"
SAMPLES_PER_SOURCE=${1:-100}
REGION="${REGION:-us-west-2}"

echo "Samples per source:      $SAMPLES_PER_SOURCE"
echo "Stochastic results dir:  $STOCHASTIC_RESULTS_DIR"
echo "Dataset path:            $DATASET_PATH"
echo "Region:                  $REGION"
echo "Output dir:              $TRAINING_DATA_DIR"
echo "Start time:              $(date)"
echo ""

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p "${TRAINING_DATA_DIR}"

# ── Step 1: Generate synthetic dataset ───────────────────────────────────────
echo "Step 1: Generating synthetic dataset..."
python "${SYNTHETIC_DIR}/generate_synthetic_dataset.py" \
    --stochastic_results_dir="${STOCHASTIC_RESULTS_DIR}" \
    --dataset_path="${DATASET_PATH}" \
    --output_dir="${TRAINING_DATA_DIR}" \
    --samples_per_source="${SAMPLES_PER_SOURCE}"

if [ ! -f "${TRAINING_DATA_DIR}/data_ts.json" ]; then
    echo "Error: Failed to create data_ts.json"
    exit 1
fi
echo "Successfully created data_ts.json"
echo ""

# ── Step 2: Generate QA benchmark (uses LLM for answer diversification) ──────
echo "Step 2: Generating QA benchmark (with LLM diversification)..."
python "${SYNTHETIC_DIR}/generate_qa_benchmark.py" \
    --data_ts_path="${TRAINING_DATA_DIR}/data_ts.json" \
    --dataset_path="${DATASET_PATH}" \
    --output_path="${TRAINING_DATA_DIR}/qa_synthetic_base.json" \
    --region="${REGION}"

if [ ! -f "${TRAINING_DATA_DIR}/qa_synthetic_base.json" ]; then
    echo "Error: Failed to create qa_synthetic_base.json"
    exit 1
fi
echo "Successfully created qa_synthetic_base.json"
echo ""

# ── Step 3: Generate MCQ benchmark ───────────────────────────────────────────
echo "Step 3: Generating MCQ benchmark..."
python "${SYNTHETIC_DIR}/generate_mcq_benchmark.py" \
    --qa_benchmark_path="${TRAINING_DATA_DIR}/qa_synthetic_base.json" \
    --output_path="${TRAINING_DATA_DIR}/rme_synthetic_easy_unfiltered.json"

if [ ! -f "${TRAINING_DATA_DIR}/rme_synthetic_easy_unfiltered.json" ]; then
    echo "Error: Failed to create rme_synthetic_easy_unfiltered.json"
    exit 1
fi
echo "Successfully created rme_synthetic_easy_unfiltered.json"
echo ""

# ── Step 4: Filter MCQ benchmark ────────────────────────────────────────────
echo "Step 4: Filtering MCQ benchmark (keeping MCQ_obs and MCQ_cause)..."
python "${SYNTHETIC_DIR}/filter_mcq_benchmark.py" \
    --input_file="${TRAINING_DATA_DIR}/rme_synthetic_easy_unfiltered.json" \
    --output_file="${TRAINING_DATA_DIR}/rme_synthetic_easy.json" \
    --keep_ability_types "MCQ_obs" "MCQ_cause"

if [ ! -f "${TRAINING_DATA_DIR}/rme_synthetic_easy.json" ]; then
    echo "Error: Failed to create filtered rme_synthetic_easy.json"
    exit 1
fi
echo "Successfully created rme_synthetic_easy.json"
echo ""

# ── Print dataset statistics ─────────────────────────────────────────────────
echo "========================================"
echo "Synthetic Benchmark Dataset Statistics"
echo "========================================"
echo "From dataset_summary.json:"
cat "${TRAINING_DATA_DIR}/dataset_summary.json"
echo ""
echo "QA benchmark count: $(python -c "import json; print(len(json.load(open('${TRAINING_DATA_DIR}/qa_synthetic_base.json'))))")"
echo "MCQ benchmark count: $(python -c "import json; print(len(json.load(open('${TRAINING_DATA_DIR}/rme_synthetic_easy.json'))))")"
echo ""
echo "========================================"
echo "Stages 3-4 Complete"
echo "Results: ${TRAINING_DATA_DIR}"
echo "End time: $(date)"
echo "========================================"
