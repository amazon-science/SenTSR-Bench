# Synthetic Time Series Benchmark Generation Pipeline

This module generates synthetic time series data and benchmarks for training time series reasoning models. The pipeline transforms real-world time series data into diverse synthetic datasets that preserve key anomaly patterns.

## Pipeline Overview

The pipeline consists of 4 stages:

### Stage 1: Iterative Model Generation (`iterative_ts_generation.py`)

Uses Claude (via AWS Bedrock) to iteratively generate Python code that models anomaly patterns from real training data.

- Input: `qa_benchmark_base_train.json` (only `what_happened` samples)
- For each sample: sends time series visualization + anomaly description to Claude
- Claude generates a `generate_synthetic_anomaly()` Python function
- 3 iterations per sample: execute code, visually compare, send feedback
- Output: `results/iterative_results/Sample_{id}/function_{1,2,3}.py`

### Stage 2: Stochastic Refinement (`stochastic_ts_generation.py`)

Simplifies Stage 1 models into sampling-based generators that produce diverse data.

- Input: Stage 1 results + original dataset
- Asks Claude to replace hardcoded parameters with probabilistic sampling
- Generates 3 different versions per sample for selection
- Output: `results/stochastic_results/Sample_{id}/stochastic_function{1,2,3}.py`
- **Manual step**: review and select the best version for each sample

### Stage 3: Dataset Generation (`generate_synthetic_dataset.py`)

Generates synthetic time series data at scale using the selected stochastic models.

- Input: Stochastic functions from Stage 2 + original dataset
- Dynamically loads Python functions via `importlib`
- Generates N samples per source with different random seeds
- Output: `results/synthetic_training_data/data_ts.json`

### Stage 4: Benchmark Generation (3 scripts, orchestrated by shell script)

Converts synthetic time series into a structured benchmark dataset:

- **4a** `generate_qa_benchmark.py`: Uses LLM to diversify the original anomaly descriptions into varied observations, root causes, and corrective actions, then generates QA pairs
- **4b** `generate_mcq_benchmark.py`: Converts to multiple-choice format with distractors from other source samples
- **4c** `filter_mcq_benchmark.py`: Filters to keep only MCQ_obs and MCQ_cause questions

Output: `results/synthetic_training_data/rme_synthetic_easy.json`

## Quick Start

### Prerequisites

- Python 3.10+
- AWS credentials configured for Bedrock access
- Required packages: `boto3`, `numpy`, `matplotlib`, `tenacity`, `scipy`

### 1. Prepare Input Data

Place your training data in `sample_data/`:
- `qa_benchmark_base_train.json`: Training samples with time series and anomaly descriptions

See `sample_data/README.md` for format details.

### 2. Run the Pipeline

```bash
# Stage 1: Generate initial models (requires Claude API access)
./scripts/run_iterative_generation.sh

# Stage 2: Refine into stochastic generators
./scripts/run_stochastic_refinement.sh

# Manual step: review results/stochastic_results/ and keep best models

# Stages 3-4: Generate full synthetic benchmark
./scripts/run_synthetic_benchmark.sh 100  # 100 samples per source
```

### 3. Outputs

All outputs are saved under `results/synthetic_training_data/`:

| File | Description |
|------|-------------|
| `data_ts.json` | Synthetic time series data |
| `dataset_summary.json` | Per-source sample statistics |
| `diversified_answers.json` | LLM-generated answer variations (cached) |
| `qa_synthetic_base.json` | QA benchmark (3 question types) |
| `rme_synthetic_easy.json` | Filtered MCQ benchmark |

## File Structure

```
dataset/synthetic/
├── README.md                          # This file
├── iterative_ts_generation.py         # Stage 1: Iterative model generation
├── stochastic_ts_generation.py        # Stage 2: Stochastic refinement
├── generate_synthetic_dataset.py      # Stage 3: Dataset generation at scale
├── generate_qa_benchmark.py           # Stage 4a: QA benchmark generation (LLM-based)
├── generate_mcq_benchmark.py          # Stage 4b: MCQ benchmark generation
├── filter_mcq_benchmark.py            # Stage 4c: Filter by ability types
├── sample_data/                       # Input data
│   └── README.md                      # Input format documentation
└── results/                           # Generated outputs (created by pipeline)
    ├── iterative_results/             # Stage 1 outputs
    ├── stochastic_results/            # Stage 2 outputs
    └── synthetic_training_data/       # Stages 3-4 outputs
```

## Data Formats

### Time Series Structure
- 3 channels: Acceleration, Velocity, Temperature
- Standardized using Median Absolute Deviation (MAD)
- Variable length (preserved from original data)

### Question Types
- `what_happened` / `MCQ_obs`: Identify the anomaly pattern
- `how_happened` / `MCQ_cause`: Identify the root cause
- `suggested_fix` / `MCQ_fix`: Recommend corrective action

### MCQ Format
Each MCQ sample includes 4 options (1 correct + 3 distractors from different source samples), with shuffled order and a `correct_index` field.
