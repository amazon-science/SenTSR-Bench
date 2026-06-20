# Sample Data for Synthetic Generation

This directory contains the input data for the synthetic data generation pipeline.

## Required Input File

### `qa_benchmark_base_train.json`

A JSON array of training samples, each with the following fields:

```json
{
  "id": "sample_001",
  "timeseries": [[...], [...], [...]],
  "cols": ["Acceleration", "Velocity", "Temperature"],
  "question": "...",
  "question_type": "what_happened",
  "answer": "vibration amplitude increases gradually over time",
  "attributes": ["vibration amplitude increases gradually over time"],
  "ability_types": ["MCQ_obs"]
}
```

Fields:
- **`id`**: Unique sample identifier
- **`timeseries`**: List of 3 channels `[acceleration, velocity, temperature]`, each a list of floats
- **`cols`**: Column names (always `["Acceleration", "Velocity", "Temperature"]`)
- **`question`**: The question text with `<ts><ts/>` placeholders for time series
- **`question_type`**: One of `what_happened`, `how_happened`, `suggested_fix`
- **`answer`**: The correct answer text (anomaly description)
- **`attributes`**: List of ground-truth attributes
- **`ability_types`**: One of `["MCQ_obs"]`, `["MCQ_cause"]`, `["MCQ_fix"]`

The synthetic generation pipeline (Stages 1-2) uses only `what_happened` samples from this file. In Stage 4a, the original `answer` fields are used as seeds for LLM-based diversification to generate varied observations, root causes, and corrective actions.

## How to Prepare Your Data

1. Prepare your real time series data with anomaly descriptions
2. Format as `qa_benchmark_base_train.json` following the schema above
3. Place the file in this directory (or specify the path via CLI arguments)
