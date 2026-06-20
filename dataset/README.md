# TSR Knowledge Injection Dataset Processing

This directory contains the scripts and data necessary for preprocessing datasets for time series reasoning (TSR) knowledge injection.

## Data Requirements

To run the preprocessing script successfully, you need the following input data files:

1. **dataset_a.json**: A JSON file containing multi-question time series reasoning entries.
   - Format: Array of JSON objects with `timeseries`, `cols`, `question`, `answer`, `attributes`, and `ability_types` fields
   - Each entry may contain multiple questions and answers that will be split into individual entries

2. **MCQ_2_TS.jsonl**: A JSONL (JSON Lines) file containing the raw TS&Language (MCQ2) dataset.
   - Format: Each line is a JSON object containing:
     - `uuid`: Unique identifier
     - `description`: Text description of the time series
     - `question`: Question text
     - `options`: Array of possible answers
     - `answer_index`: Index of the correct answer
     - `series`: Original time series data (array of numbers)
     - `new_series`: Updated time series data (array of numbers)

## Output Files

The script generates the following outputs in the specified output directory:

1. **dataset_a_split.json**: Dataset A entries split into individual questions
2. **dataset_a_split_filtered.json**: Filtered version of the split dataset, containing only entries with specific ability types
3. **mcq2_qa_eval_100.json**: 100 sampled entries from the MCQ2 dataset
4. **dataset_a_with_mcq2.json**: Final merged dataset containing both dataset_a and MCQ2 entries

## Running the Script

```bash
python preprocess_dataset.py [--dataset_a PATH] [--mcq2_source PATH] [--output_dir PATH] [--mcq2_sample_size SIZE] [--mcq2_seed SEED]
```

### Parameters:
- `--dataset_a`: Path to the dataset_a.json file (default: ./dataset_a.json)
- `--mcq2_source`: Path to the MCQ2_TS.jsonl source file (default: ./MCQ_2_TS.jsonl)
- `--output_dir`: Directory to save processed datasets (default: ./processed)
- `--mcq2_sample_size`: Number of entries to sample from MCQ2 (default: 100)
- `--mcq2_seed`: Random seed for reproducibility of MCQ2 sampling (default: 42)

## Processing Steps

1. **Split dataset_a**: Divides multi-question entries into individual questions
2. **Filter dataset_a**: Keeps only entries with ability types containing 'causal', 'deductive', or 'inductive'
3. **Sample MCQ2**: Samples entries from the MCQ2_TS.jsonl file with a fixed seed for reproducibility
4. **Merge datasets**: Combines the processed dataset_a with sampled MCQ2 entries into a single dataset

## Dependencies

- Python 3.6+
- Required libraries:
  - numpy
  - tqdm