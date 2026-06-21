#!/usr/bin/env python3
"""
Stage 3: Generate Synthetic Dataset at Scale

This script:
1. Loads all stochastic generation functions from Stage 2 results
2. Extracts the sample ID and original timeseries for each function
3. Generates multiple synthetic time series examples per sample
4. Saves the result as a structured JSON file for downstream processing
"""

import os
import sys
import json
import glob
import argparse
import importlib.util
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import traceback

# === PATH HANDLING ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def extract_id_and_timeseries(
    sample_dir: str, dataset_path: str
) -> Tuple[str, Optional[List[List[float]]]]:
    """Extract the sample ID and original timeseries from the directory name and dataset."""
    sample_id = os.path.basename(sample_dir).replace('Sample_', '')
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        for sample in dataset:
            if sample['id'] == sample_id:
                timeseries = sample.get('timeseries')
                return sample_id, timeseries
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return sample_id, None


def load_generation_function(function_path: str) -> Optional[Callable]:
    """Dynamically load a synthetic data generation function from a Python file."""
    try:
        module_name = f"synthetic_function_{os.path.basename(function_path).replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, function_path)
        if spec is None or spec.loader is None:
            print(f"Error: Failed to create module spec from {function_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, 'generate_synthetic_anomaly'):
            return module.generate_synthetic_anomaly

        for attr_name in dir(module):
            if attr_name.startswith('generate_'):
                attr = getattr(module, attr_name)
                if callable(attr):
                    print(f"Found alternative generation function: {attr_name}")
                    return attr

        print(f"Error: No suitable generation function found in {function_path}")
        return None
    except Exception as e:
        print(f"Error loading function from {function_path}: {e}")
        traceback.print_exc()
        return None


def generate_synthetic_samples(
    function: Callable,
    n_samples: int = 100,
    count: int = 100,
    base_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate multiple synthetic time series samples using the provided function."""
    samples = []
    for i in range(count):
        try:
            seed = base_seed + i
            try:
                result = function(n_samples=n_samples, seed=seed)
            except TypeError:
                try:
                    result = function(n_samples)
                except TypeError:
                    try:
                        result = function(seed=seed)
                    except TypeError:
                        result = function()

            if result and isinstance(result, tuple) and len(result) == 3:
                samples.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Generated {i+1}/{count} samples")
            else:
                print(f"Warning: Function returned invalid data for sample {i+1}")
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
    return samples


def process_sample_dir(
    sample_dir: str, dataset_path: str, samples_per_source: int = 100
) -> Optional[Dict[str, Any]]:
    """Process a sample directory to extract ID and generate synthetic data."""
    sample_id, timeseries = extract_id_and_timeseries(sample_dir, dataset_path)
    if not timeseries:
        print(f"Warning: No timeseries data found for sample {sample_id}")
        return None

    n_samples = len(timeseries[0]) if timeseries and len(timeseries) > 0 else 100
    print(f"Using {n_samples} time points from original timeseries for sample {sample_id}")

    function_paths = glob.glob(os.path.join(sample_dir, "stochastic_function*.py"))
    if not function_paths:
        print(f"Warning: No function files found for sample {sample_id}")
        return None

    if len(function_paths) > 1:
        print(f"Warning: Multiple function files found for sample {sample_id}: "
              f"{[os.path.basename(p) for p in function_paths]}")
        print(f"Using the first one: {os.path.basename(function_paths[0])}")

    function_path = function_paths[0]
    function = load_generation_function(function_path)
    if not function:
        print(f"Warning: Could not load generation function for sample {sample_id}")
        return None

    print(f"Generating {samples_per_source} samples for source {sample_id}...")
    samples = generate_synthetic_samples(function, n_samples, samples_per_source)
    if not samples:
        print(f"Warning: No samples generated for source {sample_id}")
        return None

    print(f"Successfully generated {len(samples)} samples for source {sample_id}")
    return {
        'id': sample_id,
        'samples': samples
    }


def create_data_ts_entries(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a list of synthetic time series entries from generation results."""
    data_ts = []
    sample_counter = 0

    for result in results:
        samples = result['samples']

        for i, sample in enumerate(samples):
            acceleration, velocity, temperature = sample
            synthetic_id = f"synthetic_{sample_counter:06d}"
            sample_counter += 1

            entry = {
                'id': synthetic_id,
                'timeseries': [
                    acceleration.tolist(),
                    velocity.tolist(),
                    temperature.tolist()
                ],
                'cols': ['Acceleration', 'Velocity', 'Temperature'],
                'original_id': result['id'],
                'synthetic': True
            }
            data_ts.append(entry)

    return data_ts


def generate_synthetic_dataset(
    stochastic_results_dir: str,
    dataset_path: str,
    output_dir: str,
    samples_per_source: int = 100
):
    """Generate a synthetic dataset by processing all sample directories."""
    os.makedirs(output_dir, exist_ok=True)

    sample_dirs = glob.glob(os.path.join(stochastic_results_dir, "Sample_*"))
    print(f"Found {len(sample_dirs)} sample directories")

    results = []
    for sample_dir in sample_dirs:
        result = process_sample_dir(sample_dir, dataset_path, samples_per_source)
        if result:
            results.append(result)

    data_ts = create_data_ts_entries(results)

    data_ts_path = os.path.join(output_dir, "data_ts.json")
    with open(data_ts_path, 'w') as f:
        json.dump(data_ts, f, indent=2)
    print(f"Saved {len(data_ts)} synthetic samples to {data_ts_path}")

    # Summary grouped by original source sample
    source_counts = {}
    for entry in data_ts:
        orig_id = entry['original_id']
        source_counts[orig_id] = source_counts.get(orig_id, 0) + 1

    summary = {
        'total_samples': len(data_ts),
        'sources': source_counts
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("Dataset summary:")
    for orig_id, count in source_counts.items():
        print(f"  Source {orig_id}: {count} samples")
    print(f"Total: {len(data_ts)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Generate Synthetic Dataset at Scale")
    parser.add_argument("--stochastic_results_dir", type=str,
                        default="./results/stochastic_results",
                        help="Directory containing stochastic results from Stage 2")
    parser.add_argument("--dataset_path", type=str,
                        default="./sample_data/qa_benchmark_base_train.json",
                        help="Path to the original dataset JSON file")
    parser.add_argument("--output_dir", type=str,
                        default="./results/synthetic_training_data",
                        help="Directory to save outputs")
    parser.add_argument("--samples_per_source", type=int, default=100,
                        help="Number of synthetic samples to generate per source")

    args = parser.parse_args()

    generate_synthetic_dataset(
        resolve_path(args.stochastic_results_dir),
        resolve_path(args.dataset_path),
        resolve_path(args.output_dir),
        args.samples_per_source
    )
