#!/usr/bin/env python3
"""
Stage 4c: Filter MCQ Benchmark to Keep Only MCQ_obs and MCQ_cause Questions

This script filters the generated MCQ benchmark to keep only questions with
ability_types of MCQ_obs and MCQ_cause, excluding MCQ_fix questions.
"""

import os
import json
import argparse
from typing import List, Dict, Any

# === PATH HANDLING ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def filter_mcq_benchmark(
    input_file: str,
    output_file: str,
    keep_ability_types: List[str] = None
):
    """Filter MCQ benchmark to keep only questions with specified ability types."""
    if keep_ability_types is None:
        keep_ability_types = ["MCQ_obs", "MCQ_cause"]

    print(f"Filtering {input_file} to keep only {keep_ability_types}...")

    with open(input_file, 'r') as f:
        benchmark = json.load(f)

    total_before = len(benchmark)
    filtered_benchmark = []
    counts_by_type = {}

    for item in benchmark:
        ability_types = item.get('ability_types', [])
        for ability_type in ability_types:
            counts_by_type[ability_type] = counts_by_type.get(ability_type, 0) + 1
        if any(ability_type in keep_ability_types for ability_type in ability_types):
            filtered_benchmark.append(item)

    total_after = len(filtered_benchmark)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(filtered_benchmark, f, indent=2)

    print(f"Original benchmark: {total_before} questions")
    print(f"Filtered benchmark: {total_after} questions")
    print(f"Removed: {total_before - total_after} questions")
    print("Counts by ability type:")
    for ability_type, count in counts_by_type.items():
        status = "KEPT" if ability_type in keep_ability_types else "REMOVED"
        print(f"  {ability_type}: {count} questions ({status})")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4c: Filter MCQ Benchmark by Ability Types")
    parser.add_argument("--input_file", type=str,
                        default="./results/synthetic_training_data/rme_synthetic_easy_unfiltered.json",
                        help="Path to the input MCQ benchmark file")
    parser.add_argument("--output_file", type=str,
                        default="./results/synthetic_training_data/rme_synthetic_easy.json",
                        help="Path to the output filtered benchmark file")
    parser.add_argument("--keep_ability_types", type=str, nargs="+",
                        default=["MCQ_obs", "MCQ_cause"],
                        help="List of ability types to keep")

    args = parser.parse_args()

    filter_mcq_benchmark(
        resolve_path(args.input_file),
        resolve_path(args.output_file),
        args.keep_ability_types
    )
