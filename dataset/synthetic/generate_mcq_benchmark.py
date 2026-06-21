#!/usr/bin/env python3
"""
Stage 4b: Generate Multiple Choice Questions (MCQs) for Synthetic Time Series Data

This script:
1. Takes a QA benchmark dataset (Stage 4a output) as input
2. Converts each question to a multiple-choice format
3. Groups data by original source sample and question type to create distractors
4. Outputs an MCQ benchmark dataset in JSON format
"""

import os
import json
import random
import argparse
import sys
from typing import Dict, List, Any
from collections import defaultdict

# === PATH HANDLING ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


# Constants
NUM_CHOICES = 4
MCQ_INSTRUCTION = ("\nIn your answer, start by stating your chosen option "
                   "and then provide your explanation in a separate sentence.")


def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def group_data_by_source_and_type(
    data: List[Dict]
) -> Dict[str, Dict[str, List[Dict]]]:
    """Group data by original source sample ID and question type."""
    grouped_data = defaultdict(lambda: defaultdict(list))
    for sample in data:
        orig_id = sample.get("original_id", "")
        question_type = sample.get("question_type", "")
        if orig_id and question_type:
            grouped_data[orig_id][question_type].append(sample)
    return grouped_data


def get_distractor_pool(
    grouped_data: Dict[str, Dict[str, List[Dict]]],
    target_orig_id: str, question_type: str
) -> Dict[str, str]:
    """Create a pool of distractor options from other source samples."""
    options_pool = {}
    for orig_id, type_samples in grouped_data.items():
        if orig_id == target_orig_id:
            continue
        samples = type_samples.get(question_type, [])
        if samples:
            sample = random.choice(samples)
            answer = sample.get("answer", "")
            if answer:
                options_pool[orig_id] = answer
    return options_pool


def create_mcq_for_sample(
    sample: Dict, grouped_data: Dict[str, Dict[str, List[Dict]]]
) -> Dict:
    """Create a multiple-choice question version of a sample."""
    orig_id = sample.get("original_id", "")
    question_type = sample.get("question_type", "")
    correct_answer = sample.get("answer", "")

    mcq_sample = sample.copy()

    options_pool = get_distractor_pool(grouped_data, orig_id, question_type)
    confusion_options = random.sample(
        list(options_pool.values()),
        min(NUM_CHOICES - 1, len(options_pool))
    )

    while len(confusion_options) < NUM_CHOICES - 1:
        if options_pool:
            confusion_options.append(random.choice(list(options_pool.values())))
        else:
            confusion_options.append("No clear pattern observed")

    all_options = [correct_answer] + confusion_options
    random.shuffle(all_options)
    correct_index = all_options.index(correct_answer)

    options_str = json.dumps(all_options)
    mcq_sample["question"] = (
        f"{sample['question']} Choose from: {options_str}{MCQ_INSTRUCTION}"
    )
    mcq_sample["options"] = all_options
    mcq_sample["correct_index"] = correct_index

    if "time_series" in mcq_sample:
        mcq_sample["timeseries"] = mcq_sample.pop("time_series")

    return mcq_sample


def main(qa_benchmark_path: str, output_path: str):
    """Generate MCQ benchmark from QA benchmark data."""
    print(f"Loading data from {qa_benchmark_path}...")
    qa_data = load_json(qa_benchmark_path)

    grouped_data = group_data_by_source_and_type(qa_data)

    print("Generating MCQ questions...")
    mcq_data = []
    for sample in qa_data:
        mcq_sample = create_mcq_for_sample(sample, grouped_data)
        mcq_data.append(mcq_sample)

    print(f"Generated {len(mcq_data)} MCQ questions")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mcq_data, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser(description="Stage 4b: Generate MCQ Benchmark")
    parser.add_argument("--qa_benchmark_path", type=str,
                        default="./results/synthetic_training_data/qa_synthetic_base.json",
                        help="Path to QA benchmark file from Stage 4a")
    parser.add_argument("--output_path", type=str,
                        default="./results/synthetic_training_data/rme_synthetic_easy_unfiltered.json",
                        help="Path to output file")

    args = parser.parse_args()

    main(
        resolve_path(args.qa_benchmark_path),
        resolve_path(args.output_path)
    )
