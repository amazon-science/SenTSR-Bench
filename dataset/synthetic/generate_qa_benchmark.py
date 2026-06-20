#!/usr/bin/env python3
"""
Stage 4a: Generate QA Benchmark from Synthetic Time Series Data

This script:
1. Loads synthetic time series data from data_ts.json (Stage 3 output)
2. Loads the original training data to retrieve anomaly descriptions
3. Calls Claude (via AWS Bedrock) to diversify each anomaly description into
   multiple observation, root cause, and corrective action variations
4. Generates QA benchmark entries by sampling from the diversified answers
5. No external metadata files are required beyond the original training data
"""

import os
import re
import json
import random
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

# === PATH HANDLING ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === CONFIGURATION ===
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_TOKENS = 2048
# ======================

TS_COLUMNS = ["Acceleration", "Velocity", "Temperature"]


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def load_json_data(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def should_retry(exc):
    if isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "ThrottlingException":
        return True
    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)):
        return True
    return False


@retry(
    retry=retry_if_exception(should_retry),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def invoke_claude(client, model_id, messages, system_prompt):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": messages
    }
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return json.loads(resp['body'].read())


def diversify_answers(
    client, model_id: str, original_answer: str, num_variations: int = 5
) -> Dict[str, List[str]]:
    """Use Claude to generate diverse answer variations for all 3 question types."""
    system_prompt = (
        "You are an industrial machinery expert specializing in vibration analysis "
        "and predictive maintenance. Respond ONLY with valid JSON, no markdown."
    )
    prompt = f"""Given this anomaly observation from industrial vibration and temperature sensors:
"{original_answer}"

Generate {num_variations} diverse textual variations for each of the following categories.

1. OBSERVATIONS: Rephrase the anomaly pattern description in {num_variations} different ways.
   Each should describe the same core pattern but with different wording and emphasis.
   Keep descriptions concise (1-2 sentences).

2. ROOT CAUSES: Provide {num_variations} plausible root causes that could lead to this
   anomaly pattern in industrial rotating machinery with vibration and temperature sensors.
   Each should be a concise statement (1-2 sentences).

3. CORRECTIVE ACTIONS: Provide {num_variations} appropriate corrective actions to address
   the anomaly. Each should be a concise recommendation (1-2 sentences).

Return ONLY valid JSON with this exact structure:
{{
    "observations": ["...", "...", ...],
    "root_causes": ["...", "...", ...],
    "corrective_actions": ["...", "...", ...]
}}"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = invoke_claude(client, model_id, messages, system_prompt)
    response_text = response['content'][0]['text']

    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            # Validate structure
            if all(k in result for k in ("observations", "root_causes", "corrective_actions")):
                return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback if parsing fails
    print(f"Warning: Failed to parse LLM response, using fallback for: {original_answer[:60]}...")
    return {
        "observations": [original_answer],
        "root_causes": [f"Degraded component condition leading to: {original_answer}"],
        "corrective_actions": [f"Inspect and service the affected component"]
    }


def median_absolute_deviation(data):
    """Calculate the Median Absolute Deviation (MAD) of a dataset."""
    med = np.median(data)
    return np.median(np.abs(np.array(data) - med))


def standardize_timeseries(timeseries: List[List[float]]) -> List[List[float]]:
    """
    Standardize time series data using median and MAD.
    Formula: (x - median) / (1.4826 * MAD)
    """
    standardized_ts = []
    for ts in timeseries:
        ts_array = np.array(ts)
        med = np.median(ts_array)
        mad = median_absolute_deviation(ts_array)
        if mad == 0:
            mad = 1.0
        std_ts = (ts_array - med) / (1.4826 * mad)
        standardized_ts.append(std_ts.tolist())
    return standardized_ts


def get_question_prompt(question_type: str) -> str:
    """Get the appropriate prompt for the question type."""
    prompts = {
        "what_happened": "What is the key anomalous pattern observed in these time series?",
        "how_happened": "What is the most likely cause of the anomalous pattern in these time series?",
        "suggested_fix": "What is the best corrective action for the event implied by the anomalous pattern in these time series?"
    }
    return prompts.get(question_type, "")


def generate_question_template(timeseries: List[List[float]], question_type: str) -> str:
    """Generate the complete question template with dynamic length information."""
    lengths = [len(ts) for ts in timeseries]
    template = (
        "You are a time series analysis expert. In a sensor monitoring system, the vibration "
        "(measured in velocity and acceleration) and temperature of machines are collected for monitoring. "
        "The time series data has been standardized using median and MAD (Median Absolute Deviation).\n"
    )
    for col, length in zip(TS_COLUMNS, lengths):
        template += f"\"{col}\" is a standardized time series with length of {length}: <ts><ts/>\n"
    template += (
        "Please analyze the time series features and answer the following question: "
        f"{get_question_prompt(question_type)}"
    )
    return template


# Mapping from question type to diversified answer key
QUESTION_TYPE_TO_ANSWER_KEY = {
    "what_happened": "observations",
    "how_happened": "root_causes",
    "suggested_fix": "corrective_actions"
}

QUESTION_TYPE_TO_ABILITY = {
    "what_happened": "MCQ_obs",
    "how_happened": "MCQ_cause",
    "suggested_fix": "MCQ_fix"
}


def main(data_ts_path: str, dataset_path: str, output_path: str,
         region: str, num_variations: int):
    """Generate QA benchmark using LLM-diversified answers."""
    print(f"Loading synthetic data from {data_ts_path}...")
    data_ts = load_json_data(data_ts_path)

    print(f"Loading original training data from {dataset_path}...")
    training_data = load_json_data(dataset_path)

    # Build lookup: original sample ID -> original answer (anomaly description)
    original_answers = {}
    for sample in training_data:
        if sample.get('question_type') == 'what_happened':
            original_answers[sample['id']] = sample['answer']

    # Find unique original IDs in synthetic data
    unique_originals = sorted(set(
        entry['original_id'] for entry in data_ts if entry.get('original_id')
    ))
    print(f"Found {len(unique_originals)} unique source samples to diversify")

    # Call Claude to diversify answers for each unique source
    client = boto3.client('bedrock-runtime', region_name=region)
    diversified = {}

    for i, orig_id in enumerate(unique_originals):
        original_answer = original_answers.get(orig_id)
        if not original_answer:
            print(f"Warning: No original answer found for source {orig_id}, skipping")
            continue

        print(f"Diversifying answers for source {orig_id} ({i+1}/{len(unique_originals)})...")
        result = diversify_answers(client, MODEL_ID, original_answer, num_variations)
        diversified[orig_id] = result

    # Generate QA benchmark entries
    benchmark_data = []
    for entry in data_ts:
        ts_id = entry['id']
        raw_ts = entry.get('timeseries', [])
        orig_id = entry.get('original_id')

        if not raw_ts or not orig_id or orig_id not in diversified:
            continue

        std_ts = standardize_timeseries(raw_ts)
        answers = diversified[orig_id]

        for q_type in ["what_happened", "how_happened", "suggested_fix"]:
            answer_key = QUESTION_TYPE_TO_ANSWER_KEY[q_type]
            answer_pool = answers.get(answer_key, [])
            answer = random.choice(answer_pool) if answer_pool else ""

            sample = {
                "id": ts_id,
                "timeseries": std_ts,
                "cols": TS_COLUMNS,
                "question": generate_question_template(std_ts, q_type),
                "question_type": q_type,
                "answer": answer,
                "attributes": [answer] if answer else [],
                "ability_types": [QUESTION_TYPE_TO_ABILITY[q_type]],
                "original_id": orig_id
            }
            benchmark_data.append(sample)

    print(f"Generated {len(benchmark_data)} benchmark samples")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved to {output_path}")

    # Save diversified answers cache for reproducibility
    cache_path = os.path.join(os.path.dirname(output_path), "diversified_answers.json")
    with open(cache_path, 'w') as f:
        json.dump(diversified, f, indent=2)
    print(f"Saved diversified answers cache to {cache_path}")


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser(description="Stage 4a: Generate QA Benchmark")
    parser.add_argument("--data_ts_path", type=str,
                        default="./results/synthetic_training_data/data_ts.json",
                        help="Path to data_ts.json file from Stage 3")
    parser.add_argument("--dataset_path", type=str,
                        default="./sample_data/qa_benchmark_base_train.json",
                        help="Path to original training dataset")
    parser.add_argument("--output_path", type=str,
                        default="./results/synthetic_training_data/qa_synthetic_base.json",
                        help="Path to output file")
    parser.add_argument("--region", type=str, default="us-west-2",
                        help="AWS region for Bedrock")
    parser.add_argument("--num_variations", type=int, default=10,
                        help="Number of answer variations to generate per source")

    args = parser.parse_args()

    main(
        resolve_path(args.data_ts_path),
        resolve_path(args.dataset_path),
        resolve_path(args.output_path),
        args.region,
        args.num_variations
    )
