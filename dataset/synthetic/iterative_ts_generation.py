#!/usr/bin/env python3
"""
Stage 1: Iterative Time Series Generation With Characteristics

This script is Stage 1 of the synthetic data generation pipeline. It implements
an iterative procedure to generate synthetic time series data using Claude:
1. Load real data samples from qa_benchmark_base_train.json by IDs
2. Extract time series values and "what_happened" characteristics
3. Generate Python code with Claude for synthetic data generation
4. Execute the code and compare with original data
5. Provide feedback to Claude for improvement
6. Repeat for a specified number of iterations
7. Support parallel processing for multiple IDs
"""

import os
import re
import json
import base64
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError
import concurrent.futures
from multiprocessing import cpu_count

# === PATH HANDLING ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


# === CONFIGURATION ===
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_TOKENS = 10240
THINKING_BUDGET = 4096
# ======================


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load the benchmark dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def filter_samples_by_ids(dataset: List[Dict], ids: List[str]) -> List[Dict]:
    """Filter dataset samples by specified IDs."""
    if not ids:
        return dataset
    return [sample for sample in dataset if sample['id'] in ids]


def filter_what_happened_samples(dataset: List[Dict]) -> List[Dict]:
    """Filter dataset to keep only 'what_happened' question types."""
    return [sample for sample in dataset if sample.get('question_type') == 'what_happened']


def should_retry(exc):
    """Exception checker for retries."""
    if isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "ThrottlingException":
        return True
    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)) or "ReadTimeoutError" in str(exc) or "ConnectTimeoutError" in str(exc):
        print(f"Encountered timeout error: {str(exc)}. Retrying...")
        return True
    return False


@retry(
    retry=retry_if_exception(should_retry),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def invoke_claude(client, model_id, messages, system_prompt, enable_thinking=False):
    """Invoke Claude with the specified messages and system prompt."""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0 if enable_thinking else 0.2,
        "system": system_prompt,
        "messages": messages
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return json.loads(resp['body'].read())


def parse_response(resp_body):
    """Parse the response from Claude, extracting both thinking and text content."""
    thought_chunks, text_chunks = [], []
    for chunk in resp_body.get("content", []):
        if chunk.get("type") == "thinking":
            thought_chunks.append(chunk.get("thinking", "").strip())
        elif chunk.get("type") == "text":
            text_chunks.append(chunk.get("text", "").strip())
    thought = "\n".join(thought_chunks)
    response_text = "".join(text_chunks)
    return thought, response_text


def extract_python_code(response_text: str) -> Tuple[str, List[str]]:
    """Extract Python code blocks from Claude's response."""
    code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_pattern, response_text)
    analysis_text = re.sub(code_pattern, "", response_text).strip()
    return analysis_text, code_blocks


def generate_original_image(original_data: np.ndarray, feature_names: List[str],
                            output_dir: str, sample_id: str) -> str:
    """Generate an image showing only the original data."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    sample_dir = os.path.join(output_dir, f"Sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(3):
        ax = axes[i]
        ax.plot(np.arange(original_data.shape[1]), original_data[i], 'b-', linewidth=2)
        ax.set_title(feature_names[i], fontsize=12)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time Step', fontsize=11)
    plt.suptitle(f'Original Time Series (Sample ID: {sample_id})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    img_path = os.path.join(sample_dir, "original.png")
    plt.savefig(img_path)
    plt.close(fig)
    return img_path


def generate_comparison_image(original_data: np.ndarray, synthetic_data: np.ndarray,
                              feature_names: List[str], output_dir: str, sample_id: str,
                              iteration: int) -> str:
    """Generate a comparison image showing original and synthetic data."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey='row', sharex=True)
    sample_dir = os.path.join(output_dir, f"Sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Synthetic Data', fontsize=12, fontweight='bold')
    for i in range(3):
        axes[i, 0].plot(np.arange(original_data.shape[1]), original_data[i], 'b-', linewidth=1.5)
        axes[i, 0].set_ylabel(feature_names[i], fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        synth = synthetic_data[i]
        if len(synth) > original_data.shape[1]:
            synth = synth[:original_data.shape[1]]
        elif len(synth) < original_data.shape[1]:
            synth = np.pad(synth, (0, original_data.shape[1] - len(synth)))
        axes[i, 1].plot(np.arange(len(synth)), synth, 'r-', linewidth=1.5)
        axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 1].set_xlabel('Time Step')
    plt.suptitle(f'Original vs Synthetic (Sample ID: {sample_id}, Iteration {iteration})',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    img_path = os.path.join(sample_dir, f"iteration_{iteration}.png")
    plt.savefig(img_path)
    plt.close(fig)
    return img_path


def execute_function_code(function_code: str, n_samples: int = 100, seed: int = 42):
    """Execute the function code directly and return the generated data."""
    try:
        globals_dict = {
            'np': np,
            'plt': plt,
            '__builtins__': __builtins__
        }
        try:
            import scipy
            import scipy.signal as signal
            import scipy.stats
            globals_dict['scipy'] = scipy
            globals_dict['signal'] = signal
            globals_dict['stats'] = scipy.stats
        except ImportError:
            print("Warning: SciPy not available")
        import_lines = []
        code_lines = []
        for line in function_code.splitlines():
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append(line)
            else:
                code_lines.append(line)
        if import_lines:
            import_code = '\n'.join(import_lines)
            try:
                exec(import_code, globals_dict)
            except Exception as e:
                print(f"Warning: Error executing imports: {e}")
        function_only_code = '\n'.join(code_lines)
        exec(function_only_code, globals_dict)
        function_names = [
            'generate_synthetic_anomaly',
            'generate_synthetic_data',
            'generate_time_series_data',
            'generate_ts_data'
        ]
        for function_name in function_names:
            if function_name in globals_dict:
                result = globals_dict[function_name](n_samples=n_samples, seed=seed)
                return result
        print("Error: No suitable generation function found")
        return None
    except Exception as e:
        print(f"Error executing function: {e}")
        return None


def create_initial_prompt(sample_characteristics: str) -> str:
    """Create the initial prompt for Claude with the sample characteristics."""
    prompt = f"""
# Industrial Time Series Generation

I'm analyzing these time series data that show three metrics from an industrial sensor monitoring system:
1. Acceleration vibration
2. Velocity vibration
3. Temperature

The KEY ANOMALY PATTERN observed in these time series is:
"{sample_characteristics}"

Based on this characteristic and the (normalized) time series visualization shown, please create a generative model that:
1. Reproduces the baseline pattern of the multivariate time series
2. Accurately replicates the specific anomaly pattern described above
3. Maintains the synchronous changes in acceleration and velocity (if any)
4. Properly captures ambient temperature changes over the day (24 time points) (if any)
5. Takes into consideration the consistent decreases in velocity and acceleration when the system stops working, and increases when it starts working again
6. Incorporates multiple layers of deterministic and/or random processes drawn from reasonable distributions to model various patterns (fluctuations, stops/starts, rises/decreases, sporadic spikes, etc.)

The synthetic data should closely match the statistical properties and patterns of the original time series.
Please use only NumPy and basic SciPy functions in your implementation.

Please implement a Python function with this signature:

```python
import numpy as np
import scipy.signal as signal

def generate_synthetic_anomaly(n_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Your implementation here

    # Return all three time series
    return acceleration, velocity, temperature
```

Analyze the image of the original time series carefully and develop a model that generates patterns as close as possible to the real data.
"""
    return prompt


def create_improvement_prompt(sample_characteristics: str, function_code: str,
                              error_message: Optional[str] = None) -> str:
    """Create a prompt for improving the generated code based on the comparison."""
    if error_message:
        prompt = f"""
I tried to execute your time series generation function, but encountered this error:

{error_message}

Please fix the issues and provide an improved version of your function that correctly generates synthetic data.

Remember, the KEY ANOMALY PATTERN we're trying to reproduce is:
"{sample_characteristics}"

Here's your original code:

```python
{function_code}
```

Make sure your revised function:
1. Uses correct import statements
2. Has proper error handling
3. Returns exactly three arrays in this order: (acceleration, velocity, temperature)
4. Maintains the statistical properties of each time series
5. Correctly implements the anomaly pattern described above
"""
    else:
        prompt = f"""
I've compared your generated synthetic data with the original time series in the attached image.

Please carefully examine the visual comparison and refine your model to better match the real data patterns. The KEY ANOMALY PATTERN we need to capture accurately is:
"{sample_characteristics}"

Please focus on improving:
1. The timing, magnitude, and shape of the specific anomaly pattern
2. The synchronous relationship between acceleration and velocity metrics
3. The proper representation of temperature patterns and their relationship to vibration metrics
4. The baseline behavior of the time series (including stops/starts of the system)
5. The statistical properties (variance, range, distribution) of each time series
6. The balance between deterministic patterns and stochastic variations

Here's your current code:

```python
{function_code}
```

Based on the visual comparison, provide an improved version that generates synthetic data more closely matching the patterns, correlations, and anomalies seen in the original data.
"""
    return prompt


def process_sample(args):
    """Process a single sample through the iterative generation process."""
    sample, output_dir, iterations, enable_thinking, region = args
    sample_id = sample['id']
    ts_data = np.array(sample['timeseries'])
    cols = sample['cols']
    characteristic = sample['answer']
    print(f"Processing sample {sample_id}...")
    sample_dir = os.path.join(output_dir, f"Sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)
    client = boto3.client('bedrock-runtime', region_name=region)

    initial_system_prompt = """
You are a time series expert specialized in industrial equipment sensor data analysis and modeling.

Your task is to develop a generative model that can produce synthetic multivariate time series data that matches a specific anomaly pattern described to you.
"""
    improvement_system_prompt = """
You are a time series expert specialized in industrial equipment sensor data analysis and modeling.
Your task is to improve your previously generated synthetic data model based on the comparison with the original data.
Make sure all imports are correctly specified at the top of your code.
Wrap all code in ```python code blocks.
"""

    current_code = None
    error_message = None
    sample_start_time = time.time()
    img_path = None

    for iter_num in range(1, iterations + 1):
        print(f"  Iteration {iter_num}/{iterations}")
        if iter_num == 1:
            prompt = create_initial_prompt(characteristic)
            system_prompt = initial_system_prompt
            img_path = generate_original_image(ts_data, cols, output_dir, sample_id)
            with open(img_path, "rb") as img_f:
                img_b64 = base64.b64encode(img_f.read()).decode("utf8")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                ]
            }]
        else:
            prompt = create_improvement_prompt(characteristic, current_code, error_message)
            if error_message:
                print(f"\n=== ERROR MESSAGE SENT TO CLAUDE IN ITERATION {iter_num} ===")
                print(f"{error_message}")
                print("================================================\n")
            system_prompt = improvement_system_prompt
            error_message = None
            with open(img_path, "rb") as img_f:
                img_b64 = base64.b64encode(img_f.read()).decode("utf8")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                ]
            }]

        resp = invoke_claude(client, MODEL_ID, messages, system_prompt, enable_thinking)
        if enable_thinking:
            thought, response_text = parse_response(resp)
            if thought:
                with open(os.path.join(sample_dir, f"thinking_{iter_num}.txt"), 'w') as f:
                    f.write(thought)
        else:
            response_text = resp['content'][0]['text']

        _, code_blocks = extract_python_code(response_text)
        with open(os.path.join(sample_dir, f"response_{iter_num}.txt"), 'w') as f:
            f.write(response_text)

        function_code = None
        if code_blocks:
            for block in code_blocks:
                if "def generate_synthetic" in block:
                    function_code = block
                    break
            if not function_code:
                function_code = code_blocks[0]

        if not function_code:
            print(f"  Warning: No function code found in iteration {iter_num}")
            continue

        with open(os.path.join(sample_dir, f"function_{iter_num}.py"), 'w') as f:
            f.write(function_code)

        error_message = None
        try:
            synthetic_data = execute_function_code(function_code, n_samples=ts_data.shape[1], seed=42)
            if synthetic_data is None or len(synthetic_data) != 3:
                error_message = f"Invalid output: expected tuple of 3 arrays, got {type(synthetic_data)}"
                print(f"  Error: {error_message}")
            else:
                img_path = generate_comparison_image(
                    ts_data, synthetic_data, cols, output_dir, sample_id, iter_num
                )
                print(f"  Saved comparison image to {img_path}")
                with open(os.path.join(sample_dir, f"synthetic_data_{iter_num}.json"), 'w') as f:
                    json.dump({
                        "synthetic_data": [arr.tolist() for arr in synthetic_data],
                        "cols": cols
                    }, f, indent=2)
        except Exception as e:
            error_message = str(e)
            print(f"  Error executing function: {error_message}")

        current_code = function_code

    sample_end_time = time.time()
    sample_execution_time = sample_end_time - sample_start_time
    print(f"Completed sample {sample_id} in {sample_execution_time:.2f}s ({sample_execution_time/60:.2f}min)")
    return {
        'id': sample_id,
        'characteristic': characteristic,
        'execution_time': sample_execution_time,
        'iterations': iterations
    }


def run_iterative_generation(dataset_path: str, output_dir: str, iterations: int = 3,
                             sample_ids: Optional[List[str]] = None, enable_thinking: bool = False,
                             max_workers: Optional[int] = None, region: str = 'us-west-2'):
    """Run the iterative generation process for the specified samples."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    dataset = filter_what_happened_samples(dataset)
    print(f"Filtered to {len(dataset)} 'what_happened' samples")
    if sample_ids:
        dataset = filter_samples_by_ids(dataset, sample_ids)
        print(f"Further filtered to {len(dataset)} samples based on provided IDs")
    if not dataset:
        print("Error: No samples to process after filtering.")
        return
    if max_workers is None:
        max_workers = max(1, cpu_count() // 2)
    print(f"Processing {len(dataset)} samples with up to {max_workers} workers...")
    results = []
    args_list = [(sample, output_dir, iterations, enable_thinking, region) for sample in dataset]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_sample, args_list):
            results.append(result)
    summary = {
        'total_samples': len(dataset),
        'iterations_per_sample': iterations,
        'samples': results
    }
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"All samples processed in {execution_time:.2f}s ({execution_time/60:.2f}min)")


if __name__ == "__main__":
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Stage 1: Iterative Time Series Generation")
    parser.add_argument("--dataset_path", type=str,
                        default="./sample_data/qa_benchmark_base_train.json",
                        help="Path to the benchmark dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./results/iterative_results",
                        help="Directory to save results")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of improvement iterations")
    parser.add_argument("--sample_ids", type=str, nargs="*",
                        help="Specific sample IDs to process")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for Claude")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker processes (default: half of CPU cores)")
    parser.add_argument("--region", type=str, default="us-west-2",
                        help="AWS region for Bedrock (default: us-west-2)")
    args = parser.parse_args()

    run_iterative_generation(
        resolve_path(args.dataset_path),
        resolve_path(args.output_dir),
        args.iterations,
        args.sample_ids,
        args.thinking,
        args.max_workers,
        args.region
    )

    script_end_time = time.time()
    total_execution_time = script_end_time - script_start_time
    print(f"\nTotal execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f}min)")
