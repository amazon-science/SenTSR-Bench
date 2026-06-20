#!/usr/bin/env python3
"""
Stage 2: Stochastic Time Series Generation (Sampling-Based Refinement)

This script is Stage 2 of the synthetic data generation pipeline. It builds on
the iterative generation results (Stage 1) by:
1. Loading the last iteration function code from Stage 1
2. Asking Claude to simplify into sampling-based generators
3. Replacing hardcoded parameters with probabilistic distributions
4. Generating multiple samples from each model to validate diversity
5. Visualizing the diversity of generated samples
"""

import os
import re
import json
import base64
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError
import glob
from matplotlib.gridspec import GridSpec
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
MAX_TOKENS = 4096
# ======================


def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, 'r') as f:
        return json.load(f)


def find_last_iteration_artifacts(sample_id: str, results_dir: str):
    """Find the last iteration artifacts for a given sample ID from Stage 1."""
    sample_dir = os.path.join(results_dir, f"Sample_{sample_id}")
    if not os.path.exists(sample_dir):
        print(f"Warning: No previous results found for sample {sample_id}")
        return None, None, None
    function_files = sorted(glob.glob(os.path.join(sample_dir, "function_*.py")))
    if not function_files:
        print(f"Warning: No function code found for sample {sample_id}")
        return None, None, None
    last_function = function_files[-1]
    image_files = sorted(glob.glob(os.path.join(sample_dir, "iteration_*.png")))
    last_image = image_files[-1] if image_files else None
    data_files = sorted(glob.glob(os.path.join(sample_dir, "synthetic_data_*.json")))
    sample_data = None
    if data_files:
        with open(data_files[-1], 'r') as f:
            sample_data = json.load(f)
    return last_function, last_image, sample_data


def load_function_code(function_path: str) -> str:
    with open(function_path, 'r') as f:
        return f.read()


def load_sample_info(dataset: List[Dict], sample_id: str) -> Optional[Dict]:
    for sample in dataset:
        if sample['id'] == sample_id:
            return sample
    return None


def should_retry(exc):
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
def invoke_claude(client, model_id, messages, system_prompt):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": messages
    }
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return json.loads(resp['body'].read())


def extract_python_code(response_text: str) -> Tuple[str, List[str]]:
    code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_pattern, response_text)
    analysis_text = re.sub(code_pattern, "", response_text).strip()
    return analysis_text, code_blocks


def create_simplified_prompt(function_code: str, sample_characteristics: str) -> str:
    """Create a prompt to encourage simplified, sampling-based generation."""
    prompt = f"""
# Simplified Industrial Time Series Generation

I have a synthetic data generation function that currently produces time series data with the following key anomaly pattern:
"{sample_characteristics}"

However, the current implementation is too complex and relies on hardcoded parameters. I need a simplified version that:

1. Captures only the essential components of the pattern
2. Uses sampling instead of hardcoded values
3. Models machine operation periods as a hidden state from sampling rather than fixed time points
4. Is simpler and more generalizable than the current implementation for a much more diverse time series generation

Here is the current implementation:

```python
{function_code}
```

Please revise this code to create a more simplified, sampling-based generator that:
- Uses the EXACT function name 'generate_synthetic_anomaly' (this is required for compatibility with our system)
- Focuses on simplicity - the current implementation is overfitting to the original data
- Uses proper sampling for event (spike, rise, etc.) timing and magnitude rather than hardcoding specific time points
- Treats machine operations (on/off patterns) as hidden states that can be sampled
- Is concise, interpretable, and well-commented
- Preserves only the essential characteristics that fully characterize the anomaly pattern.

NOTE:
1. For cases in `"both vibration and temperature rise sharply","a sudden parallel jump is observed in vibration and temperature","vibration and temperature increase abruptly at the same time"`,
Please properly model the jump sharply (almost vertically instead of in a sharp slope) and guarantee it is at the very end of the time series (155+).

The function signature must be:
```python
def generate_synthetic_anomaly(n_samples=100, seed=None):
    # Your code here
    return acceleration, velocity, temperature
```

The goal is to create a simplified generator that captures the core pattern while enabling generation of diverse examples through sampling.
"""
    return prompt


def execute_function_code(function_code: str, n_samples: int = 100, seed: Optional[int] = None):
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
                kwargs = {'n_samples': n_samples}
                if seed is not None:
                    kwargs['seed'] = seed
                result = globals_dict[function_name](**kwargs)
                return result
        print("Error: No suitable generation function found")
        return None
    except Exception as e:
        print(f"Error executing function: {e}")
        return None


def generate_stochastic_model(sample_id: str, dataset_path: str, results_dir: str,
                              output_dir: str, num_claude_calls: int = 3,
                              region: str = 'us-west-2') -> Optional[Dict]:
    """Generate a stochastic model for a given sample ID."""
    print(f"Processing sample {sample_id}...")
    dataset = load_dataset(dataset_path)
    sample_info = load_sample_info(dataset, sample_id)
    if sample_info is None:
        print(f"Error: Sample {sample_id} not found in dataset")
        return None
    sample_characteristic = sample_info['answer']
    function_path, image_path, _ = find_last_iteration_artifacts(sample_id, results_dir)
    if function_path is None:
        print(f"Error: No function code found for sample {sample_id}")
        return None
    function_code = load_function_code(function_path)
    client = boto3.client('bedrock-runtime', region_name=region)

    system_prompt = """
You are a time series expert specializing in industrial sensor data analysis and modeling.

Your task is to improve the synthetic data generation code provided to you.

Make sure all imports are correctly specified at the top of your code.
Wrap all code in ```python code blocks.
"""

    prompt = create_simplified_prompt(function_code, sample_characteristic)
    img_b64 = None
    if image_path:
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode("utf8")
    sample_dir = os.path.join(output_dir, f"Sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    stochastic_functions = []
    for call_index in range(num_claude_calls):
        print(f"Invoking Claude (call {call_index+1}/{num_claude_calls})...")
        if img_b64:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
            ]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        response = invoke_claude(client, MODEL_ID, messages, system_prompt)
        response_text = response['content'][0]['text']
        with open(os.path.join(sample_dir, f"stochastic_response_{call_index+1}.txt"), 'w') as f:
            f.write(response_text)
        _, code_blocks = extract_python_code(response_text)
        if not code_blocks:
            print(f"Error: No code blocks found for call {call_index+1}")
            continue
        stochastic_function = None
        for block in code_blocks:
            if "def generate_synthetic_anomaly" in block:
                stochastic_function = block
                break
        if not stochastic_function:
            stochastic_function = code_blocks[0]
        stochastic_function_path = os.path.join(sample_dir, f"stochastic_function{call_index+1}.py")
        with open(stochastic_function_path, 'w') as f:
            f.write(stochastic_function)
        print(f"Saved stochastic function {call_index+1}")
        stochastic_functions.append((stochastic_function, stochastic_function_path))

    # Generate test samples and visualizations
    ts_data = np.array(sample_info['timeseries'])
    n_samples = ts_data.shape[1]
    cols = sample_info['cols']
    samples_per_function = 3
    all_samples = []

    for func_index, (stochastic_function, _) in enumerate(stochastic_functions):
        print(f"Generating test samples for function {func_index+1}/{len(stochastic_functions)}...")
        samples = []
        for i in range(samples_per_function):
            result = execute_function_code(stochastic_function, n_samples, seed=42 + i)
            if result is not None:
                samples.append(result)
        if samples:
            n_rows = 3
            n_cols = len(samples) + 1
            fig = plt.figure(figsize=(4 * n_cols, 10), constrained_layout=True)
            gs = GridSpec(n_rows, n_cols, figure=fig)
            for i in range(3):
                ax = fig.add_subplot(gs[i, 0])
                ax.plot(np.arange(ts_data.shape[1]), ts_data[i], 'b-', linewidth=1.5)
                ax.set_ylabel(cols[i], fontsize=12)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title('Original Data', fontsize=14, fontweight='bold')
            for j, sample in enumerate(samples):
                for i in range(3):
                    ax = fig.add_subplot(gs[i, j + 1])
                    synth = sample[i]
                    if len(synth) > ts_data.shape[1]:
                        synth = synth[:ts_data.shape[1]]
                    elif len(synth) < ts_data.shape[1]:
                        synth = np.pad(synth, (0, ts_data.shape[1] - len(synth)))
                    ax.plot(np.arange(len(synth)), synth, 'r-', linewidth=1.5)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_title(f'Sample {j+1}', fontsize=14, fontweight='bold')
            img_path = os.path.join(sample_dir, f"multiple_samples_func{func_index+1}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            for i, sample in enumerate(samples):
                with open(os.path.join(sample_dir, f"stochastic_sample_func{func_index+1}_sample{i+1}.json"), 'w') as f:
                    json.dump({"synthetic_data": [arr.tolist() for arr in sample], "cols": cols}, f, indent=2)
            all_samples.extend(samples)

    return {
        'id': sample_id,
        'characteristic': sample_characteristic,
        'num_functions_generated': len(stochastic_functions),
        'num_samples_generated': len(all_samples)
    }


def process_sample_wrapper(args):
    sample_id, dataset_path, results_dir, output_dir, num_claude_calls, region = args
    return generate_stochastic_model(sample_id, dataset_path, results_dir, output_dir, num_claude_calls, region)


def run_stochastic_generation(dataset_path: str, results_dir: str, output_dir: str,
                              sample_ids: Optional[List[str]] = None,
                              max_workers: Optional[int] = None,
                              num_claude_calls: int = 3, region: str = 'us-west-2'):
    """Run stochastic generation for multiple samples in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    if not sample_ids:
        sample_dirs = [os.path.basename(d) for d in glob.glob(os.path.join(results_dir, "Sample_*"))]
        sample_ids = [d.replace("Sample_", "") for d in sample_dirs]
    print(f"Processing {len(sample_ids)} samples")
    if max_workers is None:
        max_workers = max(1, cpu_count() // 2)
    max_workers = min(len(sample_ids), max_workers) if sample_ids else 1
    print(f"Using {max_workers} parallel workers")
    args_list = [(sid, dataset_path, results_dir, output_dir, num_claude_calls, region) for sid in sample_ids]
    start_time = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_sample_wrapper, args_list):
            if result:
                results.append(result)
    end_time = time.time()
    print(f"All samples processed in {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Stochastic Time Series Generation")
    parser.add_argument("--dataset_path", type=str,
                        default="./sample_data/qa_benchmark_base_train.json")
    parser.add_argument("--results_dir", type=str,
                        default="./results/iterative_results")
    parser.add_argument("--output_dir", type=str,
                        default="./results/stochastic_results")
    parser.add_argument("--sample_ids", type=str, nargs="*")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_claude_calls", type=int, default=3)
    parser.add_argument("--region", type=str, default="us-west-2")
    args = parser.parse_args()

    start_time = time.time()
    run_stochastic_generation(
        resolve_path(args.dataset_path),
        resolve_path(args.results_dir),
        resolve_path(args.output_dir),
        args.sample_ids,
        args.max_workers,
        args.num_claude_calls,
        args.region
    )
    print(f"Total execution time: {time.time() - start_time:.2f}s")
