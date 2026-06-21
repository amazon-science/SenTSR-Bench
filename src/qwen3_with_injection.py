#!/usr/bin/env python3
"""
Text-Only GRLM with Qwen-VL Injection Script

This script injects Qwen-VL observations (thoughts and answers) into a text-only
GRLM's (Qwen3-32B or DeepSeek-R1-Distill-Qwen-32B) thinking process for enhanced
time series reasoning.

For each example it will:
  1. Load the original timeseries + question from the dataset.
  2. Load the initial thoughts and answers from Qwen-VL output JSON.
  3. Format the timeseries data as JSON text (Qwen3 is text-only).
  4. Build a prompt that:
     - User: contains the question and time series data
     - Assistant: begins with the Qwen-VL thoughts AND answer as part of the thinking
  5. Call Qwen3 with continue_final_message=true
  6. Parse out the complete answer.
  7. Save idx, question, initial thoughts, full thought, answer, success flag.

Usage:
    python qwen3_with_injection.py \\
        --dataset_path /path/to/dataset.json \\
        --injection_path /path/to/qwen_vl_output.json \\
        --output_path /path/to/output.json
"""

import os
import sys
import json
import time
import re
import argparse
import logging
import threading
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI Python client not installed. Please install with 'pip install openai'")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 with Qwen-VL injection for time series analysis")
    p.add_argument("--server_url", default="http://localhost:5001",
                   help="URL of the GRLM server (default: Qwen3 on 5001, use 5002 for R1)")
    p.add_argument("--model_name", default="qwen3",
                   help="Model name for the GRLM server (qwen3 or r1)")
    p.add_argument("--dataset_path", "-d", required=True,
                   help="Path to the JSON evaluation set")
    p.add_argument("--injection_path", "-p", required=True,
                   help="Path to the Qwen-VL output JSON (from qwen_inference.py)")
    p.add_argument("--output_path", "-o", required=True,
                   help="Where to write generated answers JSON")
    p.add_argument("--max_tokens", type=int, default=6144,
                   help="Maximum tokens to generate")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers for processing samples")
    p.add_argument("--checkpoint_interval", type=int, default=10,
                   help="Interval for saving checkpoints")
    p.add_argument("--timeout", type=int, default=120,
                   help="Timeout in seconds for API calls")
    p.add_argument("--retry_delay", type=int, default=5,
                   help="Delay between retries in seconds")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Maximum number of retry attempts")
    return p.parse_args()


class GRLMClient:
    """Client for communicating with a text-only GRLM server (Qwen3 or DeepSeek-R1) using OpenAI API."""

    def __init__(self, server_url="http://localhost:5001", model_name="qwen3", debug_mode=False):
        """Initialize the GRLM client."""
        self.server_url = server_url
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy-key")

        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.info(f"GRLMClient initialized in DEBUG mode with server URL: {server_url}")

    def check_server_health(self):
        """Check if the server is healthy."""
        import requests
        try:
            logger.info(f"Checking health of {self.model_name} server at {self.server_url}...")
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if response.status_code == 200:
                logger.info(f"{self.model_name} server is healthy")
                return True
            else:
                logger.warning(f"{self.model_name} server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking server health: {type(e).__name__}: {e}")
            return False

    def query_with_injection(
        self,
        user_prompt,
        assistant_start,
        system_message=None,
        max_tokens=6144,
        temperature=0.6,
        timeout=120,
        retry_delay=5,
        max_retries=3,
    ):
        """
        Query Qwen3 with an injected assistant start using continue_final_message.

        The assistant_start contains Qwen-VL's thoughts and answer, which Qwen3
        continues from. This requires a custom chat template that passes through
        the assistant message without modifying <think> tags.

        Args:
            user_prompt: Text prompt for the user message
            assistant_start: Initial text for the assistant response (Qwen-VL injection)
            system_message: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Model response continuation as string
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_start})

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending query to Qwen3 server (attempt {attempt+1}/{max_retries})")
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    extra_body={
                        "add_generation_prompt": False,
                        "continue_final_message": True,
                        "top_k": 20
                    }
                )

                end_time = time.time()
                elapsed = end_time - start_time

                usage = getattr(response, 'usage', None)
                if usage:
                    logger.info(
                        f"Query successful in {elapsed:.2f}s: "
                        f"prompt_tokens={usage.prompt_tokens}, "
                        f"completion_tokens={usage.completion_tokens}, "
                        f"total_tokens={usage.total_tokens}"
                    )
                else:
                    logger.info(f"Query successful, inference time: {elapsed:.2f}s")

                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Query failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                time.sleep(retry_delay)

            retry_delay = min(retry_delay * 2, 60)

        logger.critical(f"Failed to get response from Qwen3 server after {max_retries} attempts")
        raise RuntimeError("Failed to get response from Qwen3 server after multiple attempts")


def prepare_timeseries_data(ts):
    """
    Prepare timeseries data for text formatting.

    Args:
        ts: Raw timeseries data

    Returns:
        Processed timeseries data suitable for formatting
    """
    if not isinstance(ts, list):
        ts = [ts]
    elif len(ts) > 0 and not isinstance(ts[0], list):
        ts = [ts]

    if not ts or len(ts) == 0:
        ts = [[0, 1, 2, 3, 4]]
        logger.warning("Entry has empty time series data, using dummy data")

    return ts


def format_timeseries_as_json(timeseries, cols=None):
    """
    Format timeseries data as a JSON string for inclusion in prompts.

    Args:
        timeseries: The time series data as a list of lists
        cols: Optional column names for the time series

    Returns:
        Formatted JSON string representation of the time series
    """
    ts_data = prepare_timeseries_data(timeseries)

    if not cols or len(cols) != len(ts_data):
        if cols and len(cols) != len(ts_data):
            logger.error(f"Column count ({len(cols)}) doesn't match time series count ({len(ts_data)}). Using default column names.")
        cols = [f"Series {i+1}" for i in range(len(ts_data))]

    max_length = max(len(series) for series in ts_data)
    timestamps = list(range(1, max_length + 1))

    ts_json = "{\n"
    ts_json += f'  "timestamps": {timestamps},\n'

    for i, (col, series) in enumerate(zip(cols, ts_data)):
        series_values = [round(float(v), 2) for v in series]
        ts_json += f'  "{col}": {series_values}'
        if i < len(ts_data) - 1:
            ts_json += ",\n"
        else:
            ts_json += "\n"
    ts_json += "}"

    return ts_json


def build_prompt_with_injection(question, initial_thought, qwen_answer, timeseries=None, cols=None):
    """
    Build prompts that include the question as user prompt and both Qwen-VL
    thoughts and answer as the start of the assistant's response.

    Args:
        question: The original question text
        initial_thought: Initial thoughts from Qwen-VL
        qwen_answer: Answer from Qwen-VL
        timeseries: Time series data (optional)
        cols: Column names for time series (optional)

    Returns:
        Tuple of (user_prompt, assistant_start_content)
    """
    # Split off the "Now, based on ..." part if it exists
    if "Now," in question:
        q_part, rest = question.split("Now,", 1)
        question_part = q_part.strip()
        answer_format = "Now," + rest.strip()
    else:
        question_part = question.strip()
        answer_format = ""

    # Format time series data if provided
    ts_text = ""
    if timeseries is not None and cols is not None:
        ts_json = format_timeseries_as_json(timeseries, cols)
        ts_text = f"Here is the time series data in JSON format:\n{ts_json}\n\n"

    # Build the user prompt - question and time series data
    user_prompt = f"{ts_text}{question_part}"

    if answer_format:
        user_prompt += f"\n\n{answer_format}"

    # Build the assistant's starting content with TSLM thinking trace only
    # (assistant prefill for early injection)
    assistant_start = (
        f"<think>\n{initial_thought.strip()}\n\n"
        f"Wait, let me summarize and reflect on the previous observations from the time series, "
        f"and then continue my reasoning process to derive the final answer."
    )

    return user_prompt, assistant_start


def parse_response(response_content, assistant_start):
    """
    Parse Qwen3 API response continuation.

    The response_content is the continuation after the assistant_start.
    We look for </think> to separate additional thinking from the final answer.

    Args:
        response_content: The continuation text from Qwen3
        assistant_start: The initial assistant response text (for full thought reconstruction)

    Returns:
        Tuple of (full_thought, answer, success)
    """
    try:
        parts = response_content.split('</think>')

        if len(parts) > 1:
            # Found </think> - everything after is the answer
            answer_part = parts[-1].strip()

            # Construct full thought: initial injection + model's continued thinking
            clean_assistant_start = assistant_start.replace('<think>', '', 1).strip()
            full_thought = clean_assistant_start + "\n\n" + "\n\n".join(parts[:-1])
        else:
            # No </think> found - consider everything as thinking
            full_thought = assistant_start.replace('<think>', '', 1).strip() + "\n\n" + response_content
            answer_part = ""

        clean_content = answer_part

        # Try to parse as JSON with "Final Answer" key
        try:
            raw_obj = json.loads(clean_content)
            if isinstance(raw_obj, dict) and "Final Answer" in raw_obj:
                return full_thought, raw_obj["Final Answer"], True
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from code blocks
        json_pattern = re.compile(r'```(?:json)?\s*({.*?})\s*```', re.DOTALL)
        json_matches = json_pattern.findall(clean_content)

        if json_matches:
            try:
                obj = json.loads(json_matches[0])
                answer = obj.get("Final Answer", "")
                success = "Final Answer" in obj
            except Exception:
                answer = clean_content
                success = False
        else:
            # Check for inline JSON object
            json_pattern_no_blocks = re.compile(r'{\s*"Final Answer"\s*:\s*".*?"}', re.DOTALL)
            matches = json_pattern_no_blocks.findall(clean_content)

            if matches:
                try:
                    obj = json.loads(matches[0])
                    answer = obj.get("Final Answer", "")
                    success = "Final Answer" in obj
                except Exception:
                    answer = clean_content
                    success = False
            else:
                answer = clean_content
                success = True if clean_content.strip() else False

        return full_thought, answer, success

    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return "", str(e), False


def process_sample(args, client, sample, idx, injection_entry):
    """
    Process a single sample with the Qwen3 client using Qwen-VL injection.

    Args:
        args: Command line arguments
        client: GRLMClient instance
        sample: Data sample with timeseries and columns
        idx: Sample index
        injection_entry: Qwen-VL output entry with thought and response

    Returns:
        Result dictionary
    """
    try:
        ts = sample["timeseries"]
        cols = sample.get("cols", [])

        # Prepare timeseries data
        ts = prepare_timeseries_data(ts)

        # Ensure cols list is properly sized
        if not cols or len(cols) != len(ts):
            cols = [f"Series {i+1}" for i in range(len(ts))]

        # Build prompts with Qwen-VL injection
        user_prompt, assistant_start = build_prompt_with_injection(
            injection_entry["question"],
            injection_entry["thought"],
            injection_entry["response"],
            ts,
            cols
        )

        # System message
        system_message = (
            "You are a time-series expert. \n"
            "Answer **only** with a JSON object that has exactly one key, \"Final Answer\",\n"
            "whose value is the answer string.  \n"
        )

        # Query Qwen3 with injection
        response_content = client.query_with_injection(
            user_prompt,
            assistant_start,
            system_message,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries
        )

        # Parse the response
        full_thought, answer, ok = parse_response(response_content, assistant_start)

        return {
            "idx": idx,
            "question": injection_entry["question"],
            "initial_thought": injection_entry["thought"],
            "qwen_vl_answer": injection_entry["response"],
            "thought": full_thought,
            "response": answer,
            "success": ok,
            "ability_types": sample.get("ability_types", []),
            "attributes": sample.get("attributes", {})
        }

    except Exception as e:
        logger.error(f"Error processing sample {idx}: {str(e)}")
        return {
            "idx": idx,
            "question": injection_entry.get("question", ""),
            "initial_thought": injection_entry.get("thought", ""),
            "qwen_vl_answer": injection_entry.get("response", ""),
            "thought": "",
            "response": f"ERROR: {str(e)}",
            "success": False,
            "ability_types": sample.get("ability_types", []),
            "attributes": sample.get("attributes", {})
        }


def main():
    args = parse_args()

    # Initialize lock for thread-safe operations
    results_lock = threading.Lock()

    # Initialize GRLM client pool (works for both Qwen3 and DeepSeek-R1)
    clients = [GRLMClient(server_url=args.server_url, model_name=args.model_name) for _ in range(args.workers)]

    # Check server health with first client
    if not clients[0].check_server_health():
        logger.error(f"{args.model_name} server is not healthy. Please make sure it is running.")
        if args.model_name == "r1":
            logger.error("Run: src/r1_utils/start_r1_server.sh to start the server.")
        else:
            logger.error("Run: src/qwen3_utils/start_qwen3_server.sh to start the server.")
        sys.exit(1)
    else:
        logger.info(f"Using {args.workers} workers for parallel processing with {args.model_name}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    # Load Qwen-VL injection outputs
    logger.info(f"Loading Qwen-VL injection outputs from {args.injection_path}")
    with open(args.injection_path, "r") as f:
        injection_outputs = json.load(f)
    injection_map = {h["idx"]: h for h in injection_outputs}

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load existing results if any (resume support)
    results = []
    if os.path.exists(args.output_path):
        logger.info(f"Loading existing results from {args.output_path}")
        with open(args.output_path, "r") as f:
            results = json.load(f)
        processed_indices = {r["idx"] for r in results}
        logger.info(f"Resuming from {len(processed_indices)} already processed entries")
    else:
        processed_indices = set()

    # Track results
    results_dict = {r["idx"]: r for r in results}
    total_processed = len(results)
    progress_queue = Queue()

    # Determine which indices still need processing
    to_process = [i for i in sorted(injection_map) if i not in processed_indices]

    if not to_process:
        logger.info("No samples to process. All done!")
        return

    logger.info(f"Processing {len(to_process)} samples with {args.model_name} injection")

    if args.workers > 1:
        logger.info(f"Starting parallel processing with {args.workers} workers")

        def process_results():
            nonlocal total_processed

            with tqdm(total=len(to_process), desc=f"{args.model_name} injection", initial=0) as pbar:
                while True:
                    idx, result = progress_queue.get()

                    if idx == -1:  # Sentinel value to exit
                        break

                    with results_lock:
                        if result is not None:
                            results_dict[idx] = result
                            total_processed += 1

                            if total_processed % args.checkpoint_interval == 0:
                                checkpoint_results = [results_dict[k] for k in sorted(results_dict.keys())]
                                with open(args.output_path, "w") as outf:
                                    json.dump(checkpoint_results, outf, indent=2, ensure_ascii=False)
                                logger.info(f"Checkpoint saved with {total_processed} results")

                    pbar.update(1)
                    progress_queue.task_done()

        # Start progress tracking thread
        progress_thread = threading.Thread(target=process_results)
        progress_thread.start()

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            client_idx = 0

            for idx in to_process:
                if idx >= len(dataset):
                    logger.warning(f"Skipping idx={idx} - index out of range for dataset")
                    continue

                future = executor.submit(
                    process_sample,
                    args,
                    clients[client_idx % len(clients)],
                    dataset[idx],
                    idx,
                    injection_map[idx]
                )
                futures[future] = idx
                client_idx += 1

            for future in futures:
                idx = futures[future]
                try:
                    result = future.result()
                    progress_queue.put((idx, result))
                except Exception as e:
                    logger.error(f"Worker error on sample {idx}: {str(e)}")
                    progress_queue.put((idx, {
                        "idx": idx,
                        "question": injection_map[idx].get("question", ""),
                        "initial_thought": injection_map[idx].get("thought", ""),
                        "qwen_vl_answer": injection_map[idx].get("response", ""),
                        "thought": "",
                        "response": f"WORKER ERROR: {str(e)}",
                        "success": False,
                        "ability_types": dataset[idx].get("ability_types", []) if idx < len(dataset) else [],
                        "attributes": dataset[idx].get("attributes", {}) if idx < len(dataset) else {}
                    }))

        # Signal progress thread to exit
        progress_queue.put((-1, None))
        progress_thread.join()

    else:
        # Sequential processing
        logger.info("Using sequential processing (workers=1)")
        for idx in tqdm(to_process, desc=f"{args.model_name} injection"):
            if idx >= len(dataset):
                logger.warning(f"Skipping idx={idx} - index out of range for dataset")
                continue

            result = process_sample(args, clients[0], dataset[idx], idx, injection_map[idx])
            results_dict[idx] = result
            total_processed += 1

            if total_processed % args.checkpoint_interval == 0:
                checkpoint_results = [results_dict[k] for k in sorted(results_dict.keys())]
                with open(args.output_path, "w") as outf:
                    json.dump(checkpoint_results, outf, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved with {total_processed} results")

    # Final save
    final_results = [results_dict[k] for k in sorted(results_dict.keys())]
    with open(args.output_path, "w") as outf:
        json.dump(final_results, outf, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(final_results)} answers to {args.output_path}")


if __name__ == "__main__":
    main()
