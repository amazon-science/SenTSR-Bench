#!/usr/bin/env python3
"""
Qwen2.5-VL server-based inference script for time series datasets.

This script:
1. Loads a time series dataset
2. Generates all time series figures sequentially
3. Connects to a running Qwen2.5-VL server (start with start_qwen_vl_server.sh)
4. Processes figures with Qwen2.5-VL in parallel with thinking mode
5. Saves the results to a JSON file

Usage:
    python qwen_inference.py --dataset_path /path/to/dataset.json --output_path /path/to/output.json
"""

import os
import sys
import json
import time
import re
import argparse
import logging
import base64
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI Python client not installed. Please install with 'pip install openai'")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import claude_utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_utils.ts_visualization import generate_image_from_timeseries

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on a time-series QA dataset using a server")
    p.add_argument("--server_url", default="http://localhost:5003",
                   help="URL of the Qwen2.5-VL server")
    p.add_argument("--dataset_path", "-d", required=True,
                   help="Path to the JSON evaluation set")
    p.add_argument("--output_path", "-o", required=True,
                   help="Where to write generated answers JSON")
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="Maximum tokens to generate")
    p.add_argument("--max_samples", type=int, default=200,
                   help="Maximum number of samples to process")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling")
    p.add_argument("--checkpoint_interval", type=int, default=10,
                   help="Interval for saving checkpoints")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers for processing samples")
    p.add_argument("--timeout", type=int, default=120,
                   help="Timeout in seconds for API calls")
    p.add_argument("--retry_delay", type=int, default=5,
                   help="Delay between retries in seconds")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Maximum number of retry attempts")
    return p.parse_args()

class QwenVLClient:
    """Client for communicating with Qwen2.5-VL server using OpenAI API."""

    def __init__(self, server_url="http://localhost:5003", debug_mode=False):
        """Initialize the Qwen2.5-VL client."""
        self.server_url = server_url
        self.debug_mode = debug_mode
        self.client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy-key")

        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.info(f"QwenVLClient initialized in DEBUG mode with server URL: {server_url}")

    def check_server_health(self):
        """Check if the server is healthy."""
        import requests
        try:
            logger.info(f"Checking health of Qwen2.5-VL server at {self.server_url}...")
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if response.status_code == 200:
                logger.info(f"Qwen2.5-VL server is healthy")
                return True
            else:
                logger.warning(f"Qwen2.5-VL server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking server health: {type(e).__name__}: {e}")
            return False

    def query_qwen_with_image(
        self,
        image_b64,
        question,
        max_tokens=1024,
        temperature=0,
        timeout=120,
        retry_delay=5,
        max_retries=3,
    ):
        """
        Query Qwen2.5-VL with an image and question.

        Args:
            image_b64: Base64 encoded image string
            question: Question text
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Model response as string
        """
        # Create standard system message with thinking instructions
        system_message = "You are a helpful assistant that analyzes time series data."
        thinking_instruction = "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags"

        # Build image content
        image_content = []
        if image_b64:
            image_content = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }}
            ]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": f"{thinking_instruction}\n\n{question}"},
                *image_content
            ]}
        ]

        # Make the request with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending query to Qwen2.5-VL server (attempt {attempt+1}/{max_retries})")
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model="qwen_vl",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                end_time = time.time()
                elapsed = end_time - start_time

                # Get usage information if available
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
                time.sleep(retry_delay)  # Wait before retry

            # Increase retry delay for exponential backoff
            retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds

        logger.critical(f"Failed to get response from Qwen2.5-VL server after {max_retries} attempts")
        raise RuntimeError("Failed to get response from Qwen2.5-VL server after multiple attempts")

def prepare_timeseries_data(ts):
    """
    Prepare timeseries data for visualization.

    Args:
        ts: Raw timeseries data

    Returns:
        Processed timeseries data suitable for visualization
    """
    # Ensure ts is a list of lists (for multiple series)
    if not isinstance(ts, list):
        ts = [ts]  # Wrap single series
    elif len(ts) > 0 and not isinstance(ts[0], list):
        ts = [ts]  # Wrap flat list into nested list

    # Check if we got empty data
    if not ts or len(ts) == 0:
        ts = [[0, 1, 2, 3, 4]]  # Default dummy data
        logger.warning("Entry has empty time series data, using dummy data")

    return ts

def generate_all_images(data, to_process, fig_dir):
    """
    Generate all images sequentially and return image paths.

    Args:
        data: Dataset containing timeseries data
        to_process: List of indices to process
        fig_dir: Directory to save figures

    Returns:
        Dictionary mapping indices to image paths
    """
    image_paths = {}

    print(f"Generating {len(to_process)} figures sequentially...")
    for idx in tqdm(to_process, desc="Generating figures"):
        sample = data[idx]
        ts = sample["timeseries"]
        cols = sample.get("cols", [])

        # Prepare timeseries data
        ts = prepare_timeseries_data(ts)

        # Generate and save image
        path = os.path.join(fig_dir, f"{idx}.jpg")
        try:
            # Always save the image
            _ = generate_image_from_timeseries(
                case_idx=idx,
                timeseries=ts,
                cols=cols,
                fig_dir=fig_dir,
                save_image=True
            )

            # Check if the image was created successfully
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                if file_size > 0:
                    image_paths[idx] = path
                else:
                    logger.warning(f"Empty image file generated for idx={idx}")
            else:
                logger.warning(f"Image file not created for idx={idx}")
        except Exception as e:
            logger.error(f"Error generating image for idx={idx}: {e}")

    print(f"Successfully generated {len(image_paths)} figures out of {len(to_process)} requested")
    return image_paths

def get_image_base64(image_path):
    """
    Load an image from disk and convert to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        return img_b64
    except Exception as e:
        logger.error(f"Error reading image file {image_path}: {e}")
        return None

def parse_response(response):
    """
    Parse response to extract thinking and answer parts.

    Args:
        response: The raw response string from the model

    Returns:
        Tuple of (thought, answer, success_flag)
    """
    # Default values
    thought = ""
    answer = response
    success = True

    # Extract thinking section
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()

    # Extract answer section
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    # If we didn't find both sections, consider this a partial success
    # but still return the full text as the answer
    if not (think_match and answer_match):
        logger.warning("Could not parse response into thinking and answer parts")
        success = False
        if not answer:
            answer = response  # Fallback to the full response

    return thought, answer, success

def process_sample(args, client, sample, idx, image_path):
    """
    Process a single sample with the Qwen2.5-VL client.

    Args:
        args: Command line arguments
        client: QwenVLClient instance
        sample: Data sample
        idx: Sample index
        image_path: Path to the pre-generated image

    Returns:
        Result dictionary
    """
    try:
        # Extract question
        question = sample.get("question", "")

        # Load image as base64
        img_b64 = get_image_base64(image_path) if image_path else None

        # Query the Qwen2.5-VL server
        raw_answer = client.query_qwen_with_image(
            image_b64=img_b64,
            question=question,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries
        )

        # Parse the response to extract thinking and answer parts
        thought, answer, parse_ok = parse_response(raw_answer)

        # Return successful result
        return {
            "idx": idx,
            "question": question,
            "thought": thought,
            "response": answer,
            "raw_response": raw_answer,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error processing sample {idx}: {str(e)}")
        question = ""
        try:
            question = sample.get("question", "")
        except Exception:
            pass

        return {
            "idx": idx,
            "question": question,
            "thought": "",
            "response": f"ERROR: {str(e)}",
            "raw_response": f"ERROR: {str(e)}",
            "success": False
        }

def main():
    args = parse_args()

    # Initialize lock for thread-safe operations
    results_lock = threading.Lock()

    # Initialize Qwen2.5-VL client pool
    clients = [QwenVLClient(server_url=args.server_url) for _ in range(args.workers)]

    # Check server health with first client
    if not clients[0].check_server_health():
        logger.error("Qwen2.5-VL server is not healthy. Please make sure it is running.")
        logger.error("Run: src/qwen_utils/start_qwen_vl_server.sh to start the server.")
        sys.exit(1)
    else:
        logger.info(f"Using {args.workers} workers for parallel processing")

    # Load evaluation set
    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        full_dataset = json.load(f)

    # Sample if needed
    total_entries = len(full_dataset)
    if total_entries > args.max_samples:
        logger.info(f"Sampling {args.max_samples} entries from {total_entries} total")
        import random
        random.seed(args.seed)
        indices = random.sample(range(total_entries), args.max_samples)
        dataset = [full_dataset[i] for i in indices]

        # Save metadata about the sampling
        metadata_file = args.output_path.replace(".json", "_sampling_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({
                "original_size": total_entries,
                "sampled_size": args.max_samples,
                "seed": args.seed,
                "sampled_indices": indices
            }, f, indent=2)
    else:
        dataset = full_dataset

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Set up experiment-specific figure directory
    fig_dir = os.path.join(os.path.dirname(args.output_path), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    logger.info(f"Using figure directory: {fig_dir}")

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
    to_process = [i for i in range(len(dataset)) if i not in processed_indices]

    # STEP 1: Generate all images sequentially
    image_paths = generate_all_images(dataset, to_process, fig_dir)
    image_count = len(image_paths)
    logger.info(f"Generated {image_count} images out of {len(to_process)} total samples")

    # STEP 2: Process samples with Qwen2.5-VL
    if args.workers > 1:
        logger.info(f"Starting parallel processing with {args.workers} workers")

        # Function to process results and update progress
        def process_results():
            nonlocal total_processed

            with tqdm(total=len(dataset), desc="Evaluating Qwen2.5-VL", initial=len(results)) as pbar:
                while True:
                    idx, result = progress_queue.get()

                    if idx == -1:  # Sentinel value to exit
                        break

                    with results_lock:
                        results_dict[idx] = result
                        total_processed += 1

                        # Checkpoint at specified intervals
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
                # Get the image path for this index
                image_path = image_paths.get(idx, None)

                # Skip if we don't have an image
                if image_path is None:
                    logger.warning(f"Skipping idx={idx} - no image available")
                    continue

                future = executor.submit(
                    process_sample,
                    args,
                    clients[client_idx % len(clients)],
                    dataset[idx],
                    idx,
                    image_path
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
                        "question": dataset[idx].get("question", ""),
                        "thought": "",
                        "response": f"WORKER ERROR: {str(e)}",
                        "raw_response": f"WORKER ERROR: {str(e)}",
                        "success": False
                    }))

        # Signal progress thread to exit and wait for it to finish
        progress_queue.put((-1, None))
        progress_thread.join()

    else:
        # Sequential processing
        logger.info("Using sequential processing (workers=1)")
        for idx in tqdm(to_process, desc="Evaluating Qwen2.5-VL"):
            # Skip if already processed
            if idx in processed_indices:
                continue

            # Get the image path for this index
            image_path = image_paths.get(idx, None)

            # Skip if we don't have an image
            if image_path is None:
                logger.warning(f"Skipping idx={idx} - no image available")
                continue

            result = process_sample(args, clients[0], dataset[idx], idx, image_path)
            results_dict[idx] = result

            # Checkpoint at specified intervals
            if len(results_dict) % args.checkpoint_interval == 0:
                checkpoint_results = [results_dict[k] for k in sorted(results_dict.keys())]
                with open(args.output_path, "w") as outf:
                    json.dump(checkpoint_results, outf, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved with {len(results_dict)} results")

    # Final save with sorted results
    final_results = [results_dict[k] for k in sorted(results_dict.keys())]
    with open(args.output_path, "w") as outf:
        json.dump(final_results, outf, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(final_results)} answers to {args.output_path}")

if __name__ == "__main__":
    main()
