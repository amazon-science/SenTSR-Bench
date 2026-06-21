#!/usr/bin/env python3
"""
ChatTS Injection Script

This script uses a running ChatTS server to extract injection time series observations:
1. Connects to a running ChatTS server (start with start_chatts_server.sh)
2. Loads a dataset with questions and time series data
3. Submits each question to ChatTS to generate injection observations
4. Saves the observations to a JSON file for later use with Claude

Usage:
    python chatts_injection.py --dataset_path /path/to/dataset.json --output_path /path/to/observations.json
"""

import os
import sys
import json
import time
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
    p = argparse.ArgumentParser(description="Use ChatTS server to extract injection time series observations")
    p.add_argument("--server_url", default="http://localhost:5000",
                   help="URL of the ChatTS server")
    p.add_argument("--dataset_path", "-d", required=True,
                   help="Path to the dataset JSON")
    p.add_argument("--output_path", "-o", required=True,
                   help="Where to write injection observations JSON")
    p.add_argument("--max_tokens", type=int, default=3072,
                   help="Maximum tokens to generate")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers for processing samples")
    p.add_argument("--checkpoint_interval", type=int, default=10,
                   help="Interval for saving checkpoints")
    p.add_argument("--timeout", type=int, default=180,
                   help="Timeout in seconds for API calls")
    p.add_argument("--retry_delay", type=int, default=5,
                   help="Delay between retries in seconds")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Maximum number of retry attempts")
    return p.parse_args()

# For injection case, we keep the full question

def build_injection_prompt(question: str, ts_cols=None) -> str:
    """Build a prompt for the ChatTS injection."""
    # Keep the full question for injection case
    
    # Check if the question already has <ts><ts/> placeholders
    has_ts_placeholders = '<ts><ts/>' in question
    
    # If no placeholders and we have column info, add them
    ts_placeholder_text = ""
    if not has_ts_placeholders and ts_cols and len(ts_cols) > 0:
        num_series = len(ts_cols)
        ts_placeholder_text = f"There are {num_series} time series collected: "
        for i, col_name in enumerate(ts_cols):
            ts_placeholder_text += f"{col_name}:<ts><ts/>"
            if i < len(ts_cols) - 1:
                ts_placeholder_text += ", "
        ts_placeholder_text += ". "
    
    prompt_body = (
        "<|im_start|>system\n"
        "You are a helpful time series analysis assistant.\n"
        "<|im_end|>"
        "<|im_start|>user\n"
        "You are analyzing a time series to extract key quantitative observations that help answer the question.\n\n"
        f"{ts_placeholder_text}{question}\n\n"
        "Provide detailed, objective numerical observations by following these guidelines:\n"
        "1. Make numbered, precise observations about the quantitative aspects of the time series.\n"
        "2. Be specific about values, positions, and magnitudes when describing features.\n"
        "3. Begin each observation with \"Observation 1:\", \"Observation 2:\", etc.\n"
        "\n"
        "Start your response with:\n"
        "\"To answer this question, I need to carefully analyze the time series. "
        "Here are my observations: Observation 1... Observation 2...\""
        "<|im_end|>"
        "<|im_start|>assistant\n"
    )
    return prompt_body

class ChatTSClient:
    """Client for communicating with ChatTS server using OpenAI API."""
    
    def __init__(self, server_url="http://localhost:5000", debug_mode=False):
        """Initialize the ChatTS client."""
        self.server_url = server_url
        self.debug_mode = debug_mode
        self.client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy-key")
        
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.info(f"ChatTSClient initialized in DEBUG mode with server URL: {server_url}")
    
    def check_server_health(self):
        """Check if the server is healthy."""
        import requests
        try:
            logger.info(f"Checking health of ChatTS server at {self.server_url}...")
            # Try to get the models list as a health check
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if response.status_code == 200:
                logger.info(f"ChatTS server is healthy")
                return True
            else:
                logger.warning(f"ChatTS server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking server health: {type(e).__name__}: {e}")
            return False
    
    def generate_observations(
        self,
        timeseries,
        question,
        ts_cols=None,
        max_tokens=1024,
        temperature=0.01,
        timeout=120,
        retry_delay=5,
        max_retries=3,
    ):
        """
        Submit question to ChatTS for generating observations.
        
        Args:
            timeseries: Time series data
            question: Original question 
            ts_cols: Column names for the time series
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated observations as string
        """
        # Build the prompt for the injection
        prompt = build_injection_prompt(question, ts_cols)
        
        # Create messages array with timeseries data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"timeseries": ts} for ts in timeseries]
            }
        ]
        
        # Make the request with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending observation request to ChatTS server (attempt {attempt+1}/{max_retries})")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model="chatts",
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
                        f"Observation generation successful in {elapsed:.2f}s: "
                        f"prompt_tokens={usage.prompt_tokens}, "
                        f"completion_tokens={usage.completion_tokens}, "
                        f"total_tokens={usage.total_tokens}"
                    )
                else:
                    logger.info(f"Observation generation successful, inference time: {elapsed:.2f}s")
                
                return response.choices[0].message.content
                    
            except Exception as e:
                logger.error(f"Observation request failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                time.sleep(retry_delay)  # Wait before retry
            
            # Increase retry delay for exponential backoff
            retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
        
        logger.critical(f"Failed to get response from ChatTS server after {max_retries} attempts")
        raise RuntimeError("Failed to get response from ChatTS server after multiple attempts")

def process_sample(args, client, dataset, idx):
    """
    Process a single sample with the ChatTS client.
    
    Args:
        args: Command line arguments
        client: ChatTSClient instance
        dataset: Full dataset
        idx: Sample index
        
    Returns:
        Result dictionary with observations
    """
    try:
        # Get the sample
        sample = dataset[idx]
        question = sample.get("question", "")
        
        if not question:
            logger.warning(f"No question found for sample {idx}, skipping")
            return None
            
        # Get column names and timeseries data from the sample
        ts_cols = sample.get("cols", [])
        timeseries = sample["timeseries"]
        
        # For better logging
        if isinstance(timeseries, list):
            ts_shape = f"List with {len(timeseries)} elements"
            if len(timeseries) > 0 and isinstance(timeseries[0], list):
                ts_shape += f", first element length: {len(timeseries[0])}"
        else:
            ts_shape = "Not a list"
        logging.info(f"Sample {idx}: Timeseries shape: {ts_shape}")
        
        # Get the observations from the server
        observations = client.generate_observations(
            timeseries=timeseries,
            question=question,
            ts_cols=ts_cols,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries
        )
        
        # Return the result
        return {
            "idx": idx,
            "question": question,
            "observations": observations,
            "ability_types": sample.get("ability_types", []),  # Preserve any metadata from sample
            "attributes": sample.get("attributes", {})         # Preserve any metadata from sample
        }
        
    except Exception as e:
        logger.error(f"Error processing sample {idx}: {str(e)}")
        return {
            "idx": idx,
            "question": sample.get("question", "") if sample else "",
            "observations": f"ERROR: {str(e)}",
            "ability_types": sample.get("ability_types", []) if sample else [],
            "attributes": sample.get("attributes", {}) if sample else {}
        }

def main():
    args = parse_args()
    
    # Initialize lock for thread-safe operations
    results_lock = threading.Lock()
    
    # Initialize ChatTS client pool
    clients = [ChatTSClient(server_url=args.server_url) for _ in range(args.workers)]
    
    # Check server health with first client
    if not clients[0].check_server_health():
        logger.error("ChatTS server is not healthy. Please make sure it is running.")
        logger.error("Run: ./start_chatts_server.sh to start the server.")
        sys.exit(1)
    else:
        logger.info(f"Using {args.workers} workers for parallel processing")
    
    # 1) Load data
    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load existing results if any
    results = []
    if os.path.exists(args.output_path):
        logger.info(f"Loading existing results from {args.output_path}")
        with open(args.output_path, "r") as f:
            results = json.load(f)
        processed_indices = {r["idx"] for r in results}
        logger.info(f"Resuming from {len(processed_indices)} already processed entries")
    else:
        processed_indices = set()
    
    # Track results and progress in thread-safe way
    results_dict = {r["idx"]: r for r in results}
    total_processed = len(results)
    progress_queue = Queue()
    
    # Create a list of samples to process
    samples_to_process = [idx for idx in range(len(dataset)) if idx not in processed_indices]
    logger.info(f"Found {len(samples_to_process)} samples to process out of {len(dataset)} total")
    
    if not samples_to_process:
        logger.info("No new samples to process")
        return
    
    if args.workers > 1:
        logger.info(f"Starting parallel processing with {args.workers} workers")
        
        # Function to process results and update progress
        def process_results():
            nonlocal total_processed
            
            with tqdm(total=len(samples_to_process), desc="Generating observations", initial=0) as pbar:
                while True:
                    # Get the next completed task from the queue
                    idx, result = progress_queue.get()
                    
                    if idx == -1:  # Sentinel value to exit
                        break
                        
                    # Update results with thread safety
                    with results_lock:
                        if result is not None:  # Skip None results (errors)
                            results_dict[idx] = result
                            total_processed += 1
                            
                            # Checkpoint at specified intervals
                            if total_processed % args.checkpoint_interval == 0:
                                checkpoint_results = [results_dict[k] for k in sorted(results_dict.keys())]
                                with open(args.output_path, "w") as outf:
                                    json.dump(checkpoint_results, outf, indent=2, ensure_ascii=False)
                                logger.info(f"Checkpoint saved with {total_processed} results")
                    
                    # Update progress bar
                    pbar.update(1)
                    progress_queue.task_done()
        
        # Start progress tracking thread
        progress_thread = threading.Thread(target=process_results)
        progress_thread.start()
        
        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            client_idx = 0
            
            # Submit tasks to the thread pool
            for idx in samples_to_process:
                # Submit the task to the thread pool
                future = executor.submit(
                    process_sample, 
                    args,
                    clients[client_idx % len(clients)],  # Round-robin client assignment
                    dataset, 
                    idx
                )
                futures[future] = idx
                client_idx += 1
            
            # Wait for tasks to complete and collect results
            for future in futures:
                idx = futures[future]
                try:
                    result = future.result()
                    progress_queue.put((idx, result))
                except Exception as e:
                    logger.error(f"Worker error on sample {idx}: {str(e)}")
                    # Create an error result
                    error_result = {
                        "idx": idx,
                        "question": dataset[idx].get("question", "") if idx < len(dataset) else "",
                        "observations": f"WORKER ERROR: {str(e)}",
                        "ability_types": dataset[idx].get("ability_types", []) if idx < len(dataset) else [],
                        "attributes": dataset[idx].get("attributes", {}) if idx < len(dataset) else {}
                    }
                    progress_queue.put((idx, error_result))
        
        # Signal progress thread to exit and wait for it to finish
        progress_queue.put((-1, None))
        progress_thread.join()
        
    else:
        # Sequential processing
        logger.info("Using sequential processing (workers=1)")
        client = clients[0]  # Use the first client
        
        for idx in tqdm(samples_to_process, desc="Generating observations"):
            # Process the sample
            result = process_sample(args, client, dataset, idx)
            
            # Update results
            if result:
                results_dict[idx] = result
                total_processed += 1
                
                # Checkpoint at specified intervals
                if total_processed % args.checkpoint_interval == 0:
                    checkpoint_results = [results_dict[k] for k in sorted(results_dict.keys())]
                    with open(args.output_path, "w") as outf:
                        json.dump(checkpoint_results, outf, indent=2, ensure_ascii=False)
                    logger.info(f"Checkpoint saved with {total_processed} results")
    
    # Final save
    final_results = [results_dict[k] for k in sorted(results_dict.keys())]
    with open(args.output_path, "w") as outf:
        json.dump(final_results, outf, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(final_results)} injection observations to {args.output_path}")

if __name__ == "__main__":
    main()