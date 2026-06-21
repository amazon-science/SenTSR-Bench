#!/usr/bin/env python3
"""
ChatTS server-based inference benchmark script for time series datasets.

This script:
1. Loads a time series dataset
2. Connects to a running ChatTS server (start with start_chatts_server.sh)
3. Runs inference with the ChatTS server using the OpenAI-compatible API
4. Saves the results to a JSON file

Usage:
    python chatts_inference_server.py --dataset_path /path/to/dataset.json --output_path /path/to/output.json
"""

import os
import sys
import json
import time
import argparse
import logging
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

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ChatTS on a time-series QA dataset using a server")
    p.add_argument("--server_url", default="http://localhost:5000",
                   help="URL of the ChatTS server")
    p.add_argument("--dataset_path", "-d", required=True,
                   help="Path to the JSON evaluation set")
    p.add_argument("--output_path", "-o", required=True,
                   help="Where to write generated answers JSON")
    p.add_argument("--max_tokens", type=int, default=600,
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
    
    def query_chatts_with_timeseries(
        self,
        timeseries,
        question,
        max_tokens=600,
        temperature=0.01,
        timeout=120,
        retry_delay=5,
        max_retries=3,
    ):
        """
        Query ChatTS with time series data.
        
        Args:
            timeseries: Time series data array
            question: Question with <ts><ts/> markers
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response as string
        """
        # Format the question with chat template
        # formatted_question = f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|><|im_start|>user\n{question}\n<|im_end|><|im_start|>assistant\n"
        
        # logger.debug(f"Formatted question with template: {formatted_question[:100]}...")
        formatted_question = question
        
        # Create messages array with timeseries data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_question}
                ] + [{"timeseries": ts} for ts in timeseries]
            }
        ]
        
        # Make the request with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending query to ChatTS server (attempt {attempt+1}/{max_retries})")
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
        
        logger.critical(f"Failed to get response from ChatTS server after {max_retries} attempts")
        raise RuntimeError("Failed to get response from ChatTS server after multiple attempts")

def prepare_question_with_ts_placeholders(question, cols):
    """
    Prepares a question with <ts><ts/> placeholders for each time series column.
    
    Args:
        question: Original question
        cols: List of column names
        
    Returns:
        Modified question with <ts><ts/> placeholders
    """
    # Check if the question already has <ts><ts/> placeholders
    if "<ts><ts/>" in question:
        return question
        
    # Add placeholders for each column
    if cols and len(cols) > 0:
        prefix = f"There are {len(cols)} time series collected: "
        placeholder_text = ", ".join([f"{col}:<ts><ts/>" for col in cols])
        return f"{prefix}{placeholder_text}. Please analyze time series features and answer the following question:\n\n{question}"
    else:
        return question

def process_sample(args, client, sample, idx):
    """
    Process a single sample with the ChatTS client.
    
    Args:
        args: Command line arguments
        client: ChatTSClient instance
        sample: Data sample
        idx: Sample index
        
    Returns:
        Result dictionary
    """
    try:
        # Extract data
        cols = sample.get("cols", [])
        ts_data = sample["timeseries"]
        question = sample["question"]
        
        # Prepare question with placeholders if needed
        question_with_ts = prepare_question_with_ts_placeholders(question, cols)
        
        # Query the ChatTS server
        answer = client.query_chatts_with_timeseries(
            timeseries=ts_data,
            question=question_with_ts,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries
        )

        # Return successful result
        return {
            "idx": idx,
            "question": question,
            "response": answer,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error processing sample {idx}: {str(e)}")
        # Get question from sample, handling potential key errors safely
        question = ""
        try:
            question = sample.get("question", "")
        except Exception:
            pass
            
        return {
            "idx": idx,
            "question": question,
            "response": f"ERROR: {str(e)}",
            "success": False
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

    # Create a dictionary to hold futures for each worker
    if args.workers > 1:
        logger.info(f"Starting parallel processing with {args.workers} workers")
        
        # Function to process results and update progress
        def process_results():
            nonlocal total_processed
            
            with tqdm(total=len(dataset), desc="Evaluating ChatTS", initial=len(results)) as pbar:
                while True:
                    # Get the next completed task from the queue
                    idx, result = progress_queue.get()
                    
                    if idx == -1:  # Sentinel value to exit
                        break
                        
                    # Update results with thread safety
                    with results_lock:
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
            for idx, sample in enumerate(dataset):
                # Skip if already processed
                if idx in processed_indices:
                    continue
                    
                # Submit the task to the thread pool
                future = executor.submit(
                    process_sample, 
                    args,
                    clients[client_idx % len(clients)],  # Round-robin client assignment
                    sample, 
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
                    progress_queue.put((idx, {
                        "idx": idx,
                        "question": dataset[idx].get("question", ""),
                        "response": f"WORKER ERROR: {str(e)}",
                        "success": False
                    }))
        
        # Signal progress thread to exit and wait for it to finish
        progress_queue.put((-1, None))
        progress_thread.join()
        
    else:
        # Sequential processing
        logger.info("Using sequential processing (workers=1)")
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating ChatTS")):
            # Skip if already processed
            if idx in processed_indices:
                continue
            
            # Process the sample
            result = process_sample(args, clients[0], sample, idx)
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