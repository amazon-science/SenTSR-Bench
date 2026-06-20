#!/usr/bin/env python3
"""
A two-step approach for Claude inference with time series data:
1. First generates and saves all figures sequentially
2. Then processes the figures with Claude in parallel

This script:
1. Loads a multimodal dataset (timeseries + question).
2. Generate all figures sequentially to avoid thread-safety issues
3. Send each example to Claude in parallel with thinking mode enabled
4. Parse out "thought" (the chain-of-thought) and the answer.
5. Generate HTML reports for easy result inspection.

Usage:
    # Basic usage with images
    python claude_thinking_inference.py --dataset_path path/to/dataset.json --output_path path/to/output.json
    
    # Text-only mode (no images)
    python claude_thinking_inference.py --dataset_path path/to/dataset.json --output_path path/to/output.json --text_only
    
"""

import os
import json
import argparse
import random
import logging
import numpy as np
import boto3
import base64
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import utility functions
import sys
import os

# Add the parent directory to the path so we can import claude_utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from claude_utils
from claude_utils.ts_visualization import generate_image_from_timeseries
from claude_utils.claude_inference import (
    MODEL_ID,
    invoke_claude,
    parse_response,
)

# Default configuration
WORKERS = 2
FIG_DIR = "figures"  # will be created under output's parent directory
CHECKPOINT_INTERVAL = 50
MAX_SAMPLES = 450  # Default maximum number of samples to process

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
        logging.warning(f"Entry has empty time series data, using dummy data")
        
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
                    logging.warning(f"Empty image file generated for idx={idx}")
            else:
                logging.warning(f"Image file not created for idx={idx}")
        except Exception as e:
            logging.error(f"Error generating image for idx={idx}: {e}")
    
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
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return img_b64

def process_sample_with_existing_image(idx, sample, client, image_path, text_only=False):
    """
    Process a sample using a pre-generated image file.

    Args:
        idx: Sample index
        sample: Sample data containing question
        client: Boto3 client for Bedrock
        image_path: Path to the pre-generated image
        text_only: If True, only use text input (no image)

    Returns:
        Dict containing results (idx, question, thought, response, etc.)
    """
    # Extract question
    question = sample.get("question", "")
    
    # Read the image if not in text-only mode
    img_b64 = None
    if not text_only and image_path:
        try:
            img_b64 = get_image_base64(image_path)
        except Exception as e:
            logging.error(f"Error reading image for idx={idx}: {e}")
            # Continue with text-only if image loading fails
            text_only = True
    
    # Prepare message for Claude
    if text_only:
        # Text-only mode - just the question
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question}
            ]
        }]
    else:
        # With image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": img_b64
                }}
            ]
        }]
    
    # Invoke Claude
    resp = invoke_claude(client, messages)
    thought, answer, ok = parse_response(resp)
    
    # Return results in a format compatible with HTML report generator
    return {
        "idx": idx,
        "question": question,
        "analysis": thought,  # Use Claude's thought process as analysis for report
        "thought": thought,
        "response": answer,
        "success": ok,
        "image_path": image_path
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Claude inference with time series visualization (two-step approach)"
    )
    parser.add_argument(
        "--dataset_path", "-d", 
        required=True,
        help="Path to the input dataset JSON"
    )
    parser.add_argument(
        "--output_path", "-o", 
        required=True,
        help="Path to write the output JSON results"
    )
    parser.add_argument(
        "--workers", "-w", 
        type=int, 
        default=WORKERS,
        help=f"Number of parallel workers (default: {WORKERS})"
    )
    parser.add_argument(
        "--checkpoint_interval", "-c", 
        type=int, 
        default=CHECKPOINT_INTERVAL,
        help=f"Interval for saving checkpoints (default: {CHECKPOINT_INTERVAL})"
    )
    parser.add_argument(
        "--max_samples", "-m",
        type=int, 
        default=MAX_SAMPLES,
        help=f"Maximum number of samples to process (default: {MAX_SAMPLES})"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, 
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--text_only", "-t",
        action="store_true",
        help="Run inference with text-only mode (no images)"
    )
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Set up experiment-specific figure directory
    fig_dir = os.path.join(os.path.dirname(args.output_path), FIG_DIR)
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Using figure directory: {fig_dir}")
    
    # Print working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Load dataset
    logging.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        full_dataset = json.load(f)
    
    # Sample if needed
    total_entries = len(full_dataset)
    if total_entries > args.max_samples:
        logging.warning(f"Dataset has {total_entries} entries, which exceeds the maximum of {args.max_samples}.")
        logging.warning(f"Randomly sampling {args.max_samples} entries with seed {args.seed}.")
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        
        # Sample entries
        sampled_indices = random.sample(range(total_entries), args.max_samples)
        data = [full_dataset[i] for i in sampled_indices]
        
        # Create a metadata file with the sampled indices
        metadata_file = args.output_path.replace('.json', '_sampling_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                "original_size": total_entries,
                "sampled_size": args.max_samples,
                "seed": args.seed,
                "sampled_indices": sampled_indices
            }, f, indent=2)
        
        logging.info(f"Sampling metadata saved to {metadata_file}")
    else:
        data = full_dataset
        sampled_indices = list(range(len(data)))  # All indices if no sampling
        logging.info(f"Processing all {total_entries} entries (no sampling needed)")
    
    # Check for existing results to support resuming
    existing_map = {}
    if os.path.exists(args.output_path):
        print(f"Loading existing results from {args.output_path}")
        with open(args.output_path, "r") as f:
            existing = json.load(f)
            existing_map = {r["idx"]: r for r in existing}
        print(f"Resuming from {len(existing_map)} / {len(data)} already done")
    else:
        print("Starting fresh run")
    
    # Determine which indices still need processing
    to_process = [i for i in range(len(data)) if i not in existing_map]
    
    # STEP 1: Generate all images sequentially
    if not args.text_only:
        image_paths = generate_all_images(data, to_process, fig_dir)
        image_count = len(image_paths)
        print(f"Generated {image_count} images out of {len(to_process)} total samples")
    else:
        # In text-only mode, we don't need to generate images
        image_paths = {idx: None for idx in to_process}
        print("Skipping image generation in text-only mode")
    
    # Initialize Bedrock client
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    
    # STEP 2: Process samples with Claude in parallel
    print(f"Processing {len(to_process)} samples")
    
    if args.workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            
            for idx in to_process:
                # Get the image path for this index (or None if not available)
                image_path = image_paths.get(idx, None)
                
                # Skip if we don't have an image in non-text-only mode
                if not args.text_only and image_path is None:
                    logging.warning(f"Skipping idx={idx} - no image available")
                    continue
                
                # Submit the task
                futures[executor.submit(
                    process_sample_with_existing_image, 
                    idx, 
                    data[idx], 
                    client,
                    image_path,
                    args.text_only,
                )] = idx
                
            # Process futures as they complete
            for count, fut in enumerate(tqdm(futures, desc="Inference"), start=1):
                idx = futures[fut]
                try:
                    res = fut.result()
                    existing_map[idx] = res
                except Exception as e:
                    print(f"[ERROR idx={idx}] {e}")
                
                # Checkpoint at specified intervals
                if count % args.checkpoint_interval == 0 or count == len(futures):
                    # Save JSON checkpoint
                    with open(args.output_path, "w") as outf:
                        json.dump(
                            [existing_map[k] for k in sorted(existing_map)], 
                            outf, 
                            indent=2, 
                            ensure_ascii=False
                        )
    else:
        # Sequential processing
        for count, idx in enumerate(tqdm(to_process, desc="Inference"), start=1):
            # Get the image path for this index (or None if not available)
            image_path = image_paths.get(idx, None)
            
            # Skip if we don't have an image in non-text-only mode
            if not args.text_only and image_path is None:
                logging.warning(f"Skipping idx={idx} - no image available")
                continue
                
            try:
                res = process_sample_with_existing_image(
                    idx, data[idx], client, image_path, 
                    args.text_only,
                )
                
                existing_map[idx] = res
            except Exception as e:
                print(f"[ERROR idx={idx}] {e}")
            
            # Checkpoint at specified intervals
            if count % args.checkpoint_interval == 0 or count == len(to_process):
                # Save JSON checkpoint
                with open(args.output_path, "w") as outf:
                    json.dump(
                        [existing_map[k] for k in sorted(existing_map)], 
                        outf, 
                        indent=2, 
                        ensure_ascii=False
                    )

    # Final write
    with open(args.output_path, "w") as outf:
        json.dump(
            [existing_map[k] for k in sorted(existing_map)], 
            outf, 
            indent=2, 
            ensure_ascii=False
        )

    print(f"Done: {len(existing_map)}/{len(data)} entries written to {args.output_path}")

if __name__ == "__main__":
    main()