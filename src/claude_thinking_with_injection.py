#!/usr/bin/env python3
"""
A script to use ChatTS-generated injection observations with Claude to produce answers.

For each example it will:
  1. Load the original timeseries + question from the dataset.
  2. Load the injection observations from ChatTS output JSON.
  3. Generate the plot and encode to base64.
  4. Build a multimodal prompt that injects:
     - the question
     - the injection observations from ChatTS
     - a final instruction to produce a "Final Answer"
  5. Call Claude with thinking mode ON.
  6. Parse out the thought and the Final Answer.
  7. Save idx, question, injection observations, thought, answer, success flag.
"""

import os
import re
import json
import base64
import argparse
import numpy as np
# We import matplotlib.pyplot in generate_image_from_timeseries
import boto3
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID        = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_TOKENS      = 4096
THINKING_BUDGET = 2048
WORKERS         = 2
FIG_DIR         = "figures"
# ────────────────────────────────────────────────────────────────────────────────

default_system = (
    "You are a time‐series expert.  \n"
    "Answer **only** with a JSON object that has exactly one key, \"Final Answer\",\n"
    "whose value is the answer string.  \n"
)

def is_throttling(exc):
    return (
        isinstance(exc, ClientError) and
        exc.response.get("Error", {}).get("Code") == "ThrottlingException"
    )

@retry(
    retry=retry_if_exception(is_throttling),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def invoke_claude(client, messages):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": default_system,
        "messages": messages
    }
    resp = client.invoke_model(body=json.dumps(payload), modelId=MODEL_ID)
    return json.loads(resp['body'].read())

def parse_response(resp_body):
    thought_chunks, text_chunks = [], []
    for chunk in resp_body.get("content", []):
        if chunk.get("type") == "thinking":
            thought_chunks.append(chunk.get("thinking","").strip())
        elif chunk.get("type") == "text":
            text_chunks.append(chunk.get("text","").strip())
    thought = "\n".join(thought_chunks)
    raw     = "".join(text_chunks)
    clean   = re.sub(r'```(?:json)?', '', raw).strip()
    start   = clean.find('{')
    end     = clean.rfind('}')
    if start!=-1 and end>start:
        json_str = clean[start:end+1]
    else:
        json_str = clean
    try:
        obj     = json.loads(json_str)
        answer  = obj.get("Final Answer","")
        success = "Final Answer" in obj
    except Exception:
        answer  = json_str
        success = False
    return thought, answer, success

def generate_image_from_timeseries(idx, ts, cols):
    """
    Generate an image from timeseries data.
    Uses the same styling as the ts_visualization utility.
    """
    import sys
    import os
    
    # Add the parent directory to the path to import utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.ts_visualization import generate_image_from_timeseries as gen_img
    
    # Ensure directory exists
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{idx}.jpg")
    
    # Convert numpy array to list if needed
    if isinstance(ts, np.ndarray):
        # Handle different dimensions
        if len(ts.shape) == 1:
            # Single series
            ts_list = [ts.tolist()]
        else:
            # Multiple series
            ts_list = [series.tolist() for series in ts]
    else:
        ts_list = ts
    
    # Ensure we have column names
    if not cols or len(cols) != len(ts_list):
        cols = [f"Series {i+1}" for i in range(len(ts_list))]
        
    # Call the utility function with save_image=True and get the base64 string
    img_b64 = gen_img(idx, ts_list, cols, FIG_DIR, save_image=True)
    
    return path

def build_prompt_with_injection(question: str, observations: str) -> str:
    """
    Build a prompt that includes the question and injection observations from ChatTS,
    then asks Claude to provide a final answer based on these observations.
    """
    # Split off the "Now, based on ..." part if it exists
    if "Now," in question:
        q_part, rest = question.split("Now,", 1)
        question_part = q_part.strip()
        answer_format = "Now," + rest.strip()
    else:
        question_part = question.strip()
        answer_format = ""

    # Build the new user prompt (instructional proxy for early injection)
    body = (
        f"{question_part}\n\n"
        f"<thinking>{observations.strip()}\n\n"
        "Wait, let me summarize and reflect on the previous observations from the time series, "
        "and then continue my reasoning process to derive the final answer...</thinking>\n\n"
        "Please continue your thinking process from the observations above and provide your answer to the question" +
        (f" following these instructions:\n\n{answer_format}\n" if answer_format else ".\n")
    )
    return body

def generate_and_save_image(idx, sample):
    """
    Generate an image for a given sample and save it to disk.
    
    Args:
        idx: Sample index
        sample: Data sample with timeseries and columns
        
    Returns:
        Path to the generated image file or None if there was an error
    """
    try:
        # Extract timeseries and column data
        ts = sample["timeseries"]
        cols = sample["cols"]
        
        # Log structure information for debugging
        if isinstance(ts, list):
            if ts and isinstance(ts[0], list):
                lengths = [len(series) for series in ts]
                print(f"Sample {idx}: Time series is list of lists with lengths {lengths}")
            else:
                print(f"Sample {idx}: Time series is single list with length {len(ts) if ts else 0}")
        elif isinstance(ts, np.ndarray):
            print(f"Sample {idx}: Time series is numpy array with shape {ts.shape}")
        else:
            print(f"Sample {idx}: Time series has unknown type {type(ts)}")
        
        # Make sure cols list is properly sized
        if not cols or len(cols) != len(ts if isinstance(ts, list) else []):
            if isinstance(ts, list):
                cols = [f"Series {i+1}" for i in range(len(ts))]
            else:
                cols = ["Series 1"]
        
        # Generate and save the image
        img_path = generate_image_from_timeseries(idx, ts, cols)
        return img_path
    except Exception as e:
        print(f"[ERROR in generate_and_save_image idx={idx}] {e}")
        return None

def process_with_claude(idx, injection_entry, img_path, client):
    """
    Process a sample using Claude with a pre-generated image and injection observations.

    Args:
        idx: Sample index
        injection_entry: Input entry with question and injection observations
        img_path: Path to the pre-generated image
        client: Boto3 client for Bedrock

    Returns:
        Dict with results
    """
    # Read the pre-generated image file
    with open(img_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode("utf8")

    # Build the prompt
    prompt_text = build_prompt_with_injection(
        injection_entry["question"], injection_entry["observations"]
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": img_b64
            }}
        ]
    }]

    # Invoke Claude
    resp = invoke_claude(client, messages)
    thought, answer, ok = parse_response(resp)

    return {
        "idx": idx,
        "question": injection_entry["question"],
        "observations": injection_entry["observations"],
        "thought": thought,
        "response": answer,
        "success": ok,
        "ability_types": injection_entry.get("ability_types", []),  # Preserve metadata
        "attributes": injection_entry.get("attributes", {})         # Preserve metadata
    }

def main():
    p = argparse.ArgumentParser(description="Generate answers with injection observations from ChatTS")
    p.add_argument("--dataset_path", "-d", required=True,
                   help="Path to the original dataset JSON")
    p.add_argument("--injection_path", "-p", required=True,
                   help="Path to the injection observations JSON from ChatTS")
    p.add_argument("--output_path", "-o", required=True,
                   help="Where to write final results JSON")
    p.add_argument("--workers", "-w", type=int, default=WORKERS,
                   help=f"Number of parallel workers (default: {WORKERS})")
    p.add_argument("--image_workers", "-iw", type=int, default=WORKERS,
                   help=f"Number of workers for image generation (default: {WORKERS})")
    args = p.parse_args()

    # Load data & injection observations
    print(f"Loading dataset from {args.dataset_path}")
    data = json.load(open(args.dataset_path))
    print(f"Loading injection observations from {args.injection_path}")
    injections = json.load(open(args.injection_path))
    injection_map = {h["idx"]: h for h in injections}

    # Prepare output dir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Ensure figures directory exists
    fig_dir = os.path.join(os.path.dirname(args.output_path), FIG_DIR)
    os.makedirs(fig_dir, exist_ok=True)

    # Load existing results if any
    if os.path.exists(args.output_path):
        existing = json.load(open(args.output_path))
        existing_map = {r["idx"]: r for r in existing}
        print(f"Resuming from {len(existing_map)} / {len(injection_map)} already done")
    else:
        existing_map = {}
        print("Starting fresh run")

    # Determine which indices still need processing
    to_process = [i for i in sorted(injection_map) if i not in existing_map]
    
    if not to_process:
        print("No samples to process. All done!")
        return
    
    # STEP 1: Generate all images in parallel
    print(f"\nSTEP 1: Generating {len(to_process)} images in parallel with {args.image_workers} workers")
    image_paths = {}
    
    if args.image_workers > 1:
        # Use multiprocessing Pool instead of ThreadPoolExecutor
        # Create arguments for pool.map as list of tuples
        args_list = [(i, data[i]) for i in to_process if i < len(data)]
        
        # Create a multiprocessing pool
        with Pool(processes=args.image_workers) as pool:
            # Process each item and collect results
            for i, result in enumerate(tqdm(pool.starmap(generate_and_save_image, args_list), 
                                       total=len(args_list), desc="Generating images")):
                idx = to_process[i]
                if result:  # Check if image generation was successful
                    image_paths[idx] = result
                else:
                    print(f"[ERROR generating image idx={idx}] Failed to generate image")
    else:
        # Sequential image generation
        for idx in tqdm(to_process, desc="Generating images"):
            if idx < len(data):  # Make sure we have the sample in the dataset
                try:
                    img_path = generate_and_save_image(idx, data[idx])
                    if img_path:
                        image_paths[idx] = img_path
                except Exception as e:
                    print(f"[ERROR generating image idx={idx}] {e}")
    
    # Report image generation results
    print(f"Successfully generated {len(image_paths)}/{len(to_process)} images")
    
    # Filter to_process to only include samples with successful image generation
    to_process_filtered = [i for i in to_process if i in image_paths]
    if len(to_process_filtered) < len(to_process):
        print(f"WARNING: {len(to_process) - len(to_process_filtered)} samples will be skipped due to image generation failures")
    
    # STEP 2: Process with Claude in parallel
    print(f"\nSTEP 2: Processing {len(to_process_filtered)} samples with Claude using {args.workers} workers")
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    
    # Process samples with Claude
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {
                exe.submit(process_with_claude, i, injection_map[i], image_paths[i], client): i
                for i in to_process_filtered
            }
            for count, fut in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing with Claude"), start=1):
                idx = futures[fut]
                try:
                    res = fut.result()
                    existing_map[idx] = res
                except Exception as e:
                    print(f"[ERROR processing with Claude idx={idx}] {e}")
                # checkpoint every 5
                if count % 5 == 0:
                    with open(args.output_path, "w") as outf:
                        json.dump(
                            [existing_map[k] for k in sorted(existing_map)],
                            outf, indent=2, ensure_ascii=False
                        )
    else:
        for count, idx in enumerate(tqdm(to_process_filtered, desc="Processing with Claude"), start=1):
            try:
                res = process_with_claude(idx, injection_map[idx], image_paths[idx], client)
                existing_map[idx] = res
            except Exception as e:
                print(f"[ERROR processing with Claude idx={idx}] {e}")
            if count % 5 == 0:
                with open(args.output_path, "w") as outf:
                    json.dump(
                        [existing_map[k] for k in sorted(existing_map)],
                        outf, indent=2, ensure_ascii=False
                    )

    # final write
    with open(args.output_path, "w") as outf:
        json.dump(
            [existing_map[k] for k in sorted(existing_map)],
            outf, indent=2, ensure_ascii=False
        )

    print(f"Done: {len(existing_map)}/{len(injection_map)} entries written to {args.output_path}")

if __name__ == "__main__":
    main()