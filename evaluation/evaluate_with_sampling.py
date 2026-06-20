#!/usr/bin/env python3
"""
Evaluate generated answers against ground-truth data, handling sampling metadata.

This script enhances the standard evaluation script by checking for sampling metadata
and ensuring that evaluations are only performed on the sampled subset of the dataset.
If no metadata is found, it falls back to processing the full dataset.

Usage:
    python evaluate_with_sampling.py --exp EXP_NAME --dataset DATASET_PATH --generated OUTPUT_PATH
"""

import os
import json
import argparse
import logging
from tqdm import tqdm

from evaluate_qa import evaluate_batch_qa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_dataset_with_sampling(dataset_path, generated_path):
    """
    Load the dataset and generated answers, handling sampling metadata if available.
    Also matches generated answers to dataset entries based on question text.
    
    Args:
        dataset_path: Path to the full dataset
        generated_path: Path to generated answers
        
    Returns:
        tuple: (dataset, generated)
    """
    # Check if dataset exists
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load the full dataset
    with open(dataset_path, "r") as f:
        full_dataset = json.load(f)
    
    # Load generated answers
    if os.path.exists(generated_path):
        with open(generated_path, "r") as f:
            generated = json.load(f)
    else:
        generated = []
        logger.warning(f"No generated answers found at {generated_path}; proceeding with empty list.")
    
    # Check for sampling metadata
    metadata_path = generated_path.replace('.json', '_sampling_metadata.json')
    sampled_indices = None
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "sampled_indices" in metadata:
                    sampled_indices = metadata["sampled_indices"]
                    logger.info(f"Found sampling metadata: {len(sampled_indices)} samples out of {metadata['original_size']}")
        except Exception as e:
            logger.warning(f"Error loading sampling metadata: {e}")
    
    # If we have sampling metadata, filter the dataset
    if sampled_indices is not None:
        try:
            # Create a subset of the dataset with only the sampled indices
            filtered_dataset = [full_dataset[idx] for idx in sampled_indices]
            logger.info(f"Using sampled dataset with {len(filtered_dataset)} entries (out of {len(full_dataset)} total)")
            
            # Create a mapping of questions to reindex the generated answers
            aligned_generated = []
            for idx, sample in enumerate(filtered_dataset):
                sample_question = sample.get('question', '')
                # Find matching generated answer based on question text
                found = False
                for gen_answer in generated:
                    if gen_answer.get('question', '') == sample_question:
                        # Create a copy and set the idx to match the new dataset position
                        modified_answer = dict(gen_answer)
                        modified_answer['idx'] = idx
                        aligned_generated.append(modified_answer)
                        found = True
                        break
                if not found:
                    logger.warning(f"No matching generated answer found for question: {sample_question[:50]}...")
            
            logger.info(f"Aligned {len(aligned_generated)} generated answers with dataset questions")
            generated = aligned_generated
            dataset = filtered_dataset
            
        except Exception as e:
            logger.error(f"Error applying sampling indices: {e}, falling back to full dataset")
            dataset = full_dataset
    else:
        # No sampling metadata found, use the full dataset
        dataset = full_dataset
        logger.info(f"No sampling metadata found, using full dataset with {len(dataset)} entries")
    
    return dataset, generated

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated QA pairs against a ground-truth dataset, with sampling support."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment name (used to create exp/<exp>/fig folder)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the ground-truth dataset JSON (e.g. evaluation/dataset/dataset_a.json)"
    )
    parser.add_argument(
        "--generated",
        required=True,
        help="Path to the generated answers JSON (e.g. evaluation/exp/<exp>/generated_answer.json)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel workers for evaluate_batch_qa (default: 2)"
    )
    parser.add_argument(
        "--ignore_sampling",
        action="store_true",
        help="Ignore sampling metadata and use the full dataset"
    )

    args = parser.parse_args()

    EXP = args.exp
    DATASET_PATH = args.dataset
    OUTPUT_JSON = args.generated
    FIG_DIR = os.path.join("exp", EXP, "fig")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load dataset and generated answers
    if args.ignore_sampling:
        # Standard loading without sampling
        if not os.path.isfile(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
        
        with open(DATASET_PATH, "r") as f:
            dataset = json.load(f)
        
        if os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON, "r") as f:
                generated = json.load(f)
        else:
            generated = []
            logger.warning(f"No generated answers found at {OUTPUT_JSON}; proceeding with empty list.")
    else:
        # Enhanced loading with sampling support
        dataset, generated = load_dataset_with_sampling(DATASET_PATH, OUTPUT_JSON)

    logger.info(f"Loaded {len(dataset)} examples from dataset")
    logger.info(f"Loaded {len(generated)} generated answers from: {OUTPUT_JSON}")
    logger.info(f"Evaluation figures will be saved under: {FIG_DIR}")

    # Run the batch QA evaluation
    evaluate_batch_qa(dataset, generated, EXP, num_workers=args.num_workers)
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()