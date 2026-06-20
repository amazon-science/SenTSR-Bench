#!/usr/bin/env python3
"""
Script for preprocessing TSR knowledge injection datasets.

This script combines the functionality of several separate scripts:
1. Split multi-question entries in dataset_a.json into individual questions
2. Filter entries based on ability types (causal, deductive, inductive)
3. Sample 100 entries from MCQ_2_TS.jsonl (with fixed seed for reproducibility)
4. Merge dataset_a with MCQ2 dataset

Usage:
    python preprocess_dataset.py [--dataset_a PATH] [--mcq2_source PATH] [--output_dir PATH]
"""

import argparse
import json
import os
import re
import copy
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm


def split_dataset(dataset):
    """
    Split multi-question entries into individual questions.
    
    Args:
        dataset: The original dataset as a list of dictionaries
    
    Returns:
        List of dictionaries with one question per entry
    """
    print("Splitting multi-question entries...")
    
    # New dataset to store individual questions
    split_dataset = []
    
    for entry in dataset:
        # Common elements to be replicated for each subquestion
        timeseries = entry.get('timeseries')
        cols = entry.get('cols')
        
        # Extract the question text
        question = entry.get('question', '')
        
        # Split the question into prefix and subquestions
        question_parts = question.split("please analyze the time series features and answer the following questions:")
        if len(question_parts) != 2:
            # If the split doesn't work as expected, try an alternative approach
            question_parts = question.split("please analyze the time series features and answer the following question:")
            if len(question_parts) != 2:
                print(f"Warning: Could not split question properly: {question[:100]}...")
                continue
        
        question_prefix = question_parts[0] + "please analyze the time series features and answer the following question:"
        subquestions_text = question_parts[1]
        
        # Remove the formatting instructions at the end
        if "Now, based on the above questions" in subquestions_text:
            subquestions_text = subquestions_text.split("Now, based on the above questions")[0]
        
        # Extract the actual questions (numbered items)
        subquestions = re.findall(r'\n\d+\. (.*?)(?=\n\d+\.|$)', subquestions_text, re.DOTALL)
        
        # Extract answers
        answer = entry.get('answer', '')
        answers = re.findall(r'\d+\. (.*?)(?=\n\d+\.|\Z)', answer, re.DOTALL)
        
        # Extract attributes
        attributes = entry.get('attributes', [])
        
        # Extract ability types
        ability_types = entry.get('ability_types', [])
        
        # Create new entries for each subquestion
        for i, subq in enumerate(subquestions):
            if i >= len(answers):
                print(f"Warning: Missing answer for subquestion {i+1} in entry")
                continue
                
            # Create a new entry with a single question
            new_entry = {
                'timeseries': copy.deepcopy(timeseries),
                'cols': copy.deepcopy(cols),
                'question': f"{question_prefix} {subq.strip()}",
                'answer': answers[i].strip(),
            }
            
            # Handle attributes - try to match the attributes to the subquestion
            if i < len(attributes):
                new_entry['attributes'] = [attributes[i]]
            else:
                # If we can't match attributes directly, use empty list
                new_entry['attributes'] = []
            
            # Handle ability types - use the corresponding ability type if available
            if i < len(ability_types):
                new_entry['ability_types'] = [ability_types[i]]
            else:
                # If we can't match ability types directly, use the whole list
                new_entry['ability_types'] = copy.deepcopy(ability_types)
            
            split_dataset.append(new_entry)
    
    return split_dataset


def filter_dataset(dataset):
    """
    Filter the dataset to keep only entries with ability_types containing 'causal', 'deductive', or 'inductive'.
    Also adds a standard closing line to all questions.
    
    Args:
        dataset: The dataset to filter
    
    Returns:
        Filtered dataset and statistics
    """
    print("Filtering dataset based on ability types...")
    
    # Initialize counters
    total_entries = len(dataset)
    filtered_entries = 0
    causal_count = 0
    deductive_count = 0
    inductive_count = 0
    
    # Filter the dataset
    filtered_dataset = []
    
    for entry in dataset:
        ability_types = entry.get('ability_types', [])
        
        # Check if any ability type contains 'causal', 'deductive', or 'inductive'
        has_target_ability = False
        has_causal = False
        has_deductive = False
        has_inductive = False
        
        for ability in ability_types:
            if isinstance(ability, str):
                if 'causal' in ability:
                    has_causal = True
                if 'deductive' in ability:
                    has_deductive = True
                if 'inductive' in ability:
                    has_inductive = True
        
        has_target_ability = has_causal or has_deductive or has_inductive
        
        if has_target_ability:
            # Add instruction to the end of the question for causal type
            if has_causal:
                question = entry.get('question', '')
                if not question.endswith("In your answer, start by stating your chosen option and then provide your explanation in a separate sentence."):
                    entry['question'] = question + "\nIn your answer, start by stating your chosen option and then provide your explanation in a separate sentence."
            
            filtered_dataset.append(entry)
            filtered_entries += 1
            
            # Update counters
            if has_causal:
                causal_count += 1
            if has_deductive:
                deductive_count += 1
            if has_inductive:
                inductive_count += 1
    
    stats = {
        'total': total_entries,
        'filtered': filtered_entries,
        'causal': causal_count,
        'deductive': deductive_count,
        'inductive': inductive_count
    }
    
    return filtered_dataset, stats


def merge_datasets(dataset_a, mcq2_dataset):
    """
    Merge dataset_a with MCQ2 dataset.
    
    Args:
        dataset_a: Processed dataset_a
        mcq2_dataset: MCQ2 dataset
    
    Returns:
        Merged dataset
    """
    print("Merging dataset_a with MCQ2 dataset...")
    
    # Simple append - MCQ2 entries after dataset_a entries
    merged_dataset = dataset_a + mcq2_dataset
    
    return merged_dataset


def load_jsonl_data(file_path, limit=None):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            data.append(json.loads(line))
    return data


def create_evaluation_entry(entry, idx):
    """
    Create an evaluation entry for MCQ2 data.
    """
    uuid = entry.get("uuid", f"mcq2_{idx}")
    
    # Extract the formatted question parts
    description = entry.get("description", "")
    question = entry.get("question", "")
    options = entry.get("options", [])
    
    # Extract just the question part from the original question if needed
    clean_question = question.split("Options:")[0] if "Options:" in question else question
    clean_question = clean_question.split("Now, based on")[0] if "Now, based on" in clean_question else clean_question
    
    # Format the options as a string
    options_str = ", ".join([f"\"{option}\"" for option in options])
    
    # Assemble the formatted question (for evaluation, don't include tags instruction)
    formatted_question = f"""You are a time series analysis expert. {description} {clean_question.strip()} Choose from: [{options_str}]."""
    
    # Get correct answer
    correct_answer_index = entry.get("answer_index", 0)
    selected_option = options[correct_answer_index] if options and 0 <= correct_answer_index < len(options) else "Unknown"
    
    # Extract time series data
    original_series = entry.get("series", [])
    new_series = entry.get("new_series", [])
    
    # Column names
    column_names = ["original series", "updated series"]
    
    # Create evaluation format entry
    result = {
        "timeseries": [original_series, new_series],
        "cols": column_names,
        "question": formatted_question,
        "answer": selected_option,
        "attributes": [selected_option],
        "ability_types": ["MCQ2"],
        "id": uuid
    }
    
    return result


def sample_mcq2_data(mcq2_path, sample_size=100, seed=42):
    """
    Sample entries from the MCQ2_TS dataset with a fixed seed.
    
    Args:
        mcq2_path: Path to the MCQ2_TS.jsonl file
        sample_size: Number of entries to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled entries in the proper format for evaluation
    """
    print(f"Sampling {sample_size} entries from MCQ2 dataset with seed {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load original dataset
    print(f"Loading MCQ2 data from {mcq2_path}")
    data = load_jsonl_data(mcq2_path)
    print(f"Loaded {len(data)} entries from the original MCQ2 dataset")
    
    # Sample entries with the fixed random seed
    sample_size = min(sample_size, len(data))
    print(f"Sampling {sample_size} entries with seed {seed}")
    sample_indices = random.sample(range(len(data)), sample_size)
    
    # Create formatted entries from sampled data
    print(f"Creating formatted entries...")
    sampled_entries = []
    for i, idx in enumerate(tqdm(sample_indices, desc="Processing MCQ2 entries")):
        entry = data[idx]
        eval_entry = create_evaluation_entry(entry, idx)
        sampled_entries.append(eval_entry)
    
    print(f"Sampled {len(sampled_entries)} entries from {len(data)} total entries")
    
    # Print some statistics
    if sampled_entries:
        num_cols = [len(entry.get("cols", [])) for entry in sampled_entries]
        avg_cols = sum(num_cols) / len(num_cols)
        print(f"\nMCQ2 Sample Statistics:")
        print(f"- Average number of columns: {avg_cols:.2f}")
        print(f"- Range of columns: {min(num_cols)} to {max(num_cols)}")
        print(f"- First sampled index: {sample_indices[0]}")
        print(f"- Last sampled index: {sample_indices[-1]}")
    
    return sampled_entries


def process_datasets(dataset_a_path, mcq2_source_path, output_dir, mcq2_sample_size=100, mcq2_seed=42):
    """
    Main function to process the datasets.
    
    Args:
        dataset_a_path: Path to dataset_a.json
        mcq2_path: Path to MCQ2 dataset
        output_dir: Directory to save processed datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load dataset_a
    print(f"\n=== Processing dataset_a ===\n")
    print(f"Loading dataset_a from {dataset_a_path}")
    with open(dataset_a_path, 'r') as f:
        dataset_a = json.load(f)
    print(f"Loaded {len(dataset_a)} entries from dataset_a")
    
    # Step 2: Split dataset_a
    split_a = split_dataset(dataset_a)
    print(f"Split dataset_a into {len(split_a)} entries")
    
    # Save split dataset
    split_path = os.path.join(output_dir, "dataset_a_split.json")
    with open(split_path, 'w') as f:
        json.dump(split_a, f, indent=4)
    print(f"Split dataset saved to {split_path}")
    
    # Step 3: Filter split dataset
    filtered_a, filter_stats = filter_dataset(split_a)
    print(f"Filtered dataset statistics:")
    print(f"  Original: {filter_stats['total']} entries")
    print(f"  Filtered: {filter_stats['filtered']} entries")
    print(f"  Causal: {filter_stats['causal']} entries")
    print(f"  Deductive: {filter_stats['deductive']} entries")
    print(f"  Inductive: {filter_stats['inductive']} entries")
    
    # Save filtered dataset
    filtered_path = os.path.join(output_dir, "dataset_a_split_filtered.json")
    with open(filtered_path, 'w') as f:
        json.dump(filtered_a, f, indent=4)
    print(f"Filtered dataset saved to {filtered_path}")
    
    # Step 4: Sample MCQ2 dataset
    print(f"\n=== Processing MCQ2 dataset ===\n")
    mcq2_dataset = sample_mcq2_data(mcq2_source_path, mcq2_sample_size, mcq2_seed)
    
    # Save sampled MCQ2 dataset
    mcq2_path = os.path.join(output_dir, "mcq2_qa_eval_100.json")
    with open(mcq2_path, 'w') as f:
        json.dump(mcq2_dataset, f, indent=2)
    print(f"Sampled MCQ2 dataset saved to {mcq2_path}")
    
    # Step 5: Merge datasets
    print(f"\n=== Merging datasets ===\n")
    merged_dataset = merge_datasets(filtered_a, mcq2_dataset)
    print(f"Merged dataset contains {len(merged_dataset)} entries")
    
    # Save merged dataset
    merged_path = os.path.join(output_dir, "dataset_a_with_mcq2.json")
    with open(merged_path, 'w') as f:
        json.dump(merged_dataset, f, indent=2)
    print(f"Merged dataset saved to {merged_path}")


def main():
    parser = argparse.ArgumentParser(description="Process TSR knowledge injection datasets")
    parser.add_argument("--dataset_a", type=str, default="./dataset_a.json",
                      help="Path to the dataset_a.json file")
    parser.add_argument("--mcq2_source", type=str, default="./MCQ_2_TS.jsonl",
                      help="Path to the MCQ2_TS.jsonl source file")
    parser.add_argument("--output_dir", type=str, default="./processed",
                      help="Directory to save processed datasets")
    parser.add_argument("--mcq2_sample_size", type=int, default=100,
                      help="Number of entries to sample from MCQ2")
    parser.add_argument("--mcq2_seed", type=int, default=42,
                      help="Random seed for reproducibility of MCQ2 sampling")
    
    args = parser.parse_args()
    
    process_datasets(
        args.dataset_a, 
        args.mcq2_source, 
        args.output_dir,
        args.mcq2_sample_size,
        args.mcq2_seed
    )


if __name__ == "__main__":
    main()