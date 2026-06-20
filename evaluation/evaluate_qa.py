import json
import os
import numpy as np
import re
from typing import *
from loguru import logger
from tqdm import tqdm
import traceback
from multiprocessing import Pool
from evaluation.my_ragas.score import calculate_ragas_score


def split_sentences(text):
    """
    Split text into sentences while preserving decimal points and abbreviations.
    """
    abbreviations = ['max.', 'eg.', 'Mrs.', 'Dr.', 'Mr.']
    
    for abbr in abbreviations:
        escaped_abbr = re.escape(abbr)
        text = re.sub(escaped_abbr, abbr.replace('.', '<DOT>'), text)
    
    # Protect decimal points in numbers like 2.75 by temporarily replacing them
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
    
    pattern = r'[.!?。！？,;，；](?!\d)'
    sentences = re.split(pattern, text)
    
    # Restore decimal points and abbreviation dots
    sentences = [s.strip().replace('<DOT>', '.').replace('<DECIMAL>', '.') for s in sentences if s.strip()]
    
    return sentences


def split_period_sentences(text):
    """
    Split text into sentences using only periods as delimiters, while preserving decimal points and abbreviations.
    """
    abbreviations = ['max.', 'eg.', 'Mrs.', 'Dr.', 'Mr.']
    
    for abbr in abbreviations:
        escaped_abbr = re.escape(abbr)
        text = re.sub(escaped_abbr, abbr.replace('.', '<DOT>'), text)
    
    # Protect decimal points in numbers like 2.75 by temporarily replacing them
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
    
    pattern = r'[.。](?!\d)'
    sentences = re.split(pattern, text)
    
    # Restore decimal points and abbreviation dots
    sentences = [s.strip().replace('<DOT>', '.').replace('<DECIMAL>', '.') for s in sentences if s.strip()]
    
    return sentences


def match_metric_name(metric: str, sentence: str) -> bool:
    """
    Check if a metric name appears in a sentence after normalizing both strings.
    Preserves numbers for MCQ numeric options.
    """
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    sentence = re.sub(pattern, '', sentence).lower()
    metric = re.sub(pattern, '', metric).lower()

    return metric in sentence


def evaluate_trend(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for trend-related questions.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'steady' in attribute['type']:
        if 'steady' in sentences[0]:
            cate_correct = True
    elif 'decrease' in attribute['type']:
        if 'decreas' in sentences[0].lower():
            cate_correct = True
    elif 'increase' in attribute['type']:
        if 'increas' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    # Check start point
    for sentence in sentences:
        float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
        if float_numbers is None or len(float_numbers) == 0:
            continue
        if 'start' in sentence:
            if abs(attribute['start']) < 0.5:
                if abs(float_numbers[0]) < 0.5:
                    num_correct.append(1.0)
                else:
                    num_correct.append(0.0)
            else:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['start']) / abs(attribute['start']))))
            break
    else:
        num_correct.append(0.0)

    # Check amplitude
    if attribute['type'] != 'keep steady':
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'change value' in sentence or 'from left to right' in sentence:
                if abs(attribute['amplitude']) < 0.5:
                    if abs(float_numbers[0]) < 0.5:
                        num_correct.append(1.0)
                    else:
                        num_correct.append(0.0)
                else:
                    num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['amplitude']) / abs(attribute['amplitude']))))
                break
        else:
            num_correct.append(0.0)

    return [cate_correct], num_correct, [], []


def evaluate_season(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for seasonality-related questions.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'no' in attribute['type']:
        if 'no periodic' in sentences[0].lower():
            cate_correct = True
    else:
        if 'no' not in sentences[0].lower() and 'periodic' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    if attribute['type'] != 'no periodic fluctuation':
        # Check period
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'each period' in sentence:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['period']) / abs(attribute['period']))))
                break
        else:
            num_correct.append(0.0)

        # Check amplitude
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'amplitude' in sentence:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['amplitude']) / abs(attribute['amplitude']))))
                break
        else:
            num_correct.append(0.0)
    else:
        num_correct = []

    return [cate_correct], num_correct, [], []


def evaluate_noise(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for noise-related questions.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'almost no' in attribute['type']:
        if 'no noise' in sentences[0].lower():
            cate_correct = True
    else:
        if 'noisy' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    # Check noise standard deviation
    if 'noisy' in attribute['type']:
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'standard' in sentence.lower() or 'std' in sentence.lower():
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['std']) / abs(attribute['std']))))
                break
        else:
            num_correct.append(0.0)

    return [cate_correct], num_correct, [], []


def evaluate_local(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for local feature questions.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = []
    num_correct = []

    # Split into facts
    for feat in attribute:
        matched_flag = False
        pos_numerical = 0.0
        amp_numerical = 0.0
        for fact in re.split(r'[;；]', answer):
            sentences = re.split(r'[，。,;；]', fact)
            if type(feat['type']) == str:
                feat['type'] = [feat['type']]
            if any(i in sentences[0].lower() for i in feat['type']):
                # Check period and amplitude
                for sentence in sentences:
                    float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
                    if float_numbers is None or len(float_numbers) == 0:
                        continue
                    if 'position' in sentence.lower() or 'around point' in sentence.lower():
                        if abs(float_numbers[0] - feat['position']) > 64:
                            continue
                        pos_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['position']) / abs(feat['position'])))
                        matched_flag = True
                    if matched_flag and 'amplitude' in sentence.lower():
                        amp_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['amplitude']) / abs(feat['amplitude'])))
                if matched_flag:
                    break
        cate_correct.append(matched_flag)
        num_correct.append(pos_numerical)
        num_correct.append(amp_numerical)

    return cate_correct, num_correct, [], []


def evaluate_local_inductive(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for local feature questions with inductive reasoning.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = []
    num_correct = []
    reason_correct = []
    reason_details = []

    # Split into facts
    for feat in attribute:
        matched_flag = False
        pos_numerical = 0.0
        amp_numerical = 0.0
        reason_score = 0.0
        cur_detail = {}
        for fact in re.split(r'[;；]', answer):
            sentences = re.split(r'[，。,;；]', fact)
            if type(feat['type']) == str:
                feat['type'] = [feat['type']]
            if any(i in sentences[0].lower() for i in feat['type']):
                # Check period and amplitude
                for sentence in sentences:
                    float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
                    if float_numbers is None or len(float_numbers) == 0:
                        continue
                    if 'position' in sentence.lower() or 'around point' in sentence.lower():
                        if abs(float_numbers[0] - feat['position']) > 64:
                            continue
                        pos_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['position']) / abs(feat['position'])))
                        matched_flag = True
                    if matched_flag and 'amplitude' in sentence.lower():
                        amp_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['amplitude']) / abs(feat['amplitude'])))
                if matched_flag:
                    # Evaluate the inductive reasoning
                    reason_score, cur_detail = calculate_ragas_score(
                        question='Please analyze the physical meaning of this local fluctuation in one sentence.',
                        response=split_period_sentences(fact)[-1],
                        label=feat['explain']
                    )
                    cur_detail.update({
                        'label': feat['explain'],
                        'response': split_period_sentences(fact)[-1]
                    })
                    break
        cate_correct.append(matched_flag)
        num_correct.append(pos_numerical)
        num_correct.append(amp_numerical)
        reason_correct.append(reason_score)
        reason_details.append(cur_detail)

    return cate_correct, num_correct, reason_correct, reason_details


def evaluate_shape_correlation_inductive(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for shape correlation questions with inductive reasoning.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [False], [], [0.0], [{}]

    if attribute['label']:
        if 'yes' in sentences[0].lower():
            cate_correct = True
    else:
        if 'no' in sentences[0].lower():
            cate_correct = True

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they are correlated/no correlated considering their physical meaning in one sentence.',
                        response=sentences[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]


def evaluate_local_correlation_inductive(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for local correlation questions with inductive reasoning.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = False
    sentences = split_period_sentences(answer)

    # If there's nothing at all, return early
    if not sentences:
        logger.debug("evaluate_local_correlation_inductive: no sentences parsed from answer")
        return [False], [], [0.0], [{}]

    # Prepare for the case where we need sentences[1]
    has_second = len(sentences) > 1

    if attribute.get('label', False):
        # Expect a "yes" in the first sentence to proceed
        if 'yes' in sentences[0].lower():
            if not has_second:
                logger.debug("evaluate_local_correlation_inductive: expected a second sentence for fact extraction but got only one")
            else:
                # Check correlation type only when we have a second sentence
                label_cols = set(map(tuple, attribute.get('pair', [])))
                answer_cols = set()

                # Split into facts safely
                for fact in sentences[1].split(';'):
                    items = [s.strip() for s in fact.split(',')]
                    if len(items) == 2:
                        metric, corr_type = items
                        for col in cols:
                            if match_metric_name(col, metric):
                                answer_cols.add((col, corr_type))

                if label_cols == answer_cols:
                    cate_correct = True

    else:
        # Negative case: first sentence should contain "no"
        if 'no' in sentences[0].lower():
            cate_correct = True

    # For the RAG-as-a-service score, always pick the *last* sentence we have
    explanation = sentences[-1] if sentences else ""
    try:
        reason_correct, reason_detail = calculate_ragas_score(
            question="Explain why they are correlated/not correlated considering their physical meaning in one sentence.",
            response=explanation,
            label=attribute.get('explain', "")
        )
    except Exception as e:
        logger.error(f"evaluate_local_correlation_inductive: calculate_ragas_score failed: {e}")
        reason_correct, reason_detail = 0.0, {}

    return [cate_correct], [], [reason_correct], [reason_detail]


def evaluate_shape_cluster_inductive(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for shape cluster questions with inductive reasoning.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = 0.0
    num_correct = []

    label_cols = set(attribute['cols'])
    answer_cols = set()

    sentences = split_period_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [], [0.0], [{}]

    # Split into facts
    for fact in split_period_sentences(answer)[0].split(','):
        fact = fact.strip()
        for col in cols:
            if match_metric_name(col, fact):
                answer_cols.add(col)

    # Calculate f1-score for label and answer
    tp = len(label_cols & answer_cols)
    fp = len(answer_cols - label_cols)
    fn = len(label_cols - answer_cols)
    if tp + fp + fn > 0:
        cate_correct = 2 * tp / (2 * tp + fp + fn)

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they have similar overall trend considering their physical meaning in one sentence.',
                        response=split_period_sentences(answer)[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]


def evaluate_local_cluster_inductive(answer: str, attribute: dict, cols: List[str]):
    """
    Evaluate answers for local cluster questions with inductive reasoning.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    cate_correct = 0.0
    num_correct = []

    label_cols = set(zip(attribute['cols'], [i[1] for i in attribute['col_idx']]))
    answer_cols = set()

    sentences = split_period_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [], [0.0], [{}]

    # Split into facts
    for fact in split_period_sentences(answer)[0].split(';'):
        items = fact.strip().split(',')
        if len(items) == 2:
            for col in cols:
                if match_metric_name(col, items[0].strip()):
                    answer_cols.add((col, items[1].strip()))

    # Calculate f1-score for label and answer
    tp = len(label_cols & answer_cols)
    fp = len(answer_cols - label_cols)
    fn = len(label_cols - answer_cols)
    if tp + fp + fn > 0:
        cate_correct = 2 * tp / (2 * tp + fp + fn)

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they have similar local fluctuations considering their physical meaning in one sentence.',
                        response=split_period_sentences(answer)[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]


def evaluate_deductive(answer, attribute, cols):
    """
    Evaluate a yes/no (True/False) deductive question, falling back to RAGAS scoring otherwise.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    try:
        labels = split_sentences(attribute)
    except Exception as e:
        logger.error(f"evaluate_deductive: Error splitting attribute {attribute!r} into labels: {e}")
        labels = []

    try:
        sentences = split_sentences(answer)
    except Exception as e:
        logger.error(f"evaluate_deductive: Error splitting answer {answer!r} into sentences: {e}")
        sentences = []

    cur_reason_correct = 0.0
    ragas_detail = {}

    # Normalize set of boolean labels
    bool_labels = {'yes', 'no', 'true', 'false'}

    if labels and labels[0].lower().strip().rstrip('.,') in bool_labels:
        label0 = labels[0].lower().strip().rstrip('.,')
        if sentences:
            resp0 = sentences[0].lower().strip().rstrip('.,')
            cur_reason_correct = 1.0 if resp0 == label0 else 0.0
            ragas_detail = {"label": label0, "response": resp0}
        else:
            logger.warning(f"evaluate_deductive: No sentences to compare against label '{label0}', unparsable answer: {answer!r}")
            ragas_detail = {"label": label0, "response": None}
    else:
        # fallback to RAGAS
        logger.info(f"evaluate_deductive: Falling back to RAGAS for answer {answer!r}")
        try:
            ragas_score, detail = calculate_ragas_score(
                question="According to the previous information, please answer True or False and explain it in detail.",
                response=answer,
                label=attribute
            )
            cur_reason_correct = ragas_score
            ragas_detail = detail
        except Exception as e:
            logger.error(f"evaluate_deductive: Error in calculate_ragas_score: {e}")
            cur_reason_correct = 0.0
            ragas_detail = {}

    return [], [], [cur_reason_correct], [ragas_detail]


def evaluate_inductive(answer: str, attribute: str, cols: List[str]):
    """
    Evaluate an open‐ended (inductive) question by running RAGAS on the
    model's full answer.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    try:
        # Prompt the RAGAS scorer to compare the full answer against the label
        ragas_score, ragas_detail = calculate_ragas_score(
            question="According to the data and the question, please provide a correct answer and explanation.",
            response=answer,
            label=attribute
        )
    except Exception as e:
        logger.error(f"evaluate_inductive: RAGAS scoring failed: {e}")
        ragas_score, ragas_detail = 0.0, {}

    # Return only a "reason" score + detail, leaving the categorical/numerical slots empty
    return [], [], [ragas_score], [ragas_detail]


def evaluate_inductive_rme(answer: str, attribute: str, cols: List[str]):
    """
    Evaluate an open‐ended RME (inductive) question by running RAGAS on the
    model's full answer.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    try:
        # Prompt the RAGAS scorer to compare the full answer against the label
        ragas_score, ragas_detail = calculate_ragas_score(
            question="According to the data and the question, please provide a correct answer and explanation.",
            response=answer,
            label=attribute
        )
    except Exception as e:
        logger.error(f"evaluate_inductive_rme: RAGAS scoring failed: {e}")
        ragas_score, ragas_detail = 0.0, {}

    # Return only a "reason" score + detail, leaving the categorical/numerical slots empty
    return [], [], [ragas_score], [ragas_detail]


def evaluate_causal(answer: str, attribute: str, cols: List[str]):
    """
    Evaluate answers for causal questions.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    # 1) Basic input validation
    if not answer:
        logger.warning(f"evaluate_causal: empty answer for attribute={attribute!r}, cols={cols}")
        return [], [], [0.0], [{'label': attribute, 'response': ''}]
    
    if not attribute:
        logger.warning(f"evaluate_causal: empty attribute")
        return [], [], [0.0], [{'label': '', 'response': answer}]
    
    # 2) Extract first part of both strings for comparison
    # Use regex to extract the content before the first period that's not part of a number
    def extract_first_part(text):
        # Find first period not followed by a digit (to avoid splitting decimal numbers)
        match = re.search(r'(?<![0-9])[.!?。！？](?![0-9])', text)
        if match:
            return text[:match.start()].strip()
        return text.strip()
    
    label = extract_first_part(attribute).lower()
    answer_choice = extract_first_part(answer).lower()
    
    # 3) Special handling for numeric MCQ options
    # First check for direct match
    if label == answer_choice:
        return [], [], [1.0], [{'label': label, 'response': answer_choice}]
    
    # 4) Check for numeric content
    label_has_nums = bool(re.search(r'\d', label))
    answer_has_nums = bool(re.search(r'\d', answer_choice))
    
    if label_has_nums and answer_has_nums:
        # Extract all numbers for comparison
        label_nums = re.findall(r'[\d.-]+', label)
        answer_nums = re.findall(r'[\d.-]+', answer_choice)
        
        # Case 1: Check if all label numbers are in the answer
        if label_nums and all(ln in answer_choice for ln in label_nums):
            return [], [], [1.0], [{'label': label, 'response': answer_choice}]
        
        # Case 2: Special handling for truncated decimals like "7-1.9" vs "1.7-1.9"
        # This checks if there's a possible truncated decimal match
        for lnum in label_nums:
            if '.' in lnum:
                # For each decimal number in label, check if a truncated version exists in answer
                integer_part = lnum.split('.')[0]
                decimal_part = lnum.split('.')[1]
                
                # Check for truncation pattern: "1.7" appearing as "7" in response
                truncated_pattern = decimal_part
                if any(truncated_pattern in anum or anum in truncated_pattern for anum in answer_nums):
                    # Look for the specific truncation pattern in the original strings
                    if f"{integer_part}.{decimal_part}" in label and decimal_part in answer_choice:
                        return [], [], [1.0], [{'label': label, 'response': answer_choice, 'note': 'truncated decimal match'}]
        
        # Case 3: Fall back to the standard metric name match
        reason_correct = 1.0 if match_metric_name(label, answer_choice) else 0.0
    else:
        # Standard non-numeric comparison
        reason_correct = 1.0 if match_metric_name(label, answer_choice) else 0.0
    
    return [], [], [reason_correct], [{'label': label, 'response': answer_choice}]


def evaluate_causal_rme(answer: str, attribute: str, cols: List[str]):
    """
    Evaluate RME-style causal questions where there may be one or two root causes.
    
    Args:
        answer: Model's response
        attribute: Ground truth attribute
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    # 1) split into sentences and guard
    answer_sents = split_sentences(answer)
    if not answer_sents:
        logger.warning(f"evaluate_causal_rme: empty answer for attribute={attribute!r}")
        return [], [], [0.0], [{'labels': [], 'responses': []}]
    
    label_sents = split_sentences(attribute)
    if not label_sents:
        logger.warning(f"evaluate_causal_rme: empty attribute split for {attribute!r}")
        return [], [], [0.0], [{'labels': [], 'responses': []}]

    # 2) take the first sentence from each
    answer_str = answer_sents[0].lower().strip()
    label_str = label_sents[0].lower().strip()

    # 3) split on commas or the word "and"
    split_pattern = r'\s*(?:,|\band\b)\s*'
    answer_items = [frag.strip() for frag in re.split(split_pattern, answer_str) if frag.strip()]
    label_items = [frag.strip() for frag in re.split(split_pattern, label_str) if frag.strip()]

    # 4) one‐match mechanism
    match_found = any(
        match_metric_name(label_frag, ans_frag)
        for label_frag in label_items
        for ans_frag in answer_items
    )
    reason_correct = 1.0 if match_found else 0.0

    # 5) return in same format as evaluate_*: no categorical/numerical, only reason
    detail = {
        'labels': label_items,
        'responses': answer_items
    }
    return [], [], [reason_correct], [detail]


def evaluate_MCQ2(answer: str, attribute: str, cols: List[str]):
    """
    Evaluate MCQ2 questions with a robust matching procedure.
    
    Args:
        answer: Model's response
        attribute: Ground truth option/answer
        cols: Column names
        
    Returns:
        Tuple of (categorical scores, numerical scores, reason scores, reason details)
    """
    # 1) Basic input validation
    if not answer:
        logger.warning(f"evaluate_MCQ2: empty answer for attribute={attribute!r}")
        return [], [], [0.0], [{'label': attribute, 'response': ''}]
    
    if not attribute:
        logger.warning(f"evaluate_MCQ2: empty attribute")
        return [], [], [0.0], [{'label': '', 'response': answer}]
    
    # 2) Extract first part of both strings for comparison
    def extract_first_part(text):
        # Find first period not followed by a digit (to avoid splitting decimal numbers)
        match = re.search(r'(?<![0-9])[.!?。！？](?![0-9])', text)
        if match:
            return text[:match.start()].strip()
        return text.strip()
    
    label = extract_first_part(attribute).lower()
    answer_choice = extract_first_part(answer).lower()
    
    # 3) Special handling for numeric MCQ options
    # First check for direct match
    if label == answer_choice:
        return [], [], [1.0], [{'label': label, 'response': answer_choice}]
    
    # 4) Check for numeric content
    label_has_nums = bool(re.search(r'\d', label))
    answer_has_nums = bool(re.search(r'\d', answer_choice))
    
    if label_has_nums and answer_has_nums:
        # Extract all numbers for comparison
        label_nums = re.findall(r'[\d.-]+', label)
        answer_nums = re.findall(r'[\d.-]+', answer_choice)
        
        # Case 1: Check if all label numbers are in the answer
        if label_nums and all(ln in answer_choice for ln in label_nums):
            return [], [], [1.0], [{'label': label, 'response': answer_choice}]
        
        # Case 2: Special handling for truncated decimals like "7-1.9" vs "1.7-1.9"
        for lnum in label_nums:
            if '.' in lnum:
                # For each decimal number in label, check if a truncated version exists in answer
                integer_part = lnum.split('.')[0]
                decimal_part = lnum.split('.')[1]
                
                # Check for truncation pattern: "1.7" appearing as "7" in response
                truncated_pattern = decimal_part
                if any(truncated_pattern in anum or anum in truncated_pattern for anum in answer_nums):
                    # Look for the specific truncation pattern in the original strings
                    if f"{integer_part}.{decimal_part}" in label and decimal_part in answer_choice:
                        return [], [], [1.0], [{'label': label, 'response': answer_choice, 'note': 'truncated decimal match'}]
    
    # 5) Fall back to match_metric_name which removes special characters and does substring matching
    reason_correct = 1.0 if match_metric_name(label, answer_choice) else 0.0
    
    # 6) Final check for option letters (A, B, C, D, etc.) that might appear in the answer
    if reason_correct == 0.0:
        # Extract option letters from label and answer
        label_options = re.findall(r'\b([A-D])\b', label)
        answer_options = re.findall(r'\b([A-D])\b', answer_choice)
        
        if label_options and answer_options and any(lo == ao for lo in label_options for ao in answer_options):
            return [], [], [1.0], [{'label': label, 'response': answer_choice, 'note': 'option letter match'}]
    
    return [], [], [reason_correct], [{'label': label, 'response': answer_choice}]


def ability_type_to_func(ability_type: str):
    """
    Convert ability type string to evaluation function.
    
    Args:
        ability_type: String identifying the ability type
        
    Returns:
        Evaluation function
    """
    return eval("evaluate_" + ability_type.replace('-', '_'))


def evaluate_qa(answer: str, sample: dict):
    """
    Evaluate model's answer against ground truth.
    
    Args:
        answer: Model's response
        sample: Sample dictionary containing question, attributes, ability_types
        
    Returns:
        Dictionary mapping ability types to evaluation results
    """
    # Extract numbered answers from the model's response
    raw = re.findall(r'^\s*(\d+)\.\s*(.+)$', answer, re.MULTILINE)
    answer_list = [text.strip() for (_, text) in raw]
    
    num_questions = len(sample['attributes'])
    ability_types = sample['ability_types']
    matched_cnt = min(len(answer_list), num_questions)

    # Try to match answers when there's a mismatch
    if matched_cnt < num_questions and matched_cnt == 1:
        idx_pos = []
        for idx in range(1, num_questions + 1):
            sub_str = f"{idx}. "
            if sub_str in answer:
                idx_pos.append(answer.index(sub_str))
            else:
                break

        if len(idx_pos) == num_questions:
            # Successfully matched by position
            idx_pos.append(len(answer))
            answer_list = [answer[idx_pos[i] + len(f"{i+1}. "):idx_pos[i + 1]] for i in range(num_questions)]
            matched_cnt = min(len(answer_list), num_questions)
            logger.debug(f"Answer list after position matching: {answer_list}")
    elif num_questions == 1 and matched_cnt == 1 and len(answer_list[0].strip()) == 0:
        # Empty numbered answer but single question - use full answer
        answer_list[0] = answer.rstrip()
        logger.debug(f"Using full answer: {answer_list[0]}")

    # If there's only one question and no matches found, use the entire answer
    if len(answer_list) == 0 and num_questions == 1:
        answer_list = [answer.strip()]
        matched_cnt = 1

    result = {}

    # Evaluate each question type using the appropriate evaluation function
    for i in range(len(ability_types)):
        ability = ability_types[i]
        evaluate_func = ability_type_to_func(ability)
        cur_answer = answer_list[i] if i < matched_cnt else ""
        cate_correct, num_correct, reason_correct, reason_detail = evaluate_func(cur_answer, sample['attributes'][i], sample['cols'])

        if ability in result:
            # Extend current result to existing
            cate_correct = result[ability][0] + cate_correct
            num_correct = result[ability][1] + num_correct
            reason_correct = result[ability][2] + reason_correct
            reason_detail = result[ability][3] + reason_detail
        result[ability] = (cate_correct, num_correct, reason_correct, reason_detail)  

    return result


def accumulate_result(result: dict, ability: str,
                      cate_c: int, num_c: int,
                      reason_c: int, reason_det):
    """
    Safely add (cate_c, num_c, reason_c, reason_det) into result[ability].
    reason_det may be a list or a dict.
    
    Args:
        result: Existing results dictionary
        ability: Ability type to accumulate
        cate_c: Categorical score to add
        num_c: Numerical score to add
        reason_c: Reason score to add
        reason_det: Reason details to add
    """
    # Default empty state: counts = 0, reason_detail = same type as incoming
    default_detail = [] if isinstance(reason_det, list) else {}
    prev_cate, prev_num, prev_reason, prev_detail = result.get(
        ability,
        (0, 0, 0, default_detail)
    )

    # Sum counts
    new_cate = prev_cate + (cate_c if isinstance(cate_c, int) else 0)
    new_num = prev_num + (num_c if isinstance(num_c, int) else 0)
    new_reason = prev_reason + (reason_c if isinstance(reason_c, int) else 0)

    # Merge reason_detail
    if isinstance(prev_detail, list) and isinstance(reason_det, list):
        new_detail = prev_detail + reason_det
    elif isinstance(prev_detail, dict) and isinstance(reason_det, dict):
        new_detail = {**prev_detail, **reason_det}
    else:
        # fallback: overwrite or wrap single item
        new_detail = reason_det if reason_det else prev_detail

    result[ability] = (new_cate, new_num, new_reason, new_detail)


def process_sample(args):
    """
    Process a single sample for evaluation.
    
    Args:
        args: Tuple of (idx, sample, generated_answer)
        
    Returns:
        Dictionary with evaluation results
    """
    idx, sample, generated_answer = args

    app_domain = sample.get('application_domain')
    task_type = sample.get('task_type')

    # Find the model's entry for this idx
    pos = next((i for i, item in enumerate(generated_answer) if item['idx'] == idx), -1)
    if pos >= 0:
        ga_item = generated_answer[pos]
        answer = ga_item.get('response', "")
        thought = ga_item.get('thought', None)
    else:
        answer = ""
        thought = None

    # Ground-truth label and question
    label = sample['answer']
    question = sample['question']

    # Do the per-ability evaluation
    evaluation_result = evaluate_qa(answer, sample)

    return {
        'idx': idx,
        'application_domain': app_domain,
        'task_type': task_type,
        'question': question,
        'label': label,
        'thought': thought,
        'response': answer,
        'evaluation': evaluation_result
    }


def evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=8):
    """
    Evaluate a batch of QA results.
    
    Args:
        dataset: Dataset containing ground truth
        generated_answer: Model's answers
        EXP: Experiment name (for saving results)
        num_workers: Number of parallel workers
        
    Returns:
        Evaluation results
    """
    detailed_result = []
    ability_result = {'categorical': {}, 'numerical': {}, 'reason': {}}
    overall_result = {'categorical': [], 'numerical': [], 'reason': []}

    total = len(dataset)
    # 1) Identify which indices had errors
    error_count = sum(1 for item in generated_answer if '<<ERROR' in item.get('response', ''))
    error_ratio = error_count / total

    # 2) Build list of valid indices
    valid_idxs = [item['idx'] for item in generated_answer if '<<ERROR' not in item.get('response', '')]

    logger.info(f"Total samples: {total}, error responses: {error_count} ({error_ratio:.2%})")
    logger.info("Start evaluation on valid samples...")

    # 3) Only process valid samples
    work_items = [(idx, dataset[idx], generated_answer) for idx in valid_idxs]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, work_items), total=len(work_items)))

    for result in results:
        if result is None:
            continue

        detailed_result.append(result)
        evaluation_result = result['evaluation']

        if evaluation_result is None:
            logger.warning(f"evaluate_qa returned None for sample, skipping.")
            continue

        # Parse results
        for ability, (cate_correct, num_correct, reason_correct, reason_detail) in evaluation_result.items():
            ability_result['categorical'].setdefault(ability, [])
            ability_result['numerical'].setdefault(ability, [])
            ability_result['reason'].setdefault(ability, [])

            ability_result['categorical'][ability].extend(cate_correct)
            ability_result['numerical'][ability].extend(num_correct)
            ability_result['reason'][ability].extend(reason_correct)

            overall_result['categorical'].extend(cate_correct)
            overall_result['numerical'].extend(num_correct)
            overall_result['reason'].extend(reason_correct)

    # Calculate tokens
    total_tokens = 0
    for item in generated_answer:
        total_tokens += item.get('num_tokens', 0)

    # Log evaluation results
    logger.info(f"[RESULT] -----------------------------------------------------------------")
    logger.info(f"[RESULT] Experiment: {EXP}")
    logger.info(f"[RESULT] Total: {len(dataset)}, Success Evaluation: {len(detailed_result)}")
    logger.info(f"[RESULT] Detailed Categorical: {[(k, round(float(np.nanmean(v)), 4)) for (k, v) in ability_result['categorical'].items()]}")
    logger.info(f"[RESULT] Detailed Numerical: {[(k, round(float(np.nanmean(v)), 4)) for (k, v) in ability_result['numerical'].items()]}")
    logger.info(f"[RESULT] Detailed Reason: {[(k, round(float(np.nanmean(v)), 4)) for (k, v) in ability_result['reason'].items()]}")
    logger.info(f"[RESULT] Overall Categorical: {round(float(np.nanmean(overall_result['categorical'])), 4)}; Overall Numerical: {round(float(np.nanmean(overall_result['numerical'])), 4)}; Overall Reason: {round(float(np.nanmean(overall_result['reason'])), 4)}")
    logger.info(f"[RESULT] Consumed tokens: {total_tokens}")
    logger.info(f"[RESULT] -----------------------------------------------------------------")

    # Create experiment directory if it doesn't exist
    os.makedirs(f"exp/{EXP}", exist_ok=True)
    
    # Save detailed results
    json.dump(detailed_result, open(f"exp/{EXP}/detailed_result.json", "w"), ensure_ascii=False, indent=4)

    # Build the overall summary
    flat_summary = {
        'detail_categorical': {
            k: round(float(np.nanmean(v)), 4)
            for k, v in ability_result['categorical'].items()
        },
        'detail_numerical': {
            k: round(float(np.nanmean(v)), 4)
            for k, v in ability_result['numerical'].items()
        },
        'detail_reason': {
            k: round(float(np.nanmean(v)), 4)
            for k, v in ability_result['reason'].items()
        },
        'overall_categorical': round(float(np.nanmean(overall_result['categorical'])), 4),
        'overall_numerical': round(float(np.nanmean(overall_result['numerical'])), 4),
        'overall_reason': round(float(np.nanmean(overall_result['reason'])), 4),
        'consumed_tokens': total_tokens,
        'error_ratio': round(error_ratio, 4)
    }
    with open(f"exp/{EXP}/result.json", "w") as f:
        json.dump(flat_summary, f, ensure_ascii=False, indent=4)
    logger.info(f"Wrote overall summary to exp/{EXP}/result.json")

    # Generate per-task summaries if task_type metadata is available
    has_task_type = any(s.get('task_type') for s in detailed_result)
    if has_task_type:
        # Aggregate into buckets
        summary_by_task = {}
        token_map = {item['idx']: item.get('num_tokens', 0) for item in generated_answer}
        error_idxs = {item['idx'] for item in generated_answer if '<<ERROR' in item.get('response','')}

        for sr in detailed_result:
            t = sr.get('task_type') or "UNKNOWN"
            blk = summary_by_task.setdefault(t, {
                'by_ability': {},
                'token_counts': [],
                'error_count': 0,
                'sample_count': 0,
            })
            # Accumulate abilities
            for ability, (cate, num, reason, _) in sr['evaluation'].items():
                ab = blk['by_ability'].setdefault(ability, {
                    'categorical': [], 'numerical': [], 'reason': []
                })
                ab['categorical'].extend(cate)
                ab['numerical'].extend(num)
                ab['reason'].extend(reason)
            idx = sr['idx']
            blk['token_counts'].append(token_map.get(idx, 0))
            if idx in error_idxs:
                blk['error_count'] += 1
            blk['sample_count'] += 1

        # Compute summaries for tasks with >50 samples
        per_task_final = {}
        for t, blk in summary_by_task.items():
            if blk['sample_count'] <= 50:
                continue

            # Per-ability details
            detail_categorical = {
                ability: round(float(np.nanmean(vals['categorical'])), 4)
                for ability, vals in blk['by_ability'].items()
            }
            detail_numerical = {
                ability: round(float(np.nanmean(vals['numerical'])), 4)
                for ability, vals in blk['by_ability'].items()
            }
            detail_reason = {
                ability: round(float(np.nanmean(vals['reason'])), 4)
                for ability, vals in blk['by_ability'].items()
            }

            # Overall aggregates
            all_cate = np.concatenate([v['categorical'] for v in blk['by_ability'].values()]) if blk['by_ability'] else np.array([])
            all_num = np.concatenate([v['numerical'] for v in blk['by_ability'].values()]) if blk['by_ability'] else np.array([])
            all_rea = np.concatenate([v['reason'] for v in blk['by_ability'].values()]) if blk['by_ability'] else np.array([])

            per_task_final[t] = {
                'detail_categorical': detail_categorical,
                'detail_numerical': detail_numerical,
                'detail_reason': detail_reason,
                'overall_categorical': round(float(np.nanmean(all_cate)), 4) if all_cate.size else 0.0,
                'overall_numerical': round(float(np.nanmean(all_num)), 4) if all_num.size else 0.0,
                'overall_reason': round(float(np.nanmean(all_rea)), 4) if all_rea.size else 0.0,
                'avg_tokens': round(float(np.nanmean(blk['token_counts'])), 1) if blk['token_counts'] else 0.0,
                'error_ratio': round(blk['error_count']/blk['sample_count'], 4),
                'sample_count': blk['sample_count']
            }

        if per_task_final:
            with open(f"exp/{EXP}/result_by_task_type.json", "w") as f:
                json.dump(per_task_final, f, ensure_ascii=False, indent=4)
            logger.info(f"Wrote per-task-type summary to exp/{EXP}/result_by_task_type.json")
        else:
            logger.info("No task_type groups exceeding 50 samples; skipping per-task summary.")