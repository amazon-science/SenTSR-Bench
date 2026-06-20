#!/usr/bin/env python3
"""
Utility functions for Claude inference with time series data.
This module provides common functions for working with Claude API,
including response parsing, retry logic, and prompt configuration.
"""

import re
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from botocore.exceptions import ClientError

# ─── CONFIGURATION ────────────────────────────────────────────────────────────────
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_TOKENS = 4096
THINKING_BUDGET = 2048

# System prompt that requests JSON output with a single "Final Answer" key
DEFAULT_SYSTEM_PROMPT = (
    "You are a time‐series expert.  \n"
    "Answer **only** with a JSON object that has exactly one key, 'Final Answer',\n"
    "whose value is the answer string.  \n"
)

def is_throttling(exc):
    """
    Check if the exception is due to throttling.
    
    Args:
        exc: Exception to check
        
    Returns:
        bool: True if the exception is a throttling exception
    """
    return (
        isinstance(exc, ClientError) and
        exc.response.get("Error", {}).get("Code") == "ThrottlingException"
    )

@retry(
    retry=retry_if_exception(is_throttling),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def invoke_claude(client, messages, model_id=MODEL_ID, temperature=1.0, system=None):
    """
    Invoke Claude with retry logic for throttling.
    
    Args:
        client: Boto3 Bedrock client
        messages: List of message dictionaries
        model_id: Claude model ID to use
        temperature: Sampling temperature (1.0 for thinking)
        system: Optional system prompt (uses default if None)
        
    Returns:
        dict: Claude API response
    """
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": system or DEFAULT_SYSTEM_PROMPT,
        "messages": messages
    }
    resp = client.invoke_model(body=json.dumps(payload), modelId=model_id)
    return json.loads(resp['body'].read())

def parse_response(resp_body):
    """
    Parse Claude's response to extract thought and answer.
    
    Args:
        resp_body: Claude API response body
        
    Returns:
        tuple: (thought, answer, success)
    """
    thought_chunks, text_chunks = [], []
    
    for chunk in resp_body.get("content", []):
        if chunk.get("type") == "thinking":
            thought_chunks.append(chunk.get("thinking", "").strip())
        elif chunk.get("type") == "text":
            text_chunks.append(chunk.get("text", "").strip())
            
    thought = "\n".join(thought_chunks)
    raw = "".join(text_chunks)

    # Remove markdown code fences
    clean = re.sub(r'```(?:json)?', '', raw).strip()
    
    # Find first '{' and last '}' to extract JSON
    start = clean.find('{')
    end = clean.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = clean[start:end+1]
    else:
        json_str = clean

    try:
        obj = json.loads(json_str)
        answer = obj.get('Final Answer', "")
        success = 'Final Answer' in obj
    except Exception:
        answer = json_str
        success = False
        
    return thought, answer, success