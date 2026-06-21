#!/usr/bin/env python3
"""
Utility functions for interacting with the LLaMA Factory API.
"""

def ask_via_llama_fac_api(client, model, img_b64, question):
    """
    Call the local llama-factory API with one image + grouped-QA text.
    
    Args:
        client: OpenAI client object
        model: Model name to use for inference
        img_b64: Base64 encoded image string
        question: Question to ask the model
        
    Returns:
        Model response as a string
    """
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    # Build the multimodal message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
          "role": "user",
          "content": [
            {"type": "text", "text": question},
            {
              "type": "image_url",
              "image_url": {"url": data_uri}
            }
          ]
        }
    ]

    # Send with chat.completions
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096
    )

    return resp.choices[0].message.content