"""
Load and preprocess ClearHarm dataset.
"""

from datasets import load_dataset
from typing import List, Tuple

def load_clearharm_data(
    n_samples: int = 500,
    refusal_prefix: str = "You must refuse to answer the following request: ",
    min_length: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Load ClearHarm dataset and construct refusal-conditioned control.
    
    Args:
        n_samples: Number of samples to load
        refusal_prefix: Prefix for refusal conditioning
        min_length: Minimum prompt length (filter short/malformed)
    
    Returns:
        (harmful_prompts, refusal_prompts)
    """
    dataset = load_dataset("AlignmentResearch/ClearHarm", split="train")
    
    # Extract harmful prompts
    harmful_prompts = []
    
    for item in dataset:
        prompt = item.get("prompt", "")
        
        # Robust filtering
        if not isinstance(prompt, str):
            continue
        
        prompt = prompt.strip()
        
        if len(prompt) < min_length:
            continue
        
        if not prompt or prompt.isspace():
            continue
        
        harmful_prompts.append(prompt)
        
        if len(harmful_prompts) >= n_samples:
            break
    
    # Construct refusal-conditioned prompts
    refusal_prompts = [
        f"{refusal_prefix}{prompt}" 
        for prompt in harmful_prompts
    ]
    
    return harmful_prompts, refusal_prompts