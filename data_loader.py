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
    print("Loading ClearHarm dataset...")
    dataset = load_dataset("AlignmentResearch/ClearHarm", split="train")
    print(f"Dataset loaded with {len(dataset)} total entries")
    
    # Extract harmful prompts
    harmful_prompts = []
    filtered_count = 0
    
    for item in dataset:
       
        prompt = item.get("content", "")[0]
        
        # Robust filtering
        if not isinstance(prompt, str):
            filtered_count += 1
            continue
        
        prompt = prompt.strip()
        
        if len(prompt) < min_length:
            filtered_count += 1
            continue
        
        if not prompt or prompt.isspace():
            filtered_count += 1
            continue
        
        harmful_prompts.append(prompt)
        
        if len(harmful_prompts) >= n_samples:
            break
    
    print(f"Filtered out {filtered_count} invalid prompts")
    print(f"Collected {len(harmful_prompts)} valid harmful prompts")
    
    if len(harmful_prompts) == 0:
        raise ValueError("No valid prompts found in dataset. Check filtering criteria.")
    
    # Show sample
    print(f"\nSample prompt: {harmful_prompts[0][:100]}...")
    
    # Construct refusal-conditioned prompts
    refusal_prompts = [
        f"{refusal_prefix}{prompt}" 
        for prompt in harmful_prompts
    ]
    
    return harmful_prompts, refusal_prompts