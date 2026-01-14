"""
Load and preprocess harmful vs benign prompt datasets.

METHODOLOGY:
- Harmful prompts: ClearHarm dataset (curated harmful instructions)
- Benign prompts: Alpaca/Dolly (curated benign instructions)

This enables valid measurement of harm geometry rather than prefix detection.
"""

from datasets import load_dataset
from typing import List, Tuple, Optional
import random
import warnings


def load_clearharm_data(
    n_samples: int = 500,
    min_length: int = 10,
    seed: int = 42
) -> List[str]:
    """
    Load harmful prompts from ClearHarm dataset.
    
    Args:
        n_samples: Maximum number of samples to load
        min_length: Minimum prompt length (filter malformed)
        seed: Random seed for reproducibility
    
    Returns:
        List of harmful prompt strings
    """
    print("Loading ClearHarm dataset (harmful prompts)...")
    dataset = load_dataset("AlignmentResearch/ClearHarm", split="train")
    print(f"Dataset loaded with {len(dataset)} total entries")
    
    harmful_prompts = []
    filtered_count = 0
    
    for item in dataset:
        # ClearHarm schema: content is VARCHAR[] (array)
        prompt = None
        
        # Try content field first (array of strings)
        if "content" in item and item["content"]:
            content = item["content"]
            if isinstance(content, list) and len(content) > 0:
                prompt = content[0]
            elif isinstance(content, str):
                prompt = content
        
        # Fallback to instructions field
        if not prompt and "instructions" in item:
            prompt = item["instructions"]
        
        # Validate
        if not isinstance(prompt, str):
            filtered_count += 1
            continue
        
        prompt = prompt.strip()
        
        if len(prompt) < min_length or prompt.isspace():
            filtered_count += 1
            continue
        
        harmful_prompts.append(prompt)
        
        if len(harmful_prompts) >= n_samples:
            break
    
    print(f"Filtered out {filtered_count} invalid prompts")
    print(f"Collected {len(harmful_prompts)} harmful prompts")
    
    if len(harmful_prompts) == 0:
        raise ValueError("No valid prompts found in ClearHarm dataset.")
    
    # Shuffle for randomization
    random.seed(seed)
    random.shuffle(harmful_prompts)
    
    return harmful_prompts


def load_benign_data(
    n_samples: int = 500,
    min_length: int = 10,
    seed: int = 42,
    dataset_name: str = "tatsu-lab/alpaca"
) -> List[str]:
    """
    Load benign prompts from instruction-following dataset.
    
    Uses Alpaca by default - a dataset of benign instruction-following examples.
    
    Args:
        n_samples: Maximum number of samples
        min_length: Minimum prompt length
        seed: Random seed
        dataset_name: HuggingFace dataset identifier
    
    Returns:
        List of benign prompt strings
    """
    print(f"Loading benign dataset: {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Falling back to databricks/dolly-15k...")
        dataset = load_dataset("databricks/dolly-15k", split="train")
        dataset_name = "databricks/dolly-15k"
    
    print(f"Benign dataset loaded with {len(dataset)} entries")
    
    benign_prompts = []
    filtered_count = 0
    
    for item in dataset:
        prompt = None
        
        # Handle different dataset formats
        if dataset_name == "tatsu-lab/alpaca":
            # Alpaca: instruction + optional input
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
            
        elif dataset_name == "databricks/dolly-15k":
            # Dolly: instruction + optional context
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            prompt = f"{instruction}\n{context}".strip() if context else instruction
            
        else:
            # Generic: try common field names
            for field in ["instruction", "prompt", "text", "question"]:
                if field in item and item[field]:
                    prompt = item[field]
                    break
        
        if not isinstance(prompt, str) or len(prompt.strip()) < min_length:
            filtered_count += 1
            continue
        
        benign_prompts.append(prompt.strip())
    
    print(f"Filtered out {filtered_count} invalid prompts")
    
    # Shuffle and sample
    random.seed(seed)
    random.shuffle(benign_prompts)
    benign_prompts = benign_prompts[:n_samples]
    
    print(f"Selected {len(benign_prompts)} benign prompts")
    
    return benign_prompts


def load_harm_benign_pairs(
    n_samples: int = 500,
    min_length: int = 10,
    seed: int = 42,
    benign_dataset: str = "tatsu-lab/alpaca"
) -> Tuple[List[str], List[str]]:
    """
    Load matched harmful and benign prompt sets for valid comparison.
    
    This is the CORRECT methodology for studying harm geometry:
    - Harmful: Actual harmful prompts from ClearHarm
    - Benign: Actual benign prompts from instruction datasets
    
    Args:
        n_samples: Target samples per class
        min_length: Minimum prompt length
        seed: Random seed for reproducibility
        benign_dataset: HuggingFace dataset for benign prompts
    
    Returns:
        (harmful_prompts, benign_prompts) - matched lists
    """
    harmful = load_clearharm_data(n_samples, min_length, seed)
    benign = load_benign_data(n_samples, min_length, seed, benign_dataset)
    
    # Match sizes for balanced comparison
    min_size = min(len(harmful), len(benign))
    harmful = harmful[:min_size]
    benign = benign[:min_size]
    
    print(f"\n{'='*50}")
    print(f"FINAL DATASET: {min_size} prompts per class")
    print(f"{'='*50}")
    print(f"\nSample HARMFUL prompt:")
    print(f"  {harmful[0][:100]}...")
    print(f"\nSample BENIGN prompt:")
    print(f"  {benign[0][:100]}...")
    print()
    
    return harmful, benign


# =============================================================================
# DEPRECATED: Legacy functions with warnings
# =============================================================================

def load_clearharm_with_prefix(
    n_samples: int = 500,
    refusal_prefix: str = "You must refuse to answer the following request: ",
    min_length: int = 10
) -> Tuple[List[str], List[str]]:
    """
    DEPRECATED: Creates synthetic prefix-based comparison.
    
    ⚠️  WARNING: This methodology is INVALID for studying harm geometry.
    
    Problem: Comparing "prompt" vs "prefix + prompt" only measures whether
    the model can detect the prefix - not whether it can detect harm.
    
    Use load_harm_benign_pairs() instead for valid analysis.
    """
    warnings.warn(
        "\n" + "="*70 + "\n"
        "⚠️  METHODOLOGICAL WARNING ⚠️\n"
        "load_clearharm_with_prefix() creates an INVALID comparison.\n"
        "It measures PREFIX DETECTION, not HARM DETECTION.\n"
        "\n"
        "Use load_harm_benign_pairs() for valid harm geometry analysis.\n"
        "="*70,
        DeprecationWarning,
        stacklevel=2
    )
    
    harmful = load_clearharm_data(n_samples, min_length)
    refusal_conditioned = [f"{refusal_prefix}{p}" for p in harmful]
    
    return harmful, refusal_conditioned


# Legacy alias
def load_clearharm_data_legacy(
    n_samples: int = 500,
    refusal_prefix: str = "You must refuse: ",
    min_length: int = 10
) -> Tuple[List[str], List[str]]:
    """Legacy wrapper - use load_harm_benign_pairs() instead."""
    return load_clearharm_with_prefix(n_samples, refusal_prefix, min_length)