"""
Diagnostic test to identify extraction issues.
"""

import torch
from transformer_lens import HookedTransformer
from data_loader import load_clearharm_data

def test_model_loading():
    """Test if model loads correctly."""
    print("=== Testing Model Loading ===")
    try:
        model_name = "Qwen/Qwen2.5-0.5B"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        print(f"Loading {model_name}...")
        
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            torch_dtype=torch.float32
        )
        
        print(f"✓ Model loaded successfully")
        print(f"  Layers: {model.cfg.n_layers}")
        print(f"  d_model: {model.cfg.d_model}")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("\nTrying alternative model (gpt2)...")
        try:
            model = HookedTransformer.from_pretrained(
                "gpt2",
                device="cpu",
                torch_dtype=torch.float32
            )
            print(f"✓ Alternative model (gpt2) loaded successfully")
            print(f"  Note: Update MODEL_NAME in main.py to 'gpt2'")
            return model
        except Exception as e2:
            print(f"✗ Alternative model also failed: {e2}")
            return None

def test_dataset_loading():
    """Test if dataset loads correctly using data_loader.py."""
    print("\n=== Testing Dataset Loading (via data_loader.py) ===")
    try:
        # Test with small sample first
        harmful_prompts, refusal_prompts = load_clearharm_data(
            n_samples=10,
            min_length=10
        )
        
        print(f"✓ Data loader successful")
        print(f"  Harmful prompts: {len(harmful_prompts)}")
        print(f"  Refusal prompts: {len(refusal_prompts)}")
        
        if len(harmful_prompts) > 0:
            print(f"\n  First harmful prompt:")
            print(f"    {harmful_prompts[0][:100]}...")
            print(f"\n  First refusal prompt:")
            print(f"    {refusal_prompts[0][:100]}...")
        
        return harmful_prompts, refusal_prompts
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_tokenization(model, prompt):
    """Test tokenization."""
    print("\n=== Testing Tokenization ===")
    print(f"Prompt: {prompt[:100]}...")
    
    try:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        print(f"✓ Tokenization successful")
        print(f"  Token shape: {tokens.shape}")
        print(f"  Sequence length: {tokens.shape[1]}")
        print(f"  First 10 tokens: {tokens[0, :10].tolist()}")
        return tokens
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, tokens):
    """Test forward pass and cache."""
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Cache contains {len(cache)} entries")
        
        # Test residual stream access
        final_layer_idx = model.cfg.n_layers - 1
        resid_key = ("resid_pre", final_layer_idx)
        
        print(f"\n  Looking for cache key: {resid_key}")
        
        if resid_key in cache:
            resid = cache[resid_key]
            print(f"  ✓ Found resid_pre at layer {final_layer_idx}")
            print(f"    Shape: {resid.shape}")
            print(f"    Expected: [batch=1, seq_len={tokens.shape[1]}, d_model={model.cfg.d_model}]")
            
            final_pos = resid[0, -1, :]
            print(f"\n  Final position activation:")
            print(f"    Shape: {final_pos.shape}")
            print(f"    Mean: {final_pos.mean():.4f}")
            print(f"    Std: {final_pos.std():.4f}")
            print(f"    Min: {final_pos.min():.4f}")
            print(f"    Max: {final_pos.max():.4f}")
            return final_pos
        else:
            print(f"  ✗ Could not find {resid_key} in cache")
            print(f"  Available cache keys (first 10):")
            for i, key in enumerate(list(cache.keys())[:10]):
                print(f"    {key}")
            return None
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_extraction_pipeline(model, harmful_prompts, refusal_prompts):
    """Test the full extraction pipeline."""
    print("\n=== Testing Full Extraction Pipeline ===")
    
    try:
        from representation_extractor import ResidualStreamExtractor
        
        # Create extractor with the already-loaded model
        print("Creating extractor...")
        extractor = ResidualStreamExtractor.__new__(ResidualStreamExtractor)
        extractor.model = model
        extractor.device = model.cfg.device
        extractor.n_layers = model.cfg.n_layers
        extractor.d_model = model.cfg.d_model
        
        # Test single extraction
        print("\nTesting single extraction...")
        test_rep = extractor.extract_single(harmful_prompts[0])
        
        if test_rep is None:
            print("✗ Single extraction returned None")
            return False
        
        print(f"✓ Single extraction successful")
        print(f"  Shape: {test_rep.shape}")
        
        # Test batch extraction with small sample
        print("\nTesting batch extraction (3 prompts)...")
        batch_reps = extractor.extract_batch(harmful_prompts[:3])
        
        print(f"✓ Batch extraction successful")
        print(f"  Shape: {batch_reps.shape}")
        print(f"  Expected: [3, {model.cfg.d_model}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Extraction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("ClearHarm Geometry Extraction Diagnostic")
    print("="*60)
    
    # Test 1: Model loading
    model = test_model_loading()
    if model is None:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot proceed without model")
        print("="*60)
        return
    
    # Test 2: Dataset loading (using data_loader.py)
    harmful_prompts, refusal_prompts = test_dataset_loading()
    if harmful_prompts is None or len(harmful_prompts) == 0:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot proceed without valid prompts")
        print("="*60)
        return
    
    # Test 3: Tokenization
    test_prompt = harmful_prompts[0]
    tokens = test_tokenization(model, test_prompt)
    if tokens is None:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot proceed without tokenization")
        print("="*60)
        return
    
    # Test 4: Forward pass
    activation = test_forward_pass(model, tokens)
    if activation is None:
        print("\n" + "="*60)
        print("❌ FAILED: Cannot extract activations")
        print("="*60)
        return
    
    # Test 5: Full extraction pipeline
    success = test_extraction_pipeline(model, harmful_prompts, refusal_prompts)
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL DIAGNOSTIC TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the full analysis:")
        print("  python main.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nReview the errors above and check TROUBLESHOOTING.md")
    print("="*60)

if __name__ == "__main__":
    main()