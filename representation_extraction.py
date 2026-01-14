"""
Extract residual stream activations from TransformerLens model.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List
from tqdm import tqdm

class ResidualStreamExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize model and extraction utilities.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for computation
        """
        self.device = device
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
    
    def extract_single(self, prompt: str, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract residual stream activation for a single prompt.
        
        Args:
            prompt: Input text
            layer_idx: Layer index (-1 for final layer)
        
        Returns:
            Tensor of shape [d_model]
        """
        tokens = self.model.to_tokens(prompt)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        
        # Get residual stream at final token position
        if layer_idx == -1:
            layer_idx = self.n_layers - 1
        
        resid = cache["resid_pre", layer_idx]  # [batch=1, seq, d_model]
        final_pos_resid = resid[0, -1, :]  # [d_model]
        
        return final_pos_resid
    
    def extract_batch(
        self, 
        prompts: List[str], 
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract representations for multiple prompts (one at a time).
        
        Args:
            prompts: List of input texts
            layer_idx: Layer index
        
        Returns:
            Tensor of shape [n_prompts, d_model]
        """
        representations = []
        
        for prompt in tqdm(prompts, desc="Extracting representations"):
            rep = self.extract_single(prompt, layer_idx)
            representations.append(rep)
        
        return torch.stack(representations)
    
    def extract_batch_all_layers(
        self, 
        prompts: List[str]
    ) -> torch.Tensor:
        """
        Extract representations across all layers.
        
        Args:
            prompts: List of input texts
        
        Returns:
            Tensor of shape [n_layers, n_prompts, d_model]
        """
        all_layers = []
        
        for layer_idx in range(self.n_layers):
            layer_reps = self.extract_batch(prompts, layer_idx)
            all_layers.append(layer_reps)
        
        return torch.stack(all_layers)