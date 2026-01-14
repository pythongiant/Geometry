"""
Causal interventions on residual stream to test harm geometry.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List
from tqdm import tqdm
from functools import partial

class ResidualRepairHook:
    def __init__(self, model: HookedTransformer, harm_direction: torch.Tensor):
        """
        Initialize intervention mechanism.
        
        Args:
            model: TransformerLens model
            harm_direction: Harm direction vector [d_model]
        """
        self.model = model
        self.harm_direction = harm_direction.to(model.cfg.device)
        self.target_activation = None
        self.strength = 1.0
    
    def repair_hook(self, resid: torch.Tensor, hook) -> torch.Tensor:
        """
        Hook function to modify residual stream.
        
        Moves activation toward target_activation along harm direction.
        
        Args:
            resid: Residual stream [batch, seq, d_model]
            hook: TransformerLens hook object
        
        Returns:
            Modified residual stream
        """
        if self.target_activation is None:
            return resid
        
        # Modify final position only
        final_pos = resid[:, -1, :]  # [batch, d_model]
        
        # Compute shift
        shift = self.target_activation - final_pos
        
        # Project shift onto harm direction
        harm_dir = self.harm_direction.unsqueeze(0)  # [1, d_model]
        projection = (shift * harm_dir).sum(dim=-1, keepdim=True)
        directional_shift = projection * harm_dir
        
        # Apply scaled intervention
        final_pos_modified = final_pos + self.strength * directional_shift
        
        # Update residual stream
        resid_modified = resid.clone()
        resid_modified[:, -1, :] = final_pos_modified
        
        return resid_modified
    
    def intervene_single(
        self,
        prompt: str,
        target_activation: torch.Tensor,
        layer_idx: int = -1,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply intervention to single prompt and extract post-intervention representation.
        
        Args:
            prompt: Input text
            target_activation: Target representation [d_model]
            layer_idx: Layer to intervene at
            strength: Intervention strength
        
        Returns:
            Post-intervention representation [d_model]
        """
        self.target_activation = target_activation
        self.strength = strength
        
        if layer_idx == -1:
            layer_idx = self.model.cfg.n_layers - 1
        
        hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        
        tokens = self.model.to_tokens(prompt)
        
        # Use context manager to add hook
        with torch.no_grad():
            with self.model.hooks(fwd_hooks=[(hook_name, self.repair_hook)]):
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=[hook_name]
                )
        
        # Extract post-intervention representation
        resid_post = cache[hook_name][0, -1, :]
        
        return resid_post
    
    def intervene_batch(
        self,
        prompts: List[str],
        target_activation: torch.Tensor,
        layer_idx: int = -1,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply intervention to batch of prompts.
        
        Args:
            prompts: List of input texts
            target_activation: Target representation
            layer_idx: Layer to intervene at
            strength: Intervention strength
        
        Returns:
            Post-intervention representations [n, d_model]
        """
        post_reps = []
        
        for prompt in tqdm(prompts, desc="Applying interventions"):
            rep = self.intervene_single(
                prompt, 
                target_activation, 
                layer_idx, 
                strength
            )
            post_reps.append(rep)
        
        return torch.stack(post_reps)