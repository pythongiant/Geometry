"""
Visualization utilities for representation geometry.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict

class GeometryVisualizer:
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_projection_histograms(
        self,
        harmful_proj: torch.Tensor,
        refusal_proj: torch.Tensor,
        filename: str = "projection_histograms.png"
    ):
        """
        Plot histograms of projections onto harm direction.
        
        Args:
            harmful_proj: [n] projections for harmful prompts
            refusal_proj: [n] projections for refusal prompts
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        harmful_np = harmful_proj.cpu().numpy()
        refusal_np = refusal_proj.cpu().numpy()
        
        ax.hist(harmful_np, bins=50, alpha=0.6, label='Harmful', color='red')
        ax.hist(refusal_np, bins=50, alpha=0.6, label='Refusal-conditioned', color='blue')
        
        ax.axvline(harmful_np.mean(), color='red', linestyle='--', linewidth=2, label='Harmful mean')
        ax.axvline(refusal_np.mean(), color='blue', linestyle='--', linewidth=2, label='Refusal mean')
        
        ax.set_xlabel('Projection onto harm direction', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Projection Distribution: Harmful vs Refusal-Conditioned', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_pca_manifold(
        self,
        harmful_reps: torch.Tensor,
        refusal_reps: torch.Tensor,
        filename: str = "pca_manifold.png"
    ):
        """
        Plot 2D PCA projection of representation manifolds.
        
        Args:
            harmful_reps: [n, d_model]
            refusal_reps: [n, d_model]
            filename: Output filename
        """
        harmful_np = harmful_reps.cpu().numpy()
        refusal_np = refusal_reps.cpu().numpy()
        
        # Fit PCA on combined data
        combined = np.vstack([harmful_np, refusal_np])
        pca = PCA(n_components=2)
        combined_proj = pca.fit_transform(combined)
        
        n_harmful = len(harmful_np)
        harmful_proj = combined_proj[:n_harmful]
        refusal_proj = combined_proj[n_harmful:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(harmful_proj[:, 0], harmful_proj[:, 1], 
                  c='red', alpha=0.5, s=20, label='Harmful')
        ax.scatter(refusal_proj[:, 0], refusal_proj[:, 1], 
                  c='blue', alpha=0.5, s=20, label='Refusal-conditioned')
        
        # Plot means
        harmful_mean = harmful_proj.mean(axis=0)
        refusal_mean = refusal_proj.mean(axis=0)
        
        ax.scatter(harmful_mean[0], harmful_mean[1], 
                  c='darkred', marker='X', s=200, edgecolors='black', 
                  linewidths=2, label='Harmful mean', zorder=5)
        ax.scatter(refusal_mean[0], refusal_mean[1], 
                  c='darkblue', marker='X', s=200, edgecolors='black', 
                  linewidths=2, label='Refusal mean', zorder=5)
        
        # Arrow between means
        ax.annotate('', xy=refusal_mean, xytext=harmful_mean,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_explained[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.2%} variance)', fontsize=12)
        ax.set_title('PCA Projection of Residual Stream Representations', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_layerwise_separability(
        self,
        layer_stats: Dict[int, Dict[str, float]],
        filename: str = "layerwise_separability.png"
    ):
        """
        Plot separability metrics across layers.
        
        Args:
            layer_stats: Dict mapping layer to metrics
            filename: Output filename
        """
        layers = sorted(layer_stats.keys())
        cosine_dists = [layer_stats[l]['cosine_distance'] for l in layers]
        probe_accs = [layer_stats[l]['probe_accuracy'] for l in layers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cosine distance
        ax1.plot(layers, cosine_dists, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Cosine Distance', fontsize=12)
        ax1.set_title('Mean Cosine Distance: Harmful vs Refusal', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Probe accuracy
        ax2.plot(layers, probe_accs, marker='s', linewidth=2, markersize=6, color='green')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Linear Probe Accuracy', fontsize=12)
        ax2.set_title('Linear Separability Across Layers', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
        
        print(f"Saved: {filename}")