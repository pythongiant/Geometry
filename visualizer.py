"""
Visualization utilities for harm geometry analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict


class GeometryVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Use available style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            plt.style.use('ggplot')
    
    def plot_projection_histograms(
        self,
        harmful_proj: torch.Tensor,
        benign_proj: torch.Tensor,
        filename: str = "projection_histograms.png"
    ):
        """
        Plot overlapping histograms of projections onto harm direction.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        harmful_np = harmful_proj.cpu().numpy()
        benign_np = benign_proj.cpu().numpy()
        
        # Compute common bin range
        all_vals = np.concatenate([harmful_np, benign_np])
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)
        
        ax.hist(harmful_np, bins=bins, alpha=0.6, label='Harmful', color='red', density=True)
        ax.hist(benign_np, bins=bins, alpha=0.6, label='Benign', color='green', density=True)
        
        ax.axvline(harmful_np.mean(), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Harmful mean ({harmful_np.mean():.2f})')
        ax.axvline(benign_np.mean(), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Benign mean ({benign_np.mean():.2f})')
        
        ax.set_xlabel('Projection onto Harm Direction', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Projection Distribution: Harmful vs Benign Prompts', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def plot_pca_manifold(
        self,
        harmful_reps: torch.Tensor,
        benign_reps: torch.Tensor,
        filename: str = "pca_manifold.png"
    ):
        """
        Plot 2D PCA projection showing harmful vs benign clusters.
        """
        harmful_np = harmful_reps.cpu().numpy()
        benign_np = benign_reps.cpu().numpy()
        
        # Fit PCA on combined data
        combined = np.vstack([harmful_np, benign_np])
        pca = PCA(n_components=2)
        combined_proj = pca.fit_transform(combined)
        
        n_harmful = len(harmful_np)
        harmful_proj = combined_proj[:n_harmful]
        benign_proj = combined_proj[n_harmful:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(harmful_proj[:, 0], harmful_proj[:, 1], 
                   c='red', alpha=0.5, s=20, label='Harmful')
        ax.scatter(benign_proj[:, 0], benign_proj[:, 1], 
                   c='green', alpha=0.5, s=20, label='Benign')
        
        # Plot means
        harmful_mean = harmful_proj.mean(axis=0)
        benign_mean = benign_proj.mean(axis=0)
        
        ax.scatter(harmful_mean[0], harmful_mean[1], 
                   c='darkred', marker='X', s=200, edgecolors='black', 
                   linewidths=2, label='Harmful mean', zorder=5)
        ax.scatter(benign_mean[0], benign_mean[1], 
                   c='darkgreen', marker='X', s=200, edgecolors='black', 
                   linewidths=2, label='Benign mean', zorder=5)
        
        # Arrow between means (benign to harmful = harm direction)
        ax.annotate('', xy=harmful_mean, xytext=benign_mean,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
        ax.set_title('PCA: Harmful vs Benign Representations', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
    
    def plot_layerwise_separability(
        self,
        layer_stats: Dict[int, Dict[str, float]],
        filename: str = "layerwise_separability.png"
    ):
        """
        Plot separability metrics (cosine distance, probe accuracy) by layer.
        """
        layers = sorted(layer_stats.keys())
        cosine_dists = [layer_stats[l]['cosine_distance'] for l in layers]
        probe_accs = [layer_stats[l]['probe_accuracy'] for l in layers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cosine distance
        ax1.plot(layers, cosine_dists, marker='o', linewidth=2, markersize=6, color='blue')
        ax1.fill_between(layers, cosine_dists, alpha=0.2, color='blue')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Cosine Distance', fontsize=12)
        ax1.set_title('Cosine Distance: Harmful vs Benign Means', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(layers[0], layers[-1])
        
        # Probe accuracy
        ax2.plot(layers, probe_accs, marker='s', linewidth=2, markersize=6, color='green')
        ax2.fill_between(layers, probe_accs, alpha=0.2, color='green')
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance (50%)')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Probe Accuracy', fontsize=12)
        ax2.set_title('Linear Probe Accuracy by Layer', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(layers[0], layers[-1])
        ax2.set_ylim(0.4, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")