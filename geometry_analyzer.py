"""
Analyze geometric structure of harmful vs refusal representations.
"""

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np

class GeometryAnalyzer:
    @staticmethod
    def compute_harm_direction(
        harmful_reps: torch.Tensor, 
        refusal_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute harm direction as normalized difference of means.
        
        Args:
            harmful_reps: [n, d_model]
            refusal_reps: [n, d_model]
        
        Returns:
            Normalized direction vector [d_model]
        """
        harmful_mean = harmful_reps.mean(dim=0)
        refusal_mean = refusal_reps.mean(dim=0)
        
        direction = harmful_mean - refusal_mean
        direction_normalized = F.normalize(direction, dim=0)
        
        return direction_normalized
    
    @staticmethod
    def project_onto_direction(
        representations: torch.Tensor, 
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Project representations onto direction.
        
        Args:
            representations: [n, d_model]
            direction: [d_model]
        
        Returns:
            Projections [n]
        """
        direction = F.normalize(direction, dim=0)
        projections = torch.matmul(representations, direction)
        return projections
    
    @staticmethod
    def compute_projection_statistics(
        harmful_proj: torch.Tensor,
        refusal_proj: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute separation statistics for projections.
        
        Args:
            harmful_proj: [n]
            refusal_proj: [n]
        
        Returns:
            Dictionary of statistics
        """
        harmful_mean = harmful_proj.mean().item()
        harmful_std = harmful_proj.std().item()
        refusal_mean = refusal_proj.mean().item()
        refusal_std = refusal_proj.std().item()
        
        # Cohen's d effect size
        pooled_std = np.sqrt((harmful_std**2 + refusal_std**2) / 2)
        cohens_d = (harmful_mean - refusal_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            "harmful_mean": harmful_mean,
            "harmful_std": harmful_std,
            "refusal_mean": refusal_mean,
            "refusal_std": refusal_std,
            "cohens_d": cohens_d
        }
    
    @staticmethod
    def compute_cosine_distance(
        reps1: torch.Tensor, 
        reps2: torch.Tensor
    ) -> float:
        """
        Compute mean cosine distance between two sets of representations.
        
        Args:
            reps1: [n, d_model]
            reps2: [n, d_model]
        
        Returns:
            Mean cosine distance
        """
        mean1 = F.normalize(reps1.mean(dim=0), dim=0)
        mean2 = F.normalize(reps2.mean(dim=0), dim=0)
        
        cosine_sim = torch.dot(mean1, mean2).item()
        cosine_dist = 1 - cosine_sim
        
        return cosine_dist
    
    @staticmethod
    def linear_probe_accuracy(
        harmful_reps: torch.Tensor,
        refusal_reps: torch.Tensor
    ) -> float:
        """
        Train linear probe to separate harmful vs refusal.
        
        Args:
            harmful_reps: [n, d_model]
            refusal_reps: [n, d_model]
        
        Returns:
            Probe accuracy
        """
        X_harmful = harmful_reps.cpu().numpy()
        X_refusal = refusal_reps.cpu().numpy()
        
        X = np.vstack([X_harmful, X_refusal])
        y = np.hstack([
            np.ones(len(X_harmful)),
            np.zeros(len(X_refusal))
        ])
        
        # Split train/test
        n_train = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        
        accuracy = clf.score(X[test_idx], y[test_idx])
        return accuracy
    
    def layerwise_separability(
        self,
        harmful_all_layers: torch.Tensor,
        refusal_all_layers: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute separability metrics across layers.
        
        Args:
            harmful_all_layers: [n_layers, n, d_model]
            refusal_all_layers: [n_layers, n, d_model]
        
        Returns:
            Dictionary mapping layer_idx to metrics
        """
        n_layers = harmful_all_layers.shape[0]
        results = {}
        
        for layer in range(n_layers):
            harmful = harmful_all_layers[layer]
            refusal = refusal_all_layers[layer]
            
            cosine_dist = self.compute_cosine_distance(harmful, refusal)
            probe_acc = self.linear_probe_accuracy(harmful, refusal)
            
            results[layer] = {
                "cosine_distance": cosine_dist,
                "probe_accuracy": probe_acc
            }
        
        return results