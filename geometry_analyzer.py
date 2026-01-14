"""
Analyze geometric structure of harmful vs benign representations.
"""

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict
import numpy as np


class GeometryAnalyzer:
    
    @staticmethod
    def compute_harm_direction(
        harmful_reps: torch.Tensor, 
        benign_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute harm direction as normalized difference of means.
        
        The harm direction points FROM benign TO harmful:
            direction = normalize(harmful_mean - benign_mean)
        
        Positive projection = more harmful-like
        Negative projection = more benign-like
        
        Args:
            harmful_reps: [n, d_model] representations of harmful prompts
            benign_reps: [n, d_model] representations of benign prompts
        
        Returns:
            Normalized direction vector [d_model]
        """
        harmful_mean = harmful_reps.mean(dim=0)
        benign_mean = benign_reps.mean(dim=0)
        
        direction = harmful_mean - benign_mean
        direction_normalized = F.normalize(direction, dim=0)
        
        return direction_normalized
    
    @staticmethod
    def project_onto_direction(
        representations: torch.Tensor, 
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Project representations onto a direction vector.
        
        Args:
            representations: [n, d_model]
            direction: [d_model] (will be normalized)
        
        Returns:
            Scalar projections [n]
        """
        direction = F.normalize(direction, dim=0)
        projections = torch.matmul(representations, direction)
        return projections
    
    @staticmethod
    def compute_projection_statistics(
        harmful_proj: torch.Tensor,
        benign_proj: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute separation statistics for projection distributions.
        
        Args:
            harmful_proj: [n] projections for harmful prompts
            benign_proj: [n] projections for benign prompts
        
        Returns:
            Dictionary with means, stds, and Cohen's d
        """
        harmful_mean = harmful_proj.mean().item()
        harmful_std = harmful_proj.std().item()
        benign_mean = benign_proj.mean().item()
        benign_std = benign_proj.std().item()
        
        # Cohen's d: standardized mean difference
        pooled_std = np.sqrt((harmful_std**2 + benign_std**2) / 2)
        cohens_d = (harmful_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            "harmful_mean": harmful_mean,
            "harmful_std": harmful_std,
            "benign_mean": benign_mean,  # Changed from refusal_mean
            "benign_std": benign_std,    # Changed from refusal_std
            "cohens_d": cohens_d
        }
    
    @staticmethod
    def compute_cosine_distance(
        reps1: torch.Tensor, 
        reps2: torch.Tensor
    ) -> float:
        """
        Compute cosine distance between mean representations.
        
        Args:
            reps1: [n, d_model]
            reps2: [n, d_model]
        
        Returns:
            Cosine distance (1 - cosine_similarity)
        """
        mean1 = F.normalize(reps1.mean(dim=0), dim=0)
        mean2 = F.normalize(reps2.mean(dim=0), dim=0)
        
        cosine_sim = torch.dot(mean1, mean2).item()
        cosine_dist = 1 - cosine_sim
        
        return cosine_dist
    
    @staticmethod
    def linear_probe_accuracy(
        harmful_reps: torch.Tensor,
        benign_reps: torch.Tensor,
        cv_folds: int = 5
    ) -> float:
        """
        Train linear probe with cross-validation to measure separability.
        
        Uses stratified k-fold CV for more robust accuracy estimate.
        
        Args:
            harmful_reps: [n, d_model]
            benign_reps: [n, d_model]
            cv_folds: Number of cross-validation folds
        
        Returns:
            Mean cross-validated accuracy
        """
        X_harmful = harmful_reps.cpu().numpy()
        X_benign = benign_reps.cpu().numpy()
        
        X = np.vstack([X_harmful, X_benign])
        y = np.hstack([
            np.ones(len(X_harmful)),   # 1 = harmful
            np.zeros(len(X_benign))    # 0 = benign
        ])
        
        # Use cross-validation for robust estimate
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        
        # Minimum samples check
        if len(X) < cv_folds * 2:
            # Fall back to simple train/test split
            n_train = int(0.8 * len(X))
            indices = np.random.permutation(len(X))
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            clf.fit(X[train_idx], y[train_idx])
            return clf.score(X[test_idx], y[test_idx])
        
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='accuracy')
        return scores.mean()
    
    def layerwise_separability(
        self,
        harmful_all_layers: torch.Tensor,
        benign_all_layers: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute separability metrics across all layers.
        
        Args:
            harmful_all_layers: [n_layers, n_samples, d_model]
            benign_all_layers: [n_layers, n_samples, d_model]
        
        Returns:
            Dictionary: layer_idx -> {cosine_distance, probe_accuracy}
        """
        n_layers = harmful_all_layers.shape[0]
        results = {}
        
        for layer in range(n_layers):
            harmful = harmful_all_layers[layer]
            benign = benign_all_layers[layer]
            
            cosine_dist = self.compute_cosine_distance(harmful, benign)
            probe_acc = self.linear_probe_accuracy(harmful, benign)
            
            results[layer] = {
                "cosine_distance": cosine_dist,
                "probe_accuracy": probe_acc
            }
        
        return results