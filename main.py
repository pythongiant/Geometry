"""
Harmful Prompt Geometry Analysis
Study representation geometry of harmful prompts using ClearHarm dataset.
"""

import torch
from pathlib import Path
import json

from data_loader import load_clearharm_data
from representation_extractor import ResidualStreamExtractor
from geometry_analyzer import GeometryAnalyzer
from causal_interventions import ResidualRepairHook
from visualizer import GeometryVisualizer

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 0.6B closest available
    N_SAMPLES = 500  # Adjust based on computational budget
    LAYERS_TO_ANALYZE = "all"  # or list of specific layers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    DEBUG_MODE = False  # Set to True for verbose debugging
    
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # (1) Load data
    print("\n=== Loading ClearHarm Dataset ===")
    harmful_prompts, refusal_prompts = load_clearharm_data(
        n_samples=N_SAMPLES,
        refusal_prefix="You must refuse to answer the following request: "
    )
    print(f"Loaded {len(harmful_prompts)} harmful prompts")
    print(f"Created {len(refusal_prompts)} refusal-conditioned prompts")
    
    # (2) Extract representations
    print("\n=== Extracting Residual Stream Activations ===")
    extractor = ResidualStreamExtractor(MODEL_NAME, device=DEVICE)
    
    # Test extraction on a single prompt first
    print("\nTesting extraction on first prompt...")
    test_rep = extractor.extract_single(harmful_prompts[0])
    if test_rep is None:
        raise RuntimeError("Failed to extract representation from test prompt. Check model compatibility.")
    print(f"Test extraction successful: shape {test_rep.shape}")
    
    harmful_reps = extractor.extract_batch(harmful_prompts)
    refusal_reps = extractor.extract_batch(refusal_prompts)
    
    print(f"Harmful representations: {harmful_reps.shape}")
    print(f"Refusal representations: {refusal_reps.shape}")
    
    # (3) Geometry analysis
    print("\n=== Computing Harm Direction ===")
    analyzer = GeometryAnalyzer()
    
    harm_direction = analyzer.compute_harm_direction(harmful_reps, refusal_reps)
    print(f"Harm direction shape: {harm_direction.shape}")
    
    # (4) Projection analysis
    print("\n=== Projection Analysis ===")
    harmful_projections = analyzer.project_onto_direction(harmful_reps, harm_direction)
    refusal_projections = analyzer.project_onto_direction(refusal_reps, harm_direction)
    
    stats = analyzer.compute_projection_statistics(
        harmful_projections, 
        refusal_projections
    )
    print(f"Harmful mean: {stats['harmful_mean']:.4f} ± {stats['harmful_std']:.4f}")
    print(f"Refusal mean: {stats['refusal_mean']:.4f} ± {stats['refusal_std']:.4f}")
    print(f"Separation (Cohen's d): {stats['cohens_d']:.4f}")
    
    # (5) Layerwise separability
    print("\n=== Layerwise Separability Analysis ===")
    harmful_reps_all = extractor.extract_batch_all_layers(harmful_prompts)
    refusal_reps_all = extractor.extract_batch_all_layers(refusal_prompts)
    
    layer_stats = analyzer.layerwise_separability(
        harmful_reps_all, 
        refusal_reps_all
    )
    
    for layer, metrics in layer_stats.items():
        print(f"Layer {layer}: cosine_dist={metrics['cosine_distance']:.4f}, "
              f"probe_acc={metrics['probe_accuracy']:.4f}")
    
    # (6) Visualization
    print("\n=== Generating Visualizations ===")
    viz = GeometryVisualizer(OUTPUT_DIR)
    
    viz.plot_projection_histograms(
        harmful_projections, 
        refusal_projections,
        filename="projection_histograms.png"
    )
    
    viz.plot_pca_manifold(
        harmful_reps, 
        refusal_reps,
        filename="pca_manifold.png"
    )
    
    viz.plot_layerwise_separability(
        layer_stats,
        filename="layerwise_separability.png"
    )
    
    # (7) Causal intervention
    print("\n=== Causal Intervention Test ===")
    repair_hook = ResidualRepairHook(extractor.model, harm_direction)
    
    # Compute refusal target
    refusal_mean = refusal_reps.mean(dim=0)
    
    # Test on subset
    test_prompts = harmful_prompts[:10]
    
    print("Computing pre-intervention geometry...")
    pre_reps = extractor.extract_batch(test_prompts)
    pre_projections = analyzer.project_onto_direction(pre_reps, harm_direction)
    
    print("Applying intervention...")
    post_reps = repair_hook.intervene_batch(
        test_prompts, 
        target_activation=refusal_mean,
        layer_idx=-1,
        strength=0.5
    )
    post_projections = analyzer.project_onto_direction(post_reps, harm_direction)
    
    intervention_shift = (pre_projections - post_projections).mean().item()
    print(f"Mean projection shift: {intervention_shift:.4f}")
    print(f"Pre-intervention mean: {pre_projections.mean():.4f}")
    print(f"Post-intervention mean: {post_projections.mean():.4f}")
    
    # (8) Save results
    results = {
        "model": MODEL_NAME,
        "n_samples": N_SAMPLES,
        "projection_stats": stats,
        "intervention_shift": intervention_shift,
        "layer_separability": {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in layer_stats.items()
        }
    }
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to {OUTPUT_DIR} ===")
    print("Analysis complete.")

if __name__ == "__main__":
    main()