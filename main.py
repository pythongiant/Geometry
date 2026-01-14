"""
Harmful Prompt Geometry Analysis
Study representation geometry of harmful vs benign prompts.

METHODOLOGY:
- Compare genuinely harmful prompts (ClearHarm) vs genuinely benign prompts (Alpaca)
- Extract residual stream representations at final token position
- Analyze geometric separation between harm and benign clusters
- Validate with linear probes and causal interventions
"""

import torch
from pathlib import Path
import json

from data_loader import load_harm_benign_pairs
from representation_extractor import ResidualStreamExtractor
from geometry_analyzer import GeometryAnalyzer
from causal_interventions import ResidualRepairHook
from visualizer import GeometryVisualizer


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    N_SAMPLES = 500  # Per class (harmful and benign)
    BENIGN_DATASET = "tatsu-lab/alpaca"  # Source for benign prompts
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SEED = 42
    
    print("="*60)
    print("HARMFUL PROMPT GEOMETRY ANALYSIS")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples per class: {N_SAMPLES}")
    print(f"Benign source: {BENIGN_DATASET}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    # ==========================================================================
    # (1) Load Data - CORRECT METHODOLOGY
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Harmful vs Benign Datasets")
    print("="*60)
    
    harmful_prompts, benign_prompts = load_harm_benign_pairs(
        n_samples=N_SAMPLES,
        benign_dataset=BENIGN_DATASET,
        seed=SEED
    )
    
    print(f"\nLoaded {len(harmful_prompts)} harmful prompts (ClearHarm)")
    print(f"Loaded {len(benign_prompts)} benign prompts ({BENIGN_DATASET})")
    
    # ==========================================================================
    # (2) Extract Representations
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 2: Extracting Residual Stream Activations")
    print("="*60)
    
    extractor = ResidualStreamExtractor(MODEL_NAME, device=DEVICE)
    
    # Validate extraction
    print("\nValidating extraction on first prompt...")
    test_rep = extractor.extract_single(harmful_prompts[0])
    if test_rep is None:
        raise RuntimeError("Extraction failed. Check model compatibility.")
    print(f"✓ Extraction successful: shape {test_rep.shape}")
    
    # Batch extraction
    print("\nExtracting harmful representations...")
    harmful_reps = extractor.extract_batch(harmful_prompts)
    print(f"Harmful representations: {harmful_reps.shape}")
    
    print("\nExtracting benign representations...")
    benign_reps = extractor.extract_batch(benign_prompts)
    print(f"Benign representations: {benign_reps.shape}")
    
    # ==========================================================================
    # (3) Compute Harm Direction
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 3: Computing Harm Direction")
    print("="*60)
    
    analyzer = GeometryAnalyzer()
    
    # Direction: harmful_mean - benign_mean (normalized)
    harm_direction = analyzer.compute_harm_direction(harmful_reps, benign_reps)
    print(f"Harm direction computed: shape {harm_direction.shape}")
    print(f"Direction norm (should be 1.0): {harm_direction.norm().item():.6f}")
    
    # ==========================================================================
    # (4) Projection Analysis
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 4: Projection Analysis")
    print("="*60)
    
    harmful_projections = analyzer.project_onto_direction(harmful_reps, harm_direction)
    benign_projections = analyzer.project_onto_direction(benign_reps, harm_direction)
    
    stats = analyzer.compute_projection_statistics(
        harmful_projections, 
        benign_projections
    )
    
    print(f"\nProjection Statistics:")
    print(f"  Harmful:  mean = {stats['harmful_mean']:+.4f}  (std = {stats['harmful_std']:.4f})")
    print(f"  Benign:   mean = {stats['benign_mean']:+.4f}  (std = {stats['benign_std']:.4f})")
    print(f"  Separation (Cohen's d): {stats['cohens_d']:.4f}")
    
    # Interpret effect size
    if abs(stats['cohens_d']) < 0.2:
        effect_interp = "negligible"
    elif abs(stats['cohens_d']) < 0.5:
        effect_interp = "small"
    elif abs(stats['cohens_d']) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Effect size interpretation: {effect_interp}")
    
    # ==========================================================================
    # (5) Layerwise Separability
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 5: Layerwise Separability Analysis")
    print("="*60)
    
    print("\nExtracting all-layer representations (this may take a while)...")
    harmful_reps_all = extractor.extract_batch_all_layers(harmful_prompts)
    benign_reps_all = extractor.extract_batch_all_layers(benign_prompts)
    
    print(f"Harmful all-layers: {harmful_reps_all.shape}")
    print(f"Benign all-layers: {benign_reps_all.shape}")
    
    layer_stats = analyzer.layerwise_separability(
        harmful_reps_all, 
        benign_reps_all
    )
    
    print("\nLayer-by-layer separability:")
    print("-" * 50)
    for layer, metrics in sorted(layer_stats.items()):
        print(f"  Layer {layer:2d}: cosine_dist = {metrics['cosine_distance']:.4f}, "
              f"probe_acc = {metrics['probe_accuracy']:.3f}")
    
    # Find layer with best separation
    best_layer = max(layer_stats.keys(), key=lambda l: layer_stats[l]['probe_accuracy'])
    print(f"\nBest layer for linear separation: {best_layer} "
          f"(accuracy = {layer_stats[best_layer]['probe_accuracy']:.3f})")
    
    # ==========================================================================
    # (6) Visualization
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 6: Generating Visualizations")
    print("="*60)
    
    viz = GeometryVisualizer(OUTPUT_DIR)
    
    viz.plot_projection_histograms(
        harmful_projections, 
        benign_projections,
        filename="projection_histograms.png"
    )
    
    viz.plot_pca_manifold(
        harmful_reps, 
        benign_reps,
        filename="pca_manifold.png"
    )
    
    viz.plot_layerwise_separability(
        layer_stats,
        filename="layerwise_separability.png"
    )
    
    # ==========================================================================
    # (7) Causal Intervention
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 7: Causal Intervention Test")
    print("="*60)
    
    repair_hook = ResidualRepairHook(extractor.model, harm_direction)
    
    # Target: benign mean (we want to move harmful toward benign)
    benign_mean = benign_reps.mean(dim=0)
    
    # Test on subset of harmful prompts
    test_prompts = harmful_prompts[:10]
    
    print(f"Testing intervention on {len(test_prompts)} harmful prompts...")
    
    print("  Computing pre-intervention representations...")
    pre_reps = extractor.extract_batch(test_prompts)
    pre_projections = analyzer.project_onto_direction(pre_reps, harm_direction)
    
    print("  Applying intervention (strength=0.5)...")
    post_reps = repair_hook.intervene_batch(
        test_prompts, 
        target_activation=benign_mean,
        layer_idx=-1,
        strength=0.5
    )
    post_projections = analyzer.project_onto_direction(post_reps, harm_direction)
    
    intervention_shift = (pre_projections - post_projections).mean().item()
    
    print(f"\nIntervention Results:")
    print(f"  Pre-intervention mean:  {pre_projections.mean().item():+.4f}")
    print(f"  Post-intervention mean: {post_projections.mean().item():+.4f}")
    print(f"  Mean shift: {intervention_shift:.4f}")
    print(f"  Direction: {'toward benign ✓' if intervention_shift > 0 else 'toward harmful ✗'}")
    
    # ==========================================================================
    # (8) Save Results
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 8: Saving Results")
    print("="*60)
    
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "n_samples_per_class": len(harmful_prompts),
            "benign_dataset": BENIGN_DATASET,
            "seed": SEED,
            "methodology": "harmful (ClearHarm) vs benign (Alpaca) comparison"
        },
        "projection_stats": stats,
        "intervention_shift": intervention_shift,
        "layer_separability": {
            str(k): {kk: float(vv) for kk, vv in v.items()} 
            for k, v in layer_stats.items()
        },
        "best_layer": best_layer,
        "best_probe_accuracy": layer_stats[best_layer]['probe_accuracy']
    }
    
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"""
Key Findings:
  - Separation (Cohen's d): {stats['cohens_d']:.4f} ({effect_interp} effect)
  - Best probe accuracy: {layer_stats[best_layer]['probe_accuracy']:.3f} (layer {best_layer})
  - Intervention shifted projections by {intervention_shift:.4f}

Output files:
  - {OUTPUT_DIR}/projection_histograms.png
  - {OUTPUT_DIR}/pca_manifold.png
  - {OUTPUT_DIR}/layerwise_separability.png
  - {OUTPUT_DIR}/results.json
""")


if __name__ == "__main__":
    main()