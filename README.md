# Harmful Prompt Geometry Analysis

Research codebase for studying the geometric structure of harmful prompts in language model representation space, comparing harmful prompts (ClearHarm dataset) against benign prompts (Alpaca dataset).

## Key Finding

**Harmful prompts are geometrically separated from benign prompts in residual stream representation space, with near-perfect linear separability (99.7%) emerging in middle layers and a large effect size (Cohen's d = 2.87) demonstrating distinct internal representations.**

## Results Summary

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-0.5B |
| Samples per class | 179 |
| Harmful mean projection | +37.36 (std = 3.51) |
| Benign mean projection | +17.40 (std = 9.18) |
| Cohen's d | 2.87 |
| Best probe accuracy | 99.7% (layers 13-15, 18) |
| Intervention shift | 10.68 units toward benign |

## Visualizations

### Projection Distribution
![Projection Histograms](outputs/projection_histograms.png)

The histogram reveals distinct projection distributions along the harm direction:

- **Harmful prompts (red)**: Tightly clustered around mean +37.36 with low variance (std=3.51), forming a sharp peak
- **Benign prompts (green)**: Broadly distributed around mean +17.40 with higher variance (std=9.18), showing multiple modes spanning 0-35
- **Separation**: 19.96-unit gap between means with minimal overlap in the 25-35 range
- **Effect size**: Cohen's d = 2.87 indicates large practical significance

The tight clustering of harmful prompts suggests they share common geometric features, while benign prompts exhibit greater semantic diversity in representation space.

### PCA Manifold
![PCA Manifold](outputs/pca_manifold.png)

2D PCA projection of final-layer representations:

- **Variance explained**: PC1 captures 15.6%, PC2 captures 5.9% (21.5% total)
- **Harmful cluster (red)**: Concentrated in upper-right quadrant, compact and coherent
- **Benign cluster (green)**: Distributed in lower-left region with greater spread
- **Direction of separation**: Black arrow connects cluster means, visualizing the 19.96-unit geometric displacement
- **Overlap region**: Some mixing in the central area, consistent with the distributional overlap seen in projections

The PCA confirms that harmful/benign separation is not an artifact of the single harm direction but reflects genuine geometric structure in the full representation space.

### Layerwise Separability
![Layerwise Separability](outputs/layerwise_separability.png)

Analysis across all 24 transformer layers reveals progressive emergence of harm geometry:

**Cosine Distance (left panel):**
- Layer 0: 0.027 (minimal separation at embedding)
- Layers 12-16: Peak at ~0.095 (maximum geometric divergence)
- Layers 22-23: Decreases to ~0.035 (partial convergence near output)

**Linear Probe Accuracy (right panel):**
- Layer 0: 72.6% (above chance, early separation signal)
- Layer 5: 98.6% (rapid improvement in early-middle layers)
- Layers 13-15, 18: 99.7% (peak separability, near-perfect classification)
- Layers 19-23: 99.2% (sustained high accuracy through final layers)

The inverted-U pattern in cosine distance suggests the model develops maximally distinct harm representations in middle layers before partially reconverging toward output. Probe accuracy plateaus earlier and remains high, indicating robust linear separability once established.

## Methodology

### Dataset Comparison
- **Harmful**: 179 prompts from ClearHarm dataset (explicit harmful requests)
- **Benign**: 179 prompts from Alpaca dataset (general instruction-following tasks)

### Analysis Pipeline
1. **Representation extraction**: Final-layer residual stream activations via TransformerLens
2. **Harm direction**: Normalized vector from benign mean to harmful mean
3. **Projection analysis**: Scalar projection of each sample onto harm direction
4. **Layerwise probing**: Logistic regression classifier at each layer (5-fold CV)
5. **Causal intervention**: Shift harmful activations toward benign mean, measure projection change

### Causal Validation
Residual stream intervention shifted harmful projections by **10.68 units** toward benign (from +37.36 to ~+26.68), demonstrating that harm geometry is causally modifiable—not merely correlational.

## Complete Results

```json
{
  "metadata": {
    "model": "Qwen/Qwen2.5-0.5B",
    "n_samples_per_class": 179,
    "benign_dataset": "tatsu-lab/alpaca",
    "seed": 42,
    "methodology": "harmful (ClearHarm) vs benign (Alpaca) comparison"
  },
  "projection_stats": {
    "harmful_mean": 37.36,
    "harmful_std": 3.51,
    "benign_mean": 17.40,
    "benign_std": 9.18,
    "cohens_d": 2.87
  },
  "intervention_shift": 10.68,
  "best_layer": 13,
  "best_probe_accuracy": 0.997
}
```

<details>
<summary>Full layer-by-layer separability data</summary>

| Layer | Cosine Distance | Probe Accuracy |
|-------|-----------------|----------------|
| 0 | 0.027 | 72.6% |
| 1 | 0.064 | 86.9% |
| 2 | 0.040 | 92.5% |
| 3 | 0.066 | 93.3% |
| 4 | 0.073 | 94.4% |
| 5 | 0.071 | 98.6% |
| 6 | 0.068 | 98.3% |
| 7 | 0.073 | 98.6% |
| 8 | 0.062 | 98.0% |
| 9 | 0.077 | 98.6% |
| 10 | 0.075 | 98.3% |
| 11 | 0.076 | 99.2% |
| 12 | 0.090 | 99.2% |
| 13 | 0.096 | **99.7%** |
| 14 | 0.092 | **99.7%** |
| 15 | 0.092 | **99.7%** |
| 16 | 0.094 | 99.2% |
| 17 | 0.078 | 99.4% |
| 18 | 0.063 | **99.7%** |
| 19 | 0.058 | 99.2% |
| 20 | 0.058 | 99.2% |
| 21 | 0.048 | 99.2% |
| 22 | 0.033 | 99.2% |
| 23 | 0.036 | 99.2% |

</details>

## Architecture

```
.
├── main.py                      # Experiment orchestration
├── data_loader.py               # ClearHarm/Alpaca dataset loading
├── representation_extractor.py  # Residual stream extraction (TransformerLens)
├── geometry_analyzer.py         # Geometric analysis (projections, probes)
├── causal_interventions.py      # Activation interventions
├── visualizer.py                # Plotting utilities
└── outputs/
    ├── projection_histograms.png
    ├── pca_manifold.png
    ├── layerwise_separability.png
    └── results.json
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis
python main.py

# Quick diagnostic test
python test_extraction.py
```

### Configuration

Edit `main.py` to adjust parameters:
```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_SAMPLES = 179  # samples per class
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Implications

1. **Distinct geometric signatures**: Harmful and benign content occupy separable regions in representation space, with harmful prompts clustering tightly at higher projections along the harm direction.

2. **Progressive layer development**: Linear separability improves from 72.6% (layer 0) to 99.7% (layer 13), indicating the model builds increasingly refined harm representations through its forward pass.

3. **Practical detection**: Near-perfect probe accuracy suggests efficient harm classifiers could operate directly on intermediate activations.

4. **Causal steerability**: The 10.68-unit intervention shift demonstrates that representation geometry can be causally modulated, enabling potential steering approaches.

## Limitations

- Single model (Qwen2.5-0.5B, 0.5B parameters)
- Small dataset (179 samples per class)
- No controlled semantic variations (different datasets rather than matched pairs)
- Representation analysis only (no generation/behavior analysis)
- Single harm direction (full subspace may be multi-dimensional)

## Reproducibility

Results are deterministic with fixed seed (42). Runtime on M-series Mac CPU: ~15 minutes for full 24-layer analysis.

For faster iteration:
```python
N_SAMPLES = 50
layers_to_analyze = [0, 6, 12, 18, 23]
```
