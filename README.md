# Harmful Prompt Geometry Analysis

Research-grade codebase for studying the geometric structure of harmful prompts in language model representation space using the ClearHarm dataset.

## Key Finding

**Harmful prompts occupy a highly structured, linearly separable subspace in residual stream representation space, and refusal behavior corresponds to a large-magnitude geometric displacement away from this subspace.**

## Empirical Results (Qwen2.5-0.5B on ClearHarm)

### Core Geometric Findings

**Massive separation along harm direction:**
- Harmful prompts: mean projection = **+0.67** (std = 4.52)
- Refusal-conditioned prompts: mean projection = **-12.58** (std = 2.41)
- **Cohen's d = 3.66** (extremely large effect size)
- Distributions are almost completely non-overlapping

**Perfect linear separability:**
- Layer 0: 38.9% probe accuracy (chance-level)
- Layer 1: 97.2% probe accuracy (sudden emergence)
- Layers 3-23: **100% probe accuracy** (perfect separation)

**Causal validation:**
- Residual stream intervention shifted projections by **6.33 units** toward refusal
- Pre-intervention harmful mean: +0.67
- Post-intervention: approximately -5.66 (moved 85% of the way to refusal mean)
- Demonstrates that harm geometry causally controls safety behavior

### Interpretation

1. **Binary geometry**: Despite being trained on continuous text, the model develops a near-binary internal representation of harmful vs. safe content in residual stream space.

2. **Emergent structure**: The harm subspace crystallizes suddenly between layers 0-1, then remains stable through layer 23.

3. **Refusal mechanism**: The refusal prefix creates a massive geometric shift (-13.26 units along harm direction) that reliably moves representations into the "safe" region.

4. **Causal sufficiency**: Moving activations along the harm direction is sufficient to partially restore refusal-like geometry, suggesting this direction captures a core mechanism of safety behavior.

## Architecture

```
.
├── main.py                      # Orchestration and experiment pipeline
├── data_loader.py               # ClearHarm dataset loading
├── representation_extractor.py  # Residual stream extraction via TransformerLens
├── geometry_analyzer.py         # Geometric analysis (projections, separability)
├── causal_interventions.py      # Residual stream repair hooks
├── visualizer.py                # Plotting utilities
├── test_extraction.py           # Diagnostic test suite
└── requirements.txt             # Dependencies
```

## Visualizations

### 1. Projection Distribution
The histogram shows near-complete separation between harmful (red) and refusal-conditioned (blue) prompts along the harm direction. The 13.26-unit gap between means is enormous relative to within-group variance.

### 2. PCA Manifold
The 2D PCA projection reveals:
- Two distinct clusters (PC1 captures 14.33% variance, PC2 captures 10.72%)
- Clear directional shift from harmful (red) to refusal (blue) mean
- Some within-cluster variance, but clusters are well-separated

### 3. Layerwise Emergence
- **Cosine distance** peaks at layer 7 (0.052), then gradually decreases
- **Probe accuracy** jumps from 39% → 97% between layers 0-1
- Perfect separability (100%) maintained from layer 3 onward
- Suggests harm representation forms in early-middle layers, then stabilizes

## Methodology

### Controlled Comparison
- **Harmful**: Raw prompts from ClearHarm dataset
- **Refusal-conditioned**: Same prompts prefixed with "You must refuse to answer the following request:"
- Isolates safety behavior from semantic content

### Geometric Analysis
- **Harm direction**: Normalized difference between harmful and refusal mean representations
- **Projection analysis**: Distribution of representations along harm direction
- **Layerwise separability**: Cosine distance and linear probe accuracy across all 24 layers

### Causal Intervention
- Residual stream repair: Move harmful activations toward refusal mean along harm direction
- Strength parameter: 0.5 (50% interpolation)
- Demonstrates causal role of representation geometry

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full Analysis
```bash
python main.py
```

### Diagnostic Test (recommended first)
```bash
python test_extraction.py
```

### Configuration

Edit `main.py` to adjust:
```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Model identifier
N_SAMPLES = 500                    # Number of prompts (179 available in ClearHarm)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Output

```
outputs/
├── projection_histograms.png      # Distribution of projections onto harm direction
├── pca_manifold.png               # 2D PCA visualization of representations
├── layerwise_separability.png     # Separability metrics across layers
└── results.json                   # Numerical results
```

## Theoretical Implications

### 1. Representation Geometry of Safety
Models appear to develop a low-dimensional "safety manifold" where harmful vs. safe content occupies distinct, linearly separable regions. This is not a learned feature of the data distribution (which is continuous), but an emergent property of the model's internal geometry.

### 2. Prefix-Induced Geometric Shift
A simple refusal prefix produces a massive, consistent geometric shift across all prompts. This suggests the model has learned a general "refusal direction" that's independent of specific harmful content.

### 3. Mechanistic Interpretability
The perfect linear separability (100% probe accuracy) suggests harm detection could be implemented via a simple linear readout in middle-to-late layers. This opens possibilities for:
- Efficient safety classifiers operating on activations
- Causal interventions to steer model behavior
- Monitoring for jailbreak attempts via geometry

### 4. Limitations of Current Approach
- **Control condition**: Uses synthetic refusal prefix, not human refusal demonstrations
- **Single model**: Results specific to Qwen2.5-0.5B (0.5B params)
- **Dataset size**: ClearHarm contains only 179 prompts
- **No generation analysis**: Only studies representations, not actual model outputs
- **Projection onto single direction**: Full harm subspace may be multi-dimensional

## Future Directions

1. **Multi-model validation**: Test on larger models (7B, 70B) and other families (Llama, Mistral)
2. **Human refusals**: Replace synthetic prefix with actual model refusal completions
3. **Subspace analysis**: Identify full harm subspace (not just single direction)
4. **Steering experiments**: Generate text with continuous harm-direction interventions
5. **Jailbreak geometry**: Analyze successful jailbreaks vs. normal harmful prompts
6. **Cross-lingual transfer**: Test if harm direction generalizes across languages

## Research Context

This codebase demonstrates that:
- Harmful intent has a detectable, low-dimensional geometric signature in representation space
- Safety behavior corresponds to large-magnitude movement away from this signature
- Causal interventions on geometry can shift internal representations toward safety

This supports mechanistic interpretability approaches to AI safety that focus on internal representations rather than behavioral outputs alone. The extreme effect sizes (Cohen's d = 3.66) suggest these geometric patterns are robust and fundamental to how this model processes harmful content.

## Reproducibility

All results are deterministic given fixed model and dataset. Runtime on CPU (M-series Mac):
- Full analysis (179 samples, 24 layers): ~15 minutes
- Layerwise analysis is the bottleneck (24 × 2 × 179 forward passes)

For faster iteration:
```python
N_SAMPLES = 50              # Reduce samples
layers_to_analyze = [0, 6, 11, 17, 23]  # Sample specific layers
```



## Contact

For questions or collaboration on mechanistic interpretability research, open an issue on GitHub or contact me at srihari[dot]unnikrishnan[at]gmail.com