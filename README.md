# Harmful Prompt Geometry Analysis

Research-grade codebase for studying the geometric structure of harmful prompts in language model representation space using the ClearHarm dataset.

## Hypothesis

Harmful intent corresponds to a structured subspace in residual stream representation space, and safety behavior (refusal) corresponds to geometric displacement away from this subspace.

## Architecture

```
.
├── main.py                      # Orchestration and experiment pipeline
├── data_loader.py               # ClearHarm dataset loading
├── representation_extractor.py  # Residual stream extraction via TransformerLens
├── geometry_analyzer.py         # Geometric analysis (projections, separability)
├── causal_interventions.py      # Residual stream repair hooks
├── visualizer.py                # Plotting utilities
└── requirements.txt             # Dependencies
```

## Key Features

### 1. **Controlled Comparison**
- Harmful prompts from ClearHarm dataset
- Refusal-conditioned control: same prompts with refusal prefix
- Isolates safety behavior from content

### 2. **Geometric Analysis**
- **Harm direction**: Normalized difference between harmful and refusal mean representations
- **Projection analysis**: Distribution of representations along harm direction
- **Layerwise separability**: Cosine distance and linear probe accuracy across layers

### 3. **Causal Intervention**
- Residual stream repair: Move harmful activations toward refusal mean
- Demonstrates causal role of representation geometry in safety behavior

### 4. **Robustness**
- One-at-a-time tokenization (no batch corruption)
- Strict input filtering
- Extraction from `resid_pre` only

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Configuration

Edit `main.py` to adjust:
- `MODEL_NAME`: Model identifier (default: Qwen/Qwen2.5-0.5B)
- `N_SAMPLES`: Number of prompts to analyze
- `DEVICE`: `cuda` or `cpu`

## Output

```
outputs/
├── projection_histograms.png      # Distribution of projections onto harm direction
├── pca_manifold.png               # 2D PCA visualization of representations
├── layerwise_separability.png     # Separability metrics across layers
└── results.json                   # Numerical results
```

## Results Interpretation

### Expected Findings

1. **Projection separation**: Harmful prompts project more strongly onto harm direction than refusal-conditioned prompts (positive Cohen's d)

2. **Layerwise emergence**: Separability increases in middle-to-late layers

3. **Causal shift**: Residual repair intervention shifts projections toward refusal distribution

### Diagnostic Metrics

- **Cohen's d > 0.8**: Large effect size, clear geometric separation
- **Probe accuracy > 0.9**: Highly linear separability
- **Intervention shift > 0**: Causal validation of geometry

## Limitations

- Single model (Qwen 0.5B) — results may not generalize
- No multi-prompt batching (slower but more robust)
- Control condition is simple prefix, not human refusals
- Does not analyze generated text (geometry only)

## Research Context

This codebase demonstrates that:
- Harmful intent has a detectable geometric signature in representation space
- Safety behavior corresponds to movement away from this signature
- Causal interventions can partially restore this movement

This supports mechanistic interpretability approaches to AI safety that focus on internal representations rather than behavioral outputs alone.
