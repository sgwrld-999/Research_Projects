# Ablation Study Guide for Linformer-IDS

## Overview

This guide documents the complete ablation study pipeline for the Linformer-based Intrusion Detection System. The ablation study systematically evaluates the contribution of each architectural and design component to overall performance.

## Directory Structure

```
configs/ablation/
├── architecture.yaml      # Architectural ablation (k, depth, dim, heads)
├── design.yaml           # Design choice ablation (positional encoding, loss functions, etc.)
└── efficiency.yaml       # Efficiency benchmarking (latency, memory, throughput)

src/
├── run_ablation.py       # Main ablation study runner
└── ... (existing modules)

pipeline/
├── benchmark.py          # Efficiency benchmarking utilities
└── ... (existing modules)

results/ciciot_binary_classification/
├── <ablation_name>/
│   └── <timestamp>/
│       ├── results.json
│       ├── results.csv
│       ├── summary.json
│       └── analysis.txt

logs/
├── <ablation_name>/
│   └── <timestamp>/
│       └── ablation.log
```

## Ablation Studies

### 1. Architectural Ablation (`architecture.yaml`)

Studies the impact of core Linformer architectural components:

**Experiments:**
- **linformer_k_sweep**: Projection dimension k ∈ {4, 8, 16, 32, 64}
  - Tests the accuracy-efficiency tradeoff
  - k << seq_len is critical for linear complexity
  
- **depth_sweep**: Number of layers ∈ {2, 4, 6, 8, 10}
  - Evaluates model capacity and learning depth
  - Risk of overfitting with excessive depth
  
- **embedding_dim_sweep**: Embedding dimension ∈ {32, 64, 128, 256}
  - Feature representation capacity
  - Trade-off with model size
  
- **heads_sweep**: Attention heads ∈ {2, 4, 8, 16}
  - Multi-head attention expressivity
  - Must satisfy: dim % heads == 0
  
- **architecture_combinations**: Grid search of promising combinations
  - Test interaction effects between components

**Key Metrics:**
- Accuracy, Precision, Recall, F1-score
- Model size (parameters, MB)
- Inference latency
- Memory usage

### 2. Design Ablation (`design.yaml`)

Studies design choices and preprocessing strategies:

**Experiments:**
- **positional_encoding**: Enable/disable position information
  - Tests if sequence order matters for tabular IDS data
  
- **loss_function**: CrossEntropy vs. Focal Loss
  - Addresses class imbalance in CIC-IoT dataset
  
- **focal_loss_hyperparameters**: Tune α ∈ {0.1, 0.25, 0.5, 0.75}, γ ∈ {1.0, 2.0, 3.0, 4.0}
  - Optimize for attack detection
  
- **preprocessing_strategy**: Log transform and correlation pruning
  - Feature space quality and interpretability
  
- **embedding_strategy**: Initialization methods
  - Impact on convergence and final performance
  
- **attention_temperature**: Softmax temperature ∈ {0.5, 1.0, 2.0, 4.0}
  - Attention focus sharpness

**Key Metrics:**
- Class-specific precision/recall (especially for attacks)
- Training stability and convergence speed
- Feature importance distribution
- Attention weight concentration

### 3. Efficiency Ablation (`efficiency.yaml`)

Benchmarks model efficiency across configurations:

**Benchmarks:**
- **architecture_comparison**: Linformer vs. Standard Transformer
  - Validates O(n) vs. O(n²) complexity
  
- **scale_analysis**: Performance as model grows
  - Identify scaling behavior
  
- **batch_size_sensitivity**: Efficiency across batch sizes
  - Real-time deployment considerations
  
- **device_comparison**: CPU vs. GPU performance
  - Deployment viability on edge devices
  
- **quantization_analysis**: Readiness for int8 quantization
  - Further efficiency improvements

**Key Metrics:**
- Parameter count
- FLOPs (floating point operations)
- Latency (mean, std, p95, p99)
- Throughput (samples/sec)
- Peak memory
- Quantization potential

## Running Ablation Studies

### Prerequisites

```bash
pip install -r requirements.txt
# For efficiency benchmarking (optional):
pip install fvcore
```

### Quick Start

Run all architecture experiments:
```bash
python src/run_ablation.py \
  --config configs/ablation/architecture.yaml \
  --data lightweight-transformer-ids/data/raw/ciciot/ciciot_training_50_50.csv
```

Run specific experiment:
```bash
python src/run_ablation.py \
  --config configs/ablation/architecture.yaml \
  --experiment linformer_k_sweep \
  --data path/to/data.csv
```

Run on specific device:
```bash
python src/run_ablation.py \
  --config configs/ablation/design.yaml \
  --device cpu
```

### Understanding Output Structure

After each run, results are organized as:
```
results/ciciot_binary_classification/<ablation_name>/<YYYY-MM-DD_HH-MM-SS>/
├── results.json        # Raw results (one object per experiment)
├── results.csv         # Tabular format (for Excel/analysis)
├── summary.json        # Run metadata and statistics
└── analysis.txt        # Manual analysis and findings (user-added)

logs/<ablation_name>/<YYYY-MM-DD_HH-MM-SS>/
└── ablation.log        # Complete execution log
```

## Analyzing Results

### 1. Quick Review

```python
import json
import pandas as pd

# Load results
with open("results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/results.json") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.read_csv("results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/results.csv")

# Compare k values
df[df['experiment'].str.contains('linformer_k')].sort_values('model.k')
```

### 2. Visualization Examples

```python
import matplotlib.pyplot as plt

# Plot k vs. Accuracy
plt.figure(figsize=(10, 6))
k_results = df[df['experiment'].str.contains('linformer_k')]
plt.plot(k_results['model.k'], k_results['accuracy'], 'o-')
plt.xlabel('k (Projection Dimension)')
plt.ylabel('Accuracy')
plt.title('Linformer K Sweep')
plt.grid(True)
plt.savefig('k_sweep_analysis.png')
```

### 3. Detailed Analysis Template

For each ablation study, create an `analysis.txt` file:

```
ABLATION STUDY: Architecture Components
========================================

BEFORE EXPERIMENTS:
- Research Question: How do different k values affect the accuracy-efficiency tradeoff?
- Hypothesis: Larger k improves accuracy but reduces computational efficiency
- Expected Behavior: Accuracy plateau around k=16-32

EXPERIMENT RESULTS:
- k=4: 94.2% accuracy, 2.1ms latency
- k=8: 96.1% accuracy, 2.8ms latency  [BEST ACCURACY]
- k=16: 96.3% accuracy, 4.2ms latency [MINIMAL GAIN]
- k=32: 96.4% accuracy, 7.1ms latency [DIMINISHING RETURNS]
- k=64: 96.5% accuracy, 12.3ms latency [NOT WORTH THE COST]

KEY FINDINGS:
1. Sweet spot at k=8: 96.1% accuracy with reasonable latency
2. Diminishing returns beyond k=16
3. k=4 is too restrictive, sacrifices 2% accuracy

TRADEOFF ANALYSIS:
- For real-time IDS (latency < 5ms): Use k=8
- For maximum accuracy (latency not critical): Use k=16
- Not recommended: k >= 32 (poor efficiency)

NEXT STEPS:
- Use k=8 in subsequent ablations
- Investigate depth sweep to compensate for k restriction
```

## Key Concepts

### Understanding k (Projection Dimension)

```
Standard Transformer Attention:  O(n²) - projects Q, K, V to full dimension
                                 Attention = softmax(Q·K^T / √d) · V

Linformer Attention:              O(n) - projects K, V to dimension k
                                 K̄ = E·K  (E ∈ ℝ^(k×n))
                                 V̄ = F·V  (F ∈ ℝ^(k×n))
                                 Attention = softmax(Q·K̄^T / √d) · V̄
```

**Constraint**: k ≤ seq_len (78 for CIC-IoT)

**Tradeoff**:
- Small k: Better efficiency, reduced model capacity
- Large k: More capacity, approaches O(n²) complexity

### Class Imbalance in CIC-IoT

CIC-IoT dataset is imbalanced:
- Benign traffic: ~95%
- Attacks: ~5%

**Implications**:
- Standard accuracy is misleading
- Focus on attack detection (recall, F1 for attack class)
- Focal Loss helps down-weight easy negatives

## Best Practices

1. **Reproducibility**
   - All timestamps are automatic
   - Configs are saved with each experiment
   - Random seeds are fixed in ConfigManager

2. **Result Organization**
   - Each run gets unique timestamp
   - Never overwrite previous results
   - CSV format for easy Excel import

3. **Systematic Experimentation**
   - Vary one component at a time initially
   - Grid search interactions for promising combinations
   - Document reasoning before and after

4. **Analysis Quality**
   - Don't just report metrics, explain WHY
   - Analyze statistically (confidence intervals, trends)
   - Consider practical implications (deployment constraints)

## Common Patterns

### Single Variable Sweep
```yaml
experiments:
  - name: "linformer_k_sweep"
    variables:
      k: [4, 8, 16, 32, 64]
```

### Grid Search
```yaml
experiments:
  - name: "depth_k_grid"
    variables_grid:
      - depth: [2, 4, 6]
        k: [8, 16, 32]
```

### Boolean Flags
```yaml
experiments:
  - name: "positional_encoding"
    variables_boolean:
      enable_positional_encoding: [true, false]
```

## Troubleshooting

### Out of Memory
- Reduce batch size in base profile
- Lower embedding dimension
- Use CPU device instead of GPU

### Slow Training
- Reduce number of epochs in base profile
- Use smaller k values
- Reduce depth

### Results Not Saved
- Check logs in `logs/<ablation_name>/<timestamp>/ablation.log`
- Ensure write permissions to `results/` directory
- Verify config file path is correct

## Next Steps After Ablation Study

1. **Aggregate Results**
   - Combine CSV files from multiple ablations
   - Create master comparison table

2. **Statistical Analysis**
   - Compute confidence intervals
   - Test significance of differences
   - Identify optimal configurations

3. **Final Model Selection**
   - Choose configuration balancing accuracy and efficiency
   - Validate on held-out test set
   - Document final architecture

4. **Publication**
   - Include ablation study results in paper/report
   - Show reasoning before and after each ablation
   - Justify all design choices
