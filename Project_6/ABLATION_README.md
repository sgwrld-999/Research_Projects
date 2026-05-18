# Complete Ablation Study Pipeline for Linformer-IDS

## Quick Start

```bash
# Run architecture ablation study
python lightweight-transformer-ids/src/run_ablation.py \
  --config lightweight-transformer-ids/configs/ablation/architecture.yaml \
  --data lightweight-transformer-ids/data/raw/ciciot/ciciot_training_50_50.csv

# Run design choices ablation
python lightweight-transformer-ids/src/run_ablation.py \
  --config lightweight-transformer-ids/configs/ablation/design.yaml

# Run efficiency benchmarking
python lightweight-transformer-ids/src/run_ablation.py \
  --config lightweight-transformer-ids/configs/ablation/efficiency.yaml
```

## Overview

This ablation study pipeline provides a **complete, reproducible framework** for systematically evaluating every component of the Linformer-IDS architecture:

### 🎯 Three Ablation Studies

1. **Architecture Ablation** (`configs/ablation/architecture.yaml`)
   - Linformer projection dimension (k)
   - Model depth (number of layers)
   - Embedding dimension
   - Number of attention heads
   - Interaction effects

2. **Design Ablation** (`configs/ablation/design.yaml`)
   - Positional encoding (on/off)
   - Loss functions (CrossEntropy vs. Focal Loss)
   - Focal Loss hyperparameters
   - Preprocessing strategies
   - Embedding initialization
   - Attention temperature

3. **Efficiency Ablation** (`configs/ablation/efficiency.yaml`)
   - Architecture comparison (Linformer vs. Transformer)
   - Scaling behavior
   - Batch size sensitivity
   - Device comparison (CPU vs. GPU)
   - Quantization readiness

## Directory Structure

```
lightweight-transformer-ids/
├── configs/ablation/
│   ├── architecture.yaml        # ← Architectural experiments
│   ├── design.yaml              # ← Design choice experiments
│   └── efficiency.yaml          # ← Efficiency benchmarks
│
├── src/
│   ├── run_ablation.py          # ← Main ablation runner
│   └── ... (existing modules)
│
├── pipeline/
│   ├── benchmark.py             # ← Efficiency measurement utilities
│   └── ... (existing modules)
│
├── examples/
│   └── run_ablation_example.py  # ← Usage examples
│
├── ABLATION_STUDY_GUIDE.md      # ← Detailed guide
├── ANALYSIS_TEMPLATE.md         # ← Template for analysis
└── ... (existing files)

results/ciciot_binary_classification/
├── architecture_ablation/
│   └── 2026-05-03_14-32-10/     # Timestamp-based organization
│       ├── results.json
│       ├── results.csv
│       ├── summary.json
│       └── analysis.txt         # ← Add your analysis here
│
├── design_ablation/
│   └── 2026-05-03_15-45-20/
│       ├── results.json
│       ├── results.csv
│       └── ...
│
└── efficiency_ablation/
    └── 2026-05-03_16-20-15/
        ├── results.json
        ├── results.csv
        └── ...

logs/
├── architecture_ablation/
│   └── 2026-05-03_14-32-10/
│       └── ablation.log
├── design_ablation/
│   └── ...
└── efficiency_ablation/
    └── ...
```

## Key Features

### ✅ Fully Automated
- Config-driven experiments
- Automatic timestamp generation
- Reproducible random seeds
- No manual file management

### ✅ Systematic Organization
- Timestamped results directories
- Separate logs for each run
- CSV export for easy analysis
- JSON for programmatic access

### ✅ Complete Reasoning
- Document hypothesis before experiments
- Log reasoning after results
- Track decision-making process
- Enable retrospective analysis

### ✅ Production-Grade
- Proper error handling
- Comprehensive logging
- Experiment metadata
- Configuration snapshots

### ✅ Extensible
- Easy to add new experiments
- Reuses existing training/evaluation
- Supports config overrides
- Device-agnostic (CPU/GPU)

## Detailed Usage

### 1. Running a Single Ablation Study

```bash
python lightweight-transformer-ids/src/run_ablation.py \
  --config lightweight-transformer-ids/configs/ablation/architecture.yaml \
  --data /path/to/data.csv \
  --device cuda
```

**Output**:
```
results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/
├── results.json      # All experiment results
├── results.csv       # Tabular format
├── summary.json      # Metadata
└── logs/ablation.log # Execution log
```

### 2. Running a Specific Experiment

```bash
python lightweight-transformer-ids/src/run_ablation.py \
  --config lightweight-transformer-ids/configs/ablation/architecture.yaml \
  --experiment linformer_k_sweep \
  --data /path/to/data.csv
```

### 3. Analyzing Results

```python
import pandas as pd
import json

# Load results
df = pd.read_csv("results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/results.csv")

# Find best experiment
best = df.loc[df['f1'].idxmax()]
print(f"Best: {best['experiment']} with F1={best['f1']:.4f}")

# Compare k values
k_results = df[df['experiment'].str.contains('linformer_k')]
print(k_results[['model.k', 'accuracy', 'f1', 'latency_ms_mean']])
```

### 4. Benchmarking Model Efficiency

```python
from pipeline.benchmark import ModelBenchmark
from src.model import ModelFactory
from src.config_manager import ModelConfig

model = ModelFactory.create_model(
    model_type="linformer",
    input_seq_len=78,
    num_classes=2,
    config=ModelConfig(dim=64, depth=4, heads=4, k=16)
)

benchmark = ModelBenchmark(model, input_size=(78,), device="cuda")
metrics = benchmark.benchmark()

print(f"Parameters: {metrics['total_parameters']:,}")
print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
print(f"Latency: {metrics['latency_ms_mean']:.2f}ms")
print(f"Throughput: {metrics['throughput_bs1_samples_per_sec']:.0f} samples/sec")
```

## Configuration File Format

### Simple Variable Sweep

```yaml
experiments:
  - name: "linformer_k_sweep"
    description: "Effect of k on performance"
    variables:
      k: [4, 8, 16, 32, 64]
    reasoning_before:
      - "Larger k increases capacity"
      - "But reduces computational efficiency"
```

### Grid Search

```yaml
experiments:
  - name: "depth_k_grid"
    description: "Interaction of depth and k"
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

## Typical Workflow

### 1. **Planning Phase**
- Choose which component to ablate
- Define research question
- List hypotheses in config file

### 2. **Execution Phase**
```bash
python src/run_ablation.py --config configs/ablation/architecture.yaml
```

### 3. **Analysis Phase**
- Review results CSV
- Check logs for any issues
- Analyze metrics and trends
- Fill in ANALYSIS_TEMPLATE.md

### 4. **Decision Phase**
- Identify optimal configuration
- Update base config if needed
- Plan next ablation

### 5. **Documentation Phase**
- Save analysis to `analysis.txt`
- Create plots/visualizations
- Add to research notes

## Understanding Results

### Results CSV Columns

| Column | Description |
|--------|-------------|
| `experiment` | Experiment name |
| `model.k` | Ablation value |
| `accuracy` | Classification accuracy |
| `precision` | Positive predictive value |
| `recall` | True positive rate |
| `f1` | F1-score (harmonic mean) |
| `latency_ms_mean` | Average inference latency |
| `peak_memory_mb` | Maximum GPU memory used |
| `timestamp` | When experiment was run |

### Key Metrics

- **Accuracy**: Overall correctness (useful but misleading on imbalanced data)
- **Precision**: False positive rate (important for IDS)
- **Recall**: False negative rate (critical for attack detection)
- **F1**: Harmonic mean (balanced metric)
- **Latency**: Real-time deployment requirement
- **Memory**: Edge device constraint

## Best Practices

### 1. Systematic Variation
```
❌ BAD: Change k, depth, AND dim simultaneously
✅ GOOD: Change k while holding others constant
```

### 2. Proper Documentation
```
❌ BAD: No analysis, just numbers
✅ GOOD: Fill in ANALYSIS_TEMPLATE.md with reasoning
```

### 3. Reproducibility
```
❌ BAD: Run once, don't record config
✅ GOOD: Every run is timestamped and config is saved
```

### 4. Statistical Rigor
```
❌ BAD: Report single number
✅ GOOD: Report mean ± std over multiple runs
```

## Troubleshooting

### Issue: "CUDA out of memory"
```bash
# Use smaller model or batch size
python src/run_ablation.py \
  --config configs/ablation/architecture.yaml \
  --device cpu
```

### Issue: "Results not found"
```bash
# Check logs
cat logs/architecture_ablation/2026-05-03_14-32-10/ablation.log

# Check write permissions
ls -la results/ciciot_binary_classification/
```

### Issue: "Training is very slow"
```bash
# Run with small data sample first
# Modify base_profile in YAML to use fewer epochs
```

## Advanced Usage

### Running Multiple Ablations in Sequence

```python
from src.run_ablation import AblationStudy

studies = [
    "configs/ablation/architecture.yaml",
    "configs/ablation/design.yaml",
    "configs/ablation/efficiency.yaml"
]

for study_config in studies:
    ablation = AblationStudy(study_config)
    results = ablation.run_ablation_study(
        data_path="data/raw/ciciot/ciciot_training_50_50.csv"
    )
    print(f"Completed {study_config}")
```

### Combining Results for Comparison

```python
import pandas as pd
from pathlib import Path

results_dir = Path("results/ciciot_binary_classification")
all_results = []

for ablation_dir in results_dir.iterdir():
    timestamp_dirs = sorted([d for d in ablation_dir.iterdir()])
    if timestamp_dirs:
        latest = timestamp_dirs[-1]
        df = pd.read_csv(latest / "results.csv")
        df["ablation"] = ablation_dir.name
        all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("all_ablation_results.csv", index=False)
```

## File Locations Reference

| Purpose | Location |
|---------|----------|
| Architecture ablation | `configs/ablation/architecture.yaml` |
| Design ablation | `configs/ablation/design.yaml` |
| Efficiency ablation | `configs/ablation/efficiency.yaml` |
| Ablation runner | `src/run_ablation.py` |
| Benchmark utilities | `pipeline/benchmark.py` |
| Usage examples | `examples/run_ablation_example.py` |
| Full guide | `lightweight-transformer-ids/ABLATION_STUDY_GUIDE.md` |
| Analysis template | `ANALYSIS_TEMPLATE.md` |
| Results storage | `results/ciciot_binary_classification/<ablation>/<timestamp>/` |
| Log storage | `logs/<ablation>/<timestamp>/` |

## Next Steps

1. **Start small**: Run one experiment (e.g., k sweep)
2. **Analyze results**: Use provided templates
3. **Iterate**: Based on findings, adjust configurations
4. **Document**: Fill in analysis template
5. **Compile**: Aggregate results across ablations
6. **Decide**: Choose optimal configuration
7. **Validate**: Test final model on held-out set

## Support and Questions

- Check `ABLATION_STUDY_GUIDE.md` for detailed documentation
- Review examples in `examples/run_ablation_example.py`
- Check logs in `logs/<ablation>/<timestamp>/ablation.log` for errors
- Use `ANALYSIS_TEMPLATE.md` to organize findings

---

**Last Updated**: 2026-05-03
**Version**: 1.0
**Status**: Production Ready
