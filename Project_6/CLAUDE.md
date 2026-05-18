# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Linformer-IDS** is a lightweight Transformer-based Intrusion Detection System that uses the Linformer architecture to achieve O(n) computational complexity instead of the standard O(n²) of traditional Transformers. It's designed for efficient, scalable network intrusion detection suitable for edge devices while maintaining high accuracy.

The project supports multiple datasets:
- **NSL-KDD** and **UNSW-NB15**: Standard IDS datasets
- **CIC-IoT** and **Edge-IIoT**: IoT-specific attack datasets

Both binary (normal vs. attack) and multi-class classification tasks are supported.

## Setup and Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability (optional)
python tests/verify_gpu.py
```

## Project Architecture

### Core Structure

```
lightweight-transformer-ids/
├── src/                      # Core implementation
│   ├── model.py             # Linformer architecture (LinformerSelfAttention, LinformerIDS)
│   ├── trainer.py           # Training loop with early stopping, checkpointing, FocalLoss
│   ├── evaluator.py         # Evaluation metrics and visualizations
│   ├── config_manager.py    # YAML-based configuration management (ConfigManager)
│   ├── logger.py            # Logging setup
│   └── utils.py             # Utility functions, validation helpers
│
├── pipeline/                # Data preprocessing (ETL)
│   ├── ciciot_pipeline.py   # CIC-IoT specific preprocessing
│   └── edge_iiot_pipeline.py # Edge-IIoT specific preprocessing
│
├── configs/                 # YAML configuration files
│   ├── config.yaml          # Main config with profiles (default, pi, etc.)
│   └── preprocessing_config.yaml
│
├── data/                    # Dataset location
│   └── raw/                 # Raw dataset files (not committed)
│
├── models/                  # Saved model checkpoints
├── results/                 # Training results and visualizations
├── logs/                    # Training logs
├── main_ciciot.py          # Main entry point for CIC-IoT training
├── test_all_datasets_binary_ciciot.py        # Comprehensive testing for CIC-IoT
└── test_all_datasets_binary_edge_iiot.py     # Comprehensive testing for Edge-IIoT
```

### Key Modules

**src/model.py:**
- `LinformerSelfAttention`: Core attention mechanism that projects K, V to dimension k for O(n) complexity
- `LinformerIDS`: Full model combining Linformer encoder layers with classification head
- Validates k ≤ seq_len and proper head/dimension divisibility

**src/trainer.py:**
- `Trainer`: Handles training loop with early stopping, gradient clipping, model checkpointing
- `FocalLoss`: Addresses class imbalance (useful for IDS where attacks are rare)
- Supports both PyTorch native and custom loss functions

**src/config_manager.py:**
- `ConfigManager`: Loads YAML configs with profile support (default, pi for Raspberry Pi, etc.)
- `ModelConfig`, `TrainingConfig`, `PreprocessingConfig`: Dataclass-based configuration
- Validates parameters at initialization time

**src/evaluator.py:**
- `MetricsCalculator`: Computes accuracy, precision, recall, F1-score
- Generates confusion matrices, ROC curves, precision-recall curves
- Supports statistical significance tests (paired t-test, permutation test, ANOVA)

**pipeline/:** 
- `DataPreprocessor`: Sklearn-based ETL with label encoding, standardization, feature selection
- Configuration-driven preprocessing (methods use config values automatically)

## Common Development Commands

### Training

```bash
# Train on CIC-IoT with default configuration
python main_ciciot.py --mode train --config-profile default --input-file data/raw/ciciot.csv

# Train with custom hyperparameters
python main_ciciot.py --mode train --config-profile default --input-file data/raw/ciciot.csv \
  --epochs 50 --batch-size 32 --learning-rate 1e-3

# View all training options
python main_ciciot.py --help
```

### Testing and Evaluation

```bash
# Comprehensive test suite for CIC-IoT
python test_all_datasets_binary_ciciot.py

# Comprehensive test suite for Edge-IIoT
python test_all_datasets_binary_edge_iiot.py

# GPU verification
python tests/verify_gpu.py

# Example config usage
python tests/example_config_usage.py
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
flake8 src/ --max-line-length=100

# Code formatting
black src/ --line-length=100
```

## Configuration System

Configurations are defined in `configs/config.yaml` using profiles. Each profile defines:

- **Model Config**: dim (embedding), depth (layers), heads, k (projection dim), dropout, ff_hidden_mult
- **Training Config**: epochs, batch_size, learning_rate, weight_decay, gradient_clip_max_norm
- **Preprocessing Config**: binary_encode_column, positive_class, columns_to_drop, scaling method

### Loading Configuration

```python
from src.config_manager import ConfigManager

# Load with profile
config = ConfigManager.load_config("configs/config.yaml", profile="default")

# Access nested configs
model_dim = config.model.dim
batch_size = config.training.batch_size
```

## Data Handling

### Dataset Format

Raw datasets should be placed in `data/raw/` as CSV files with:
- Features as columns
- A `label` column (or configure in preprocessing_config.yaml)
- No missing values (or handle during preprocessing)

### Preprocessing Flow

The `DataPreprocessor` class handles:
1. **Encoding**: Label encoding for categorical features
2. **Binary Encoding**: Convert multi-class labels to binary (e.g., Normal=0, Attack=1)
3. **Feature Scaling**: StandardScaler normalization
4. **Feature Selection**: VarianceThreshold to remove constant features
5. **Dropping Columns**: Remove specified columns (e.g., duplicates, identifiers)

## Algorithm Reference

The core Linformer attention mechanism (Algorithm 1 in docs/algorithm.md):

1. Projects Keys and Values to dimension k using learnable matrices E and F
2. Computes attention against compressed dimensions instead of full n×n matrix
3. Achieves O(n) complexity instead of O(n²) of standard Transformers
4. Supports multi-head attention for richer representations

Key parameters:
- `k`: Projection dimension (typically much smaller than n)
- `heads`: Number of attention heads (dim must be divisible by heads)
- `dim`: Embedding dimension per token
- `depth`: Number of encoder layers

## Testing Approach

Tests are primarily integration/functional tests:
- `test_all_datasets_binary_ciciot.py`: Tests across different train/test splits
- `example_config_usage.py`: Demonstrates configuration-driven preprocessing
- `verify_gpu.py`: Checks CUDA availability and PyTorch GPU support

**Note**: No pytest unit test suite exists yet. Focus on integration testing through main scripts.

## Important Patterns and Conventions

### Configuration-First Design
All hyperparameters and preprocessing steps are configuration-driven. Use ConfigManager to load and apply settings—don't hardcode values in scripts.

### Modular Pipeline
- Data preprocessing is separated from model training
- Each pipeline class (DataPreprocessor) is composable and reusable
- Configuration objects validate parameters at initialization

### Device Agnostic
Code should work on CPU and GPU. Model and data are moved using `.to(device)`. Test on both when possible.

### Metric-Driven Evaluation
Evaluation uses comprehensive metrics (accuracy, precision, recall, F1, confusion matrix, ROC) to assess model performance across different aspects.

## Deployment Considerations

The project supports deployment profiles (default, pi for Raspberry Pi). When adding new features or changing hyperparameters:
- Update both profile configurations in `config.yaml`
- Ensure the model respects resource constraints on target devices
- Document memory/compute requirements for each profile

## Ablation Study System

A comprehensive ablation study pipeline has been implemented to systematically evaluate architectural components, design choices, and efficiency metrics of the Linformer-IDS model.

### Quick Start

```bash
# Run all ablation studies on CPU
./run_ablation_studies.sh

# Run specific ablation study on GPU
./run_ablation_studies.sh architecture cuda

# Run quick test ablation (minimal epochs)
./run_ablation_studies.sh test cpu

# Show help
./run_ablation_studies.sh --help
```

### Ablation Study Types

**Architecture Ablation** (`architecture.yaml`)
- Linformer projection dimension (k): [4, 8, 16, 32, 64]
- Model depth: [2, 4, 6, 8, 10]
- Embedding dimension: [32, 64, 128, 256]
- Attention heads: [2, 4, 8, 16]
- Interaction effects: Grid search of promising combinations

**Design Ablation** (`design.yaml`)
- Positional encoding: on/off
- Loss functions: CrossEntropy vs. Focal Loss
- Focal Loss hyperparameters: alpha and gamma tuning
- Preprocessing strategies: log transform, correlation pruning
- Embedding initialization strategies
- Attention temperature effects

**Efficiency Ablation** (`efficiency.yaml`)
- Architecture comparison: Linformer vs. Transformer
- Scaling analysis: Small, base, and large models
- Batch size sensitivity analysis
- Device comparison: CPU vs. GPU performance
- Quantization readiness assessment

### Core Components

**Main Runner Script:**
- `run_ablation_studies.sh`: Bash script to execute studies (see above for usage)

**Python Implementation:**
- `src/run_ablation.py`: Core ablation study engine with experiment execution, logging, result saving
- `pipeline/benchmark.py`: Efficiency benchmarking utilities (FLOPs, latency, memory)

**Configuration Files:**
- `configs/ablation/architecture.yaml`: Architecture study definitions
- `configs/ablation/design.yaml`: Design study definitions
- `configs/ablation/efficiency.yaml`: Efficiency study definitions

### Result Structure

Results are organized by study type with automatic timestamping:

```
results/ciciot_binary_classification/
├── architecture_ablation/
│   └── 2026-05-03_14-32-10/
│       ├── results.json          # Structured results (all experiments)
│       ├── results.csv           # Tabular format for analysis
│       └── summary.json          # Metadata and statistics
│
├── design_ablation/
│   └── 2026-05-03_15-45-20/
│       └── ...
│
└── efficiency_ablation/
    └── 2026-05-03_16-20-15/
        └── ...

logs/
├── architecture_ablation/
│   └── 2026-05-03_14-32-10/
│       └── linformer_ids_2026-05-03_14-32-10.log
├── design_ablation/
└── efficiency_ablation/
```

### Using Results

**Load Results in Python:**

```python
import pandas as pd
import json

# Load results as DataFrame
df = pd.read_csv('results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/results.csv')

# Find best experiment by F1-score
best = df.loc[df['f1'].idxmax()]
print(f"Best: {best['experiment']} with F1={best['f1']:.4f}")

# Load structured results
with open('results/ciciot_binary_classification/architecture_ablation/2026-05-03_14-32-10/results.json') as f:
    results = json.load(f)
```

**Analyze in Excel/Sheets:**
- Open `results.csv` files directly in spreadsheet applications
- Sort/filter by metrics (accuracy, F1, latency, memory)
- Create comparison charts

### Key Features

✓ **Fully Automated**: Config-driven experiments, no manual intervention needed
✓ **Reproducible**: Automatic timestamping, seed control, config snapshots
✓ **Organized**: Timestamped directories prevent result overwriting
✓ **Tracked**: Logs reasoning before/after each experiment
✓ **Flexible**: Simple variable sweeps, grid searches, boolean flags
✓ **Integrated**: Reuses existing model, trainer, evaluator—no core code changes
✓ **Device Agnostic**: Works on CPU and GPU

### Important Notes

- **Data**: By default uses `data/raw/ciciot_2023/ciciot_training_50_50.csv` (50-50 benign-attack split)
- **Timing**: Each full ablation study takes 20+ hours on CPU; use GPU for faster execution
- **Configuration Overrides**: Training epochs, batch size, and other parameters can be overridden in ablation configs
- **Analysis**: Always fill in `ANALYSIS_TEMPLATE.md` for each ablation run to document findings

### Documentation

- `ABLATION_STUDY_GUIDE.md`: Comprehensive ablation study documentation with detailed examples
- `ANALYSIS_TEMPLATE.md`: Template for documenting findings from each ablation study

## Documentation

- `docs/algorithm.md`: Detailed Linformer algorithm and mathematical formulation
- `docs/proposed_methodology.md`: High-level methodology and attack classification
- `data/DATA_README.md`: Dataset format and feature descriptions
- README.md: General project overview and installation
