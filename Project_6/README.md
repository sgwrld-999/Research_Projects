# Lightweight Transformer Intrusion Detection System (Linformer-IDS)

## Overview

This repository provides a full implementation of an **Intrusion Detection System (IDS)** supporting multiple architectures for comprehensive model comparison and ablation studies. The project includes:

- **Linformer**: Lightweight Transformer using low-rank projections to reduce complexity from O(n²) to O(n)
- **Transformer (Full)**: Standard transformer with O(n²) full attention for comparison baseline
- **LSTM**: Bidirectional LSTM treating features as time sequences
- **CNN**: 1D convolutional architecture with adaptive pooling
- **MLP**: Multi-layer perceptron baseline for feature-wise processing

The Linformer architecture approximates the attention matrix with low-rank projections, significantly reducing computational and memory requirements. By supporting multiple architectures, this project enables systematic evaluation and ablation studies for efficient and scalable network intrusion detection suitable for edge devices.

**Supported Datasets:**
- **CIC-IoT 2023**: IoT-specific attack dataset with comprehensive attack scenarios
- **Edge-IIoT**: Industrial IoT attack dataset for edge computing environments

Both binary (normal vs. attack) and multi-class classification tasks are supported.

The project includes:
- ✓ Modular architecture with SOLID principles
- ✓ Configuration-driven training and preprocessing
- ✓ **Comprehensive ablation study system** for architectural evaluation
- ✓ Multiple baseline models for comparison
- ✓ Training efficiency tracking (epoch times, convergence metrics)
- ✓ Extensive evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC curves)
- ✓ Reproducible training scripts

## Repository Structure

```
Project_6/
├── lightweight-transformer-ids/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration definitions
│   │   ├── config_manager.py      # YAML-based configuration management
│   │   ├── data_loader.py         # Data loading and preprocessing
│   │   ├── model.py               # Model implementations (Linformer, Transformer, LSTM, CNN, MLP)
│   │   ├── trainer.py             # Training loop with early stopping, checkpointing
│   │   ├── evaluator.py           # Evaluation metrics and visualizations
│   │   ├── logger.py              # Logging setup
│   │   └── utils.py               # Utility functions
│   │
│   ├── pipeline/
│   │   ├── benchmark.py           # Efficiency benchmarking utilities
│   │   └── pipeline.py            # Data preprocessing pipeline
│   │
│   ├── configs/
│   │   ├── config.yaml            # Main configuration with profiles
│   │   └── ablation/              # Ablation study configurations
│   │       ├── architecture.yaml   # Architecture ablation settings
│   │       ├── design.yaml         # Design ablation settings
│   │       └── efficiency.yaml     # Efficiency ablation settings
│   │
│   ├── data/
│   │   └── raw/                   # Raw dataset files (not committed)
│   │
│   ├── models/                    # Saved model checkpoints
│   ├── results/                   # Training results and visualizations
│   ├── logs/                      # Training logs
│   │
│   ├── main_ciciot.py            # Main entry point for CIC-IoT training
│   ├── test_all_datasets_binary_ciciot.py        # Comprehensive CIC-IoT testing
│   └── test_all_datasets_binary_edge_iiot.py     # Comprehensive Edge-IIoT testing
│
├── run_ablation_studies.sh        # Bash script to run ablation studies
├── train_7class_chunked.py        # Multi-class training script
├── CLAUDE.md                      # Developer guide
├── ABLATION_README.md             # Ablation study documentation
└── README.md                      # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Project_6/lightweight-transformer-ids
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU availability** (optional):
   ```bash
   python tests/verify_gpu.py
   ```

## Preparing Datasets

Download datasets from their official sources and place CSV files in `data/raw/`:

```
data/raw/ciciot_2023/ciciot_training_50_50.csv
data/raw/edge_iiot_training.csv
```

The data loader expects:
- CSV format with features as columns
- A `label` column (or configure in preprocessing_config.yaml)
- No missing values (or handle during preprocessing)

## Usage

### Basic Training

Train on CIC-IoT 2023 with default configuration:
```bash
python main_ciciot.py --mode train --config-profile default \
    --input-file data/raw/ciciot_2023/ciciot_training_50_50.csv
```

Train with custom hyperparameters:
```bash
python main_ciciot.py --mode train --config-profile default \
    --input-file data/raw/ciciot_2023/ciciot_training_50_50.csv \
    --epochs 50 --batch-size 32 --learning-rate 1e-3
```

Train with specific model type:
```bash
python main_ciciot.py --mode train --config-profile default \
    --input-file data/raw/ciciot.csv \
    --model-type linformer  # or: transformer, lstm, cnn, mlp
```

View all available options:
```bash
python main_ciciot.py --help
```

### Model Selection

The project supports multiple models for comparison:

- **linformer**: Default lightweight transformer with O(n) complexity
- **transformer**: Full attention transformer (O(n²)) for baseline comparison
- **lstm**: Bidirectional LSTM processing features as sequences
- **cnn**: 1D CNN with convolutional blocks
- **mlp**: Multi-layer perceptron for simple baseline

### Ablation Studies

Run comprehensive ablation studies to evaluate architectural components:

```bash
# Run all ablation studies (architecture, design, efficiency)
./run_ablation_studies.sh

# Run specific ablation type on GPU
./run_ablation_studies.sh architecture cuda

# Run quick test ablation (minimal epochs)
./run_ablation_studies.sh test cpu

# Show help
./run_ablation_studies.sh --help
```

**Ablation Study Types:**

1. **Architecture Ablation** - Evaluates core model parameters:
   - Linformer projection dimension (k)
   - Model depth (number of layers)
   - Embedding dimension
   - Attention heads
   - Interaction effects and combinations

2. **Design Ablation** - Evaluates design choices:
   - Positional encoding (on/off)
   - Loss functions (CrossEntropy vs. Focal Loss)
   - Focal Loss hyperparameters
   - Preprocessing strategies
   - Embedding initialization
   - Attention temperature

3. **Efficiency Ablation** - Evaluates computational efficiency:
   - Architecture comparison (Linformer vs. Transformer)
   - Model scaling (Small, Base, Large)
   - Batch size sensitivity
   - Device performance (CPU vs. GPU)
   - Quantization readiness

Results are automatically organized by study type with timestamps:
```
results/ciciot_binary_classification/
├── architecture_ablation/2026-05-03_14-32-10/
│   ├── results.csv
│   ├── results.json
│   └── summary.json
├── design_ablation/...
└── efficiency_ablation/...
```

### Testing and Evaluation

Comprehensive test suites:
```bash
# Test on CIC-IoT
python test_all_datasets_binary_ciciot.py

# Test on Edge-IIoT
python test_all_datasets_binary_edge_iiot.py
```

## Configuration System

Configurations are YAML-based with profile support. Each profile defines:

- **Model Config**: dim (embedding), depth (layers), heads, k (projection dim), dropout
- **Training Config**: epochs, batch_size, learning_rate, weight_decay, gradient_clip
- **Preprocessing Config**: encoding, scaling, feature selection methods

Load configuration in code:
```python
from src.config_manager import ConfigManager

config = ConfigManager.load_config("configs/config.yaml", profile="default")
model_dim = config.model.dim
batch_size = config.training.batch_size
```

## Training Efficiency Tracking

The trainer automatically tracks:
- **Epoch times**: Execution time per epoch (seconds)
- **Convergence epoch**: When the best validation score was achieved
- **Total training time**: Complete training duration
- **Average epoch time**: Mean time per epoch

These metrics are logged at the end of training and help identify:
- Model computational efficiency
- Training convergence speed
- Resource utilization patterns

Example logging output:
```
Best validation accuracy: 0.9876
Convergence epoch: 15
Total training time: 450.25s
Avg epoch time: 22.51s
```

## Project Structure and Modularity

### Core Models (src/model.py)

All models follow a consistent interface:

```python
class ModelName(nn.Module):
    def __init__(self, input_seq_len: int, num_classes: int, **config):
        # Initialize model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
    
    def count_parameters(self) -> int:
        # Return trainable parameter count
```

### ModelFactory for Dynamic Creation

Create models dynamically from configuration:

```python
from src.model import ModelFactory

model = ModelFactory.create(
    input_seq_len=42,
    num_classes=2,
    model_config=config.model,
    model_type="linformer"  # or: transformer, lstm, cnn, mlp
)
```

## Extending the Project

The modular architecture allows easy extension:

**Add a new dataset:**
- Implement a loader function in `src/data_loader.py`
- Update configuration in `configs/config.yaml`
- No changes needed to training pipeline

**Add a new model:**
- Create model class in `src/model.py` following the standard interface
- Add initialization logic to `ModelFactory.create()`
- Update ablation configs as needed
- Trainer works without modification

**Add a new ablation study:**
- Create config file in `configs/ablation/`
- Define parameter sweeps and combinations
- Update `src/run_ablation.py` to recognize new study type

## Code Quality

Type checking and linting:
```bash
mypy src/
flake8 src/ --max-line-length=100
black src/ --line-length=100
```

## Documentation

- `CLAUDE.md` - Comprehensive developer guide
- `ABLATION_README.md` - Ablation study system documentation
- `lightweight-transformer-ids/ABLATION_STUDY_GUIDE.md` - Detailed ablation examples
- `lightweight-transformer-ids/ANALYSIS_TEMPLATE.md` - Template for documenting results
- `docs/` - Algorithm specifications and methodology

## Key Features

✓ **Multiple Models**: Compare Linformer, Transformer, LSTM, CNN, and MLP baselines
✓ **Automated Ablation Studies**: Systematically evaluate architectural components
✓ **Configuration-Driven**: All hyperparameters and preprocessing via YAML configs
✓ **Efficiency Tracking**: Monitor training time, convergence, and computational metrics
✓ **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrix, ROC curves
✓ **Reproducible**: Automatic timestamping, seed control, config snapshots
✓ **Device Agnostic**: Works on CPU and GPU with automatic device management
✓ **Modular Design**: Easy to extend with new datasets, models, or evaluation methods

## References

- **Linformer**: Sinai Berger et al. "Linformer: Self-Attention with Linear Complexity"
- **CIC-IoT 2023**: Canadian Institute for Cybersecurity IoT Attack Dataset
- **Edge-IIoT**: Industrial IoT Intrusion Detection Dataset

## Authors

Developed at IIIT Guwahati for intrusion detection research.

## License

See LICENSE file for details.
