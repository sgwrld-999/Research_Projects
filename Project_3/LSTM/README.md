# LSTM Network Intrusion Detection System

A professional implementation of LSTM-based neural networks for network intrusion detection, with **CUDA-accelerated PyTorch** support for high-performance GPU training.

## 🚀 Quick Start - CUDA Training

### Prerequisites
- NVIDIA GPU with CUDA support
- PyTorch with CUDA installed
- Conda environment: `pyTorch_LSTM`

### Running CUDA-Accelerated Training

**Option 1: Using Batch Script (Windows)**
```bash
cd C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\LSTM
run_cuda_training.bat
```

**Option 2: Using PowerShell**
```powershell
cd C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\LSTM
.\run_cuda_training.ps1
```

**Option 3: Manual Execution**
```bash
# Activate environment
conda activate pyTorch_LSTM

# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run training
python scirpts\train_pytorch_cuda.py
```

### What the Training Does

✅ **Automatic CUDA Verification** - Checks GPU availability and prints device information  
✅ **No CPU Fallback** - Training fails if CUDA is not available (as required)  
✅ **Timestamp-Based Results** - All outputs saved in `results/results_YYYY-MM-DD_HH-MM-SS/`  
✅ **Comprehensive Visualizations**:
  - ROC Curves with AUC (4 decimal places)
  - Confusion Matrix
  - Training/Validation Curves (Loss & Accuracy)
✅ **Detailed Metrics**:
  - `classification_report.txt` - Detailed per-class metrics
  - `classification_metrics.csv` - Structured metrics data
✅ **Reproducible** - Fixed random seed (42) for consistent results

---

## 📁 Project Structure (Updated)

```
LSTM/
├── config/                         # Configuration files
│   ├── default.yaml
│   └── lstm_config_experiment_1.yaml
├── lstm/                          # Core LSTM package
│   ├── __init__.py
│   ├── config_loader.py          # Pydantic configuration
│   ├── builder.py                # TensorFlow model builder
│   └── lstm_with_softmax.py
├── scirpts/                       # Training scripts
│   ├── train.py                  # TensorFlow CPU training
│   ├── train_pytorch_cuda.py     # 🆕 PyTorch CUDA training
│   └── evaluate.py
├── logs/                          # Training logs
│   └── training_cuda_*.log       # Timestamped CUDA logs
├── models/                        # Saved models
│   └── saved_Models/
├── results/                       # 🆕 Timestamped results
│   └── results_YYYYMMDD_HHMMSS/
│       ├── classification_report.txt
│       ├── classification_metrics.csv
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       └── training_curves.png
├── run_cuda_training.bat         # 🆕 Windows batch launcher
├── run_cuda_training.ps1         # 🆕 PowerShell launcher
└── README.md
```

---

## 🎯 CUDA Training Features

### 1. Automatic CUDA Verification
The training script verifies CUDA before starting:
```
===================================================================
CUDA DEVICE VERIFICATION
===================================================================
✓ CUDA is available!
✓ CUDA Version: 12.1
✓ Number of CUDA devices: 1
✓ Current device: 0
✓ Device name: NVIDIA GeForce RTX 3080
✓ Device capability: 8.6
✓ Total memory: 10.00 GB
✓ Multi-processor count: 68
===================================================================
```

### 2. PyTorch LSTM Architecture
```python
class PyTorchLSTM(nn.Module):
    - LSTM layers with CUDA optimization
    - Automatic gradient computation
    - Bidirectional support
    - Dropout regularization
    - Fully connected output layer
```

### 3. Training Pipeline
- **Data Loading**: Automatic sequence generation
- **Train/Val Split**: Stratified splitting (80/20)
- **CUDA Tensors**: All data moved to GPU automatically
- **Batch Processing**: Memory-efficient DataLoader
- **Early Stopping**: Patience-based stopping
- **LR Scheduling**: ReduceLROnPlateau
- **Gradient Clipping**: Prevents exploding gradients

### 4. Results Organization
All results saved in timestamped folders:
```
results/results_20251023_143052/
├── classification_report.txt      # Detailed metrics per class
├── classification_metrics.csv     # Structured CSV metrics
├── confusion_matrix.png           # Visual confusion matrix
├── roc_curves.png                 # ROC curves with AUC (4 decimals)
└── training_curves.png            # Loss & Accuracy plots
```

---

## 🔧 Configuration

### YAML Configuration File
```yaml
# config/lstm_config_experiment_1.yaml
input_dim: 14           # Number of features
seq_len: 64             # Sequence length
num_classes: 5          # Number of classes
lstm_units: 16          # LSTM hidden units
num_layers: 2           # Number of LSTM layers
dropout: 0.3            # Dropout rate
bidirectional: false    # Bidirectional LSTM
learning_rate: 0.001    # Adam learning rate
batch_size: 16          # Batch size
epochs: 15              # Maximum epochs
validation_split: 0.2   # Validation split
early_stopping_patience: 2  # Early stopping patience
```

---

## 📊 Output Metrics

### Classification Report (TXT)
```
CUDA-Accelerated PyTorch LSTM Classification Report
============================================================

              precision    recall  f1-score   support

     Class_0       0.95      0.94      0.94       500
     Class_1       0.92      0.93      0.93       450
     ...
```

### Classification Metrics (CSV)
```csv
Per-Class Metrics
Class,Precision,Recall,F1-Score
Class_0,0.9523,0.9401,0.9462
Class_1,0.9234,0.9311,0.9272
...

Overall Metrics
Metric,Value
Accuracy,0.9456
Precision (Macro),0.9401
Recall (Macro),0.9378
F1-Score (Macro),0.9389
```

### ROC Curves
- Individual ROC curve for each class
- AUC values rounded to **4 decimal places** (e.g., 0.9827)
- Clear legend and grid
- High-resolution PNG (300 DPI)

---

## 🧠 Theoretical Foundation (Original Content Below)

## 📚 Theoretical Foundation

### LSTM Architecture Theory

Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem when processing long sequences.

#### Key Components:

1. **Cell State (C_t)**: The memory highway that carries information across time steps
2. **Hidden State (h_t)**: The filtered output at each time step
3. **Three Gates**:
   - **Forget Gate**: Decides what information to discard
   - **Input Gate**: Determines what new information to store
   - **Output Gate**: Controls what parts of cell state to output

#### Mathematical Formulation:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t ⊙ tanh(C_t)                  # Hidden state update
```

Where:
- σ = sigmoid function
- ⊙ = element-wise multiplication
- W = weight matrices
- b = bias vectors

## 🏗️ Project Structure

```
LSTM/
├── config/                     # Configuration files
│   ├── default.yaml           # Default configuration
│   └── lstm_config_experiment_1.yaml  # Experiment configuration
├── lstm/                      # Core LSTM package
│   ├── __init__.py           # Package initialization and exports
│   ├── config_loader.py      # Configuration management with Pydantic
│   ├── builder.py            # Model architecture construction
│   └── lstm_with_softmax.py  # Complete LSTM implementation
├── scripts/                   # Training and evaluation scripts
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Model evaluation
├── logs/                      # Training logs and metrics
├── models/                    # Saved model artifacts
└── README.md                  # This file
```

## 🔧 Installation and Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install tensorflow>=2.0.0
pip install pydantic
pip install pandas
pip install scikit-learn
pip install pyyaml
pip install numpy
```

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd LSTM

# Create virtual environment (recommended)
python -m venv lstm_env
source lstm_env/bin/activate  # On Windows: lstm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Configuration

The system uses YAML configuration files for all parameters:

```yaml
# config/lstm_config_experiment_1.yaml
input_dim: 13
seq_len: 120
num_classes: 5
lstm_units: 32
num_layers: 2
dropout: 0.3
learning_rate: 0.001
```

### 2. Loading Configuration

```python
from lstm import LSTMConfig

# Load configuration from YAML
config = LSTMConfig.from_yaml('config/lstm_config_experiment_1.yaml')

# Access parameters with validation
print(f"LSTM units: {config.lstm_units}")
print(f"Dropout rate: {config.dropout}")
```

### 3. Building Models

```python
from lstm import build_lstm_model

# Build model from configuration
model = build_lstm_model(config)

# Display model architecture
model.summary()
```

### 4. Training

```python
# Run training pipeline
python scripts/train.py
```

## 📖 Detailed Usage Guide

### Configuration Management

The configuration system uses Pydantic for robust validation:

```python
from lstm.config_loader import LSTMConfig

# Load from YAML
config = LSTMConfig.from_yaml('config/lstm_config.yaml')

# Create programmatically
config = LSTMConfig(
    input_dim=13,
    seq_len=120,
    num_classes=5,
    lstm_units=64,
    num_layers=3,
    dropout=0.4,
    learning_rate=0.001
)

# Validate and save
config.to_yaml('config/new_config.yaml')
```

### Model Building

The builder supports various LSTM configurations:

```python
from lstm.builder import LSTMModelBuilder

# Create builder
builder = LSTMModelBuilder(config)

# Build model
model = builder.build_model()

# Get detailed summary
print(builder.get_model_summary())
```

### Data Preprocessing

```python
from scripts.train import DataProcessor

# Initialize processor
processor = DataProcessor(config)

# Load and validate data
data = processor.load_and_validate_data('data/network_data.csv')

# Preprocess for LSTM
X, y = processor.preprocess_data(data)
```

### Training Pipeline

```python
from scripts.train import LSTMTrainer

# Initialize trainer
trainer = LSTMTrainer(config)

# Train model
model, history = trainer.train(X, y)
```

## 🔬 Advanced Features

### 1. Bidirectional LSTMs

```yaml
# Enable bidirectional processing
bidirectional: true
```

This processes sequences in both directions, capturing future context.

### 2. Layer Stacking

```yaml
# Stack multiple LSTM layers
num_layers: 3
lstm_units: 64
```

Deeper networks can capture more complex patterns.

### 3. Custom Metrics

```yaml
metrics:
  - accuracy
  - precision
  - recall
  - f1_score
```

Monitor multiple performance indicators.

### 4. Advanced Regularization

The system includes multiple regularization techniques:

- **Dropout**: Randomly disables neurons during training
- **Layer Normalization**: Normalizes layer inputs
- **L1/L2 Regularization**: Penalizes large weights
- **Early Stopping**: Prevents overfitting

## 📊 Monitoring and Logging

### Training Logs

All training activities are logged:

```python
# Logs are saved to logs/training.log
# Metrics are saved to logs/training_metrics.csv
```

### TensorBoard Integration

```bash
# Start TensorBoard (if enabled)
tensorboard --logdir=logs/tensorboard
```

### Model Checkpointing

Best models are automatically saved during training:

```python
# Models saved to path specified in config
# checkpoint callback saves best validation performance
```

## 🧪 Experimentation

### Hyperparameter Tuning

Create multiple configuration files for experiments:

```bash
config/
├── experiment_1.yaml  # Baseline
├── experiment_2.yaml  # More layers
├── experiment_3.yaml  # Higher dropout
└── experiment_4.yaml  # Bidirectional
```

### Experiment Tracking

```python
# Each experiment creates detailed logs
# Compare results across configurations
# Track model performance over time
```

## 🎯 Best Practices Implemented

### 1. Code Organization

- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Injection**: Configuration passed to components
- **PEP 8 Compliance**: Professional Python styling
- **Type Hints**: Full type annotation for better IDE support

### 2. Error Handling

```python
# Comprehensive error handling
try:
    config = LSTMConfig.from_yaml(config_path)
except FileNotFoundError:
    logger.error(f"Configuration file not found: {config_path}")
except ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
```

### 3. Logging

```python
# Professional logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 4. Reproducibility

```python
# Set random seeds for consistent results
np.random.seed(42)
tf.random.set_seed(42)
```

## 🔍 Understanding the Code

### Configuration Validation

```python
class LSTMConfig(BaseModel):
    input_dim: int = Field(ge=1, description="Number of input features")
    dropout: float = Field(ge=0.0, le=1.0, description="Dropout rate")
    
    @validator('metrics')
    def validate_metrics(cls, metrics_list):
        # Ensure metrics are supported
        return metrics_list
```

### Model Architecture

```python
def _add_lstm_layers(self, model: Sequential) -> None:
    for layer_idx in range(self.config.num_layers):
        return_sequences = (layer_idx < self.config.num_layers - 1)
        
        lstm_layer = LSTM(
            units=self.config.lstm_units,
            return_sequences=return_sequences
        )
        
        if self.config.bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        
        model.add(lstm_layer)
```

### Data Processing

```python
def create_sequences(self, data, target):
    sequences = []
    for i in range(len(data) - self.config.seq_len + 1):
        sequence = data[i:i + self.config.seq_len]
        sequences.append(sequence)
    return np.array(sequences)
```

## 🚨 Common Issues and Solutions

### 1. Memory Issues

```python
# Reduce batch size
batch_size: 16

# Reduce model complexity
lstm_units: 32
num_layers: 2
```

### 2. Overfitting

```python
# Increase regularization
dropout: 0.5

# Add early stopping
# (automatically included in training pipeline)
```

### 3. Slow Convergence

```python
# Increase learning rate
learning_rate: 0.01

# Add learning rate scheduling
# (automatically included in callbacks)
```

## 📈 Performance Optimization

### GPU Utilization

```python
# Automatic GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Batch Processing

```python
# Optimal batch sizes for different scenarios
batch_size: 32   # Good default
batch_size: 64   # For larger datasets
batch_size: 16   # For limited memory
```

## 🤝 Contributing

When modifying the code:

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
5. Update configuration schema if needed

## 📚 Learning Resources

### LSTM Theory
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Implementation Details
- [TensorFlow LSTM Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)

### Best Practices
- [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Machine Learning Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
