# Project_1: Two-Phase VAE–BiLSTM Intrusion Detection System

**A research prototype for IoT network intrusion detection on Edge-IIoT / CIC-IoT datasets.**

**🔥 Now powered by PyTorch with CUDA GPU acceleration! 🚀**

---

## Overview

This project implements a **two-phase hybrid IDS**:

1. **Phase-1 (VAE)**: Unsupervised anomaly detection via Variational Autoencoder trained on benign traffic. Computes reconstruction loss and uses KDE-based threshold to detect anomalies.
   - **Framework**: PyTorch with CUDA support
   - **Preprocessing**: Scikit-learn Pipeline for reproducibility
   
2. **Phase-2 (CNN–BiLSTM)**: Supervised sequence classifier for binary/multi-class attack detection using hybrid CNN + Bidirectional LSTM architecture.

All modules are fully logged, reproducible, and configurable.

### Key Features

✅ **PyTorch Implementation** - Modern deep learning framework with CUDA GPU acceleration  
✅ **Scikit-learn Pipeline** - Reproducible preprocessing with serialization support  
✅ **CUDA Optimized** - Automatic GPU detection and memory management  
✅ **Comprehensive Logging** - Timestamped logs for all training runs  
✅ **Checkpoint System** - 8-checkpoint training pipeline with validation  
✅ **Model Serialization** - Save/load models and pipelines for inference

---

## Repository Structure

```
Project_1/
├── data/                          # Dataset directory (not tracked)
│   ├── edge_iiot/
│   │   ├── normal_training.csv
│   │   ├── normal_testing.csv
│   │   └── multiclass_train_*.csv
│   └── cic_iot/
├── docs/                          # Architecture & metrics documentation
│   ├── feature_engineering_pipeline.txt
│   ├── metrics.txt
│   ├── phase_1_model.txt
│   ├── phase_2_model.txt
│   └── VAE_BiLSTM_IDS__A_Two_Phase_Hybrid_Framework_for_IoT_Network_Security.pdf
├── src/                           # Core implementation
│   ├── utils.py                   # Logging, seed setting, directory utils
│   ├── binary_class_data_processing.py
│   ├── multi_class_data_processing.py
│   ├── phase_1_model.py           # VAE implementation
│   ├── Phase_2_model.py           # CNN-BiLSTM implementation
│   ├── metrics.py                 # Phase-1 & Phase-2 metrics
│   └── plots.py                   # Visualization utilities
├── logs/                          # Timestamped run logs (auto-created)
├── plots/                         # Output plots (auto-created)
├── models/                        # Saved models & checkpoints (auto-created)
├── run_example.py                 # Minimal end-to-end example with synthetic data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Setup

### 1. Install PyTorch with CUDA Support

For GPU acceleration, install PyTorch with CUDA:

```powershell
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA installation:
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 2. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Prepare Data

Place your datasets in the `data/` folder:

- **Edge-IIoT**: `data/edge_iiot/normal_training.csv`, `multiclass_train_*.csv`, etc.
- **CIC-IoT**: `data/cic_iot/` (structure as needed)

See `docs/feature_engineering_pipeline.txt` for the expected 20-feature set.

---

## Usage

### Quick Test (Synthetic Data)

Run the minimal example to verify installation and module integration:

```powershell
python run_example.py
```

This will:
- Generate synthetic data
- Train a VAE (Phase-1)
- Compute reconstruction losses and KDE threshold
- Train a CNN-BiLSTM (Phase-2)
- Run K-Fold evaluation
- Save logs to `logs/run_YYYYMMDD_HHMMSS.log` and plots to `plots/`

---

### Real Workflow

#### Step 1: Phase-1 (Unsupervised VAE with PyTorch + CUDA)

```python
import torch
from src import utils
from src.binary_class_data_processing import load_csv, create_preprocessing_pipeline, fit_transform_pipeline, build_phase1_split
from src.phase_1_model import train_phase1, compute_reconstruction_losses, compute_kde_threshold, save_threshold, device

utils.set_global_seed(42)
logger = utils.get_logger('phase1_real_run')

print(f"Using device: {device}")  # Will show 'cuda' if GPU is available

# Load and preprocess benign training data using sklearn Pipeline
df = load_csv('data/edge_iiot/normal_training.csv')
pipeline = create_preprocessing_pipeline()
df_processed, fitted_pipeline = fit_transform_pipeline(df, pipeline, save_path='models/preprocessing_pipeline.pkl')

# Extract features and split
X = df_processed.drop(columns=['Attack_label', 'Attack_label_original'], errors='ignore').values
y = df_processed['Attack_label'].values if 'Attack_label' in df_processed.columns else None
X_tr, X_val = build_phase1_split(X, y, val_fraction=0.2)

# Train VAE (automatically uses CUDA if available)
vae, history = train_phase1(X_tr, X_val, epochs=50, batch_size=64, run_dir='models/phase1_real')

# Threshold estimation
losses = compute_reconstruction_losses(vae, X_tr)
thresh_cfg = compute_kde_threshold(losses, percentile=95.0)
save_threshold(thresh_cfg, 'models/phase1_real/threshold.json')
logger.info(f"Threshold: {thresh_cfg['threshold']}")

# Save PyTorch model
torch.save(vae.state_dict(), 'models/phase1_real/vae_full.pth')
```

#### Step 2: Phase-2 (Supervised CNN–BiLSTM)

```python
from src.binary_class_data_processing import load_csv, preprocess
from src.Phase_2_model import train_phase2, evaluate_kfold
from src.plots import plot_training_history

# Load full dataset (normal + attack)
df = load_csv('data/edge_iiot/combined_train.csv')
X, y, _ = preprocess(df, label_col='Attack_label')

# Train Phase-2
model, hist = train_phase2(X, y, seq_len=10, num_classes=2, epochs=30, batch_size=16, run_dir='models/phase2_real')

# K-Fold evaluation
results = evaluate_kfold(X, y, seq_len=10, n_splits=3)
logger.info(f"K-Fold results: {results}")

# Plot
plot_training_history(hist, run_id='phase2_real')
```

---

## Logging

All modules use Python's `logging` module:

- **Log files**: `logs/run_YYYYMMDD_HHMMSS.log`
- **Console output**: Timestamped messages to stdout/stderr
- **Logged items**:
  - Config (paths, hyperparameters)
  - Data shapes at each processing step
  - Training progress (epoch, loss, metrics)
  - Final metrics & saved model paths

---

## Module Descriptions

### `src/utils.py`
- **`set_global_seed(seed)`**: Sets random seeds for reproducibility.
- **`get_logger(name, run_id, log_dir, level)`**: Creates a logger with file and console handlers.
- **`ensure_dir(path)`**: Creates directories if they don't exist.
- **`save_json(obj, path)`** / **`load_json(path)`**: JSON I/O helpers.

### `src/binary_class_data_processing.py`
- **`load_normal_train(base_path)`** / **`load_normal_test(base_path)`**: Load normal traffic CSVs.
- **`select_features(df, feature_list, label_col)`**: Filter to final 20 features.
- **`preprocess(df, ...)`**: Median imputation + MinMax scaling to [0,1].
- **`build_phase1_split(X, y, val_fraction)`**: Split benign samples for Phase-1.
- **`save_npz(path, **arrays)`** / **`load_npz(path)`**: NumPy array I/O.

### `src/multi_class_data_processing.py`
- **`find_multiclass_files(base_dir)`**: Auto-discover multiclass CSVs.
- **`load_all_multiclass(base_dir)`**: Concatenate all multiclass files.
- **`preprocess_multiclass(df, ...)`**: Same preprocessing as binary, returns encoded labels.
- **`encode_labels(y, save_map)`**: Map attack type names to integers and save mapping.

### `src/phase_1_model.py`
- **`Layer1AutoencoderVAE`**: Custom Keras VAE with encoder/decoder and reparameterization.
- **`train_phase1(X_train, X_val, ...)`**: Train VAE with MSE + KL loss.
- **`compute_reconstruction_losses(vae, X)`**: Compute per-sample MSE.
- **`compute_kde_threshold(losses, bandwidth, percentile)`**: Fit KDE and return threshold at percentile.
- **`save_threshold(cfg, path)`**: Save threshold config as JSON.

### `src/Phase_2_model.py`
- **`build_cnn_bilstm_model(seq_len, num_features, num_classes, params)`**: Builds CNN + BiLSTM Keras model.
- **`build_sequences(X, seq_len)`**: Creates sliding windows for sequence input.
- **`train_phase2(X, y, seq_len, ...)`**: Train Phase-2 with train/val split, EarlyStopping, ModelCheckpoint.
- **`evaluate_kfold(X, y, seq_len, n_splits, ...)`**: K-fold cross-validation with per-fold metrics.

### `src/metrics.py`
- **`phase1_metrics(y_true, losses, threshold, logger)`**: Computes TPR, FPR, AUC, confusion matrix for Phase-1.
- **`phase2_metrics(y_true, y_pred, ...)`**: Computes accuracy, precision, recall, F1 (weighted), confusion matrix for Phase-2.

### `src/plots.py`
- **`plot_reconstruction_kde(losses, threshold, ...)`**: KDE plot with threshold line.
- **`plot_training_history(history, ...)`**: Training/validation accuracy & loss curves.
- **`plot_confusion_matrix(cm, labels, ...)`**: Heatmap of confusion matrix.
- **`plot_pca_importance(X, ...)`**: PCA cumulative variance explained plot.

---

## Configuration & Hyperparameters

All training functions accept configurable hyperparameters via function arguments:

- **Phase-1 VAE**:
  - `epochs=50`, `batch_size=64`, `latent_dim` (auto-computed or manual), `l2_reg=1e-5`, `dropout=0.2`
- **Phase-2 CNN-BiLSTM**:
  - `seq_len=10`, `epochs=30`, `batch_size=16`
  - `params` dict: `filters=28`, `kernel_size=3`, `lstm_units=29`, `dense_units=19`, `dropout_rate=0.35`, `learning_rate=7.9e-4`

To customize, pass your own `params` dictionary to `build_cnn_bilstm_model` or `train_phase2`.

---

## Advanced: Adding CLI Entrypoints

For production experiments, consider adding `argparse`-based CLI scripts:

```python
# train_phase1_cli.py
import argparse
from src import utils
from src.binary_class_data_processing import load_normal_train, preprocess, build_phase1_split
from src.phase_1_model import train_phase1

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/edge_iiot/', help='Path to dataset')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

utils.set_global_seed(42)
logger = utils.get_logger('phase1_cli')

df = load_normal_train(args.data_path)
X_train, y, _ = preprocess(df)
X_tr, X_val = build_phase1_split(X_train, y)

vae, hist = train_phase1(X_tr, X_val, epochs=args.epochs, batch_size=args.batch_size, run_dir='models/phase1_cli')
logger.info("Training complete.")
```

Run as:
```powershell
python train_phase1_cli.py --data_path data/edge_iiot/ --epochs 50 --batch_size 64
```

---

## Troubleshooting

### Import Errors
- Ensure `src/` is in your Python path or run from the project root: `python -m run_example`

### TensorFlow/GPU Issues
- If training is slow, ensure TensorFlow detects your GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- For mixed precision (faster training): add `tf.keras.mixed_precision.set_global_policy('mixed_float16')` in your scripts

### Missing Features in Dataset
- Check `docs/feature_engineering_pipeline.txt` for the expected 20 features
- Update `DEFAULT_FEATURES` in `binary_class_data_processing.py` to match your dataset columns

---

## Next Steps

1. **Run full experiments**: Train on real Edge-IIoT / CIC-IoT datasets with full epochs.
2. **Generate all plots**: Use `plots.py` functions to create figures for papers.
3. **Multi-class experiments**: Use `multi_class_data_processing.py` to run Phase-2 with 14 attack classes.
4. **Hyperparameter tuning**: Experiment with different `params` dicts in Phase-2.
5. **Add CI/CD**: Automate testing with pytest, linting with flake8/black.

---

## References

- Paper: `docs/VAE_BiLSTM_IDS__A_Two_Phase_Hybrid_Framework_for_IoT_Network_Security.pdf`
- Phase-1 architecture: `docs/phase_1_model.txt`
- Phase-2 architecture: `docs/phase_2_model.txt`
- Metrics & plots: `docs/metrics.txt`
- Feature engineering: `docs/feature_engineering_pipeline.txt`

---

## License

Research prototype. Not for production use without further validation.

---

## Contact

For questions or collaboration, reach out to the repository owner.
