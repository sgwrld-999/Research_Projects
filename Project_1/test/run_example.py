"""
Minimal end-to-end example runner for VAE-BiLSTM IDS pipeline.
Uses small synthetic data to demonstrate the workflow.
"""
from pathlib import Path
import numpy as np
from src import utils
from src.binary_class_data_processing import preprocess, build_phase1_split, save_npz
from src.phase_1_model import train_phase1, compute_reconstruction_losses, compute_kde_threshold, save_threshold
from src.Phase_2_model import train_phase2, evaluate_kfold
from src.metrics import phase1_metrics, phase2_metrics
from src.plots import plot_reconstruction_kde, plot_training_history, plot_confusion_matrix

# Setup
utils.set_global_seed(42)
run_id = utils.get_run_id()
logger = utils.get_logger('example_runner', run_id=run_id)

logger.info("=== Starting VAE-BiLSTM IDS example run ===")

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 20
X_synth = np.random.randn(n_samples, n_features).astype('float32')
y_synth = np.random.randint(0, 2, size=n_samples)  # binary labels
logger.info(f"Generated synthetic data X: {X_synth.shape}, y: {y_synth.shape}")

# Phase-1: VAE training on benign samples
X_benign = X_synth[y_synth == 0]
logger.info(f"Phase-1: Training VAE on benign samples (n={len(X_benign)})")
vae, hist = train_phase1(X_benign, epochs=5, batch_size=32, run_dir='models/phase1_example')

# Compute reconstruction loss on all data
losses = compute_reconstruction_losses(vae, X_synth)
thresh_cfg = compute_kde_threshold(losses, percentile=95.0)
save_threshold(thresh_cfg, 'models/phase1_example/threshold.json')

# Phase-1 metrics
metrics_p1 = phase1_metrics(y_synth, losses, thresh_cfg['threshold'], logger=logger)
logger.info(f"Phase-1 metrics: {metrics_p1}")

# Plot Phase-1
plot_reconstruction_kde(losses, thresh_cfg['threshold'], run_id=run_id)

# Phase-2: CNN-BiLSTM training
logger.info("Phase-2: Training CNN-BiLSTM classifier")
y_cat = y_synth.copy()
model2, hist2 = train_phase2(X_synth, y_cat, seq_len=5, num_classes=2, epochs=3, batch_size=16, run_dir='models/phase2_example')

# K-fold evaluation (quick 2-fold)
logger.info("Phase-2: K-Fold evaluation")
kfold_results = evaluate_kfold(X_synth, y_cat, seq_len=5, n_splits=2, num_classes=2)
logger.info(f"K-Fold results: {kfold_results}")

# Plot Phase-2
plot_training_history(hist2, run_id=run_id)

logger.info("=== Example run complete ===")
