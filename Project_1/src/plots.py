from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from utils import get_logger, ensure_dir

LOGGER = get_logger('plots')


def ensure_plots_dir(base_dir = None):
    """Ensure plots directory exists. Accepts str or Path."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'plots'
    else:
        base_dir = Path(base_dir)
    ensure_dir(base_dir)
    return base_dir


def plot_reconstruction_kde(losses, threshold: float, out_path = None, run_id: str = None):
    """Plot reconstruction loss KDE. out_path accepts str or Path."""
    out_dir = ensure_plots_dir()
    if out_path is None:
        out_path = out_dir / f'reconstruction_kde_{run_id or "latest"}.png'
    else:
        out_path = Path(out_path)
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5))
    sns.kdeplot(losses, fill=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold {threshold:.4f}')
    plt.xlabel('Reconstruction loss')
    plt.title('Reconstruction Loss KDE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    LOGGER.info(f'Saved reconstruction KDE plot to {out_path}')
    return out_path


def plot_training_history(history: dict, out_path = None, run_id: str = None):
    """Plot training history. out_path accepts str or Path."""
    out_dir = ensure_plots_dir()
    if out_path is None:
        out_path = out_dir / f'training_history_{run_id or "latest"}.png'
    else:
        out_path = Path(out_path)
    plt.figure(figsize=(10, 4))
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='train_acc')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='val_acc')
    if 'loss' in history:
        plt.plot(history['loss'], label='train_loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Training History')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    LOGGER.info(f'Saved training history plot to {out_path}')
    return out_path


def plot_confusion_matrix(cm, labels, out_path = None, run_id: str = None):
    """Plot confusion matrix heatmap. out_path accepts str or Path."""
    out_dir = ensure_plots_dir()
    if out_path is None:
        out_path = out_dir / f'confusion_matrix_{run_id or "latest"}.png'
    else:
        out_path = Path(out_path)
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    LOGGER.info(f'Saved confusion matrix plot to {out_path}')
    return out_path


def plot_pca_importance(X, out_path = None, run_id: str = None):
    """Plot PCA cumulative variance. out_path accepts str or Path."""
    out_dir = ensure_plots_dir()
    if out_path is None:
        out_path = out_dir / f'pca_importance_{run_id or "latest"}.png'
    else:
        out_path = Path(out_path)
    pca = PCA()
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(var_ratio), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    LOGGER.info(f'Saved PCA cumulative variance plot to {out_path}')
    return out_path
