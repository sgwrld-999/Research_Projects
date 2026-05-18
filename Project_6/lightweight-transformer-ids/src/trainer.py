"""Training module for the Linformer-IDS model.

This module implements the training loop as specified in Algorithm 1 (steps 7-10)
with production-grade features including early stopping, model checkpointing,
gradient clipping, and comprehensive logging.

Example:
    >>> from src.config_manager import ConfigManager
    >>> from src.trainer import Trainer
    >>> config = ConfigManager.load_config("configs/config.yaml", profile="first")
    >>> trainer = Trainer(model, train_loader, val_loader, config)
    >>> best_model_path = trainer.train()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .logger import get_logger

if TYPE_CHECKING:
    from .config_manager import AppConfig

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    Focal Loss down-weights well-classified examples and focuses training
    on hard negatives. Particularly useful for imbalanced intrusion detection datasets.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Attributes:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples.
        gamma: Focusing parameter for modulating loss (gamma >= 0).
        reduction: Loss reduction method ('mean' or 'sum').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for class balance.
            gamma: Focusing parameter.
            reduction: How to reduce loss ('mean' or 'sum').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits from model of shape (batch, num_classes).
            targets: Ground truth labels of shape (batch,).

        Returns:
            Computed focal loss (scalar if reduction='mean').
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping handler to prevent overfitting.

    Monitors a metric and stops training when the metric stops improving
    for a specified number of epochs (patience).

    Attributes:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in monitored metric to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy/F1 (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' depending on metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False


class Trainer:
    """Trainer class for the Linformer-IDS model.

    This class orchestrates the training process following Algorithm 1,
    including the training loop (steps 7-10), validation, model checkpointing,
    and early stopping.

    Attributes:
        model: The Linformer-IDS model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        criterion: Loss function (Focal Loss or Cross-Entropy).
        optimizer: Optimizer (Adam).
        device: Computation device ('cuda' or 'cpu').
        epochs: Maximum number of training epochs.
        model_save_path: Path to save best model checkpoint.
        early_stopping: Early stopping handler.
        gradient_clip_val: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: AppConfig,
        model_save_path: str = "models/best_model.pth",
        use_focal_loss: bool = False,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: Linformer-IDS model.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            config: Configuration object containing training parameters.
            model_save_path: Path to save the best model.
            use_focal_loss: Whether to use Focal Loss instead of Cross-Entropy.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Set device with intelligent fallback
        if config.training.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✓ Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
                logger.info(f"✓ CUDA Version: {torch.version.cuda}")
                logger.info(f"✓ GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                logger.error("=" * 70)
                logger.error("CUDA requested but not available!")
                logger.error("Your PyTorch installation does not support CUDA.")
                logger.error("Please install CUDA-enabled PyTorch:")
                logger.error("  pip uninstall torch torchvision torchaudio")
                logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                logger.error("=" * 70)
                logger.warning("Falling back to CPU (training will be slower)")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (set device='cuda' in config to use GPU)")

        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Initialize loss function (step 3 of Algorithm 1)
        if use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.loss.focal_loss_alpha,
                gamma=config.loss.focal_loss_gamma,
                reduction=config.loss.reduction
            )
            logger.info(f"Using Focal Loss: alpha={config.loss.focal_loss_alpha}, gamma={config.loss.focal_loss_gamma}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using Cross-Entropy Loss for training")

        # Initialize optimizer (step 2 of Algorithm 1)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        logger.info(
            f"Initialized Adam optimizer: lr={config.training.learning_rate}, "
            f"weight_decay={config.training.weight_decay}"
        )

        # Early stopping
        if config.training.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.early_stopping.min_delta,
                mode=config.early_stopping.mode
            )
            logger.info(
                f"Early stopping enabled: patience={config.training.early_stopping_patience}, "
                f"min_delta={config.early_stopping.min_delta}, mode={config.early_stopping.mode}"
            )
        else:
            self.early_stopping = None

        self.epochs = config.training.epochs
        self.gradient_clip_val = config.training.gradient_clip_val

        self.best_val_score = 0.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

        # Training efficiency tracking
        self.epoch_times: List[float] = []      # seconds per epoch
        self.total_training_time: float = 0.0
        self.convergence_epoch: int = 0         # epoch where best val score was last improved

    def train_epoch(self) -> float:
        """Train the model for one epoch.

        Implements steps 8-9 of Algorithm 1: iterate through mini-batches,
        compute loss, backpropagate, and update weights.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for X_batch, y_batch in pbar:
            # Move data to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass: ŷ ← Linformer-IDS(x; ω)
            logits = self.model(X_batch)

            # Compute loss: loss ← L(ŷ, y)
            loss = self.criterion(logits, y_batch)

            # Backward pass: loss.backward()
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional)
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            # Update weights: optimizer.step()
            self.optimizer.step()

            # Accumulate loss
            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / num_samples
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model on the validation set.

        Implements step 10 of Algorithm 1: evaluate performance on validation data.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            # Accumulate loss
            running_loss += loss.item() * X_batch.size(0)

            # Compute accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_score: float) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            val_score: Validation score (accuracy or F1).
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_score": val_score,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }

        torch.save(checkpoint, self.model_save_path)
        logger.info(f"Saved model checkpoint to {self.model_save_path}")

    def load_checkpoint(self) -> None:
        """Load model checkpoint from disk."""
        if not self.model_save_path.exists():
            logger.warning(f"Checkpoint not found: {self.model_save_path}")
            return

        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from {self.model_save_path}")

    def train(self) -> str:
        """Execute the complete training loop.

        Implements the main training loop of Algorithm 1 (steps 7-11):
        - Train for E epochs
        - Validate after each epoch
        - Save best model checkpoint
        - Apply early stopping if configured

        Returns:
            Path to the saved best model.

        Example:
            >>> trainer = Trainer(model, train_loader, val_loader, config)
            >>> best_model_path = trainer.train()
            >>> print(f"Best model saved at: {best_model_path}")
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        _train_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.epochs}")
            logger.info(f"{'='*60}")

            _epoch_start = time.perf_counter()

            # Training phase (step 8-9)
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"Training Loss: {train_loss:.4f}")

            # Validation phase (step 10)
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                logger.info(f"Validation Loss: {val_loss:.4f}")
                logger.info(f"Validation Accuracy: {val_acc:.4f}")

                # Save best model checkpoint
                if val_acc > self.best_val_score:
                    self.best_val_score = val_acc
                    self.convergence_epoch = epoch
                    self.save_checkpoint(epoch, val_acc)
                    logger.info(f"★ New best validation accuracy: {val_acc:.4f}")

                # Early stopping check
                if self.early_stopping is not None:
                    if self.early_stopping(val_acc):
                        logger.info(f"Early stopping at epoch {epoch}")
                        self.epoch_times.append(time.perf_counter() - _epoch_start)
                        break
            else:
                # No validation set, save model every epoch
                self.save_checkpoint(epoch, train_loss)

            self.epoch_times.append(time.perf_counter() - _epoch_start)

        self.total_training_time = time.perf_counter() - _train_start

        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_score:.4f}")
        logger.info(f"Convergence epoch: {self.convergence_epoch}")
        logger.info(f"Total training time: {self.total_training_time:.2f}s")
        logger.info(f"Avg epoch time: {sum(self.epoch_times)/len(self.epoch_times):.2f}s")
        logger.info(f"Best model saved at: {self.model_save_path}")
        logger.info("="*60)

        return str(self.model_save_path)
