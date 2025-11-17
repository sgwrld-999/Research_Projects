"""
CUDA-Accelerated PyTorch LSTM Training Pipeline

This module implements a comprehensive training pipeline for LSTM-based neural networks
using PyTorch with CUDA acceleration for multiclass classification.

THEORY - CUDA Acceleration:
==========================
CUDA (Compute Unified Device Architecture) enables parallel computing on NVIDIA GPUs:
- Thousands of cores for parallel operations
- Optimized for matrix operations (core of neural networks)
- 10-100x faster than CPU for deep learning
- Automatic memory management between CPU and GPU
"""

import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from lstm import LSTMConfig

# Configure logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_cuda_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename)
    ]
)
logger = logging.getLogger(__name__)


class PyTorchLSTM(nn.Module):
    """
    PyTorch LSTM Model for Multiclass Classification
    
    THEORY - PyTorch LSTM Architecture:
    ==================================
    PyTorch's LSTM implementation:
    - Supports stacking multiple LSTM layers
    - Bidirectional processing capability
    - Efficient CUDA implementation
    - Automatic gradient computation
    """
    
    def __init__(self, config: LSTMConfig):
        super(PyTorchLSTM, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.lstm_units,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        lstm_output_size = config.lstm_units * (2 if config.bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, config.num_classes)
        
        # Softmax for classification
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get the output from the last time step
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            out = h_n[-1,:,:]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class CUDADeviceManager:
    """
    CUDA Device Management and Verification
    
    THEORY - GPU Computing:
    ======================
    - Verify CUDA availability
    - Select appropriate device
    - Monitor GPU memory usage
    - Handle device placement
    """
    
    @staticmethod
    def verify_cuda() -> torch.device:
        """
        Verify CUDA availability and return device.
        
        Returns:
            torch.device: CUDA device if available
            
        Raises:
            RuntimeError: If CUDA is not available
        """
        logger.info("=" * 70)
        logger.info("CUDA DEVICE VERIFICATION")
        logger.info("=" * 70)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            error_msg = "CUDA is not available! This script requires GPU acceleration."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Get CUDA information
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_properties = torch.cuda.get_device_properties(current_device)
        
        logger.info(f"✓ CUDA is available!")
        logger.info(f"✓ CUDA Version: {cuda_version}")
        logger.info(f"✓ Number of CUDA devices: {device_count}")
        logger.info(f"✓ Current device: {current_device}")
        logger.info(f"✓ Device name: {device_name}")
        logger.info(f"✓ Device capability: {device_properties.major}.{device_properties.minor}")
        logger.info(f"✓ Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
        logger.info(f"✓ Multi-processor count: {device_properties.multi_processor_count}")
        logger.info("=" * 70)
        
        device = torch.device("cuda")
        return device
    
    @staticmethod
    def log_memory_usage(device: torch.device):
        """Log current GPU memory usage."""
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


class DataProcessor:
    """Data Processing for LSTM Training"""
    
    def __init__(self, config: LSTMConfig):
        self.config = config
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data from CSV file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        
        if data.empty:
            raise ValueError("Loaded dataset is empty")
        
        logger.info(f"Data loaded successfully: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        
        return data
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series data into LSTM-compatible sequences.
        
        Args:
            data: Feature array of shape (n_samples, n_features)
            target: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (sequences, targets) for LSTM training
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.seq_len + 1):
            sequence = data[i:i + self.config.seq_len]
            target_value = target[i + self.config.seq_len - 1]
            
            sequences.append(sequence)
            targets.append(target_value)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.int64)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.config.seq_len}")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare preprocessed data for LSTM training.
        
        Args:
            data: Preprocessed pandas DataFrame
            
        Returns:
            Tuple of (X, y) ready for LSTM training
        """
        logger.info("Preparing preprocessed data for LSTM training...")
        
        # Separate features and target (assuming target is last column)
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(np.int64)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        # Validate dimensions
        if X.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.config.input_dim}, "
                f"got {X.shape[1]}"
            )
        
        # Validate classes
        unique_classes = np.unique(y)
        logger.info(f"Unique classes: {unique_classes}")
        
        # Ensure labels start from 0
        min_label = np.min(y)
        if min_label != 0:
            logger.info(f"Adjusting labels: subtracting {min_label}")
            y = y - min_label
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y)
        
        logger.info(f"Final sequence shape: {X_sequences.shape}")
        logger.info(f"Final target shape: {y_sequences.shape}")
        logger.info(f"Class distribution: {np.bincount(y_sequences)}")
        
        return X_sequences, y_sequences


class LSTMTrainer:
    """PyTorch LSTM Training Pipeline with CUDA Acceleration"""
    
    def __init__(self, config: LSTMConfig, device: torch.device):
        self.config = config
        self.device = device
        self.current_time = current_time
        
        # Set random seeds
        self._set_random_seeds()
        
        # Initialize model
        self.model = PyTorchLSTM(config).to(device)
        logger.info("Model initialized and moved to CUDA device")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _set_random_seeds(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seeds set to {seed} for reproducibility")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute the complete training pipeline.
        
        Args:
            X: Input sequences
            y: Target labels
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info("Starting PyTorch LSTM training with CUDA acceleration...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.validation_split,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Display model architecture
        logger.info("\nModel Architecture:")
        logger.info(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("\nStarting training loop...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Log memory usage
            CUDADeviceManager.log_memory_usage(self.device)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self._load_checkpoint('best_model.pth')
        
        logger.info("Training completed successfully")
        
        # Create comprehensive evaluation
        self._create_evaluation(X_val_tensor, y_val_tensor)
        
        # Save training history
        self._save_training_history()
        
        return self.model, self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = PROJECT_ROOT / 'models' / 'saved_Models' / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.dict()
        }, checkpoint_path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = PROJECT_ROOT / 'models' / 'saved_Models' / filename
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_training_history(self):
        """Save training history to CSV."""
        history_df = pd.DataFrame(self.history)
        history_path = PROJECT_ROOT / 'logs' / f'training_metrics_{self.current_time}.csv'
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")
    
    def _create_evaluation(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Create comprehensive evaluation metrics and visualizations."""
        # Create results directory with timestamp
        results_dir = PROJECT_ROOT / 'results' / f'results_{self.current_time}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating evaluation metrics and plots in {results_dir}")
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        # Move to CPU for sklearn compatibility
        y_true = y_val.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_proba = probabilities.cpu().numpy()
        
        # 1. Classification Report
        self._save_classification_report(y_true, y_pred, results_dir)
        
        # 2. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, results_dir)
        
        # 3. ROC Curves
        self._plot_roc_curves(y_true, y_proba, results_dir)
        
        # 4. Training Curves
        self._plot_training_curves(results_dir)
        
        # 5. Classification Metrics CSV
        self._save_classification_metrics_csv(y_true, y_pred, results_dir)
        
        logger.info(f"All evaluation results saved to {results_dir}")
    
    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path):
        """Save detailed classification report."""
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        report_path = results_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("CUDA-Accelerated PyTorch LSTM Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        logger.info(f"Classification report saved to {report_path}")
    
    def _save_classification_metrics_csv(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path):
        """Save classification metrics to CSV."""
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Overall metrics
        overall_metrics = pd.DataFrame([{
            'Metric': 'Accuracy',
            'Value': (y_true == y_pred).mean()
        }, {
            'Metric': 'Precision (Macro)',
            'Value': precision_score(y_true, y_pred, average='macro')
        }, {
            'Metric': 'Recall (Macro)',
            'Value': recall_score(y_true, y_pred, average='macro')
        }, {
            'Metric': 'F1-Score (Macro)',
            'Value': f1_score(y_true, y_pred, average='macro')
        }])
        
        metrics_path = results_dir / 'classification_metrics.csv'
        with open(metrics_path, 'w') as f:
            f.write("Per-Class Metrics\n")
            metrics_df.to_csv(f, index=False)
            f.write("\n\nOverall Metrics\n")
            overall_metrics.to_csv(f, index=False)
        
        logger.info(f"Classification metrics saved to {metrics_path}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrix plot saved")

    def _plot_training_curves(self, results_dir: Path):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves plot saved")


def main():
    """Main training pipeline execution."""
    try:
        logger.info("="* 70)
        logger.info("CUDA-ACCELERATED PYTORCH LSTM TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Training session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verify CUDA availability
        device = CUDADeviceManager.verify_cuda()
        
        # Load configuration
        config_path = PROJECT_ROOT / "config" / "lstm_config_32_units_reg.yaml"
        config = LSTMConfig.from_yaml(str(config_path))
        logger.info("Configuration loaded successfully")
        logger.info(f"\n{config.get_model_summary()}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load data
        data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
        
        # Prepare data
        raw_data = processor.load_and_validate_data(str(data_path))
        X, y = processor.prepare_data(raw_data)
        
        # Initialize trainer
        trainer = LSTMTrainer(config, device)
        
        # Train model
        model, history = trainer.train(X, y)
        
        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Logs saved to: {log_filename}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Ensure directories exist
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'models' / 'saved_Models').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'results').mkdir(exist_ok=True)
    
    # Run main pipeline
    main()
