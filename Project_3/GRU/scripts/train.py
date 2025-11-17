# standard imports
from pathlib import Path
import sys
import os
import time
from typing import Tuple, List, Any, Optional, Dict
from datetime import datetime
import warnings
import logging

# third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)

# ignoring warning
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# import custom modules
from gru import GRUConfig, build_gru_model

# Create timestamp modules 
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f"gru_training_{current_time}.log"

# Configure logging with professional formatting and timestamped files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'training.log')  # Keep general log too
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data Processing and Preparation for GRU Training
    
    THEORY - Data Preprocessing for GRUs:
    ======================================
    
    GRUs require specific data formatting:
    1. 3D input shape: (samples, time_steps, features)
    2. Proper sequence generation from time series
    3. Data validation and quality checks
    
    Note: This implementation assumes data is already preprocessed
    (normalized, scaled, and encoded) and focuses on sequence generation
    and validation for GRU training.
    """
    
    def __init__(self, config: GRUConfig):
        """
        Initialize data processor with configuration.
        
        Args:
            config: GRU configuration object
        """
        self.config = config
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with comprehensive validation.
        
        THEORY - Data Validation:
        ========================
        Data validation prevents silent failures and ensures model quality:
        - Schema validation: correct columns and types
        - Range validation: features within expected bounds
        - Completeness validation: acceptable missing data levels
        - Consistency validation: logical relationships between features
        
        Args:
            file_path: Path to the CSV data file
            
        Returns:
            Validated pandas DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data with error handling
            data = pd.read_csv(file_path)
            
            # Basic validation
            if data.empty:
                raise ValueError("Loaded dataset is empty")
            
            if len(data.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (features + target)")
            
            # Check for excessive missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.1:  # More than 10% missing
                logger.warning(f"High missing data ratio: {missing_ratio:.2%}")
            
            # Validate data types
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < len(data.columns) - 1:  # Assuming last column is target
                logger.warning("Non-numeric features detected. May need encoding.")
            
            logger.info(f"Data loaded successfully: {data.shape}")
            logger.info(f"Features: {list(data.columns[:-1])}")
            logger.info(f"Target: {data.columns[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series data into GRU-compatible sequences with optimized performance.
        
        THEORY - Sequence Generation:
        ============================
        
        GRUs process sequences of fixed length. For time series data:
        1. Sliding window approach creates overlapping sequences
        2. Window size = seq_len from configuration
        3. Each sequence predicts the next value or class
        
        Example:
        Data: [1, 2, 3, 4, 5, 6, 7, 8]
        seq_len: 3
        Sequences: [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]]
        Targets:   [4, 5, 6, 7, 8]
        
        Args:
            data: Feature array of shape (n_samples, n_features)
            target: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (sequences, targets) for GRU training
        """
        n_samples = len(data) - self.config.seq_len + 1
        n_features = data.shape[1]
        
        # Pre-allocate arrays for better performance
        sequences = np.zeros((n_samples, self.config.seq_len, n_features), dtype=np.float32)
        targets = np.zeros(n_samples, dtype=target.dtype)
        
        # Vectorized sequence creation for speed
        logger.info("Creating sequences with optimized vectorization...")
        for i in range(n_samples):
            sequences[i] = data[i:i + self.config.seq_len]
            targets[i] = target[i + self.config.seq_len - 1]
            
            # Progress indicator for large datasets
            if i % 50000 == 0 and i > 0:
                logger.info(f"Processed {i}/{n_samples} sequences ({i/n_samples*100:.1f}%)")
        
        logger.info(f"Created {len(sequences)} sequences of length {self.config.seq_len}")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def prepare_preprocessed_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare already preprocessed data for GRU training.
        
        THEORY - Working with Preprocessed Data:
        =======================================
        
        When data is already preprocessed, we focus on:
        1. Data validation and quality checks
        2. Sequence generation for GRU input format
        3. Ensuring proper data types and shapes
        4. Minimal transformation while preserving preprocessing
        
        This approach is preferred when:
        - Data has been carefully preprocessed offline
        - Preprocessing is part of a larger pipeline
        - You want to maintain exact control over normalization
        - Reproducibility requires fixed preprocessing
        
        Args:
            data: Preprocessed pandas DataFrame with features and target
            
        Returns:
            Tuple of (X, y) ready for GRU training
            
        Raises:
            ValueError: If data dimensions don't match configuration
        """
        logger.info("Preparing preprocessed data for GRU training...")
        
        # Separate features and target (assuming target is last column)
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(np.int32)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        # Validate feature dimensions
        expected_features = self.config.input_dim
        actual_features = X.shape[1]
        
        if actual_features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_features}, "
                f"got {actual_features}. Please update config.input_dim or check your data."
            )
        
        # Validate target classes
        unique_classes = np.unique(y)
        expected_classes = self.config.num_classes
        actual_classes = len(unique_classes)
        
        if actual_classes != expected_classes:
            logger.warning(
                f"Class count mismatch: expected {expected_classes}, "
                f"found {actual_classes} classes: {unique_classes}. "
                f"Consider updating config.num_classes."
            )
        
        # Ensure labels start from 0 (required for sparse_categorical_crossentropy)
        min_label = np.min(y)
        if min_label != 0:
            logger.info(f"Adjusting labels: subtracting {min_label} to start from 0")
            y = y - min_label
        
        # Create sequences for GRU
        X_sequences, y_sequences = self.create_sequences(X, y)
        
        # Final validation
        if len(X_sequences) == 0:
            raise ValueError(
                f"No sequences created. Data length ({len(X)}) may be smaller "
                f"than sequence length ({self.config.seq_len})"
            )
        
        logger.info("Preprocessed data preparation completed successfully")
        logger.info(f"Final sequence shape: {X_sequences.shape}")
        logger.info(f"Final target shape: {y_sequences.shape}")
        logger.info(f"Class distribution: {np.bincount(y_sequences)}")
        
        return X_sequences, y_sequences


class PlottingUtilities:
    """
    Comprehensive plotting utilities for training visualization and evaluation.
    
    This class provides methods to create publication-quality plots for:
    - Training history visualization
    - Confusion matrices  
    - ROC curves for multi-class classification
    - Precision-Recall curves
    
    All plots follow consistent styling and are saved with timestamps.
    """
    
    def __init__(self, timestamp: str, plots_dir: str):
        """
        Initialize plotting utilities.
        
        Args:
            timestamp: Current timestamp for file naming
            plots_dir: Directory to save plots
        """
        self.timestamp = timestamp
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent plot styling
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot comprehensive training history including all tracked metrics.
        
        Args:
            history: Training history dictionary from model.fit()
        """
        metrics_to_plot = []
        
        # Determine which metrics were tracked
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1_score']:
            if metric in history and f'val_{metric}' in history:
                metrics_to_plot.append(metric)
        
        if not metrics_to_plot:
            logger.warning("No training metrics found to plot")
            return
            
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            epochs = range(1, len(history[metric]) + 1)
            
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
            ax.plot(epochs, history[f'val_{metric}'], 'r-', label=f'Validation {metric}', linewidth=2)
            
            ax.set_title(f'{metric.capitalize()} Over Epochs', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            if metric == 'loss':
                best_val = min(history[f'val_{metric}'])
                best_epoch = history[f'val_{metric}'].index(best_val) + 1
            else:
                best_val = max(history[f'val_{metric}'])
                best_epoch = history[f'val_{metric}'].index(best_val) + 1
                
            ax.annotate(f'Best: {best_val:.4f} (epoch {best_epoch})',
                       xy=(best_epoch, best_val), xytext=(10, 10),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'training_history_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to: {plot_path}")
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Plot confusion matrix and calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes for labeling
            
        Returns:
            Dictionary containing per-class metrics
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        
        plt.title(f'Confusion Matrix - {self.timestamp}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Save plot
        plot_path = self.plots_dir / f'confusion_matrix_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to: {plot_path}")
        
        # Calculate per-class metrics
        metrics = {}
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(len(cm)):
            class_name = class_names[i] if class_names else f'Class_{i}'
            metrics[f'{class_name}_precision'] = precision_scores[i]
            metrics[f'{class_name}_recall'] = recall_scores[i]
            metrics[f'{class_name}_f1_score'] = f1_scores[i]
            
        # Add overall metrics
        metrics['overall_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['overall_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['overall_f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return metrics
        
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels (integer labels for multi-class)
            y_pred_proba: Predicted probabilities of shape (n_samples, n_classes)
            class_names: Names of classes for labeling
            
        Returns:
            Dictionary containing AUC scores for each class
        """
        n_classes = y_pred_proba.shape[1]
        
        # Convert integer labels to binary format for ROC calculation
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Handle the case where there are only 2 classes (binary classification)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true.reshape(-1, 1), y_true.reshape(-1, 1)])
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        auc_scores = {}
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            class_name = class_names[i] if class_names else f'Class {i}'
            
            # Calculate ROC curve for each class
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            auc_scores[f'{class_name}_auc'] = auc_score
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'{class_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multi-Class ROC Curves - {self.timestamp}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.plots_dir / f'roc_curves_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves plot saved to: {plot_path}")
        
        # Add macro and micro average AUC for multi-class
        try:
            auc_scores['macro_avg_auc'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                       average='macro', multi_class='ovr')
            auc_scores['weighted_avg_auc'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                          average='weighted', multi_class='ovr')
        except ValueError as e:
            logger.warning(f"Could not calculate macro/weighted AUC: {e}")
        
        return auc_scores


class GRUTrainer:
    """
    Professional GRU Training Pipeline
    
    THEORY - Training Strategy:
    ==========================
    
    1. BATCH TRAINING:
       - Process data in small batches for memory efficiency
       - Enables training on datasets larger than memory
       - Provides regularization through mini-batch gradient descent
    
    2. VALIDATION STRATEGY:
       - Hold-out validation set for unbiased performance estimation
       - Early stopping prevents overfitting
       - Learning rate scheduling adapts to training progress
    
    3. MONITORING:
       - Track multiple metrics during training
       - Log training progress for analysis
       - Save best model based on validation performance
    """
    
    def __init__(self, config: GRUConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: GRU configuration object
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Store the current timestamp for consistent file naming
        self.current_time = current_time
        
        # Initialize plotting utilities
        plots_dir = "/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/GRU/plots"
        self.plotter = PlottingUtilities(self.current_time, plots_dir)
        
        # Define class names for better visualization
        self.class_names = ['Recon', 'Exploitation', 'C&C', 'Attack', 'Benign']
        
        # Configure PyTorch and store GPU availability
        self.using_gpu = self._configure_pytorch()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Set device for training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _set_random_seeds(self, seed: int = 42) -> None:
        """
        Set random seeds for reproducible results.
        
        THEORY - Reproducibility in Deep Learning:
        ==========================================
        Neural networks use random initialization and stochastic training.
        Setting seeds ensures:
        - Consistent results across runs
        - Fair comparison between experiments
        - Debugging capability with deterministic behavior
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seeds set to {seed} for reproducibility")
    
    def _configure_pytorch(self) -> bool:
        """
        Configure PyTorch for GPU operation with maximum performance.
        
        THEORY - GPU Configuration:
        ===========================
        - Enable GPU mode: Use CUDA-compatible GPU for training
        - GPU Verification: Ensure GPU is available and properly configured
        """
        
        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            
            logger.info(f"=== GPU MODE ENABLED ===")
            logger.info(f"GPU Device: {gpu_name}")
            logger.info(f"Number of GPUs available: {gpu_count}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU is ready for training!")
            
            return True  # Using GPU
        else:
            logger.warning("=== NO GPU DETECTED ===")
            logger.warning("Training will run on CPU (slower performance)")
            
            # Configure for CPU as fallback
            cpu_count = psutil.cpu_count(logical=True)
            available_memory = psutil.virtual_memory().available / (1024**3)
            logger.info(f"System Info: {cpu_count} CPU cores, {available_memory:.1f}GB available RAM")
            
            return False  # Not using GPU
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute the complete training pipeline with PyTorch.
        
        THEORY - Training Process:
        =========================
        
        1. DATA SPLITTING:
           - Training set: model learns from this data
           - Validation set: monitors generalization during training
           - Test set: final evaluation (not used during training)
        
        2. MODEL TRAINING:
           - Optimizer: how model updates weights
           - Loss function: what model optimizes
           - Metrics: what we monitor (not optimized directly)
        
        3. TRAINING LOOP:
           - Forward pass: compute predictions
           - Loss calculation: compare with targets
           - Backward pass: compute gradients
           - Weight update: apply gradients via optimizer
        
        Args:
            X: Input sequences of shape (samples, seq_len, features)
            y: Target labels of shape (samples,)
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info("Starting GRU model training...")
        start_time = time.time()
        
        # Optimize data types for speed
        logger.info("Optimizing data types for performance...")
        X = X.astype(np.float32) if X.dtype != np.float32 else X
        y = y.astype(np.int64) if y.dtype != np.int64 else y
        
        # Split data with progress indicator
        logger.info("Splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Validation samples: {len(X_val):,}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Build model with progress tracking
        logger.info("Building model architecture...")
        model_start = time.time()
        self.model = build_gru_model(self.config)
        self.model = self.model.to(self.device)
        model_time = time.time() - model_start
        logger.info(f"Model built successfully in {model_time:.2f} seconds")

        # Display model architecture
        logger.info(f"Model architecture:\n{self.model}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Setup optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )

        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Train model with timing
        logger.info("=== STARTING TRAINING PROCESS ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Total batches per epoch: {len(train_loader)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create model save directory
        checkpoint_path = Path(self.config.export_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV logging setup
        csv_filename = f"training_metrics_{self.current_time}.csv"
        csv_path = PROJECT_ROOT / 'logs' / csv_filename
        with open(csv_path, 'w') as f:
            f.write("epoch,loss,accuracy,val_loss,val_accuracy,lr\n")
        
        training_start = time.time()
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Log to CSV
            with open(csv_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_accuracy:.6f},{val_loss:.6f},{val_accuracy:.6f},{current_lr:.8f}\n")
            
            # Print progress
            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, str(checkpoint_path).replace('.keras', '.pth'))
                logger.info(f"Model saved with validation loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("Training completed successfully")
        
        training_time = time.time() - training_start
        logger.info(f"Total training time: {training_time/60:.2f} minutes")

        # Log final metrics
        final_metrics = {
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss']),
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'total_epochs': len(history['loss'])
        }

        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Log training session summary
        logger.info(f"Training session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Timestamped log saved as: {log_filename}")
        logger.info(f"CSV metrics saved as: training_metrics_{current_time}.csv")

        # Generate training plots
        logger.info("Generating training visualization plots...")
        self.plotter.plot_training_history(history)
        
        # Evaluate model on validation set and create plots
        logger.info("Evaluating model and generating evaluation plots...")
        self._evaluate_and_plot(X_val, y_val)

        self.history = history
        return self.model, history
    
    def _evaluate_and_plot(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Evaluate model on validation set and generate comprehensive plots.
        
        Args:
            X_val: Validation input sequences
            y_val: Validation target labels
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Convert to tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            y_pred_proba = self.model(X_val_tensor).cpu().numpy()
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Plot confusion matrix and get per-class metrics
        cm_metrics = self.plotter.plot_confusion_matrix(
            y_val, y_pred, self.class_names
        )
        
        # Plot ROC curves and get AUC metrics
        roc_metrics = self.plotter.plot_roc_curves(
            y_val, y_pred_proba, self.class_names
        )
        
        # Log comprehensive metrics
        logger.info("=== VALIDATION METRICS ===")
        for metric, value in cm_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        for metric, value in roc_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Generate classification report
        report = classification_report(
            y_val, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        logger.info("=== DETAILED CLASSIFICATION REPORT ===")
        report_str = classification_report(
            y_val, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        logger.info(f"\n{report_str}")
        
        logger.info("Model evaluation and plotting completed successfully")


def main():
    """
    Main training pipeline execution.
    
    THEORY - Pipeline Orchestration:
    ===============================
    
    A well-structured main function for preprocessed data:
    1. Configuration loading and validation
    2. Data loading and sequence preparation  
    3. Model training and validation
    4. Results logging and model saving
    5. Error handling and cleanup
    
    Note: This pipeline assumes data is already preprocessed
    (normalized, scaled, encoded) and focuses on GRU-specific
    preparation and training.
    """
    try:
        # Log training session start
        logger.info(f"Training session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Session logs will be saved to: {log_filename}")
        
        # Load configuration
        config_path = PROJECT_ROOT / "config" / "gru_experiment_2.yaml"
        config = GRUConfig.from_yaml(str(config_path))
        logger.info("Configuration loaded successfully")
        logger.info(f"Model configuration:\n{config.get_model_summary()}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load and prepare data (assumes data is already preprocessed)
        data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            logger.info("Please place your CSV data file in the correct location")
            return
        
        # Load and prepare preprocessed data
        raw_data = processor.load_and_validate_data(str(data_path))
        X, y = processor.prepare_preprocessed_data(raw_data)
        
        # Initialize trainer and train model
        trainer = GRUTrainer(config)
        model, history = trainer.train(X, y)
        
        # Save final model
        final_model_path = config.export_path.replace('.keras', '_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history
        }, final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Ensure log directory exists
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    
    # Run main training pipeline
    main()