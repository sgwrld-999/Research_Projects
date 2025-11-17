"""
Random Forest Model Training Pipeline

This module implements a comprehensive training pipeline for Random Forest-based models,
following machine learning best practices and software engineering principles.

THEORY - Random Forest Training Pipeline Design:
================================================

A well-designed Random Forest training pipeline consists of several key stages:

1. DATA LOADING & VALIDATION:
   - Verify data integrity and format
   - Handle missing values and outliers
   - Validate data schemas and types
   - Feature quality assessment

2. DATA PREPROCESSING:
   - Feature scaling (optional for Random Forest)
   - Categorical encoding if needed
   - Train/validation/test splits with stratification
   - Handle class imbalance if present

3. MODEL CONSTRUCTION:
   - Tree ensemble configuration
   - Hyperparameter optimization
   - Bootstrap sampling and feature selection

4. TRAINING PROCESS:
   - Parallel tree construction
   - Out-of-bag (OOB) scoring
   - Feature importance calculation
   - Model checkpointing

5. EVALUATION & VALIDATION:
   - Multiple performance metrics
   - Cross-validation
   - Feature importance analysis
   - Model interpretability

THEORY - Random Forest Training Best Practices:
===============================================

1. HYPERPARAMETER TUNING:
   - n_estimators: Start with 100, increase for better performance
   - max_depth: Control overfitting, use cross-validation
   - min_samples_split/leaf: Prevent overfitting on small datasets
   - max_features: Control randomness and feature selection

2. FEATURE ENGINEERING:
   - Random Forest handles mixed data types well
   - Feature scaling not usually necessary
   - Focus on feature selection and creation
   - Handle categorical variables appropriately

3. ENSEMBLE CONSIDERATIONS:
   - Bootstrap sampling introduces randomness
   - Feature subsampling at each split
   - Out-of-bag samples for validation
   - Parallel tree construction for efficiency

4. REGULARIZATION:
   - Tree depth and leaf size constraints
   - Minimum samples for splitting
   - Bootstrap sampling size
   - Feature subset selection

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import os
import sys
import logging
import warnings
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Custom imports
from random_forest.config_loader import RandomForestConfig

# ============================================================================
# PyTorch Model Definition (Embedded)
# ============================================================================

class TabularDeepClassifier(nn.Module):
    """Deep Learning Classifier optimized for tabular data with GPU acceleration."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        super(TabularDeepClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build the network
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        identity = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropouts[i](x)
            
            if self.use_residual and i > 0 and identity.shape[1] == x.shape[1]:
                x = x + identity
            
            identity = x
        
        logits = self.output_layer(x)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities using softmax."""
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class RandomForestTrainingPipeline:
    """
    Comprehensive training pipeline for Random Forest models.
    
    This class orchestrates the entire training process including data loading,
    preprocessing, model training, evaluation, and result logging.
    
    Attributes:
        config (RandomForestConfig): Configuration object
        model (RandomForestClassifier): The trained model
        training_metrics (Dict): Training performance metrics
        
    Example:
        >>> config = RandomForestConfig.from_yaml("config.yaml")
        >>> pipeline = RandomForestTrainingPipeline(config)
        >>> pipeline.run_training("data.csv")
    """
    
    def __init__(self, config: RandomForestConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config (RandomForestConfig): Configuration object
        """
        self.config = config
        self.model = None
        self.training_metrics = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Performance tracking
        self.training_time = 0.0
        self.inference_time = 0.0
        self.peak_memory_usage = 0.0
        self.gpu_memory_allocated = 0.0
        self.gpu_memory_reserved = 0.0
        
        # Setup GPU device
        self.device = self._setup_device()
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        logger.info("PyTorch GPU Training Pipeline initialized")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup and verify CUDA device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This script requires GPU.")
        
        device = torch.device('cuda')
        logger.info(f"✓ CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA Version: {torch.version.cuda}")
        logger.info(f"✓ PyTorch Version: {torch.__version__}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        return device
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        base_dir = Path(__file__).parent.parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.results_dir = base_dir / "results" / f"results_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        directories = [
            Path(self.config.export_path).parent,
            base_dir / "logs",
            base_dir / "models",
            self.results_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results will be saved to: {self.results_dir}")
            
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent.parent
        log_filename = f"pytorch_training_{timestamp}.log"
        log_path = base_dir / "logs" / log_filename
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        self.log_filename = log_filename
        logger.info(f"Logging to {log_path}")
        
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate dataset.
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully: {data.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
            
        # Basic validation
        if data.empty:
            raise ValueError("Dataset is empty")
            
        # Check target column exists
        if not hasattr(self.config, 'target_column'):
            raise ValueError("Target column not defined in config")
            
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data. Available columns: {data.columns.tolist()}")
            
        # Log basic statistics
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Target column: {self.config.target_column}")
        logger.info(f"Number of features: {data.shape[1] - 1}")
        
        # Check class distribution
        class_counts = data[self.config.target_column].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        else:
            logger.info("No missing values found")
            
        return data
        
    def prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets from the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, and feature names
        """
        logger.info("Preparing features and targets...")
        
        # Separate features and targets
        if hasattr(self.config, 'feature_columns') and self.config.feature_columns:
            # Use specified feature columns
            feature_columns = self.config.feature_columns
            if not all(col in data.columns for col in feature_columns):
                missing_cols = [col for col in feature_columns if col not in data.columns]
                raise ValueError(f"Feature columns not found: {missing_cols}")
        else:
            # Use all columns except target
            feature_columns = [col for col in data.columns if col != self.config.target_column]
            
        X = data[feature_columns].values
        y = data[self.config.target_column].values
        
        # Convert to appropriate types
        X = X.astype(np.float32)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Feature matrix shape: {X_scaled.shape}")
        logger.info(f"Target vector shape: {y_encoded.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Class labels: {self.label_encoder.classes_}")
        logger.info(f"Features: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        # Check for infinite or NaN values
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            logger.warning("NaN or infinite values found in features")
            
        return X_scaled, y_encoded, feature_columns
        
    def create_data_splits(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Create train/validation/test splits.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Tuple[np.ndarray, ...]: Train/validation/test splits
        """
        logger.info("Creating data splits...")
        
        # First split: separate test set
        test_size = getattr(self.config, 'test_size', 0.2)
        val_size = getattr(self.config, 'validation_size', 0.2)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        
        # Log class distributions
        unique_classes = np.unique(y)
        for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            class_dist = [(cls, np.sum(split_y == cls)) for cls in unique_classes]
            logger.info(f"{split_name} class distribution: {class_dist}")
            
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str]) -> Any:
        """
        Train the PyTorch model on GPU.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            feature_names (List[str]): Names of features
            
        Returns:
            Any: Trained model
        """
        logger.info("Starting PyTorch GPU model training...")
        logger.info(f"Training on device: {self.device}")
        
        # Start tracking training time and memory
        training_start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        # Create data loaders
        batch_size = getattr(self.config, 'batch_size', 256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        model = TabularDeepClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=getattr(self.config, 'hidden_dims', [256, 128, 64]),
            dropout_rate=getattr(self.config, 'dropout_rate', 0.3),
            use_batch_norm=True
        ).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        learning_rate = getattr(self.config, 'learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        num_epochs = getattr(self.config, 'num_epochs', 100)
        best_val_loss = float('inf')
        patience = getattr(self.config, 'patience', 15)
        patience_counter = 0
        
        logger.info(f"Training for {num_epochs} epochs with batch size {batch_size}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), Path(self.config.export_path).parent / 'best_model_checkpoint.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(Path(self.config.export_path).parent / 'best_model_checkpoint.pth'))
        
        # Calculate training time
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        
        # Calculate peak memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory_usage = final_memory - initial_memory
        
        # Get GPU memory usage
        if torch.cuda.is_available():
            self.gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB
        
        logger.info("Model training completed successfully")
        logger.info(f"Training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        logger.info(f"Peak CPU memory usage: {self.peak_memory_usage:.2f} MB")
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory allocated: {self.gpu_memory_allocated:.2f} MB")
            logger.info(f"Peak GPU memory reserved: {self.gpu_memory_reserved:.2f} MB")
        
        return model
        
    def evaluate_model(self, model: nn.Module, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained PyTorch model.
        
        Args:
            model (nn.Module): Trained PyTorch model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Evaluating model performance on GPU...")
        
        model.eval()
        
        # Start tracking inference time
        inference_start_time = time.time()
        
        # Convert to tensors and move to device
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Get predictions in batches
        batch_size = 1024
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size]
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        y_proba = np.vstack(all_probs)
        y_pred = np.concatenate(all_preds)
        
        # Calculate inference time
        inference_end_time = time.time()
        self.inference_time = inference_end_time - inference_start_time
        
        # Calculate inference metrics
        num_samples = len(X_test)
        avg_inference_time_per_sample = (self.inference_time / num_samples) * 1000  # ms
        throughput = num_samples / self.inference_time  # samples/sec
        
        logger.info(f"Inference time: {self.inference_time:.4f} seconds")
        logger.info(f"Average inference time per sample: {avg_inference_time_per_sample:.4f} ms")
        logger.info(f"Throughput: {throughput:.2f} samples/second")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'inference_time_total': self.inference_time,
            'inference_time_per_sample_ms': avg_inference_time_per_sample,
            'throughput_samples_per_sec': throughput,
            'training_time_total': self.training_time,
            'training_time_minutes': self.training_time / 60,
            'peak_cpu_memory_mb': self.peak_memory_usage,
            'peak_gpu_memory_allocated_mb': self.gpu_memory_allocated,
            'peak_gpu_memory_reserved_mb': self.gpu_memory_reserved
        }
        
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
        
        # Generate classification report
        class_names = self.label_encoder.classes_
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC AUC for each class
        roc_auc_scores = {}
        try:
            for i, class_name in enumerate(class_names):
                y_test_binary = (y_test == i).astype(int)
                roc_auc_scores[class_name] = roc_auc_score(y_test_binary, y_proba[:, i])
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            
        evaluation_results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'training_history': self.training_history,
            'roc_auc_scores': roc_auc_scores,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
        
        return evaluation_results
        
    def generate_plots(self, model: nn.Module, 
                      evaluation_results: Dict[str, Any]) -> None:
        """
        Generate and save visualization plots including ROC curves.
        
        Args:
            model (nn.Module): Trained PyTorch model
            evaluation_results (Dict[str, Any]): Evaluation results
        """
        logger.info("Generating visualization plots...")
        
        # 1. Training and Validation Curves
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            epochs = range(1, len(self.training_history['train_loss']) + 1)
            
            # Loss curves
            ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
            ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
            ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Training curves saved")
        except Exception as e:
            logger.warning(f"Could not generate training curves: {e}")
            
        # 2. Confusion Matrix
        try:
            conf_matrix = np.array(evaluation_results['confusion_matrix'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix saved")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {e}")
            
        # 3. ROC Curves for each class
        try:
            y_test = evaluation_results['y_test']
            y_proba = evaluation_results['y_proba']
            class_names = self.label_encoder.classes_
            n_classes = len(class_names)
            
            # Compute ROC curve and AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                y_test_binary = (y_test == i).astype(int)
                fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
            
            for i, color in enumerate(colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Multiclass Classification')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("ROC curves saved")
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")
            
    def save_results(self, model: nn.Module, 
                    evaluation_results: Dict[str, Any]) -> None:
        """
        Save PyTorch model and results with timestamp organization.
        
        Args:
            model (nn.Module): Trained PyTorch model
            evaluation_results (Dict[str, Any]): Evaluation results
        """
        logger.info("Saving model and results...")
        
        # Save PyTorch model
        model_dir = Path(self.config.export_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / Path(self.config.export_path).name.replace('.joblib', '.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'training_history': self.training_history
        }, model_path)
        
        logger.info(f"PyTorch model saved to: {model_path}")
        
        # Save classification metrics CSV
        class_names = self.label_encoder.classes_
        metrics_data = []
        
        for class_name in class_names:
            if class_name in evaluation_results['classification_report']:
                class_metrics = evaluation_results['classification_report'][class_name]
                roc_auc = evaluation_results['roc_auc_scores'].get(class_name, 0.0)
                metrics_data.append({
                    'Class': class_name,
                    'Accuracy': evaluation_results['metrics']['accuracy'],
                    'Precision': class_metrics['precision'],
                    'Recall': class_metrics['recall'],
                    'F1 Score': class_metrics['f1-score'],
                    'ROC AUC': round(roc_auc, 4)
                })
        
        # Add overall metrics
        metrics_data.append({
            'Class': 'Overall',
            'Accuracy': evaluation_results['metrics']['accuracy'],
            'Precision': evaluation_results['metrics']['precision_macro'],
            'Recall': evaluation_results['metrics']['recall_macro'],
            'F1 Score': evaluation_results['metrics']['f1_macro'],
            'ROC AUC': np.mean(list(evaluation_results['roc_auc_scores'].values()))
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv_path = self.results_dir / "classification_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        logger.info(f"Classification metrics saved to: {metrics_csv_path}")
        
        # Save performance metrics CSV
        performance_data = [{
            'Metric': 'Training Time (seconds)',
            'Value': f"{self.training_time:.4f}"
        }, {
            'Metric': 'Training Time (minutes)',
            'Value': f"{self.training_time / 60:.4f}"
        }, {
            'Metric': 'Inference Time (seconds)',
            'Value': f"{self.inference_time:.4f}"
        }, {
            'Metric': 'Inference Time per Sample (ms)',
            'Value': f"{evaluation_results['metrics']['inference_time_per_sample_ms']:.4f}"
        }, {
            'Metric': 'Throughput (samples/sec)',
            'Value': f"{evaluation_results['metrics']['throughput_samples_per_sec']:.2f}"
        }, {
            'Metric': 'Peak CPU Memory Usage (MB)',
            'Value': f"{self.peak_memory_usage:.2f}"
        }, {
            'Metric': 'Peak GPU Memory Allocated (MB)',
            'Value': f"{self.gpu_memory_allocated:.2f}"
        }, {
            'Metric': 'Peak GPU Memory Reserved (MB)',
            'Value': f"{self.gpu_memory_reserved:.2f}"
        }]
        
        performance_df = pd.DataFrame(performance_data)
        performance_csv_path = self.results_dir / "performance_metrics.csv"
        performance_df.to_csv(performance_csv_path, index=False)
        logger.info(f"Performance metrics saved to: {performance_csv_path}")
        
        # Save detailed classification report
        report_path = self.results_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PYTORCH GPU-ACCELERATED MULTICLASS CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Time: {self.training_time:.4f} seconds ({self.training_time/60:.4f} minutes)\n")
            f.write(f"Inference Time: {self.inference_time:.4f} seconds\n")
            f.write(f"Inference Time per Sample: {evaluation_results['metrics']['inference_time_per_sample_ms']:.4f} ms\n")
            f.write(f"Throughput: {evaluation_results['metrics']['throughput_samples_per_sec']:.2f} samples/second\n\n")
            
            f.write("Memory Usage:\n")
            f.write(f"  Peak CPU Memory: {self.peak_memory_usage:.2f} MB\n")
            f.write(f"  Peak GPU Memory Allocated: {self.gpu_memory_allocated:.2f} MB\n")
            f.write(f"  Peak GPU Memory Reserved: {self.gpu_memory_reserved:.2f} MB\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL ARCHITECTURE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Input Dimension: {self.config.input_dim}\n")
            f.write(f"Number of Classes: {self.config.num_classes}\n")
            f.write(f"Hidden Layers: {getattr(self.config, 'hidden_dims', [256, 128, 64])}\n")
            f.write(f"Dropout Rate: {getattr(self.config, 'dropout_rate', 0.3)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Epochs Trained: {len(self.training_history['train_loss'])}\n")
            f.write(f"Batch Size: {getattr(self.config, 'batch_size', 256)}\n")
            f.write(f"Learning Rate: {getattr(self.config, 'learning_rate', 0.001)}\n")
            f.write(f"Best Validation Accuracy: {max(self.training_history['val_acc']):.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("CLASSIFICATION METRICS\n")
            f.write("=" * 80 + "\n\n")
            for metric, value in evaluation_results['metrics'].items():
                if not metric.startswith(('inference_', 'training_', 'peak_', 'throughput')):
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Per-Class Metrics:\n")
            f.write("=" * 80 + "\n\n")
            
            for class_name in class_names:
                if class_name in evaluation_results['classification_report']:
                    f.write(f"{class_name}:\n")
                    class_metrics = evaluation_results['classification_report'][class_name]
                    f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_metrics['support']}\n")
                    if class_name in evaluation_results['roc_auc_scores']:
                        f.write(f"  ROC AUC: {evaluation_results['roc_auc_scores'][class_name]:.4f}\n")
                    f.write("\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        # Save full results as joblib
        results_joblib_path = self.results_dir / "evaluation_results.joblib"
        joblib.dump(evaluation_results, results_joblib_path)
        logger.info(f"Full results saved to: {results_joblib_path}")
        
    def run_training(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path (str): Path to the training dataset
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING RANDOM FOREST TRAINING PIPELINE")
            logger.info("=" * 50)
            
            # Step 1: Load and validate data
            data = self.load_and_validate_data(data_path)
            
            # Step 2: Prepare features and targets
            X, y, feature_names = self.prepare_features_and_targets(data)
            
            # Step 3: Create data splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_data_splits(X, y)
            
            # Step 4: Train model
            self.model = self.train_model(X_train, y_train, X_val, y_val, feature_names)
            
            # Step 5: Evaluate model
            evaluation_results = self.evaluate_model(self.model, X_test, y_test)
            
            # Step 6: Generate plots
            self.generate_plots(self.model, evaluation_results)
            
            # Step 7: Save results
            self.save_results(self.model, evaluation_results)
            
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run PyTorch GPU training."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Train PyTorch GPU-accelerated classifier")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/random_forest_experiment_1.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to training dataset CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration from YAML directly
        config_path = Path(__file__).parent.parent / args.config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create a simple class to hold configuration
        class SimpleConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Create config object with default PyTorch training parameters
        config = SimpleConfig(**config_data)
        
        # Add PyTorch-specific parameters
        config.target_column = 'label_stage_encoded'
        config.batch_size = 256
        config.num_epochs = 100
        config.learning_rate = 0.001
        config.patience = 15
        config.hidden_dims = [256, 128, 64]
        config.dropout_rate = 0.3
        
        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"Target column set to: {config.target_column}")
        logger.info("PyTorch Training Parameters:")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Max Epochs: {config.num_epochs}")
        logger.info(f"  Learning Rate: {config.learning_rate}")
        logger.info(f"  Hidden Layers: {config.hidden_dims}")
        
        # Create pipeline with this config
        pipeline = RandomForestTrainingPipeline(config)
        
        # Train the model
        results = pipeline.run_training(args.data)
        
        # Print summary
        print("\n" + "=" * 80)
        print("✓ PYTORCH GPU TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Device Used: {pipeline.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Training Time: {pipeline.training_time:.2f} seconds ({pipeline.training_time/60:.2f} minutes)")
        print(f"Inference Time: {pipeline.inference_time:.4f} seconds")
        print(f"Inference Time per Sample: {results['metrics']['inference_time_per_sample_ms']:.4f} ms")
        print(f"Throughput: {results['metrics']['throughput_samples_per_sec']:.2f} samples/second")
        print(f"\nPeak CPU Memory Usage: {pipeline.peak_memory_usage:.2f} MB")
        print(f"Peak GPU Memory Allocated: {pipeline.gpu_memory_allocated:.2f} MB")
        print(f"Peak GPU Memory Reserved: {pipeline.gpu_memory_reserved:.2f} MB")
        print(f"\n--- CLASSIFICATION METRICS ---")
        print(f"Final Test Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"F1-Score (Macro): {results['metrics']['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
        print(f"Precision (Macro): {results['metrics']['precision_macro']:.4f}")
        print(f"Recall (Macro): {results['metrics']['recall_macro']:.4f}")
        print(f"\nResults saved to: {pipeline.results_dir}")
        print(f"Log saved to: {Path(__file__).parent.parent}/logs/{pipeline.log_filename}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
