"""
XGBoost Model Training Pipeline

This module implements a comprehensive training pipeline for XGBoost-based models,
following machine learning best practices and software engineering principles.

THEORY - XGBoost Training Pipeline Design:
==========================================

A well-designed XGBoost training pipeline consists of several key stages:

1. DATA LOADING & VALIDATION:
   - Verify data integrity and format
   - Handle missing values (XGBoost handles this automatically)
   - Validate data schemas and types
   - Check for data leakage and quality issues

2. DATA PREPROCESSING:
   - Feature scaling (optional, often not needed for tree-based models)
   - Categorical encoding if needed
   - Train/validation/test splits with stratification
   - Handle class imbalance if present

3. MODEL CONSTRUCTION:
   - Architecture definition based on configuration
   - Hyperparameter validation and tuning
   - Cross-validation for robust evaluation

4. TRAINING PROCESS:
   - Iterative boosting with early stopping
   - Progress monitoring and logging
   - Feature importance tracking
   - Model checkpointing for fault tolerance

5. EVALUATION & VALIDATION:
   - Multiple performance metrics calculation
   - Validation on held-out data
   - Model diagnostics and analysis
   - Learning curve generation

THEORY - XGBoost Training Best Practices:
========================================

1. HYPERPARAMETER TUNING:
   - Start with default parameters
   - Tune learning_rate and n_estimators together
   - Adjust max_depth and min_child_weight for regularization
   - Use early stopping to prevent overfitting

2. FEATURE ENGINEERING:
   - XGBoost handles missing values automatically
   - Feature scaling often not necessary
   - Focus on feature selection and creation
   - Monitor feature importance for insights

3. CROSS-VALIDATION:
   - Use stratified k-fold for imbalanced datasets
   - Monitor both training and validation metrics
   - Use multiple evaluation metrics

4. REGULARIZATION:
   - L1 (alpha) and L2 (lambda) regularization
   - Subsample and colsample_bytree for randomness
   - Early stopping based on validation performance

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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Custom imports
from xgboost_custom.config_loader import XGBoostConfig
from xgboost_custom.xgboost_with_softmax import XGBoostWithSoftmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class XGBoostTrainingPipeline:
    """
    Comprehensive training pipeline for XGBoost models.
    
    This class orchestrates the entire training process including data loading,
    preprocessing, model training, evaluation, and result logging.
    
    Attributes:
        config (XGBoostConfig): Configuration object
        model (XGBoostWithSoftmax): The trained model
        training_metrics (Dict): Training performance metrics
        
    Example:
        >>> config = XGBoostConfig.from_yaml("config.yaml")
        >>> pipeline = XGBoostTrainingPipeline(config)
        >>> pipeline.run_training("data.csv")
    """
    
    def __init__(self, config: XGBoostConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config (XGBoostConfig): Configuration object
        """
        self.config = config
        self.model = None
        self.training_metrics = {}
        self.label_encoder = LabelEncoder()
        
        # Performance tracking
        self.training_time = 0.0
        self.inference_time = 0.0
        self.peak_memory_usage = 0.0
        self.gpu_memory_allocated = 0.0
        self.gpu_memory_reserved = 0.0
        
        # Verify GPU availability for XGBoost
        self._verify_gpu_availability()
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        logger.info("XGBoost GPU Training Pipeline initialized")
    
    def _verify_gpu_availability(self) -> None:
        """Verify GPU availability for XGBoost training."""
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This script requires GPU for XGBoost.")
        
        logger.info("=" * 70)
        logger.info("GPU VERIFICATION FOR XGBOOST")
        logger.info("=" * 70)
        logger.info(f"✓ CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"✓ PyTorch Version: {torch.__version__}")
        
        # Force GPU usage in XGBoost config (XGBoost 3.1+ uses 'device' with tree_method='hist')
        self.config.tree_method = 'hist'  # Changed from 'gpu_hist' to 'hist' for XGBoost 3.1+
        self.config.gpu_id = 0  # This will be converted to device='cuda:0' in to_xgboost_params()
        logger.info(f"✓ XGBoost tree_method set to: {self.config.tree_method}")
        logger.info(f"✓ XGBoost will use GPU (device=cuda:0)")
        logger.info("=" * 70)
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs with timestamp."""
        base_dir = Path(__file__).parent.parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.results_dir = base_dir / "results" / f"results_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        directories = [
            Path(self.config.export_path).parent,
            base_dir / "logs",
            self.results_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results will be saved to: {self.results_dir}")
            
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent.parent
        log_filename = f"xgboost_gpu_training_{timestamp}.log"
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
            
        # Check for target column
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")

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
        if self.config.feature_columns:
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
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        logger.info(f"Features: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        # Check for infinite or NaN values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or infinite values found in features")
            
        return X, y, feature_columns
        
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
                   feature_names: List[str]) -> XGBoostWithSoftmax:
        """
        Train the XGBoost model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            feature_names (List[str]): Names of features
            
        Returns:
            XGBoostWithSoftmax: Trained model
        """
        logger.info("Starting XGBoost model training...")
        
        # Start tracking training time and memory
        training_start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear GPU cache if using PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        
        # Create and train model
        model = XGBoostWithSoftmax(self.config)
        model.feature_names = feature_names
        
        # Train with validation
        model.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )
        
        # Calculate training time
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        
        # Calculate peak memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory_usage = final_memory - initial_memory
        
        # Get GPU memory usage if available
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                self.gpu_memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB
        except ImportError:
            pass
        
        logger.info("Model training completed successfully")
        logger.info(f"Training time: {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        logger.info(f"Peak CPU memory usage: {self.peak_memory_usage:.2f} MB")
        if self.gpu_memory_allocated > 0:
            logger.info(f"Peak GPU memory allocated: {self.gpu_memory_allocated:.2f} MB")
            logger.info(f"Peak GPU memory reserved: {self.gpu_memory_reserved:.2f} MB")
        
        return model
        
    def evaluate_model(self, model: XGBoostWithSoftmax, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            model (XGBoostWithSoftmax): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Start tracking inference time
        inference_start_time = time.time()
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
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
        metrics = model.evaluate(X_test, y_test, verbose=True)
        
        # Add performance metrics
        metrics['inference_time_total'] = self.inference_time
        metrics['inference_time_per_sample_ms'] = avg_inference_time_per_sample
        metrics['throughput_samples_per_sec'] = throughput
        metrics['training_time_total'] = self.training_time
        metrics['training_time_minutes'] = self.training_time / 60
        metrics['peak_cpu_memory_mb'] = self.peak_memory_usage
        metrics['peak_gpu_memory_allocated_mb'] = self.gpu_memory_allocated
        metrics['peak_gpu_memory_reserved_mb'] = self.gpu_memory_reserved
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        evaluation_results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'training_history': model.training_history
        }
        
        return evaluation_results
        
    def generate_plots(self, model: XGBoostWithSoftmax, 
                      evaluation_results: Dict[str, Any],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> None:
        """
        Generate and save visualization plots.
        
        Args:
            model (XGBoostWithSoftmax): Trained model
            evaluation_results (Dict[str, Any]): Evaluation results
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        logger.info("Generating visualization plots...")
        
        # Use results directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature Importance Plot
        try:
            fig = model.plot_feature_importance(top_n=20)
            plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Feature importance plot saved")
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
            
        # 2. Confusion Matrix Plot
        try:
            conf_matrix = np.array(evaluation_results['confusion_matrix'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - XGBoost GPU')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix plot: {e}")
            
        # 3. ROC Curves
        try:
            y_proba = model.predict_proba(X_test)
            n_classes = y_proba.shape[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Binarize the output for multi-class ROC
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                        label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - XGBoost GPU Multi-class Classification')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(plots_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("ROC curves saved")
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")
            
        # 4. Training History Plot
        try:
            if 'training_history' in evaluation_results and evaluation_results['training_history']:
                history = evaluation_results['training_history']
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('XGBoost GPU Training History', fontsize=16)
                
                # Plot training metrics if available
                if 'train' in history:
                    train_metrics = history['train']
                    iterations = range(len(train_metrics))
                    
                    axes[0, 0].plot(iterations, train_metrics, label='Training')
                    if 'validation' in history:
                        axes[0, 0].plot(iterations, history['validation'], label='Validation')
                    axes[0, 0].set_xlabel('Iteration')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].set_title('Training Progress')
                    axes[0, 0].legend()
                    axes[0, 0].grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Training history plot saved")
        except Exception as e:
            logger.warning(f"Could not generate training history plot: {e}")
            
    def save_results(self, model: XGBoostWithSoftmax, 
                    evaluation_results: Dict[str, Any],
                    y_test: np.ndarray,
                    y_pred: np.ndarray) -> None:
        """
        Save model and results.
        
        Args:
            model (XGBoostWithSoftmax): Trained model
            evaluation_results (Dict[str, Any]): Evaluation results
            y_test (np.ndarray): True test labels
            y_pred (np.ndarray): Predicted test labels
        """
        logger.info("Saving model and results...")
        
        # Save model
        model_path = self.results_dir / "xgboost_model.joblib"
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save evaluation results
        results_path = self.results_dir / "evaluation_results.joblib"
        joblib.dump(evaluation_results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Save classification report as TXT
        report_txt_path = self.results_dir / "classification_report.txt"
        class_report = classification_report(y_test, y_pred)
        with open(report_txt_path, 'w') as f:
            f.write("XGBoost GPU Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(class_report)
        logger.info(f"Classification report saved to: {report_txt_path}")
        
        # Save detailed classification metrics as CSV
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Create detailed metrics DataFrame
        metrics_data = []
        for class_label, metrics in class_report_dict.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_data.append({
                    'Class': class_label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        # Add overall metrics
        metrics_data.append({
            'Class': 'Overall',
            'Precision': class_report_dict['macro avg']['precision'],
            'Recall': class_report_dict['macro avg']['recall'],
            'F1-Score': class_report_dict['macro avg']['f1-score'],
            'Support': class_report_dict['macro avg']['support']
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv_path = self.results_dir / "classification_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.16f')
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
        
        # Save training metrics summary
        summary_path = self.results_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("XGBoost GPU Training Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Training Time: {self.training_time:.4f} seconds ({self.training_time/60:.4f} minutes)\n")
            f.write(f"Inference Time: {self.inference_time:.4f} seconds\n")
            f.write(f"Inference Time per Sample: {evaluation_results['metrics']['inference_time_per_sample_ms']:.4f} ms\n")
            f.write(f"Throughput: {evaluation_results['metrics']['throughput_samples_per_sec']:.2f} samples/second\n\n")
            
            f.write("Memory Usage:\n")
            f.write(f"  Peak CPU Memory: {self.peak_memory_usage:.2f} MB\n")
            if self.gpu_memory_allocated > 0:
                f.write(f"  Peak GPU Memory Allocated: {self.gpu_memory_allocated:.2f} MB\n")
                f.write(f"  Peak GPU Memory Reserved: {self.gpu_memory_reserved:.2f} MB\n")
            f.write("\n")
            
            f.write("CLASSIFICATION METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {evaluation_results['metrics']['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {evaluation_results['metrics']['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {evaluation_results['metrics']['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {evaluation_results['metrics']['f1_weighted']:.4f}\n")
        logger.info(f"Training summary saved to: {summary_path}")
        
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
            logger.info("STARTING XGBOOST GPU TRAINING PIPELINE")
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
            
            # Get predictions for plots and saving
            y_pred = self.model.predict(X_test)
            
            # Step 6: Generate plots
            self.generate_plots(self.model, evaluation_results, X_test, y_test)
            
            # Step 7: Save results
            self.save_results(self.model, evaluation_results, y_test, y_pred)
            
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run XGBoost training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/xgboost_experiment_2.yaml",
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
        # Load configuration - handle relative paths from project root
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Get project root (parent directory of scripts)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / config_path
        
        config = XGBoostConfig.from_yaml(config_path)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Create and run training pipeline
        pipeline = XGBoostTrainingPipeline(config)
        results = pipeline.run_training(args.data)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Training Time: {pipeline.training_time:.2f} seconds ({pipeline.training_time/60:.2f} minutes)")
        print(f"Inference Time: {pipeline.inference_time:.4f} seconds")
        print(f"Inference Time per Sample: {results['metrics']['inference_time_per_sample_ms']:.4f} ms")
        print(f"Throughput: {results['metrics']['throughput_samples_per_sec']:.2f} samples/second")
        print(f"\nPeak CPU Memory Usage: {pipeline.peak_memory_usage:.2f} MB")
        if pipeline.gpu_memory_allocated > 0:
            print(f"Peak GPU Memory Allocated: {pipeline.gpu_memory_allocated:.2f} MB")
            print(f"Peak GPU Memory Reserved: {pipeline.gpu_memory_reserved:.2f} MB")
        print("\n--- CLASSIFICATION METRICS ---")
        print(f"Final Test Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"F1-Score (Macro): {results['metrics']['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
        print(f"\nResults saved to: {pipeline.results_dir}")
        print(f"Log saved to: logs/{pipeline.log_filename}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
