"""
XGBoost Model Builder Module

This module provides a comprehensive implementation for building and configuring
XGBoost-based models for multiclass classification tasks such as network intrusion
detection, fraud detection, and general classification problems.

THEORY - XGBoost (eXtreme Gradient Boosting):
=============================================

XGBoost is an optimized distributed gradient boosting library designed to be highly
efficient, flexible, and portable. It implements machine learning algorithms under
the Gradient Boosting framework.

KEY CONCEPTS:

1. GRADIENT BOOSTING FRAMEWORK:
   - Ensemble method that builds models sequentially
   - Each new model corrects errors made by previous models
   - Final prediction is the sum of predictions from all models

2. GRADIENT BOOSTING MATH:
   
   Given training data (x_i, y_i) where i = 1, ..., n:
   
   a) OBJECTIVE FUNCTION:
      Obj = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
      where:
      - l(y_i, ŷ_i) is the loss function
      - Ω(f_k) is the regularization term for tree k
      - ŷ_i = Σ f_k(x_i) is the prediction

   b) ADDITIVE TRAINING:
      ŷ_i^(t) = ŷ_i^(t-1) + f_t(x_i)
      where f_t is the tree added at iteration t

   c) SECOND-ORDER APPROXIMATION:
      Uses both first and second derivatives for optimization:
      g_i = ∂l(y_i, ŷ_i^(t-1))/∂ŷ_i^(t-1)  (gradient)
      h_i = ∂²l(y_i, ŷ_i^(t-1))/∂ŷ_i^(t-1)² (hessian)

3. REGULARIZATION:
   
   Ω(f) = γT + (1/2)λ||w||²
   where:
   - T is the number of leaves
   - w is the vector of leaf weights
   - γ controls tree complexity
   - λ controls leaf weight regularization

4. TREE LEARNING:
   
   For each leaf j, the optimal weight is:
   w_j = -G_j/(H_j + λ)
   where G_j = Σ g_i and H_j = Σ h_i for instances in leaf j

5. SPLIT FINDING:
   
   Gain from splitting = (1/2)[G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ

THEORY - XGBoost Advantages:
===========================

1. EFFICIENCY:
   - Parallel tree construction
   - Cache-aware access patterns
   - Block structure for out-of-core computation

2. ACCURACY:
   - Second-order optimization
   - Advanced regularization
   - Handles missing values automatically

3. FLEXIBILITY:
   - Custom objective functions
   - Custom evaluation metrics
   - Cross-validation built-in

Classes:
    XGBoostModelBuilder: Main class for building XGBoost models

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import warnings
import os

# Third-party imports
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Custom imports
from .config_loader import XGBoostConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class XGBoostModelBuilder:
    """
    Builds an XGBoost-based model for multiclass classification.
    
    This class provides a comprehensive interface for building, training, and
    evaluating XGBoost models with advanced features like early stopping,
    cross-validation, and feature importance analysis.
    
    Attributes:
        config (XGBoostConfig): Configuration object containing model parameters
        model (xgb.XGBClassifier): The XGBoost classifier instance
        label_encoder (LabelEncoder): Encoder for target labels
        feature_names (List[str]): Names of input features
        
    Example:
        >>> config = XGBoostConfig.from_yaml("config.yaml")
        >>> builder = XGBoostModelBuilder(config)
        >>> model = builder.build_model(input_shape=(100,))
        >>> builder.train(X_train, y_train, X_val, y_val)
    """
    
    def __init__(self, config: XGBoostConfig):
        """
        Initialize the XGBoost model builder.
        
        Args:
            config (XGBoostConfig): Configuration object with model parameters
            
        Raises:
            ValueError: If configuration validation fails
        """
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Validate configuration
        self.validate_config()
        
        logger.info("XGBoost Model Builder initialized successfully")
        
    def validate_config(self) -> None:
        """
        Validates the configuration parameters to ensure they are within acceptable ranges.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate basic parameters
        if self.config.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
            
        if not 0 < self.config.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
            
        if not 0 <= self.config.max_depth <= 20:
            raise ValueError("max_depth must be between 0 and 20")
            
        if not 0 <= self.config.subsample <= 1:
            raise ValueError("subsample must be between 0 and 1")
            
        if not 0 <= self.config.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be between 0 and 1")
            
        if self.config.reg_alpha < 0:
            raise ValueError("reg_alpha must be non-negative")
            
        if self.config.reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
            
        # Validate objective
        valid_objectives = ['multi:softprob', 'multi:softmax', 'binary:logistic']
        if self.config.objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
            
        # Performance warnings
        if self.config.n_estimators > 1000:
            logger.warning(
                f"High n_estimators ({self.config.n_estimators}) may lead to long training times"
            )
            
        if self.config.max_depth > 10:
            logger.warning(
                f"High max_depth ({self.config.max_depth}) may lead to overfitting"
            )
            
        logger.info("Configuration validation completed successfully")
        
    def build_model(self, input_shape: tuple, num_classes: Optional[int] = None) -> xgb.XGBClassifier:
        """
        Build the XGBoost classifier model.
        
        Args:
            input_shape (tuple): Shape of input features (n_features,)
            num_classes (Optional[int]): Number of target classes. If None, uses config value
            
        Returns:
            xgb.XGBClassifier: Configured XGBoost classifier
            
        Raises:
            ValueError: If input_shape is invalid
        """
        if len(input_shape) != 1:
            raise ValueError("input_shape must be 1D tuple representing number of features")
            
        n_features = input_shape[0]
        if n_features <= 0:
            raise ValueError("Number of features must be positive")
            
        # Use provided num_classes or fall back to config
        if num_classes is not None:
            self.config.num_classes = num_classes
            
        # Configure XGBoost parameters
        xgb_params = {
            'n_estimators': self.config.n_estimators,
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'objective': self.config.objective,
            'num_class': self.config.num_classes if self.config.objective == 'multi:softmax' else None,
            'eval_metric': self.config.eval_metric,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbosity': 0,  # Reduce XGBoost verbosity
        }
        
        # Remove None values
        xgb_params = {k: v for k, v in xgb_params.items() if v is not None}
        
        # Enable GPU if specified (XGBoost 3.1+ uses 'device' instead of 'gpu_id' and 'hist' instead of 'gpu_hist')
        if self.config.tree_method == 'gpu_hist':
            xgb_params['tree_method'] = 'hist'  # In XGBoost 3.1+, use 'hist' with device parameter
            xgb_params['device'] = 'cuda:0'
            logger.info("GPU acceleration enabled for XGBoost")
        else:
            xgb_params['tree_method'] = 'hist'  # Use histogram-based algorithm for speed
            
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(**xgb_params)
        
        logger.info("XGBoost model built successfully")
        logger.info(f"Model configuration: {xgb_params}")
        
        return self.model
        
    def get_model_summary(self) -> str:
        """
        Generate a comprehensive summary of the model configuration.
        
        Returns:
            str: Formatted model summary
        """
        if self.model is None:
            return "Model not built yet. Call build_model() first."
            
        summary = f"""
XGBoost Model Configuration Summary:
===================================
Architecture:
  - Algorithm: XGBoost (eXtreme Gradient Boosting)
  - Objective: {self.config.objective}
  - Number of classes: {self.config.num_classes}
  - Tree method: {self.config.tree_method}

Boosting Parameters:
  - Number of estimators: {self.config.n_estimators}
  - Learning rate: {self.config.learning_rate}
  - Max depth: {self.config.max_depth}
  - Min child weight: {self.config.min_child_weight}

Regularization:
  - L1 regularization (alpha): {self.config.reg_alpha}
  - L2 regularization (lambda): {self.config.reg_lambda}
  - Subsample ratio: {self.config.subsample}
  - Feature subsample ratio: {self.config.colsample_bytree}

Training Setup:
  - Evaluation metric: {self.config.eval_metric}
  - Random state: {self.config.random_state}
  - Number of jobs: {self.config.n_jobs}

Performance Features:
  - Feature importance: Available after training
  - Cross-validation: Built-in support
  - Missing value handling: Automatic
  - Memory optimization: Histogram-based algorithm
"""
        
        if hasattr(self.model, 'n_features_in_'):
            summary += f"  - Input features: {self.model.n_features_in_}\n"
            
        return summary
        
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (Union[str, Path]): Path to save the model
            
        Raises:
            ValueError: If model is not trained
            IOError: If save operation fails
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the model using joblib (recommended for sklearn-compatible models)
            joblib.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'config': self.config
            }, filepath)
            
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
            
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (Union[str, Path]): Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            IOError: If load operation fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            # Load the model
            saved_data = joblib.load(filepath)
            
            self.model = saved_data['model']
            self.label_encoder = saved_data['label_encoder']
            self.feature_names = saved_data['feature_names']
            self.config = saved_data['config']
            self.is_trained = True
            
            logger.info(f"Model loaded successfully from {filepath}")
            
        except Exception as e:
            raise IOError(f"Failed to load model: {str(e)}")
            
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type (str): Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            pd.DataFrame: Feature importance scores
            
        Raises:
            ValueError: If model is not trained or invalid importance type
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        valid_types = ['weight', 'gain', 'cover']
        if importance_type not in valid_types:
            raise ValueError(f"importance_type must be one of {valid_types}")
            
        # Get feature importance
        importance_scores = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        if self.feature_names is not None:
            # Use provided feature names
            feature_names = self.feature_names
        else:
            # Use default feature names
            feature_names = [f'f{i}' for i in range(len(importance_scores))]
            
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance_scores.get(feature, 0.0)}
            for feature in feature_names
        ]).sort_values('importance', ascending=False)
        
        return importance_df


def create_xgboost_model(config: XGBoostConfig, input_shape: tuple, 
                        num_classes: Optional[int] = None) -> xgb.XGBClassifier:
    """
    Factory function to create an XGBoost model.
    
    Args:
        config (XGBoostConfig): Configuration object
        input_shape (tuple): Shape of input features
        num_classes (Optional[int]): Number of target classes
        
    Returns:
        xgb.XGBClassifier: Configured XGBoost model
    """
    builder = XGBoostModelBuilder(config)
    return builder.build_model(input_shape, num_classes)


# Example usage and testing
if __name__ == "__main__":
    # This section is for testing and demonstration
    print("XGBoost Model Builder Module")
    print("============================")
    
    # Create a simple test configuration
    from .config_loader import XGBoostConfig
    
    # Test configuration
    test_config = XGBoostConfig(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_classes=5
    )
    
    # Test model building
    builder = XGBoostModelBuilder(test_config)
    model = builder.build_model(input_shape=(20,))
    
    print("Model built successfully!")
    print(builder.get_model_summary())
