"""
Configuration Management for XGBoost Models

This module provides robust configuration management for XGBoost-based classifiers,
following software engineering best practices and ensuring type safety through
Pydantic validation.

THEORY - Configuration Management in Machine Learning:
=========================================================
Configuration management is crucial in ML projects because:
1. Reproducibility: Same config = same results
2. Experimentation: Easy parameter tuning without code changes
3. Validation: Prevents invalid parameter combinations
4. Documentation: Self-documenting parameter constraints
5. Version Control: Track configuration changes alongside code

THEORY - Pydantic for Data Validation:
======================================
Pydantic provides runtime type checking and data validation:
- Automatic type conversion when possible
- Clear error messages for invalid data
- IDE support with type hints
- JSON/YAML serialization support
- Custom validators for complex constraints

THEORY - XGBoost Hyperparameters:
=================================
XGBoost has numerous hyperparameters that control different aspects of training:

1. GENERAL PARAMETERS:
   - booster: Type of model (gbtree, gblinear, dart)
   - n_jobs: Parallel processing threads
   - random_state: Seed for reproducibility

2. TREE BOOSTER PARAMETERS:
   - n_estimators: Number of boosting rounds
   - max_depth: Maximum depth of trees
   - min_child_weight: Minimum sum of instance weights in child
   - subsample: Fraction of samples used for training each tree
   - colsample_bytree: Fraction of features used for training each tree

3. LEARNING TASK PARAMETERS:
   - objective: Learning objective (multi:softmax, multi:softprob)
   - eval_metric: Evaluation metric for validation
   - num_class: Number of classes (for multiclass)

4. REGULARIZATION PARAMETERS:
   - reg_alpha: L1 regularization on weights
   - reg_lambda: L2 regularization on weights
   - gamma: Minimum loss reduction for split

5. LEARNING CONTROL:
   - learning_rate: Step size shrinkage (eta)
   - early_stopping_rounds: Stop if no improvement
   - verbose: Control output verbosity

Code Writing style: PEP 8 compliant, well-documented, modular functions and classes
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Third-party imports
import yaml
from pydantic import BaseModel, Field, validator

# Ignoring specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class XGBoostConfig(BaseModel):
    """Configuration class for XGBoost model.

    This class defines the configuration parameters for an XGBoost-based
    classifier with comprehensive validation and type checking.
    
    THEORY - XGBoost Configuration:
    ===============================
    
    XGBoost hyperparameters are organized into several categories:
    
    1. GENERAL PARAMETERS:
       - Control the overall functioning of XGBoost
       - booster: Type of weak learner
       - n_jobs: Parallelization settings
    
    2. BOOSTER PARAMETERS:
       - Control individual boosters (trees)
       - max_depth: Tree complexity
       - learning_rate: Shrinkage to prevent overfitting
       - subsample: Stochastic training
    
    3. LEARNING TASK PARAMETERS:
       - Define the learning scenario
       - objective: Loss function to minimize
       - eval_metric: Performance measurement
    
    4. REGULARIZATION:
       - Control overfitting
       - reg_alpha: L1 regularization
       - reg_lambda: L2 regularization
       - gamma: Minimum split loss
    """
    
    # === Core Architecture Parameters ===
    input_dim: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of input features"
    )
    
    num_classes: int = Field(
        ...,
        ge=2,
        le=1000,
        description="Number of output classes for classification"
    )
    
    # === General Parameters ===
    booster: str = Field(
        default="gbtree",
        description="Type of booster to use"
    )
    
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of parallel threads (-1 for all cores)"
    )
    
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    # === Tree Booster Parameters ===
    n_estimators: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of boosting rounds"
    )
    
    max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum depth of trees"
    )
    
    min_child_weight: float = Field(
        default=1.0,
        ge=0,
        description="Minimum sum of instance weights needed in child"
    )
    
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Step size shrinkage to prevent overfitting"
    )
    
    subsample: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Fraction of samples used for training each tree"
    )
    
    colsample_bytree: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Fraction of features used for training each tree"
    )
    
    colsample_bylevel: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Fraction of features used for each level"
    )
    
    colsample_bynode: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Fraction of features used for each split"
    )
    
    # === Regularization Parameters ===
    reg_alpha: float = Field(
        default=0.0,
        ge=0,
        description="L1 regularization term on weights"
    )
    
    reg_lambda: float = Field(
        default=1.0,
        ge=0,
        description="L2 regularization term on weights"
    )
    
    gamma: float = Field(
        default=0.0,
        ge=0,
        description="Minimum loss reduction required for split"
    )
    
    # === Learning Task Parameters ===
    objective: str = Field(
        default="multi:softprob",
        description="Learning objective function"
    )
    
    eval_metric: Union[str, List[str]] = Field(
        default="mlogloss",
        description="Evaluation metric for validation data"
    )
    
    # === Training Control ===
    verbose: bool = Field(
        default=False,
        description="Print messages during training"
    )
    
    # === Advanced Parameters ===
    max_delta_step: float = Field(
        default=0.0,
        ge=0,
        description="Maximum delta step for weight estimation"
    )
    
    scale_pos_weight: float = Field(
        default=1.0,
        gt=0,
        description="Balancing of positive and negative weights"
    )
    
    # === Tree Method ===
    tree_method: str = Field(
        default="auto",
        description="Tree construction algorithm"
    )
    
    grow_policy: str = Field(
        default="depthwise",
        description="Tree growth policy"
    )
    
    max_leaves: int = Field(
        default=0,
        ge=0,
        description="Maximum number of leaves (0 for no limit)"
    )
    
    max_bin: int = Field(
        default=256,
        ge=2,
        le=65536,
        description="Maximum number of discrete bins for features"
    )
    
    # === GPU Parameters ===
    gpu_id: int = Field(
        default=-1,
        ge=-1,
        description="GPU device ID (-1 for CPU)"
    )
    
    # === Training Configuration ===
    test_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of data for testing"
    )
    
    validation_split: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of training data for validation"
    )
    
    # === Feature Engineering ===
    feature_selection: bool = Field(
        default=False,
        description="Whether to perform feature selection"
    )
    
    feature_selection_method: str = Field(
        default="importance",
        description="Method for feature selection"
    )
    
    n_features_to_select: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of features to select"
    )
    
    # === Cross-Validation ===
    use_cv: bool = Field(
        default=False,
        description="Whether to use cross-validation for training"
    )
    
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of cross-validation folds"
    )
    
    # === Evaluation Metrics ===
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score"],
        description="Metrics to evaluate model performance"
    )
    
    # === Data Configuration ===
    target_column: str = Field(
        default="label_stage_encoded",
        description="Name of the target column in the dataset"
    )
    
    feature_columns: Optional[List[str]] = Field(
        default=None,
        description="List of feature column names. If None, all columns except target will be used"
    )
    
    scale_features: bool = Field(
        default=False,
        description="Whether to apply StandardScaler to features. Set to False if data is already normalized"
    )
    
    # === Model Persistence ===
    export_path: str = Field(
        default="models/saved_Models/xgboost_experiment1.joblib",
        description="Path to save the trained model"
    )
    
    # === Validation Methods ===
    @validator('booster')
    def validate_booster(cls, v):
        """Validate booster parameter."""
        allowed_boosters = {'gbtree', 'gblinear', 'dart'}
        if v not in allowed_boosters:
            raise ValueError(f"booster must be one of {allowed_boosters}")
        return v
    
    @validator('objective')
    def validate_objective(cls, v):
        """Validate objective parameter."""
        allowed_objectives = {
            'multi:softmax', 'multi:softprob', 'binary:logistic',
            'reg:squarederror', 'reg:logistic'
        }
        if v not in allowed_objectives:
            raise ValueError(f"objective must be one of {allowed_objectives}")
        return v
    
    @validator('eval_metric')
    def validate_eval_metric(cls, v):
        """Validate evaluation metric parameter."""
        allowed_metrics = {
            'rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss',
            'auc', 'aucpr', 'ndcg', 'map'
        }
        
        if isinstance(v, str):
            if v not in allowed_metrics:
                raise ValueError(f"eval_metric must be one of {allowed_metrics}")
        elif isinstance(v, list):
            for metric in v:
                if metric not in allowed_metrics:
                    raise ValueError(f"eval_metric '{metric}' not in allowed metrics")
        
        return v
    
    @validator('tree_method')
    def validate_tree_method(cls, v):
        """Validate tree method parameter."""
        allowed_methods = {'auto', 'exact', 'approx', 'hist', 'gpu_hist'}
        if v not in allowed_methods:
            raise ValueError(f"tree_method must be one of {allowed_methods}")
        return v
    
    @validator('grow_policy')
    def validate_grow_policy(cls, v):
        """Validate grow policy parameter."""
        allowed_policies = {'depthwise', 'lossguide'}
        if v not in allowed_policies:
            raise ValueError(f"grow_policy must be one of {allowed_policies}")
        return v
    
    @validator('feature_selection_method')
    def validate_feature_selection_method(cls, v):
        """Validate feature selection method."""
        allowed_methods = {'importance', 'gain', 'cover', 'weight'}
        if v not in allowed_methods:
            raise ValueError(f"feature_selection_method must be one of {allowed_methods}")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate metric names."""
        allowed_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'confusion_matrix', 'classification_report',
            'feature_importance'
        }
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Unknown metric: {metric}")
        return v
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "XGBoostConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            XGBoostConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.dict(), file, default_flow_style=False, indent=2)
    
    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get parameters formatted for XGBoost.
        
        Returns:
            Dict[str, Any]: Parameters ready for XGBoost
        """
        xgb_params = {
            # General parameters
            'booster': self.booster,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            
            # Tree parameters
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'colsample_bynode': self.colsample_bynode,
            
            # Regularization
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            
            # Learning task
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'num_class': self.num_classes,
            
            # Advanced
            'max_delta_step': self.max_delta_step,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': self.tree_method,
            'grow_policy': self.grow_policy,
            'max_leaves': self.max_leaves,
            'max_bin': self.max_bin,
            
            # Training control
            'verbose': self.verbose
        }
        
        # Add GPU support if specified (XGBoost 3.1+ uses 'device' instead of 'gpu_id' and 'hist' instead of 'gpu_hist')
        if self.gpu_id >= 0:
            xgb_params['device'] = f'cuda:{self.gpu_id}'
            if self.tree_method == 'auto' or self.tree_method == 'gpu_hist':
                xgb_params['tree_method'] = 'hist'  # In XGBoost 3.1+, use 'hist' with device parameter
        
        # Remove None values and objective-specific adjustments
        xgb_params = {k: v for k, v in xgb_params.items() if v is not None}
        
        # For multi-class classification
        if self.objective in ['multi:softmax', 'multi:softprob']:
            xgb_params['num_class'] = self.num_classes
        
        return xgb_params


def load_config(config_path: str) -> XGBoostConfig:
    """Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        XGBoostConfig: Validated configuration object
        
    Example:
        >>> config = load_config('config/xgboost_experiment_1.yaml')
        >>> print(f"Number of estimators: {config.n_estimators}")
    """
    config_file = Path(config_path)
    return XGBoostConfig.from_yaml(config_file)
