"""
Configuration Management for LSTM Models

This module provides robust configuration management for LSTM-based neural networks,
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
"""

from pathlib import Path
from typing import List, Optional
import warnings

import yaml
from pydantic import BaseModel, Field, validator

#ignoring warning
warnings.filterwarnings("ignore", category=UserWarning)


class LSTMConfig(BaseModel):
    """
    LSTM Configuration with Comprehensive Validation
    
    LSTM Architecture Parameters:
    =====================================
    
    1. INPUT_DIM: Number of features per time step
       - In time series: number of variables (temperature, humidity, etc.)
       - In NLP: embedding dimension (word vector size)
       - In network intrusion: number of network features
    
    2. SEQ_LEN: Sequence length (time steps)
       - Critical for LSTM memory capacity
       - Too short: loses long-term dependencies
       - Too long: vanishing gradient problems
       - Rule of thumb: 10-100 for most applications
    
    3. NUM_CLASSES: Output categories
       - Binary classification: 2
       - Multi-class: 3+
       - Affects final layer activation (sigmoid vs softmax)
    
    4. LSTM_UNITS: Hidden state dimension
       - Controls model capacity and memory
       - More units = more complex patterns
       - Common values: 32, 64, 128, 256, 512
       - Balance between underfitting and overfitting
    
    5. NUM_LAYERS: Depth of LSTM stack
       - Deeper networks can learn more complex patterns
       - Risk of vanishing gradients increases with depth
       - Typically 1-4 layers for most applications
    
    6. DROPOUT: Regularization technique
       - Randomly sets neurons to 0 during training
       - Prevents overfitting by reducing co-adaptation
       - Values: 0.1-0.5 (10%-50% neurons dropped)
    
    7. BIDIRECTIONAL: Information flow direction
       - Forward: past → future
       - Bidirectional: past ↔ future
       - Better for sequence labeling, not prediction
    
    8. LEARNING_RATE: Optimization step size
       - Too high: overshooting, instability
       - Too low: slow convergence
       - Adaptive optimizers (Adam) handle this better
    """
    
    # === Core Architecture Parameters ===
    input_dim: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of input features per time step"
    )
    
    seq_len: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of time steps in input sequences"
    )
    
    num_classes: int = Field(
        ...,
        ge=2,
        le=1000,
        description="Number of output classes for classification"
    )
    
    # === LSTM Layer Configuration ===
    lstm_units: int = Field(
        ...,
        ge=1,
        le=2048,
        description="Number of LSTM units (hidden state dimension)"
    )
    
    num_layers: int = Field(
        ...,
        ge=1,
        le=10,
        description="Number of stacked LSTM layers"
    )
    
    # === Regularization Parameters ===
    dropout: float = Field(
        0.2,
        ge=0.0,
        le=0.9,
        description="Dropout rate for regularization (0.0-0.9)"
    )
    
    weight_decay: float = Field(
        0.0,
        ge=0.0,
        le=0.1,
        description="L2 regularization (weight decay) factor"
    )
    
    # === Architecture Variants ===
    bidirectional: bool = Field(
        False,
        description="Use bidirectional LSTM (processes sequences in both directions)"
    )
    
    # === Training Configuration ===
    learning_rate: float = Field(
        0.001,
        gt=0.0,
        le=1.0,
        description="Learning rate for optimizer (Adam)"
    )
    
    batch_size: int = Field(
        32,
        ge=1,
        le=1024,
        description="Batch size for training"
    )
    
    epochs: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of training epochs"
    )
    
    # === Monitoring and Evaluation ===
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy"],
        description="Metrics to monitor during training"
    )
    
    validation_split: float = Field(
        0.2,
        ge=0.0,
        le=0.5,
        description="Fraction of data to use for validation"
    )
    
    early_stopping_patience: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of epochs with no improvement before stopping training"
    )
    
    # === Model Persistence ===
    export_path: str = Field(
        "models/lstm_model.keras",
        description="Path to save the trained model"
    )

    
    
    
    @validator('export_path')
    def validate_export_path(cls, path: str) -> str:
        """Ensure export directory exists or can be created."""
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        return str(export_path)
    
    # === Utility Methods ===
    @classmethod
    def from_yaml(cls, config_path: str) -> "LSTMConfig":
        """
        Load configuration from YAML file with comprehensive error handling.
        
        THEORY - Configuration as Code:
        ==============================
        YAML is preferred for ML configurations because:
        1. Human readable and writable
        2. Supports comments for documentation
        3. Hierarchical structure for complex configs
        4. Version control friendly
        5. Language agnostic
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            LSTMConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValidationError: If configuration values are invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        
        if config_data is None:
            raise ValueError(f"Empty configuration file: {config_path}")
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(
                self.dict(),
                file,
                default_flow_style=False,
                sort_keys=True,
                indent=2
            )
    
    def get_model_summary(self) -> str:
        """
        Generate a human-readable summary of the model configuration.
        
        Returns:
            Formatted string describing the model architecture
        """
        bidirectional_info = "Bidirectional " if self.bidirectional else ""
        total_params = self._estimate_parameters()
        
        return f"""
LSTM Model Configuration Summary:
================================
Architecture:
  - Input: {self.seq_len} time steps × {self.input_dim} features
  - {self.num_layers}x {bidirectional_info}LSTM layers ({self.lstm_units} units each)
  - Output: {self.num_classes} classes
  - Estimated parameters: ~{total_params:,}

Training Setup:
  - Learning rate: {self.learning_rate}
  - Batch size: {self.batch_size}
  - Max epochs: {self.epochs}
  - Dropout: {self.dropout}
  - Validation split: {self.validation_split}
  - Early stopping patience: {self.early_stopping_patience}

Monitoring:
  - Metrics: {', '.join(self.metrics)}
  - Model save path: {self.export_path}
        """.strip()
    
    def _estimate_parameters(self) -> int:
        """
        Estimate the number of trainable parameters in the model.
        
        THEORY - Parameter Estimation:
        =============================
        LSTM parameter calculation (per layer):
        - Input gate: 4 * (input_size + hidden_size + 1) * hidden_size
        - For stacked LSTMs: input_size changes for subsequent layers
        - Bidirectional: doubles the parameters
        """
        params = 0
        
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                input_size = self.input_dim
            else:
                input_size = self.lstm_units * (2 if self.bidirectional else 1)
            
            # LSTM parameters: 4 gates × (input weights + hidden weights + bias)
            layer_params = 4 * (
                (input_size * self.lstm_units) +      # Input weights
                (self.lstm_units * self.lstm_units) + # Hidden weights  
                self.lstm_units                       # Bias terms
            )
            
            if self.bidirectional:
                layer_params *= 2
            
            params += layer_params
        
        # Dense output layer
        final_input_size = self.lstm_units * (2 if self.bidirectional else 1)
        params += (final_input_size + 1) * self.num_classes
        
        return params


def load_config(config_path: str) -> LSTMConfig:
    """
    Convenience function to load LSTM configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        LSTMConfig: Validated configuration object
    """
    return LSTMConfig.from_yaml(config_path)


def save_config(config: LSTMConfig, path: Path) -> None:
    """Save LSTMConfig object into a YAML file.

    Args:
        config (LSTMConfig): The LSTMConfig object to save.
        path (Path): Path to the YAML config file.
    """
    with open(path, "w") as f:
        yaml.dump(config.dict(), f, sort_keys=False)
