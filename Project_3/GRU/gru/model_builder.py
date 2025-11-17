# standard imports
from pathlib import Path
from typing import Optional, List
import warnings

# third-party imports
import torch
import torch.nn as nn
import torch.optim as optim


# custom imports
from .config_loader import GRUConfig

# ignore warnings
warnings.filterwarnings("ignore")

class GRUModel(nn.Module):
    """
    PyTorch GRU-based neural network model.
    """
    def __init__(self, config: GRUConfig):
        super(GRUModel, self).__init__()
        self.config = config
        
        # Build GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            input_size = config.input_dim if i == 0 else (config.gru_units * 2 if config.bidirectional else config.gru_units)
            
            gru_layer = nn.GRU(
                input_size=input_size,
                hidden_size=config.gru_units,
                num_layers=1,
                batch_first=True,
                bidirectional=config.bidirectional
            )
            
            self.gru_layers.append(gru_layer)
            
            if config.dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(config.dropout))
        
        # Output layer
        final_input_size = config.gru_units * 2 if config.bidirectional else config.gru_units
        self.output_layer = nn.Linear(final_input_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        for i, gru_layer in enumerate(self.gru_layers):
            x, _ = gru_layer(x)
            
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        
        # Take the last time step output
        x = x[:, -1, :]
        
        # Pass through output layer
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x


class GRUModelBuilder:
    """
    Builds a GRU-based neural network model based on the provided configuration.
    """
    def __init__(self, config):
        self.config = config
        self.validate_config()
        
    def validate_config(self):
        """
        Validates the configuration parameters to ensure they are within acceptable ranges.
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        
        # checking for potential issues 
        estimated_params = self._estimated_memory_usage()
        if estimated_params > 1e7:
            print(
                f"WARNING: Estimated number of parameters is very high: {estimated_params}. "
                "This may lead to high memory usage and slow training."
            )  
            
        # validate bidirectional + classification combination
        if self.config.bidirectional and self.config.num_classes > 100:
            print(
                "WARNING: Using bidirectional GRU with a large number of classes may lead to "
                "excessive memory usage. Consider using unidirectional GRU or reducing "
                "the number of classes."
            )
            
        # validate dropout rate
        if not (0.0 <= self.config.dropout < 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")
        
        # validate GRU units
        if self.config.gru_units <= 0:
            raise ValueError("gru_units must be a positive integer.")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if self.config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for classification tasks.")
        if self.config.input_dim is None or self.config.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer representing the number of features.")
        if not self.config.metrics or not all(isinstance(m, str) for m in self.config.metrics):
            raise ValueError("metrics must be a non-empty list of strings.")
        if self.config.export_path is not None and not isinstance(self.config.export_path, str):
            raise ValueError("export_path must be a string if provided.")
        
    def build_model(self) -> nn.Module:
        """Builds the GRU model based on the configuration.

        Returns:
            nn.Module: The constructed GRU model.
            
        Args:
            config (GRUConfig): Configuration parameters for building the model.
            
        Description:
            The model consists of multiple GRU layers (either bidirectional or unidirectional),
            followed by dropout and dense layers. The final layer uses softmax activation for
            multi-class classification.
        """
        model = GRUModel(self.config)
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model
    
    def _estimated_memory_usage(self) -> int:
        params = 0
        
        # GRU layers
        for layer_idx in range(self.config.num_layers):
            if layer_idx == 0:
                input_size = self.config.input_dim
            else:
                input_size = self.config.gru_units
                if self.config.bidirectional:
                    input_size *= 2

            # GRU parameters: 3 * (input_size * units + units * units + units)
            layer_params = 3 * (
                input_size * self.config.gru_units +
                self.config.gru_units * self.config.gru_units +
                self.config.gru_units
            )
            
            if self.config.bidirectional:
                layer_params *= 2
            
            params += layer_params
        # Dense output layer
        final_input = self.config.gru_units
        if self.config.bidirectional:
            final_input *= 2
        params += (final_input + 1) * self.config.num_classes
        
        return params
    
    def get_model_summary(self, model: nn.Module) -> None:
        """Prints the summary of the model.

        Args:
            model (nn.Module): The PyTorch model whose summary will be printed.
        """
        
        model = self.build_model()
        
        return f"""
            {self.config.get_model_summary()}

            PyTorch Model Architecture:
            ============================
            {model}
                    """.strip()
    
    @staticmethod
    def build_gru_model(config: GRUConfig) -> nn.Module:
        """
        Builds and compiles a GRU-based neural network model based on the provided configuration.

        Args:
            config (GRUConfig): Configuration parameters for building the model.
        """
        
        builder = GRUModelBuilder(config)
        model = builder.build_model()
        
        return model


# Module-level function to expose the functionality
def build_gru_model(config: GRUConfig) -> nn.Module:
    """
    Builds and compiles a GRU-based neural network model based on the provided configuration.

    Args:
        config (GRUConfig): Configuration parameters for building the model.

    Returns:
        nn.Module: PyTorch model ready for training.
    """
    return GRUModelBuilder.build_gru_model(config)