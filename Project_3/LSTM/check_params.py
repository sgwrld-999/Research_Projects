import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from scirpts.train_pytorch_cuda import LSTMConfig, PyTorchLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Load the reduced config
    config_path = PROJECT_ROOT / "config" / "lstm_config_reduced.yaml"
    print(f"Loading config from {config_path}")
    config = LSTMConfig.from_yaml(str(config_path))
    
    print(f"LSTM Units: {config.lstm_units}")
    
    model = PyTorchLSTM(config)
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")
    
    # Compare with original (128 units)
    config_orig = LSTMConfig.from_yaml(str(config_path))
    config_orig.lstm_units = 128
    model_orig = PyTorchLSTM(config_orig)
    params_orig = count_parameters(model_orig)
    print(f"Original parameters (128 units): {params_orig:,}")
    
    reduction = (params_orig - params) / params_orig * 100
    print(f"Reduction: {reduction:.2f}%")

if __name__ == "__main__":
    main()
