import sys
import os
import re
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from scirpts.train_pytorch_cuda import LSTMTrainer, LSTMConfig, DataProcessor, CUDADeviceManager

def parse_training_log(log_path):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Parsing log file: {log_path}")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(r'Train Loss: ([\d\.]+), Train Acc: ([\d\.]+), Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)', line)
                if match:
                    history['train_loss'].append(float(match.group(1)))
                    history['train_acc'].append(float(match.group(2)))
                    history['val_loss'].append(float(match.group(3)))
                    history['val_acc'].append(float(match.group(4)))
    except Exception as e:
        print(f"Error parsing log: {e}")
        
    print(f"Parsed {len(history['train_loss'])} epochs.")
    return history

def safe_create_evaluation(trainer, X_val_tensor, y_val_tensor, config):
    # Create results directory
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / 'results' / f'results_{current_time}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating evaluation metrics and plots in {results_dir}")
    
    # Batch prediction
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_outputs = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch_X, _ in val_loader:
            outputs = trainer.model(batch_X)
            all_outputs.append(outputs)
            
    outputs = torch.cat(all_outputs)
    probabilities = torch.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)
    
    # Move to CPU
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_proba = probabilities.cpu().numpy()
    
    # 1. Classification Report
    trainer._save_classification_report(y_true, y_pred, results_dir)
    
    # 2. Confusion Matrix
    trainer._plot_confusion_matrix(y_true, y_pred, results_dir)
    
    # 3. ROC Curves
    trainer._plot_roc_curves(y_true, y_proba, results_dir)
    
    # 4. Training Curves
    # Use trainer.history which we populated
    trainer._plot_training_curves(results_dir)
    
    # 5. Classification Metrics CSV
    trainer._save_classification_metrics_csv(y_true, y_pred, results_dir)
    
    print(f"All evaluation results saved to {results_dir}")

def main():
    print("Starting evaluation of saved model...")
    
    # Config
    config_path = PROJECT_ROOT / "config" / "lstm_config_high_acc.yaml"
    print(f"Loading config from {config_path}")
    config = LSTMConfig.from_yaml(str(config_path))
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    processor = DataProcessor(config)
    data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
    raw_data = processor.load_and_validate_data(str(data_path))
    X, y = processor.prepare_data(raw_data)
    
    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.validation_split,
        random_state=42,
        stratify=y
    )
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Trainer
    print("Initializing trainer...")
    trainer = LSTMTrainer(config, device)
    
    # Load best model
    print("Loading best model...")
    try:
        trainer._load_checkpoint('best_model.pth')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Parse logs
    log_file = PROJECT_ROOT / 'logs' / 'training_cuda_20251222_202301.log'
    if log_file.exists():
        trainer.history = parse_training_log(log_file)
    else:
        print(f"Log file not found: {log_file}")
        trainer.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Evaluate safely
    print("Running evaluation...")
    safe_create_evaluation(trainer, X_val_tensor, y_val_tensor, config)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
