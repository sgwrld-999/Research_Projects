import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy.interpolate import make_interp_spline

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from scirpts.train_pytorch_cuda import LSTMTrainer, LSTMConfig, DataProcessor, PyTorchLSTM

def plot_roc_curves_smoothed(y_true, y_proba, save_path):
    n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(n_classes):
        try:
            if len(fpr[i]) > 100:
                unique_fpr, unique_indices = np.unique(fpr[i], return_index=True)
                unique_tpr = tpr[i][unique_indices]
                sorted_indices = np.argsort(unique_fpr)
                unique_fpr = unique_fpr[sorted_indices]
                unique_tpr = unique_tpr[sorted_indices]
                
                if len(unique_fpr) > 10:
                    x_new = np.linspace(unique_fpr.min(), unique_fpr.max(), 300)
                    spl = make_interp_spline(unique_fpr, unique_tpr, k=3)
                    y_smooth = spl(x_new)
                    y_smooth = np.clip(y_smooth, 0, 1)
                    x_plot, y_plot = x_new, y_smooth
                else:
                    x_plot, y_plot = fpr[i], tpr[i]
            else:
                x_plot, y_plot = fpr[i], tpr[i]
        except:
            x_plot, y_plot = fpr[i], tpr[i]
            
        plt.plot(x_plot, y_plot, color=colors[i % len(colors)], lw=3,
                 label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.4)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, framealpha=0.95, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting evaluation for 32 units model...")
    
    # Config
    config_path = PROJECT_ROOT / "config" / "lstm_config_32_units_reg.yaml"
    print(f"Loading config from {config_path}")
    config = LSTMConfig.from_yaml(str(config_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    processor = DataProcessor(config)
    data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
    raw_data = processor.load_and_validate_data(str(data_path))
    X, y = processor.prepare_data(raw_data)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.validation_split,
        random_state=42,
        stratify=y
    )
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Model
    model_path = PROJECT_ROOT / 'models' / 'saved_Models' / 'best_model.pth'
    print(f"Loading model from {model_path}")
    model = PyTorchLSTM(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inference
    print("Running inference...")
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    
    # Results dir - use the timestamp from the log if possible, or new one
    # I'll use a new one to be safe
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / 'results' / f'results_{current_time}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Save Report
    report = classification_report(y_true, y_pred, digits=4)
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
        
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(results_dir / 'confusion_matrix.png')
    plt.close()
    
    # Save ROC
    plot_roc_curves_smoothed(y_true, y_proba, results_dir / 'roc_curves.png')
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
