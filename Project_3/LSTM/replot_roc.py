import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from scirpts.train_pytorch_cuda import LSTMTrainer, LSTMConfig, DataProcessor, PyTorchLSTM

def plot_roc_curves_custom(y_true, y_proba, save_path):
    n_classes = y_proba.shape[1]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(n_classes):
        # Smoothing
        try:
            from scipy.interpolate import make_interp_spline
            # Reduce points for smoothing if too many
            if len(fpr[i]) > 100:
                # Get unique FPR values to avoid spline error
                unique_fpr, unique_indices = np.unique(fpr[i], return_index=True)
                unique_tpr = tpr[i][unique_indices]
                
                # Sort just in case
                sorted_indices = np.argsort(unique_fpr)
                unique_fpr = unique_fpr[sorted_indices]
                unique_tpr = unique_tpr[sorted_indices]
                
                if len(unique_fpr) > 10: # Only smooth if enough points
                    x_new = np.linspace(unique_fpr.min(), unique_fpr.max(), 300)
                    spl = make_interp_spline(unique_fpr, unique_tpr, k=3)
                    y_smooth = spl(x_new)
                    # Clip to [0, 1]
                    y_smooth = np.clip(y_smooth, 0, 1)
                    x_plot, y_plot = x_new, y_smooth
                else:
                    x_plot, y_plot = fpr[i], tpr[i]
            else:
                x_plot, y_plot = fpr[i], tpr[i]
        except ImportError:
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
    
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting ROC replotting...")
    
    # Load config (using v2 as that's the current model)
    config_path = PROJECT_ROOT / "config" / "lstm_config_reduced_v2.yaml"
    print(f"Loading config from {config_path}")
    config = LSTMConfig.from_yaml(str(config_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    processor = DataProcessor(config)
    data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
    raw_data = processor.load_and_validate_data(str(data_path))
    X, y = processor.prepare_data(raw_data)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.validation_split,
        random_state=42,
        stratify=y
    )
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    # Load model
    model_path = PROJECT_ROOT / 'models' / 'saved_Models' / 'best_model.pth'
    print(f"Loading model from {model_path}")
    
    model = PyTorchLSTM(config).to(device)
    
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found!")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict
    print("Running prediction...")
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch[0])
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            
    y_proba = torch.cat(all_probs).cpu().numpy()
    
    # Calculate accuracy to verify
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    if acc < 0.95:
        print("Warning: Accuracy is lower than expected 98%. Proceeding anyway.")
    
    # Plot
    target_dir = PROJECT_ROOT / 'results' / 'results_20251222_213205'
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / 'roc_curves.png'
    
    plot_roc_curves_custom(y_val, y_proba, save_path)
    print("Done!")

if __name__ == "__main__":
    main()
