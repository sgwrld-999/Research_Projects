import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from scirpts.train_pytorch_cuda import LSTMTrainer, LSTMConfig, DataProcessor, PyTorchLSTM

def main():
    print("Starting SHAP explanation (KernelExplainer)...", flush=True)
    
    # Config
    config_path = PROJECT_ROOT / "config" / "lstm_config_32_units_reg.yaml"
    config = LSTMConfig.from_yaml(str(config_path))
    
    device = torch.device("cpu") # KernelExplainer on CPU
    
    # Data
    processor = DataProcessor(config)
    data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
    
    df = pd.read_csv(data_path, nrows=1)
    feature_names = df.columns[:-1].tolist()
    
    raw_data = processor.load_and_validate_data(str(data_path))
    X, y = processor.prepare_data(raw_data)
    
    # Subset
    background_size = 100
    test_size = 20 # Small for speed
    
    indices = np.random.choice(X.shape[0], background_size + test_size, replace=False)
    background_indices = indices[:background_size]
    test_indices = indices[background_size:]
    
    X_background = X[background_indices]
    X_test = X[test_indices]
    
    # Flatten for KernelExplainer
    # Input to explainer must be 2D (samples, features)
    # We will reshape inside the prediction function
    X_background_flat = X_background.reshape(X_background.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Model
    model_path = PROJECT_ROOT / 'models' / 'saved_Models' / 'best_model.pth'
    model = PyTorchLSTM(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prediction function
    def predict_fn(x_flat):
        # x_flat: (samples, seq_len * features)
        # Reshape to (samples, seq_len, features)
        x_reshaped = x_flat.reshape(-1, config.seq_len, config.input_dim)
        x_tensor = torch.FloatTensor(x_reshaped).to(device)
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    # SHAP
    print("Initializing KernelExplainer...", flush=True)
    # Use kmeans to summarize background
    background_summary = shap.kmeans(X_background_flat, 10)
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    
    print("Computing SHAP values...", flush=True)
    shap_values = explainer.shap_values(X_test_flat, nsamples=100)
    # shap_values is a list of arrays (one per class), each (samples, seq_len*features)
    
    print("Processing SHAP values...", flush=True)
    
    results_dir = PROJECT_ROOT / 'results' / 'shap_results_kernel'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # shap_values seems to be (samples, seq_len*features, classes)
    shap_values = np.array(shap_values)
    print(f"SHAP values shape: {shap_values.shape}", flush=True)
    
    # We want to plot for Class 0
    # Class 0 SHAP: (samples, seq_len*features)
    shap_values_class0 = shap_values[:, :, 0]
    
    # Reshape to (samples, seq_len, features)
    shap_values_class0_reshaped = shap_values_class0.reshape(-1, config.seq_len, config.input_dim)
    
    # Mean over time: (samples, features)
    shap_values_class0_mean = np.mean(shap_values_class0_reshaped, axis=1)
    
    print(f"Class 0 Mean SHAP shape: {shap_values_class0_mean.shape}", flush=True)
    
    # X_test mean for plotting
    X_test_mean = np.mean(X_test, axis=1)
    
    # Plot Class 0
    print("Plotting...", flush=True)
    plt.figure()
    shap.summary_plot(shap_values_class0_mean, X_test_mean, feature_names=feature_names, show=False)
    plt.savefig(results_dir / 'shap_summary_class_0.png', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values_class0_mean, X_test_mean, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(results_dir / 'shap_bar_class_0.png', bbox_inches='tight')
    plt.close()
    
    print(f"SHAP plots saved to {results_dir}", flush=True)

if __name__ == "__main__":
    main()
