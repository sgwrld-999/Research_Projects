import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data, drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
from Phase_2_model import build_sequences, CNN_BiLSTM
from utils import get_logger, ensure_dir, set_global_seed

LOGGER = get_logger('plot_simplified_auc')

def get_lstm_probs(model, X, seq_len=10, device='cpu'):
    model.eval()
    y_dummy = np.zeros(len(X))
    X_seq, _ = build_sequences(X, y_dummy, seq_len)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            batch = X_tensor[i:i+256]
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            
    return np.concatenate(probs_list, axis=0)

def plot_roc(model_path, dataset_name, test_files, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model (Need to know input shape)
    # We'll infer input shape from the first test file
    first_df = pd.read_csv(test_files[0])
    # ... preprocessing to get shape ...
    # This is tricky without the pipeline. 
    # But for plotting, we can just re-run preprocessing.
    # We need the pipeline used during training to be exact (e.g. scaler).
    # But `phase_2_simplified` re-fit pipeline on training data.
    # We should ideally load the pipeline. But for now, we'll re-fit on training data.
    
    base_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data")
    if dataset_name == 'edge-iiot':
        train_file = base_data_dir / "edge_iiot/multiclass_train_80_20.csv"
    else:
        train_file = base_data_dir / "cic_iot/ciciot_processed_datasets_corrected/ciciot_training_50_50.csv"
        
    LOGGER.info(f"Loading training data for {dataset_name} to fit pipeline...")
    train_df = pd.read_csv(train_file)
    
    if dataset_name == 'ciciot':
        if 'label' in train_df.columns:
             train_df['Attack_label'] = (train_df['label'] != 'BenignTraffic').astype(int)
             train_df.drop(columns=['label'], inplace=True)
             
    label_col = 'Attack_label'
    processed_train, pipeline = preprocess_multiclass_data(train_df, label_col=label_col)
    le = pipeline['le']
    
    num_features = processed_train.shape[1] - 1 # minus label
    num_classes = 2
    
    model = CNN_BiLSTM(input_shape=(10, num_features), num_classes=num_classes, lstm_units=64, dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    plt.figure(figsize=(10, 8))
    
    for path in test_files:
        name = Path(path).stem
        LOGGER.info(f"Processing {name}...")
        test_df = pd.read_csv(path)
        
        # Preprocess
        if dataset_name == 'ciciot':
            if 'label' in test_df.columns:
                if 'Attack_label' not in test_df.columns:
                    test_df['Attack_label'] = (test_df['label'] != 'BenignTraffic').astype(int)
                    test_df.drop(columns=['label'], inplace=True)
            elif 'Attack_label' not in test_df.columns:
                continue
                
        if label_col not in test_df.columns: continue
        
        df = test_df
        y_test_raw = df[label_col].copy()
        df = df.drop(columns=[label_col])
        
        df = drop_zero_value_features(df)
        df = drop_zero_variance_columns(df)
        df = handle_missing_values(df)
        df = engineer_time_features(df)
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        df = convert_object_to_numeric(df)
        
        if 'minmax_scaler' in pipeline:
            scaler = pipeline['minmax_scaler']
            scaler_features = getattr(scaler, 'feature_names_in_', [])
            for f in scaler_features:
                if f not in df.columns: df[f] = 0
            if len(scaler_features) > 0:
                 df[scaler_features] = scaler.transform(df[scaler_features])
            
        df = apply_log_transformation(df)
        if 'zscore_stats' in pipeline:
            df, _ = apply_zscore_normalization(df, stats=pipeline['zscore_stats'])
        df = create_aggregated_features(df)
        
        feature_cols = processed_train.drop(columns=[label_col]).columns.tolist()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
        df = df[feature_cols]
        
        X_test = df.values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = y_test_raw.values
        
        probs = get_lstm_probs(model, X_test, seq_len=10, device=device)
        y_test_aligned = y_test[9:]
        
        fpr, tpr, _ = roc_curve(y_test_aligned, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name} (Simplified Model)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    LOGGER.info(f"Saved plot to {output_path}")

def main():
    set_global_seed(42)
    models_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\models")
    plots_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\plots")
    
    # Edge-IIoT
    edge_run_id = "edge-iiot_phase_2_20251221_123409"
    edge_model = models_dir / edge_run_id / "best_model.pth"
    edge_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\edge_iiot")
    edge_tests = [
        edge_data_dir / "multiclass_test_50_50.csv",
        edge_data_dir / "multiclass_test_60_40.csv",
        edge_data_dir / "multiclass_test_70_30.csv",
        edge_data_dir / "multiclass_test_80_20.csv"
    ]
    plot_roc(edge_model, 'edge-iiot', edge_tests, plots_dir / f"{edge_run_id}_roc.png")
    
    # CICIoT
    ciciot_run_id = "ciciot_phase_2_20251221_123546"
    ciciot_model = models_dir / ciciot_run_id / "best_model.pth"
    ciciot_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\cic_iot\ciciot_processed_datasets_corrected")
    ciciot_tests = [p for p in ciciot_data_dir.glob("*.csv") if p.name != "ciciot_training_50_50.csv"]
    plot_roc(ciciot_model, 'ciciot', ciciot_tests, plots_dir / f"{ciciot_run_id}_roc.png")

if __name__ == "__main__":
    main()
