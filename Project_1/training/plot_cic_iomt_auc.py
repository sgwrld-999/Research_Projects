import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob
import re

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data, drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
from Phase_2_model import build_sequences, CNN_BiLSTM
from utils import get_logger, ensure_dir, set_global_seed

LOGGER = get_logger('plot_cic_iomt_auc')

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

def plot_roc_for_split(model_path, train_file, test_file, output_path, split_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LOGGER.info(f"Processing Split: {split_name}")
    LOGGER.info(f"Loading training data to fit pipeline: {train_file}")
    train_df = pd.read_csv(train_file)
    
    if 'label' in train_df.columns and 'Attack_label' not in train_df.columns:
         train_df['Attack_label'] = (train_df['label'] != 'BenignTraffic').astype(int)
         train_df.drop(columns=['label'], inplace=True)
             
    label_col = 'Attack_label'
    processed_train, pipeline = preprocess_multiclass_data(train_df, label_col=label_col)
    
    num_features = processed_train.shape[1] - 1 
    num_classes = 2
    
    # Reduced Units: 51
    model = CNN_BiLSTM(input_shape=(10, num_features), num_classes=num_classes, lstm_units=51, dropout_rate=0.16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    plt.figure(figsize=(10, 8))
    
    LOGGER.info(f"Evaluating on {test_file}...")
    test_df = pd.read_csv(test_file)
    
    # Preprocess
    if 'label' in test_df.columns and 'Attack_label' not in test_df.columns:
        test_df['Attack_label'] = (test_df['label'] != 'BenignTraffic').astype(int)
        test_df.drop(columns=['label'], inplace=True)
            
    if label_col not in test_df.columns:
        LOGGER.error(f"Label column missing in {test_file}")
        return
    
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
    
    plt.plot(fpr, tpr, lw=2, label=f'{split_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - CIC_IoMT {split_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    LOGGER.info(f"Saved plot to {output_path}")

def main():
    set_global_seed(42)
    models_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\models")
    plots_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\plots")
    data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\cici_omt")
    
    # Find the latest run for each split
    splits = ["50_50", "60_40", "70_30", "80_20"]
    
    for split in splits:
        # Find model folder
        # Pattern: cic_iomt_50_50_YYYYMMDD_HHMMSS
        pattern = f"cic_iomt_{split}_*"
        candidates = list(models_dir.glob(pattern))
        if not candidates:
            LOGGER.warning(f"No model found for split {split}")
            continue
            
        # Sort by name (timestamp)
        latest_model_dir = sorted(candidates, key=lambda x: x.name)[-1]
        model_path = latest_model_dir / "best_model.pth"
        
        train_file = data_dir / f"cic_iomt_training_{split}.csv"
        test_file = data_dir / f"cic_iomt_testing_{split}.csv"
        
        output_plot = plots_dir / f"{latest_model_dir.name}_roc.png"
        
        plot_roc_for_split(model_path, train_file, test_file, output_plot, split)

if __name__ == "__main__":
    main()
