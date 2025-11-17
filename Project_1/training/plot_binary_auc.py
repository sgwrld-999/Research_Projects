import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data
from Phase_2_model import build_sequences, CNN_BiLSTM
from utils import get_logger, ensure_dir, set_global_seed

LOGGER = get_logger('plot_binary_auc')

def load_and_process_data(file_paths, label_col='Attack_label'):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    processed_df, pipeline = preprocess_multiclass_data(full_df, label_col=label_col)
    return processed_df, pipeline

def get_lstm_probs(model, X, seq_len=10, device='cpu'):
    model.eval()
    y_dummy = np.zeros(len(X))
    X_seq, _ = build_sequences(X, y_dummy, seq_len)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    batch_size = 256
    probs_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            
    return np.concatenate(probs_list, axis=0)

def main():
    set_global_seed(42)
    
    base_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\edge_iiot")
    models_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\models")
    plots_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\plots")
    ensure_dir(plots_dir)
    
    # Model Run ID
    run_id = "binary_ensemble_20251219_155814"
    model_path = models_dir / run_id / "best_model.pth"
    
    # 1. Retrain RF (Fast)
    train_files = [
        base_data_dir / "multiclass_train_50_50.csv",
        base_data_dir / "multiclass_train_70_30.csv",
        base_data_dir / "multiclass_train_80_20.csv" 
    ]
    
    LOGGER.info("Loading training data to retrain RF...")
    train_df, pipeline = load_and_process_data(train_files, label_col='Attack_label')
    le = pipeline['le']
    label_col = 'Attack_label'
    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    LOGGER.info("Retraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 2. Load LSTM
    LOGGER.info("Loading LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reconstruct model architecture
    # We need to know input_shape and num_classes
    # input_shape[1] is num_features. X_train.shape[1]
    # num_classes is 2
    num_features = X_train.shape[1]
    num_classes = 2
    
    lstm_model = CNN_BiLSTM(input_shape=(10, num_features), num_classes=num_classes, lstm_units=96, dropout_rate=0.203)
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    
    # 3. Evaluate and Plot
    test_files = [
        ("Test_50_50", base_data_dir / "multiclass_test_50_50.csv"),
        ("Test_60_40", base_data_dir / "multiclass_test_60_40.csv"),
        ("Test_70_30", base_data_dir / "multiclass_test_70_30.csv"),
        ("Test_80_20", base_data_dir / "multiclass_test_80_20.csv")
    ]
    
    plt.figure(figsize=(10, 8))
    
    for name, path in test_files:
        LOGGER.info(f"Evaluating on {name}...")
        test_df = pd.read_csv(path)
        
        # Preprocess (Simplified replication of pipeline)
        from multi_class_data_processing import drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
        
        df = test_df
        df[label_col] = le.transform(df[label_col].astype(str))
        
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
            scaler_features = getattr(scaler, 'feature_names_in_', ["tcp.seq", "udp.time_delta", "icmp.transmit_timestamp"])
            for f in scaler_features:
                if f not in df.columns: df[f] = 0
            df[scaler_features] = scaler.transform(df[scaler_features])
            
        df = apply_log_transformation(df)
        
        if 'zscore_stats' in pipeline:
            y_tmp = df[label_col].copy()
            df_feats = df.drop(columns=[label_col])
            df_feats, _ = apply_zscore_normalization(df_feats, stats=pipeline['zscore_stats'])
            df = pd.concat([df_feats, y_tmp], axis=1)
            
        df = create_aggregated_features(df)
        
        feature_cols = train_df.drop(columns=[label_col]).columns.tolist()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
        df = df[feature_cols + [label_col]]
        
        y_test = df[label_col].values
        X_test = df.drop(columns=[label_col]).values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # Predictions
        seq_len = 10
        lstm_probs = get_lstm_probs(lstm_model, X_test, seq_len=seq_len, device=device)
        rf_probs = rf_model.predict_proba(X_test)[seq_len-1:]
        
        ensemble_probs = (rf_probs + lstm_probs) / 2
        y_test_aligned = y_test[seq_len-1:]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test_aligned, ensemble_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Binary Ensemble)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    out_path = plots_dir / f"{run_id}_roc_curves.png"
    plt.savefig(out_path)
    LOGGER.info(f"Saved ROC curves to {out_path}")

if __name__ == "__main__":
    main()
