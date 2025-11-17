import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data
from Phase_2_model import train_phase2, build_sequences, CNN_BiLSTM
from utils import get_logger, ensure_dir, set_global_seed, save_json
from plots import plot_confusion_matrix

LOGGER = get_logger('phase2_ensemble')

def load_and_process_data(file_paths, label_col='Attack_type'):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    processed_df, pipeline = preprocess_multiclass_data(full_df, label_col=label_col)
    return processed_df, pipeline

def get_lstm_probs(model, X, seq_len=10, device='cpu'):
    model.eval()
    # Build sequences (dummy y)
    y_dummy = np.zeros(len(X))
    X_seq, _ = build_sequences(X, y_dummy, seq_len)
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    # Process in batches to avoid OOM
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
    
    # Paths
    base_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\edge_iiot")
    models_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\models")
    logs_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\logs")
    plots_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\plots")
    
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)
    
    run_id = datetime.now().strftime("ensemble_%Y%m%d_%H%M%S")
    LOGGER.info(f"Starting Ensemble Training Run: {run_id}")
    
    # 1. Load Training Data
    train_files = [
        base_data_dir / "multiclass_train_50_50.csv",
        base_data_dir / "multiclass_train_70_30.csv",
        base_data_dir / "multiclass_train_80_20.csv" 
    ]
    
    LOGGER.info("Loading and processing training data...")
    train_df, pipeline = load_and_process_data(train_files)
    le = pipeline['le']
    
    label_col = 'Attack_type'
    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values.astype(np.float32)
    
    # Handle NaNs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Train Random Forest
    LOGGER.info("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    LOGGER.info("Random Forest Trained.")
    
    # 3. Train CNN-BiLSTM
    LOGGER.info("Training CNN-BiLSTM...")
    # Compute Class Weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    seq_len = 10
    num_classes = len(le.classes_)
    model_save_dir = models_dir / run_id
    
    lstm_model, _ = train_phase2(
        X_train, y_train, 
        seq_len=seq_len, 
        num_classes=num_classes, 
        run_dir=model_save_dir,
        epochs=30,
        batch_size=256,
        class_weights=class_weights,
        lstm_units=96,
        dropout_rate=0.203,
        learning_rate=0.000691
    )
    LOGGER.info("CNN-BiLSTM Trained.")
    
    # 4. Evaluation
    test_files = [
        ("Test_50_50", base_data_dir / "multiclass_test_50_50.csv"),
        ("Test_60_40", base_data_dir / "multiclass_test_60_40.csv"),
        ("Test_70_30", base_data_dir / "multiclass_test_70_30.csv"),
        ("Test_80_20", base_data_dir / "multiclass_test_80_20.csv")
    ]
    
    overall_results = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, path in test_files:
        LOGGER.info(f"Evaluating on {name}...")
        test_df = pd.read_csv(path)
        
        # Preprocess test data (Simplified - assuming same pipeline steps)
        # We need to replicate the exact preprocessing steps. 
        # Ideally we should refactor preprocessing to be reusable, but for now we'll use the pipeline dict if possible
        # or just re-run the steps. Since pipeline dict only has 'le', 'minmax', 'zscore', we need to run the steps.
        
        from multi_class_data_processing import drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
        
        df = test_df
        df = df[df[label_col].isin(le.classes_)]
        df[label_col] = le.transform(df[label_col].astype(str))
        
        df = drop_zero_value_features(df)
        df = drop_zero_variance_columns(df)
        df = handle_missing_values(df)
        df = engineer_time_features(df)
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        df = convert_object_to_numeric(df)
        
        # Scaling
        if 'minmax_scaler' in pipeline:
            scaler = pipeline['minmax_scaler']
            scaler_features = getattr(scaler, 'feature_names_in_', ["tcp.seq", "udp.time_delta", "icmp.transmit_timestamp"])
            for f in scaler_features:
                if f not in df.columns: df[f] = 0
            df[scaler_features] = scaler.transform(df[scaler_features])
            
        df = apply_log_transformation(df)
        
        # Z-score
        if 'zscore_stats' in pipeline:
            y_tmp = df[label_col].copy()
            df_feats = df.drop(columns=[label_col])
            df_feats, _ = apply_zscore_normalization(df_feats, stats=pipeline['zscore_stats'])
            df = pd.concat([df_feats, y_tmp], axis=1)
            
        df = create_aggregated_features(df)
        
        # Align columns
        feature_cols = train_df.drop(columns=[label_col]).columns.tolist()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
        df = df[feature_cols + [label_col]]
        
        y_test = df[label_col].values
        X_test = df.drop(columns=[label_col]).values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # --- Ensemble Prediction ---
        
        # 1. RF Probs (N, num_classes)
        # RF is trained on individual samples.
        # LSTM is trained on sequences of length 10.
        # To ensemble, we must evaluate on the SAME targets.
        # LSTM evaluation drops the first 9 samples.
        # So we must align RF predictions to LSTM predictions.
        
        # LSTM Probs
        lstm_probs = get_lstm_probs(lstm_model, X_test, seq_len=seq_len, device=device)
        # lstm_probs shape: (N - seq_len + 1, num_classes)
        
        # RF Probs
        # We need to take the RF predictions for the SAME samples that LSTM predicts.
        # LSTM predicts for index i+seq_len-1.
        # So we need RF probs for indices [seq_len-1, seq_len, ..., N-1]
        
        rf_probs_full = rf_model.predict_proba(X_test)
        rf_probs = rf_probs_full[seq_len-1:]
        
        # Align y_test
        y_test_aligned = y_test[seq_len-1:]
        
        # Combine
        ensemble_probs = (rf_probs + lstm_probs) / 2
        y_pred = np.argmax(ensemble_probs, axis=1)
        
        # Metrics
        acc = accuracy_score(y_test_aligned, y_pred)
        LOGGER.info(f"{name} Ensemble Accuracy: {acc:.4f}")
        
        overall_results[name] = acc
        
        # Save Report
        report = classification_report(y_test_aligned, y_pred, target_names=le.classes_, output_dict=True)
        save_json(report, logs_dir / f"{run_id}_{name}_ensemble_report.json")
        
        # Plot CM
        cm = confusion_matrix(y_test_aligned, y_pred)
        plot_confusion_matrix(cm, le.classes_, out_path=plots_dir / f"{run_id}_{name}_ensemble_cm.png")
        
    LOGGER.info("Overall Ensemble Results:")
    LOGGER.info(overall_results)
    save_json(overall_results, logs_dir / f"{run_id}_ensemble_results.json")

if __name__ == "__main__":
    main()
