import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from datetime import datetime
import argparse
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data, drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
from Phase_2_model import train_phase2, build_sequences, CNN_BiLSTM
from utils import get_logger, ensure_dir, set_global_seed, save_json
from plots import plot_confusion_matrix

LOGGER = get_logger('phase2_cic_iomt')

def load_and_process_data(file_paths, label_col='Attack_label'):
    dfs = []
    for path in file_paths:
        LOGGER.info(f"Reading {path}...")
        df = pd.read_csv(path)
        
        # Create Binary Label if not present
        if 'label' in df.columns and 'Attack_label' not in df.columns:
            LOGGER.info("Creating binary Attack_label from 'label' column")
            df['Attack_label'] = (df['label'] != 'BenignTraffic').astype(int)
            df.drop(columns=['label'], inplace=True)
            
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    processed_df, pipeline = preprocess_multiclass_data(full_df, label_col=label_col)
    return processed_df, pipeline

def main():
    set_global_seed(42)
    
    base_data_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\cici_omt")
    models_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\models")
    logs_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\logs")
    plots_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\plots")
    
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)
    
    # Define Splits
    splits = [
        ("50_50", "cic_iomt_training_50_50.csv", "cic_iomt_testing_50_50.csv"),
        ("60_40", "cic_iomt_training_60_40.csv", "cic_iomt_testing_60_40.csv"),
        ("70_30", "cic_iomt_training_70_30.csv", "cic_iomt_testing_70_30.csv"),
        ("80_20", "cic_iomt_training_80_20.csv", "cic_iomt_testing_80_20.csv")
    ]
    
    overall_results = {}
    
    for split_name, train_file, test_file in splits:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"cic_iomt_{split_name}_{timestamp}"
        LOGGER.info(f"Starting Reduced Training Run for Split {split_name}: {run_id}")
        
        # 1. Load Training Data
        train_path = base_data_dir / train_file
        if not train_path.exists():
            LOGGER.error(f"Training file not found: {train_path}")
            continue
            
        LOGGER.info(f"Loading training data: {train_file}")
        train_df, pipeline = load_and_process_data([train_path], label_col='Attack_label')
        le = pipeline['le']
        
        label_col = 'Attack_label'
        y_train = train_df[label_col].values
        X_train = train_df.drop(columns=[label_col]).values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0)
        
        LOGGER.info(f"Classes: {le.classes_}") 
        
        # 2. Train Reduced CNN-BiLSTM
        LOGGER.info("Training Reduced CNN-BiLSTM (51 Units)...")
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
            epochs=16, 
            batch_size=256,
            class_weights=class_weights,
            lstm_units=51, 
            dropout_rate=0.16, 
            learning_rate=0.0008 
        )
        LOGGER.info("Model Trained.")
        
        # 3. Evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_model.eval()
        
        test_path = base_data_dir / test_file
        LOGGER.info(f"Evaluating on {test_file}...")
        
        if not test_path.exists():
             LOGGER.error(f"Test file not found: {test_path}")
             continue

        test_df = pd.read_csv(test_path)
        
        # Create Binary Label
        if 'label' in test_df.columns:
            if 'Attack_label' not in test_df.columns:
                test_df['Attack_label'] = (test_df['label'] != 'BenignTraffic').astype(int)
                test_df.drop(columns=['label'], inplace=True)
        elif 'Attack_label' not in test_df.columns:
            LOGGER.warning(f"Skipping {test_file}: Label column missing.")
            continue
        
        # Preprocess
        df = test_df
        if label_col not in df.columns:
             LOGGER.warning(f"Skipping {test_file}: {label_col} missing.")
             continue

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
        
        # Align columns
        feature_cols = train_df.drop(columns=[label_col]).columns.tolist()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
        df = df[feature_cols]
        
        y_test = y_test_raw.values
        X_test = df.values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # Prediction
        y_dummy = np.zeros(len(X_test))
        X_seq, _ = build_sequences(X_test, y_dummy, seq_len)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch = X_tensor[i:i+256]
                outputs = lstm_model(batch)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())
        
        y_probs = np.concatenate(probs_list, axis=0)
        y_pred = np.argmax(y_probs, axis=1)
        y_test_aligned = y_test[seq_len-1:]
        
        # Metrics
        acc = accuracy_score(y_test_aligned, y_pred)
        try:
            auc_score = roc_auc_score(y_test_aligned, y_probs[:, 1])
        except:
            auc_score = 0.0
            
        LOGGER.info(f"{split_name} Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
        overall_results[split_name] = {"accuracy": acc, "auc": auc_score}
        
        # Save Report
        target_names = [str(c) for c in le.classes_]
        report = classification_report(y_test_aligned, y_pred, target_names=target_names, output_dict=True)
        save_json(report, logs_dir / f"{run_id}_report.json")
        
        # Plot CM
        cm = confusion_matrix(y_test_aligned, y_pred)
        plot_confusion_matrix(cm, target_names, out_path=plots_dir / f"{run_id}_cm.png")
        
    LOGGER.info("Overall Results for CIC_IoMT Splits:")
    LOGGER.info(overall_results)
    save_json(overall_results, logs_dir / f"cic_iomt_results.json")

if __name__ == "__main__":
    main()
