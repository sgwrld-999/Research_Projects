from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data
from Phase_2_model import train_phase2, evaluate_model
from utils import get_logger, ensure_dir, set_global_seed, save_json
from plots import plot_training_history, plot_confusion_matrix

LOGGER = get_logger('phase2_train')

def load_and_process_data(file_paths, label_col='Attack_type', le=None):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    processed_df, pipeline = preprocess_multiclass_data(full_df, label_col=label_col)
    
    return processed_df, pipeline

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
    
    run_id = datetime.now().strftime("phase2_%Y%m%d_%H%M%S")
    LOGGER.info(f"Starting Phase 2 Training Run: {run_id}")
    
    # 1. Training Datasets
    # Note: User listed multiclass_test_80_20.csv in training list, assuming typo and using multiclass_train_80_20.csv
    train_files = [
        base_data_dir / "multiclass_train_50_50.csv",
        base_data_dir / "multiclass_train_70_30.csv",
        base_data_dir / "multiclass_train_80_20.csv" 
    ]
    
    LOGGER.info("Loading and processing training data...")
    train_df, pipeline = load_and_process_data(train_files)
    le = pipeline['le']
    
    # Prepare X and y
    label_col = 'Attack_type'
    if label_col not in train_df.columns:
         # Try to find the encoded label column if name changed or just use the one from le
         # In preprocess_multiclass_data, it keeps the same name
         pass

    # Update feature_cols after dropping
    feature_cols = train_df.drop(columns=[label_col]).columns.tolist()
    
    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values

    # Debug: Check for non-numeric columns
    non_numeric_cols = train_df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        LOGGER.warning(f"Found non-numeric columns: {non_numeric_cols.tolist()}")
        # Force drop or convert
        train_df = train_df.drop(columns=non_numeric_cols)
        LOGGER.info("Dropped non-numeric columns.")
        # Re-create X_train
        X_train = train_df.drop(columns=[label_col]).values

    # Update feature_cols after dropping non-numeric (ensure consistency)
    feature_cols = train_df.drop(columns=[label_col]).columns.tolist()

    # Ensure X_train is float32
    X_train = X_train.astype(np.float32)
    
    # Check for NaNs or Infs
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        LOGGER.warning("X_train contains NaNs or Infs. Replacing with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute Class Weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    LOGGER.info(f"Computed Class Weights: {class_weights}")
    
    # 2. Train Model
    LOGGER.info("Starting Model Training...")
    seq_len = 10
    model_save_dir = models_dir / run_id
    
    num_classes = len(le.classes_)
    
    # Identify minority classes for augmentation
    target_names = ['Injection', 'Backdoor', 'DDoS_ICMP', 'Ransomware', 'XSS', 'Scanning', 'Password']
    augment_classes = []
    for name in target_names:
        if name in le.classes_:
            idx = le.transform([name])[0]
            augment_classes.append(int(idx))
    LOGGER.info(f"Augmenting classes: {target_names} -> Indices: {augment_classes}")

    model, history = train_phase2(
        X_train, y_train, 
        seq_len=seq_len, 
        num_classes=num_classes, 
        run_dir=model_save_dir,
        epochs=30,
        batch_size=256,
        class_weights=class_weights,
        lstm_units=96,
        dropout_rate=0.203,
        learning_rate=0.000691,
        augment_classes=None
    )
    
    # Save History
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_save_dir / 'training_history.csv', index=False)
    LOGGER.info(f"Saved training history to {model_save_dir / 'training_history.csv'}")
    
    # Save Plots
    save_json(history, logs_dir / f"{run_id}_history.json")
    plot_training_history(history, out_path=plots_dir / f"{run_id}_training_history.png")
    
    # 3. Evaluation on Test Datasets
    test_files = [
        ("Test_50_50", base_data_dir / "multiclass_test_50_50.csv"),
        ("Test_60_40", base_data_dir / "multiclass_test_60_40.csv"),
        ("Test_70_30", base_data_dir / "multiclass_test_70_30.csv"),
        ("Test_80_20", base_data_dir / "multiclass_test_80_20.csv")
    ]
    
    overall_results = {}
    
    for name, path in test_files:
        LOGGER.info(f"Evaluating on {name}...")
        test_df = pd.read_csv(path)
        
        # Preprocess test data
        from multi_class_data_processing import drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
        
        df = test_df
        
        # Transform labels using training le
        # Filter out unknown classes if any
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
        
        # Apply MinMax scaling using training scaler
        if 'minmax_scaler' in pipeline:
            scaler = pipeline['minmax_scaler']
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = scaler.feature_names_in_
            else:
                # Fallback if not fitted with feature names (should not happen with pandas)
                scaler_features = ["tcp.seq", "udp.time_delta", "icmp.transmit_timestamp"]
            
            # Ensure all features exist
            for f in scaler_features:
                if f not in df.columns:
                    df[f] = 0
            
            # Transform using the correct order
            df[scaler_features] = scaler.transform(df[scaler_features])
        
        df = apply_log_transformation(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)
            
        # Apply Z-score normalization using training stats
        if 'zscore_stats' in pipeline:
            # We need to pass stats to apply_zscore_normalization
            # But we need to exclude label_col first
            y = df[label_col].copy()
            df_features = df.drop(columns=[label_col])
            
            df_features, _ = apply_zscore_normalization(df_features, stats=pipeline['zscore_stats'])
            
            df = pd.concat([df_features, y], axis=1)
            
        df = create_aggregated_features(df)
        
        try:
            # Align columns with training data
            LOGGER.info(f"Aligning columns. df type: {type(df)}")
            if isinstance(df, pd.DataFrame):
                LOGGER.info(f"df columns: {len(df.columns)}")
            
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = 0
                    
            # Drop extra cols and reorder
            # Ensure label_col is present
            if label_col not in df.columns:
                 LOGGER.warning(f"Label column {label_col} missing in {name} after preprocessing!")
                 
            df = df[feature_cols + [label_col]]
            
            y_test = df[label_col].values
            X_test = df.drop(columns=[label_col]).values
            
            # Ensure X_test is float32 and handle NaNs
            X_test = X_test.astype(np.float32)
            if np.isnan(X_test).any() or np.isinf(X_test).any():
                LOGGER.warning(f"X_test for {name} contains NaNs or Infs. Replacing with 0.")
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
            
            # Evaluate
            y_true, y_pred = evaluate_model(model, X_test, y_test, seq_len=seq_len)
            
            acc = accuracy_score(y_true, y_pred)
            LOGGER.info(f"{name} Accuracy: {acc:.4f}")
            
            # Save results
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, le.classes_, out_path=plots_dir / f"{run_id}_{name}_cm.png")
            
            report = classification_report(y_true, y_pred, labels=range(len(le.classes_)), target_names=le.classes_, output_dict=True)
            save_json(report, logs_dir / f"{run_id}_{name}_report.json")
            
            overall_results[name] = acc
        except Exception as e:
            LOGGER.error(f"Error evaluating {name}: {e}", exc_info=True)

        
    LOGGER.info("Overall Results:")
    LOGGER.info(overall_results)
    save_json(overall_results, logs_dir / f"{run_id}_overall_results.json")

if __name__ == "__main__":
    main()
