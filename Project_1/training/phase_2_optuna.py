import optuna
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from Phase_2_model import train_phase2
from multi_class_data_processing import preprocess_multiclass_data
from utils import get_logger, ensure_dir

LOGGER = get_logger('phase2_optuna')

def objective(trial):
    # Hyperparameters to tune
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 96])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    
    # Fixed parameters for tuning to save time
    epochs = 10 
    batch_size = 256
    seq_len = 10
    
    # Load Data (Load once globally if possible, but here for safety)
    # Assuming data is already available or loading it here
    # For efficiency, we should load data outside objective if possible
    # But for simplicity in this script, we'll rely on the main block to pass data or load it
    
    # Using global variables for data to avoid reloading every trial
    global X_train, y_train, num_classes, class_weights
    
    run_dir = Path(__file__).resolve().parent.parent / 'models' / f'optuna_trial_{trial.number}'
    ensure_dir(run_dir)
    
    try:
        model, history = train_phase2(
            X_train, y_train, 
            seq_len=seq_len, 
            num_classes=num_classes, 
            run_dir=run_dir, 
            epochs=epochs, 
            batch_size=batch_size, 
            class_weights=class_weights,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Return validation accuracy of the last epoch (or best)
        val_acc = max(history['val_acc'])
        return val_acc
        
    except Exception as e:
        LOGGER.error(f"Trial failed: {e}")
        return 0.0

if __name__ == "__main__":
    # Load Data Once
    DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
    train_path = DATA_DIR / 'edge_iiot' / 'multiclass_train_80_20.csv' # Using 80_20 for tuning
    
    LOGGER.info("Loading training data for Optuna...")
    df_train = pd.read_csv(train_path)
    
    # Preprocess
    df_train, pipeline = preprocess_multiclass_data(df_train, label_col='Attack_type')
    le = pipeline['le']
    num_classes = len(le.classes_)
    
    label_col = 'Attack_type'
    y_train = df_train[label_col].values
    X_train = df_train.drop(columns=[label_col]).values
    
    # Ensure float32 and handle NaNs
    X_train = X_train.astype(np.float32)
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        LOGGER.warning("X_train contains NaNs or Infs. Replacing with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute Class Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    # Run Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    LOGGER.info(f"Best trial: {study.best_trial.params}")
    
    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv('optuna_results.csv', index=False)
    LOGGER.info("Saved Optuna results to optuna_results.csv")
