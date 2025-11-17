import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

# Setup Logging
log_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Paths
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\models"
RESULTS_BASE_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\results"
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"test_run_{TIMESTAMP}")
DATASET_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test Datasets
TEST_FILES = {
    "Exploitation": os.path.join(DATASET_DIR, "exploitation_10k.csv"),
    "Recon": os.path.join(DATASET_DIR, "Recon_10k.csv"),
    "Mix": [os.path.join(DATASET_DIR, "exploitation_10k.csv"), os.path.join(DATASET_DIR, "Recon_10k.csv")],
    "Balanced_5k": os.path.join(DATASET_DIR, "balanced_test_5k.csv")
}

def load_test_data():
    datasets = {}
    
    # Load individual files
    for name, path in TEST_FILES.items():
        if isinstance(path, list):
            continue # Skip lists, handled separately if needed or just ignore for now as Mix is hardcoded
            
        if os.path.exists(path):
            logging.info(f"Loading {name} from {path}")
            df = pd.read_csv(path)
            datasets[name] = df
        else:
            logging.warning(f"File not found: {path}")
            
    # Create Mix
    if "Exploitation" in datasets and "Recon" in datasets:
        logging.info("Creating Mix dataset...")
        df_mix = pd.concat([datasets["Exploitation"], datasets["Recon"]], ignore_index=True)
        # Shuffle
        df_mix = df_mix.sample(frac=1, random_state=42).reset_index(drop=True)
        datasets["Mix"] = df_mix
        
    return datasets

def prepare_data(df):
    # Assuming 'label_stage_encoded' is the target
    target_col = 'label_stage_encoded'
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{model_name}_{dataset_name}_confusion_matrix.png'))
    plt.close()

def run_testing():
    datasets = load_test_data()
    
    # Load Models
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            logging.info(f"Loading model: {model_name}")
            models[model_name] = joblib.load(os.path.join(MODEL_DIR, file))
            
    if not models:
        logging.error("No models found!")
        return

    all_results = []

    for dataset_name, df in datasets.items():
        logging.info(f"Testing on {dataset_name} ({len(df)} samples)...")
        X, y = prepare_data(df)
        
        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating {model_name} on {dataset_name}...")
                
                y_pred = model.predict(X)
                
                # Metrics
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                
                logging.info(f"{model_name} on {dataset_name}: Acc={acc:.4f}, F1={f1:.4f}")
                
                all_results.append({
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1_Score": f1
                })
                
                # Plots
                plot_confusion_matrix(y, y_pred, model_name, dataset_name, RESULTS_DIR)
                
                # Classification Report
                report = classification_report(y, y_pred, output_dict=True)
                pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}_classification_report.csv"))
                
            except Exception as e:
                logging.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}", exc_info=True)

    # Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "testing_pipeline_results.csv"), index=False)
    logging.info("Testing pipeline completed.")

if __name__ == "__main__":
    run_testing()
