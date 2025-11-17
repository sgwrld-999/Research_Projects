import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import logging
from datetime import datetime
import traceback

# Setup Logging
log_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"training_balanced_4class_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Paths
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DATA_PATH = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1_balanced_4class_encoded_normalised.csv"
MODEL_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\models_balanced_4class"
RESULTS_BASE_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\classicalMachineLearning\results_balanced_4class"
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"run_{TIMESTAMP}")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    logging.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    target_col = 'label_stage_encoded'
    if target_col not in df.columns:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logging.info(f"Data Shape: {df.shape}")
    logging.info(f"Classes: {y.unique()}")
    
    return X, y

def split_data(X, y):
    # Split 80/20 for internal validation
    logging.info("Splitting data 80/20 for internal validation.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logging.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_test, y_prob, model_name, save_dir, n_classes):
    present_classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        if i in present_classes:
            try:
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
            except ValueError as e:
                logging.warning(f"Could not calculate ROC for class {i}: {e}")
        else:
             logging.warning(f"Class {i} not present in test set. Skipping ROC for this class.")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'))
    plt.close()

def train_and_evaluate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    n_classes = 5
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, num_class=5, objective='multi:softprob'),
        "LightGBM": lgb.LGBMClassifier(random_state=42, num_class=5, objective='multiclass')
    }
    
    results = []
    
    for name, model in models.items():
        try:
            logging.info(f"Training {name}...")
            start_time = datetime.now()
            
            model.fit(X_train, y_train)
            
            train_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"Training {name} completed in {train_time:.2f}s")
            
            # Save Model
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Ensure y_prob has columns for all 5 classes
            if y_prob.shape[1] < n_classes:
                logging.info(f"Padding probabilities for {name}. Got {y_prob.shape[1]} columns, expected {n_classes}.")
                model_classes = model.classes_
                y_prob_full = np.zeros((y_prob.shape[0], n_classes))
                for i, cls in enumerate(model_classes):
                    if cls < n_classes:
                        y_prob_full[:, int(cls)] = y_prob[:, i]
                y_prob = y_prob_full
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logging.info(f"{name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            
            results.append({
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "Training_Time": train_time
            })
            
            # Plots
            plot_confusion_matrix(y_test, y_pred, name, RESULTS_DIR)
            plot_roc_curve(y_test, y_prob, name, RESULTS_DIR, n_classes)
            
            # Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)
            pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, f"{name}_classification_report.csv"))
            
        except Exception as e:
            logging.error(f"Failed to train/evaluate {name}: {e}", exc_info=True)

    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "all_models_metrics.csv"), index=False)
    logging.info("Training pipeline completed.")

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        logging.error("An error occurred during training:", exc_info=True)
        with open("error_dump_balanced_4class.txt", "w") as f:
            traceback.print_exc(file=f)
