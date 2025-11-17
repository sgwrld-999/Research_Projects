import pandas as pd
import os

# Paths
dataset_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"

# Configuration
EXPECTED_COUNT = 10000
CLASSES = {
    "attack": 0,
    "benign": 1,
    "c_and_c_communication": 2,
    "exploitation": 3,
    "Recon": 4
}

def verify_file(filename, expected_label):
    path = os.path.join(dataset_dir, filename)
    print(f"Verifying {filename}...")
    
    if not os.path.exists(path):
        print(f"  FAIL: File not found: {path}")
        return False
    
    try:
        df = pd.read_csv(path)
        count = len(df)
        unique_labels = df['label_stage_encoded'].unique()
        
        if count != EXPECTED_COUNT:
            print(f"  FAIL: Expected {EXPECTED_COUNT} rows, found {count}")
            return False
            
        if len(unique_labels) != 1:
            print(f"  FAIL: Found multiple labels: {unique_labels}")
            return False
            
        if unique_labels[0] != expected_label:
            print(f"  FAIL: Expected label {expected_label}, found {unique_labels[0]}")
            return False
            
        print(f"  PASS: Verification successful.")
        return True
    except Exception as e:
        print(f"  FAIL: Error reading file: {e}")
        return False

# Run verification
all_passed = True
for class_name, label_code in CLASSES.items():
    filename = f"{class_name}_10k.csv"
    if not verify_file(filename, label_code):
        all_passed = False

if all_passed:
    print("\nALL CHECKS PASSED.")
else:
    print("\nVERIFICATION FAILED.")
