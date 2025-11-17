import pandas as pd
import numpy as np

path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1_encoded_normalised.csv"

print(f"Verifying {path}...")
df = pd.read_csv(path)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check columns
expected_cols = [str(i) for i in range(14)] + ['label_stage_encoded']
if df.columns.tolist() != expected_cols:
    print(f"FAIL: Columns mismatch. Expected {expected_cols}")
else:
    print("PASS: Columns match.")

# Check value ranges
print("\nValue Ranges:")
for col in df.columns:
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"  {col}: {min_val} - {max_val}")
    
    if col != 'label_stage_encoded':
        if min_val < 0 or max_val > 1.000001: # Allow slight float error
            print(f"    WARNING: {col} out of range [0, 1]")

print("\nLabel Distribution:")
print(df['label_stage_encoded'].value_counts())
