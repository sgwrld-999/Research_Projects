import pandas as pd
import numpy as np

# Paths
raw_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced.csv"
encoded_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv"

print("Reading raw labels...")
# Read only the label column
df_raw = pd.read_csv(raw_path, usecols=['label_stage'])
unique_raw = df_raw['label_stage'].unique()
print(f"Unique raw labels ({len(unique_raw)}): {sorted(unique_raw)}")

print("Reading encoded labels...")
df_encoded = pd.read_csv(encoded_path, usecols=['label_stage_encoded'])
unique_encoded = df_encoded['label_stage_encoded'].unique()
print(f"Unique encoded labels ({len(unique_encoded)}): {sorted(unique_encoded)}")

# We can't assume row alignment for the whole file without checking, 
# but we can check the first N rows where we find different labels.
# Let's find the index of the first occurrence of each label in the raw dataset
# and check the corresponding value in the encoded dataset.

label_indices = {}
for label in unique_raw:
    # Find first index
    idx = df_raw[df_raw['label_stage'] == label].index[0]
    label_indices[label] = idx

print("\nMapping based on first occurrence:")
mapping = {}
for label, idx in label_indices.items():
    encoded_val = df_encoded.iloc[idx]['label_stage_encoded']
    mapping[label] = encoded_val
    print(f"{label} -> {encoded_val}")

print("\nTarget Labels for Extraction:")
targets = ['Recon', 'exploitation']
for t in targets:
    if t in mapping:
        print(f"{t}: {mapping[t]}")
    else:
        print(f"{t}: NOT FOUND")
