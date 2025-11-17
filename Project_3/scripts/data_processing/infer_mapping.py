import pandas as pd
import numpy as np

# Paths
raw_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced.csv"
encoded_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv"

# Load samples
print("Loading datasets...")
df_raw = pd.read_csv(raw_path, nrows=5000)
df_enc = pd.read_csv(encoded_path, nrows=5000)

print(f"Raw shape: {df_raw.shape}")
print(f"Encoded shape: {df_enc.shape}")

# Raw columns (excluding one-hot states for now to see what matches)
raw_cols = ['protocol', 'service', 'duration', 'source_bytes', 'destination_bytes', 
            'local_source', 'local_destination', 'missed_bytes', 'history', 
            'source_pkts', 'source_ip_bytes', 'destination_pkts', 'destination_ip_bytes']

# Encoded columns (0-13)
enc_cols = [str(i) for i in range(14)]

print("\n--- Analyzing Feature Mapping ---")

for enc_col in enc_cols:
    enc_vals = df_enc[enc_col]
    best_match = None
    best_corr = 0
    match_type = "None"
    
    # Check correlation with numerical raw columns
    for raw_col in raw_cols:
        if df_raw[raw_col].dtype in [np.float64, np.int64]:
            # Calculate correlation
            try:
                corr = np.corrcoef(df_raw[raw_col].fillna(0), enc_vals)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_match = raw_col
                    match_type = "Numerical"
            except:
                pass
                
    # Check if it matches a categorical column (Label Encoded)
    # For categorical, we check if unique values map 1-to-1
    for raw_col in raw_cols:
        if df_raw[raw_col].dtype == object:
            # Create a simple integer encoding for comparison
            raw_codes = df_raw[raw_col].astype('category').cat.codes
            try:
                corr = np.corrcoef(raw_codes, enc_vals)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_match = raw_col
                    match_type = "Categorical"
            except:
                pass

    print(f"Encoded {enc_col} matches {best_match} ({match_type}) with correlation {best_corr:.4f}")
    
    # If numerical match, estimate Min/Max
    if match_type == "Numerical" and abs(best_corr) > 0.9:
        raw_min = df_raw[best_match].min()
        raw_max = df_raw[best_match].max()
        print(f"  Estimated Scale: Min={raw_min}, Max={raw_max}")

print("\n--- Checking One-Hot Columns ---")
# Check if any encoded columns correspond to conn_state one-hot
conn_state_cols = [c for c in df_raw.columns if 'conn_state' in c]
for enc_col in enc_cols:
    enc_vals = df_enc[enc_col]
    for state_col in conn_state_cols:
        corr = np.corrcoef(df_raw[state_col], enc_vals)[0, 1]
        if corr > 0.9:
            print(f"Encoded {enc_col} matches {state_col} with correlation {corr:.4f}")

