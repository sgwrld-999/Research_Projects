import pandas as pd
import numpy as np

# Paths
raw_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced.csv"
encoded_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv"

# Load samples
print("Loading datasets...")
df_raw = pd.read_csv(raw_path, nrows=10000)
df_enc = pd.read_csv(encoded_path, nrows=10000)

# Raw columns of interest (excluding one-hot for now)
# We suspect the 14 columns are:
# protocol, service, duration, source_bytes, destination_bytes, 
# local_source, local_destination, missed_bytes, history, 
# source_pkts, source_ip_bytes, destination_pkts, destination_ip_bytes
# + maybe conn_state (reconstructed from one-hot)

# Reconstruct conn_state from one-hot
conn_state_cols = [c for c in df_raw.columns if 'conn_state_' in c]
def get_conn_state(row):
    for c in conn_state_cols:
        if row[c] == 1:
            return c.replace('conn_state_', '')
    return 'OTH'

df_raw['conn_state'] = df_raw.apply(get_conn_state, axis=1)

potential_features = ['protocol', 'service', 'duration', 'source_bytes', 'destination_bytes', 
                      'local_source', 'local_destination', 'missed_bytes', 'history', 
                      'source_pkts', 'source_ip_bytes', 'destination_pkts', 'destination_ip_bytes',
                      'conn_state']

print("\n--- Feature Mapping Analysis ---")
mapping = {}
scales = {}

with open('mapping_results.txt', 'w') as f:
    for i in range(14):
        enc_col = str(i)
        enc_vals = df_enc[enc_col]
        
        best_feat = None
        best_corr = 0
        
        for feat in potential_features:
            raw_vals = df_raw[feat]
            
            # Handle categorical
            if raw_vals.dtype == object:
                # Try label encoding for correlation check
                codes = raw_vals.astype('category').cat.codes
                corr = np.corrcoef(codes, enc_vals)[0, 1]
            else:
                corr = np.corrcoef(raw_vals, enc_vals)[0, 1]
                
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_feat = feat
                
        f.write(f"Encoded {i} -> {best_feat} (Corr: {best_corr:.4f})\n")
        mapping[i] = best_feat
        
        # Calculate Scale (Min/Max) from RAW data for this feature
        if best_feat:
            if df_raw[best_feat].dtype == object:
                unique_raw = df_raw[best_feat].unique()
                unique_enc = df_enc[enc_col].unique()
                f.write(f"  Categorical: {len(unique_raw)} raw values, {len(unique_enc)} encoded values\n")
                scales[best_feat] = {'type': 'categorical', 'values': unique_raw.tolist()}
            else:
                raw_min = df_raw[best_feat].min()
                raw_max = df_raw[best_feat].max()
                f.write(f"  Numerical: Min={raw_min}, Max={raw_max}\n")
                scales[best_feat] = {'type': 'numerical', 'min': float(raw_min), 'max': float(raw_max)}

    f.write("\n--- Scales ---\n")
    f.write(str(scales))
