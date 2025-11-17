import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Paths
dataset_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
raw_ref_path = os.path.join(dataset_dir, "combined_dataset_short_balanced.csv")
target_path = os.path.join(dataset_dir, "conn1.csv")
output_path = os.path.join(dataset_dir, "conn1_encoded_normalised.csv")

# Scales (Min, Max) from inference
SCALES = {
    'service': (0.0, 10.0),
    'duration': (0.0, 7443.286078),
    'source_bytes': (0.0, 36176.0),
    'destination_bytes': (0.0, 1049392.0),
    'local_source': (0.0, 1.0),
    'local_destination': (0.0, 1.0),
    'missed_bytes': (0.0, 6387.0),
    'history': (14.0, 735.0),
    'source_pkts': (0.0, 1187.0),
    'source_ip_bytes': (0.0, 82148.0),
    'destination_pkts': (0.0, 1279.0),
    'destination_ip_bytes': (0.0, 1079124.0)
}

# Label Mapping
# Detailed Label -> Encoded Value
LABEL_MAPPING = {
    'Benign': 1,
    'C&C': 2,
    'PartOfAHorizontalPortScan': 4,
    'DDoS': 0,
    'Okiru': 3, # Assuming Okiru is Exploitation/Infection
    'Attack': 0,
    'Recon': 4,
    'C&C-HeartBeat': 2,
    'C&C-FileDownload': 2,
    'C&C-Torii': 2,
    'C&C-HeartBeat-FileDownload': 2,
    'FileDownload': 2,
    'C&C-Mirai': 2,
    'Okiru-Attack': 3
}

class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        # Check if values are in classes
        y = np.array(y)
        unseen = ~np.isin(y, self.classes_)
        if np.any(unseen):
            # Map unseen to the first class (usually 0 or most common)
            y[unseen] = self.classes_[0] 
        return super().transform(y)

def load_and_preprocess_conn1(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Replace '-' with 0
    df.replace('-', 0, inplace=True)
    df.replace('(empty)', 0, inplace=True)
    
    # Rename columns to match our internal mapping keys
    rename_map = {
        'orig_bytes': 'source_bytes',
        'resp_bytes': 'destination_bytes',
        'local_orig': 'local_source',
        'local_resp': 'local_destination',
        'orig_pkts': 'source_pkts',
        'orig_ip_bytes': 'source_ip_bytes',
        'resp_pkts': 'destination_pkts',
        'resp_ip_bytes': 'destination_ip_bytes',
        'proto': 'protocol'
    }
    df.rename(columns=rename_map, inplace=True)
    
    return df

def fit_encoders(ref_path):
    print(f"Fitting encoders using {ref_path}...")
    # Read available categorical columns
    df_ref = pd.read_csv(ref_path, usecols=['history', 'protocol'])
    
    encoders = {}
    
    # History and Protocol
    for col in ['history', 'protocol']:
        le = SafeLabelEncoder()
        df_ref[col] = df_ref[col].astype(str)
        le.fit(df_ref[col])
        encoders[col] = le
        
    # Conn State (Manually defined from one-hot columns)
    conn_state_classes = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'RSTRH', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH', 'SHR']
    le_cs = SafeLabelEncoder()
    le_cs.fit(conn_state_classes)
    encoders['conn_state'] = le_cs
        
    return encoders

def process_data():
    # 1. Fit Encoders
    encoders = fit_encoders(raw_ref_path)
    
    # 2. Load Target
    df_target = load_and_preprocess_conn1(target_path)
    
    # 3. Create Output DataFrame
    df_out = pd.DataFrame()
    
    # List of features in order 0-13
    features = [
        'service', 'duration', 'source_bytes', 'destination_bytes', 
        'local_source', 'local_destination', 'missed_bytes', 'history', 
        'source_pkts', 'source_ip_bytes', 'destination_pkts', 'destination_ip_bytes', 
        'conn_state', 'protocol'
    ]
    
    for i, feat in enumerate(features):
        print(f"Processing feature {i}: {feat}")
        col_data = df_target[feat]
        
        # Handle Encoding
        if feat in encoders:
            col_data = col_data.astype(str)
            encoded = encoders[feat].transform(col_data)
            vals = encoded.astype(float)
        elif feat == 'service':
            vals = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(float)
        else:
            vals = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(float)
            
        # Handle Normalization
        feat_scale = None
        if feat in SCALES:
            feat_scale = SCALES[feat]
        elif feat == 'conn_state':
            feat_scale = (0.0, 12.0)
        elif feat == 'protocol':
            feat_scale = (0.0, 1.0) 
            
        if feat_scale:
            min_val, max_val = feat_scale
            if max_val > min_val:
                vals = (vals - min_val) / (max_val - min_val)
            else:
                vals = 0.0
                
        # Clip to [0, 1]
        vals = vals.clip(0.0, 1.0)
        
        df_out[str(i)] = vals

    # 4. Handle Label
    print("Processing label...")
    
    # Identify the merged column
    merged_col = [c for c in df_target.columns if 'label' in c and 'tunnel' in c]
    
    if merged_col:
        print(f"Found merged label column: {merged_col[0]}")
        # Split by whitespace
        split_cols = df_target[merged_col[0]].astype(str).str.split(expand=True)
        
        # Check if we got 3 columns (tunnel, label, detailed-label)
        if split_cols.shape[1] >= 3:
            label_col = split_cols[1]
            detailed_col = split_cols[2]
            
            # Create a Series for final labels
            final_labels = pd.Series(index=df_target.index, dtype=int)
            
            # Map Benign
            final_labels[label_col == 'Benign'] = LABEL_MAPPING['Benign']
            
            # Map Malicious based on detailed label
            malicious_mask = label_col == 'Malicious'
            # Map detailed labels, default to Exploitation (3) if not found in mapping
            mapped_detailed = detailed_col.map(LABEL_MAPPING).fillna(3).astype(int)
            final_labels[malicious_mask] = mapped_detailed[malicious_mask]
            
            df_out['label_stage_encoded'] = final_labels.fillna(0).astype(int) # Default to 0 if something goes wrong
            
        elif split_cols.shape[1] >= 2:
             # Fallback if only 2 columns (maybe just Benign?)
             label_col = split_cols[1]
             df_out['label_stage_encoded'] = label_col.map(LABEL_MAPPING).fillna(1).astype(int)
        else:
            print("Warning: Could not split merged column correctly. Assigning 0.")
            df_out['label_stage_encoded'] = 0
            
    elif 'label' in df_target.columns:
        df_out['label_stage_encoded'] = df_target['label'].map(LABEL_MAPPING).fillna(1).astype(int)
    elif 'detailed-label' in df_target.columns:
         df_out['label_stage_encoded'] = df_target['detailed-label'].map(LABEL_MAPPING).fillna(1).astype(int)
    else:
        print("Warning: No label column found. Assigning 0.")
        df_out['label_stage_encoded'] = 0

    # 5. Save
    print(f"Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    process_data()
