import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Paths
dataset_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
input_path = os.path.join(dataset_dir, "conn1_balanced_4class.csv")
output_path = os.path.join(dataset_dir, "conn1_balanced_4class_encoded_normalised.csv")

# Scales (Min, Max) - Reused for consistency
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
LABEL_MAPPING = {
    'Benign': 1,
    'C&C': 2,
    'PartOfAHorizontalPortScan': 4,
    'DDoS': 0,
    'Okiru': 3,
    'Attack': 0,
    'Recon': 4,
    'Reconnaissance': 4, # Added alias
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
        y = np.array(y)
        unseen = ~np.isin(y, self.classes_)
        if np.any(unseen):
            y[unseen] = self.classes_[0] 
        return super().transform(y)

def process_data():
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean and Rename
    df.replace('-', 0, inplace=True)
    df.replace('(empty)', 0, inplace=True)
    
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
    
    # Fit Encoders on current data
    encoders = {}
    for col in ['history', 'protocol', 'conn_state']:
        le = SafeLabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le
        
    # Create Output DataFrame
    df_out = pd.DataFrame()
    
    features = [
        'service', 'duration', 'source_bytes', 'destination_bytes', 
        'local_source', 'local_destination', 'missed_bytes', 'history', 
        'source_pkts', 'source_ip_bytes', 'destination_pkts', 'destination_ip_bytes', 
        'conn_state', 'protocol'
    ]
    
    for i, feat in enumerate(features):
        print(f"Processing feature {i}: {feat}")
        if feat not in df.columns:
             # Handle missing columns (e.g. service might be missing or named differently)
             if feat == 'service':
                 vals = 0.0
             else:
                 print(f"Warning: Feature {feat} not found in input. Filling with 0.")
                 vals = 0.0
        else:
            col_data = df[feat]
            
            if feat in encoders:
                col_data = col_data.astype(str)
                encoded = encoders[feat].transform(col_data)
                vals = encoded.astype(float)
            else:
                vals = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(float)
            
        # Normalization
        feat_scale = SCALES.get(feat)
        if feat == 'conn_state': feat_scale = (0.0, 12.0)
        elif feat == 'protocol': feat_scale = (0.0, 1.0)
        
        if feat_scale:
            min_val, max_val = feat_scale
            if max_val > min_val:
                vals = (vals - min_val) / (max_val - min_val)
            else:
                vals = 0.0
        
        if isinstance(vals, (pd.Series, np.ndarray)):
             vals = np.clip(vals, 0.0, 1.0)
             
        df_out[str(i)] = vals

    # Label Handling
    print("Processing label...")
    if 'detailed-label' in df.columns:
        df_out['label_stage_encoded'] = df['detailed-label'].map(LABEL_MAPPING).fillna(0).astype(int)
    elif 'label_stage' in df.columns:
         df_out['label_stage_encoded'] = df['label_stage'].map(LABEL_MAPPING).fillna(0).astype(int)
    else:
        print("Warning: No label column found. Assigning 0.")
        df_out['label_stage_encoded'] = 0
        
    print(f"Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    process_data()
