import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Paths
dataset_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
raw_ref_path = os.path.join(dataset_dir, "combined_dataset_short_balanced.csv")
target_path = os.path.join(dataset_dir, "conn1.csv")
output_path = os.path.join(dataset_dir, "conn1_encoded_normalised.csv")

# Feature Mapping (Target -> Source)
# 0: service -> service
# 1: duration -> duration
# 2: source_bytes -> orig_bytes
# 3: destination_bytes -> resp_bytes
# 4: local_source -> local_orig
# 5: local_destination -> local_resp
# 6: missed_bytes -> missed_bytes
# 7: history -> history
# 8: source_pkts -> orig_pkts
# 9: source_ip_bytes -> orig_ip_bytes
# 10: destination_pkts -> resp_pkts
# 11: destination_ip_bytes -> resp_ip_bytes
# 12: conn_state -> conn_state
# 13: protocol -> proto

# Scales (Min, Max) from inference
SCALES = {
    'service': (0.0, 10.0),
    'duration': (0.0, 7443.286078),
    'source_bytes': (0.0, 36176.0),
    'destination_bytes': (0.0, 1049392.0),
    'local_source': (0.0, 1.0),
    'local_destination': (0.0, 1.0),
    'missed_bytes': (0.0, 6387.0),
    'history': (14.0, 735.0), # Note: This is for label encoded values
    'source_pkts': (0.0, 1187.0),
    'source_ip_bytes': (0.0, 82148.0),
    'destination_pkts': (0.0, 1279.0),
    'destination_ip_bytes': (0.0, 1079124.0)
}

# Label Mapping
LABEL_MAPPING = {
    'attack': 0,
    'benign': 1, 'Benign': 1,
    'c_and_c_communication': 2, 'C&C': 2, 'C&C-HeartBeat': 2,
    'exploitation': 3, 'Infection': 3, 'Discovery': 3,
    'Recon': 4, 'Port Scanning': 4, 'propogation': 4
}

def load_and_preprocess_conn1(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Replace '-' with 0 or appropriate value
    df.replace('-', 0, inplace=True)
    df.replace('(empty)', 0, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 
                    'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

def fit_encoders(ref_path):
    print(f"Fitting encoders using {ref_path}...")
    # Read only categorical columns to save memory
    df_ref = pd.read_csv(ref_path, usecols=['history', 'conn_state', 'protocol', 'service'])
    
    encoders = {}
    for col in ['history', 'conn_state', 'protocol', 'service']:
        le = LabelEncoder()
        # Ensure all values are strings
        df_ref[col] = df_ref[col].astype(str)
        le.fit(df_ref[col])
        encoders[col] = le
        
    return encoders

def process_data():
    import os # Ensure os is imported
    
    # 1. Fit Encoders
    encoders = fit_encoders(raw_ref_path)
    
    # 2. Load Target
    df_target = load_and_preprocess_conn1(target_path)
    
    # 3. Create Output DataFrame
    df_out = pd.DataFrame()
    
    # --- Numerical Features (Normalize) ---
    # Map: Target Name -> (Source Name, Scale Key)
    num_map = {
        '0': ('service', 'service'), # Service is numerical in encoded file (0-10)?? Wait, inference said service is 0-10. 
                                     # But in raw it's categorical (http, dns...). 
                                     # Let's check if service was label encoded then normalized.
                                     # Inference said: Encoded 0 -> service (Corr: 1.0000), Numerical: Min=0, Max=10
                                     # This implies service was Label Encoded then MinMax scaled? Or just Label Encoded?
                                     # If Min=0, Max=10, it looks like Label Encoding.
                                     # Let's assume Label Encoding for service.
        '1': ('duration', 'duration'),
        '2': ('orig_bytes', 'source_bytes'),
        '3': ('resp_bytes', 'destination_bytes'),
        '4': ('local_orig', 'local_source'),
        '5': ('local_resp', 'local_destination'),
        '6': ('missed_bytes', 'missed_bytes'),
        '8': ('orig_pkts', 'source_pkts'),
        '9': ('orig_ip_bytes', 'source_ip_bytes'),
        '10': ('resp_pkts', 'destination_pkts'),
        '11': ('resp_ip_bytes', 'destination_ip_bytes')
    }
    
    # Handle Service separately (Categorical -> Label Encoded -> Normalized?)
    # Actually, let's look at the scales again. Service max is 10. 
    # If we label encode, we get integers. Then we might normalize.
    # But wait, if max is 10, maybe it's just label encoded?
    # Let's just apply Label Encoding for service and see if it falls in 0-10.
    
    # --- Categorical Features (Label Encode) ---
    # 7: history
    # 12: conn_state
    # 13: protocol
    
    # Apply Encoders
    # Service
    print("Encoding service...")
    df_target['service'] = df_target['service'].astype(str)
    # Handle unseen labels
    known_services = set(encoders['service'].classes_)
    df_target['service'] = df_target['service'].apply(lambda x: x if x in known_services else encoders['service'].classes_[0]) # Default to first class if unknown
    df_out['0'] = encoders['service'].transform(df_target['service'])
    # Normalize service? If the reference is 0-10, it might be normalized or just small integer range.
    # The reference file is "normalised", so likely everything is 0-1. 
    # BUT inference said Min=0, Max=10. That contradicts "normalised" usually meaning 0-1.
    # Unless "normalised" just means standard format?
    # Wait, let's check the inference output again for other columns.
    # duration: Min=1e-06, Max=7443.
    # This means the "encoded_normalised" file is NOT 0-1 normalized!
    # It seems "normalised" might be a misnomer or refers to something else, OR my inference script read the RAW values?
    # Ah! My inference script calculated Min/Max from the RAW dataset (`df_raw`), NOT the encoded one!
    # "Calculate Scale (Min/Max) from RAW data for this feature" -> Yes.
    # So I need to verify if the ENCODED file is actually 0-1.
    # Let's check the encoded file values quickly before proceeding.
    
    return df_out

if __name__ == "__main__":
    # We need to verify normalization first.
    # Let's just print the head of encoded file again to check range.
    df_enc = pd.read_csv(os.path.join(dataset_dir, "combined_dataset_short_balanced_encoded_normalised.csv"), nrows=5)
    print("Encoded file head:")
    print(df_enc.head())
