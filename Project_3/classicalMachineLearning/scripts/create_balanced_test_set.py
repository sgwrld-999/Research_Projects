import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATASET_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
OUTPUT_FILE = os.path.join(DATASET_DIR, "balanced_test_5k.csv")

# Source files
SOURCE_FILES = [
    "Recon_10k.csv",
    "attack_10k.csv",
    "benign_10k.csv",
    "c_and_c_communication_10k.csv",
    "exploitation_10k.csv"
]

SAMPLES_PER_FILE = 1000

def create_balanced_dataset():
    combined_df = pd.DataFrame()
    
    for file_name in SOURCE_FILES:
        file_path = os.path.join(DATASET_DIR, file_name)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            continue
            
        try:
            logging.info(f"Reading {file_name}...")
            df = pd.read_csv(file_path)
            
            if len(df) < SAMPLES_PER_FILE:
                logging.warning(f"{file_name} has only {len(df)} samples, taking all.")
                sampled_df = df
            else:
                sampled_df = df.sample(n=SAMPLES_PER_FILE, random_state=42)
                
            logging.info(f"Extracted {len(sampled_df)} samples from {file_name}")
            combined_df = pd.concat([combined_df, sampled_df], ignore_index=True)
            
        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    if not combined_df.empty:
        # Shuffle the final dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        combined_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Successfully created {OUTPUT_FILE} with {len(combined_df)} samples.")
    else:
        logging.error("No data collected. Dataset creation failed.")

if __name__ == "__main__":
    create_balanced_dataset()
