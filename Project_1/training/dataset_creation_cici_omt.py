import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

# Configuration
SOURCE_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Datasets\CIC_IoT\wataiData\csv\CICIoT2023"
TARGET_DIR = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\cici_omt"
os.makedirs(TARGET_DIR, exist_ok=True)

# Target Counts (from Edge-IIoT)
TARGETS = {
    "cic_iomt_training_50_50.csv": 168000,
    "cic_iomt_testing_50_50.csv": 168000,
    "cic_iomt_training_60_40.csv": 140000,
    "cic_iomt_testing_60_40.csv": 140000,
    "cic_iomt_training_70_30.csv": 120000,
    "cic_iomt_testing_70_30.csv": 120000,
    "cic_iomt_training_80_20.csv": 105000,
    "cic_iomt_testing_80_20.csv": 105000
}

def create_dataset(filename, target_count):
    output_path = os.path.join(TARGET_DIR, filename)
    if os.path.exists(output_path):
        print(f"Skipping {filename}, already exists.")
        return

    print(f"Creating {filename} with {target_count} rows...")
    
    # Get list of all part files
    all_files = glob.glob(os.path.join(SOURCE_DIR, "part-*.csv"))
    np.random.shuffle(all_files) # Shuffle to get random parts
    
    collected_rows = []
    current_count = 0
    
    # Iterate through files until we have enough data
    for f in all_files:
        if current_count >= target_count:
            break
            
        try:
            # Read a chunk
            chunk = pd.read_csv(f)
            
            # Take a random sample from this chunk
            # We don't want to take ALL rows from one file to avoid bias
            # Let's take up to 10% of the target count from each file
            sample_size = min(len(chunk), int(target_count * 0.1))
            
            if sample_size > 0:
                sampled_chunk = chunk.sample(n=sample_size, random_state=42)
                collected_rows.append(sampled_chunk)
                current_count += len(sampled_chunk)
                print(f"  Collected {len(sampled_chunk)} rows from {os.path.basename(f)}. Total: {current_count}/{target_count}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            
    if collected_rows:
        full_df = pd.concat(collected_rows, ignore_index=True)
        # Shuffle again
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Trim to exact count
        final_df = full_df.head(target_count)
        
        final_df.to_csv(output_path, index=False)
        print(f"Saved {filename} with {len(final_df)} rows.")
    else:
        print(f"Failed to collect data for {filename}")

def main():
    # Process sequentially to avoid disk thrashing
    for filename, count in TARGETS.items():
        create_dataset(filename, count)

if __name__ == "__main__":
    main()
