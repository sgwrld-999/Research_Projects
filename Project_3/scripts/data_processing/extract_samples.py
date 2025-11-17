import pandas as pd
import os

# Paths
dataset_dir = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset"
input_path = os.path.join(dataset_dir, "combined_dataset_short_balanced_encoded_normalised.csv")

# Configuration
SAMPLE_SIZE = 10000
CLASSES = {
    "attack": 0,
    "benign": 1,
    "c_and_c_communication": 2,
    "exploitation": 3,
    "Recon": 4
}

print(f"Loading dataset from {input_path}...")
# Load dataset
df = pd.read_csv(input_path)
print("Dataset loaded. Shape:", df.shape)

for class_name, label_code in CLASSES.items():
    output_filename = f"{class_name}_10k.csv"
    output_path = os.path.join(dataset_dir, output_filename)
    
    print(f"Extracting {SAMPLE_SIZE} samples for {class_name} (Label {label_code})...")
    
    class_df = df[df['label_stage_encoded'] == label_code]
    count = len(class_df)
    
    if count >= SAMPLE_SIZE:
        sample_df = class_df.sample(n=SAMPLE_SIZE, random_state=42)
        sample_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_filename}")
    else:
        print(f"  Error: Not enough samples. Found {count}, required {SAMPLE_SIZE}")

print("Extraction complete.")
