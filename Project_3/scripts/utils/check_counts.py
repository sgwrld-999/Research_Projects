import pandas as pd

# Path
raw_path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced.csv"

# Read only label column
df = pd.read_csv(raw_path, usecols=['label_stage'])

# Count values
print("Label Counts:")
print(df['label_stage'].value_counts())
