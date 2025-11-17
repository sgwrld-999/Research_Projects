import pandas as pd
from pathlib import Path

data_path = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
df = pd.read_csv(data_path, nrows=1)
feature_names = df.columns[:-1].tolist()
print(f"Feature names ({len(feature_names)}): {feature_names}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
