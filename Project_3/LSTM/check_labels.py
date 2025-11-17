import pandas as pd

try:
    df = pd.read_csv(r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced_encoded_normalised.csv")
    print("Columns:", df.columns.tolist())
    print("Unique labels:", df.iloc[:, -1].unique())
    print("Number of unique labels:", len(df.iloc[:, -1].unique()))
except Exception as e:
    print(e)
