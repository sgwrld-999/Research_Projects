import pandas as pd

df = pd.read_csv(r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1.csv")
if 'label' in df.columns:
    print("Unique labels:", df['label'].unique())
if 'detailed-label' in df.columns:
    print("Unique detailed-labels:", df['detailed-label'].unique())
