import pandas as pd

df = pd.read_csv(r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\combined_dataset_short_balanced.csv", usecols=['service'])
print("Unique services in reference:", df['service'].unique())
