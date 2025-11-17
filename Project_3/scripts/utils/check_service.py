import pandas as pd

df = pd.read_csv(r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1.csv")
print("Unique services:", df['service'].unique())
