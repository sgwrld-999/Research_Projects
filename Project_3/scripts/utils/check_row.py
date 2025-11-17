import pandas as pd

df = pd.read_csv(r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1.csv", nrows=1)
print("Header columns:", len(df.columns))
print("First row values:", len(df.iloc[0]))
print("Last column value:", df.iloc[0][-1])
