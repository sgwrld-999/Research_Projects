import pandas as pd
import glob
import os

path = r'c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\data\edge_iiot\multiclass_*.csv'
files = glob.glob(path)
print(f"Found {len(files)} files.")

for f in files:
    try:
        # Reading only one column for speed
        df = pd.read_csv(f, usecols=[0])
        print(f"{os.path.basename(f)}: {len(df)}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
