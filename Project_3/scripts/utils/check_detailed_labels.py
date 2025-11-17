import pandas as pd

path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1.csv"
df = pd.read_csv(path)

# Identify the merged column
merged_col = [c for c in df.columns if 'label' in c and 'tunnel' in c]

if merged_col:
    # Split by whitespace
    split_cols = df[merged_col[0]].astype(str).str.split(expand=True)
    
    if split_cols.shape[1] >= 3:
        label_col = split_cols[1]
        detailed_col = split_cols[2]
        
        # Filter for Malicious
        malicious_indices = label_col == 'Malicious'
        if malicious_indices.any():
            print("Detailed labels for Malicious:", detailed_col[malicious_indices].unique())
        else:
            print("No Malicious labels found.")
    else:
        print("Split produced fewer than 3 columns.")
