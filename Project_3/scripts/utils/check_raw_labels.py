import pandas as pd

path = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\dataset\conn1.csv"
print(f"Loading {path}...")
df = pd.read_csv(path)

# Identify the merged column
merged_col = [c for c in df.columns if 'label' in c and 'tunnel' in c]

if merged_col:
    print(f"Found merged label column: {merged_col[0]}")
    # Split by whitespace
    split_cols = df[merged_col[0]].astype(str).str.split(expand=True)
    
    if split_cols.shape[1] >= 2:
        extracted_label = split_cols[1]
        print("Unique extracted labels:", extracted_label.unique())
    else:
        print("Could not split merged column.")
else:
    print("No merged label column found. Checking other columns.")
    if 'label' in df.columns:
        print("Unique labels:", df['label'].unique())
    if 'detailed-label' in df.columns:
        print("Unique detailed-labels:", df['detailed-label'].unique())
