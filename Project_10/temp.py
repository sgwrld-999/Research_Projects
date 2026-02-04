import pandas as pd 
import numpy as np

df = pd.read_csv("/Users/siddhantgond/Desktop/Research_Internship_Under_Dr_Rakesh_Matam/Project_10/data/raw/ciciot_training_100_0.csv")

print(df["label"].value_counts())