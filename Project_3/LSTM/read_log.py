import os

log_file = r"c:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_3\LSTM\logs\training_cuda_20251222_202301.log"

try:
    with open(log_file, 'r') as f:
        lines = f.readlines()
        print("".join(lines[-20:]))
except Exception as e:
    print(e)
