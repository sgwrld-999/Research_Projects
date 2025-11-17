import os
import time
from pathlib import Path

path = "models/saved_Models/best_model.pth"
if os.path.exists(path):
    mtime = os.path.getmtime(path)
    print(f"{path} last modified: {time.ctime(mtime)}")
else:
    print(f"{path} does not exist")

print("\nSearching for save logic in train_pytorch_cuda.py:")
with open("scirpts/train_pytorch_cuda.py", "r") as f:
    for i, line in enumerate(f):
        if "save" in line.lower() or "best_model" in line.lower():
            print(f"{i+1}: {line.strip()}")
