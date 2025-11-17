import os
import glob

print("Checking logs/training_output.txt size:")
try:
    size = os.path.getsize("logs/training_output.txt")
    print(f"Size: {size} bytes")
except Exception as e:
    print(f"Error: {e}")

print("\nChecking results directory:")
results_dirs = glob.glob("results/results_*")
results_dirs.sort(key=os.path.getmtime, reverse=True)
for d in results_dirs[:3]:
    print(f"{d} - {os.path.getmtime(d)}")
    # List files in the directory
    try:
        files = os.listdir(d)
        print(f"  Files: {files}")
    except:
        pass
