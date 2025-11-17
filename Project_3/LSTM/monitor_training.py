import os
import glob
import time

# Find the latest log file
log_files = glob.glob("logs/training_cuda_*.log")
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Latest log file: {latest_log}")
    print(f"Size: {os.path.getsize(latest_log)} bytes")
    print(f"Last modified: {time.ctime(os.path.getmtime(latest_log))}")
    
    # Read last 20 lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print(f"\nTotal lines: {len(lines)}")
            print("\nLast 20 lines:")
            print("".join(lines[-20:]))
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("No log files found")
