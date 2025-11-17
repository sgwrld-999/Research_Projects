import os
import glob

# Find the latest training log
log_files = glob.glob("logs/training_cuda_*.log")
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Latest log: {latest_log}")
    print(f"Size: {os.path.getsize(latest_log)} bytes")
    
    # Read and print last 30 lines
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(f"\nTotal lines: {len(lines)}")
            print("\n=== Last 30 lines ===")
            for line in lines[-30:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No log files found")
