"""
Quick verification script to check environment and imports.
Run this before running the full pipeline.
"""
import sys

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check core dependencies
dependencies = [
    'numpy',
    'pandas',
    'sklearn',
    'torch',
    'matplotlib',
    'seaborn'
]

missing = []
for dep in dependencies:
    try:
        mod = __import__(dep)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {dep}: {version}")
    except ImportError:
        print(f"✗ {dep}: NOT INSTALLED")
        missing.append(dep)

# Check PyTorch CUDA
if 'torch' not in missing:
    import torch
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  (Running on CPU)")

# Check src modules
print("\n" + "=" * 60)
print("PROJECT MODULE IMPORTS")
print("=" * 60)

modules = [
    'src.utils',
    'src.binary_class_data_processing',
    'src.multi_class_data_processing',
    'src.phase_1_model',
    'src.Phase_2_model',
    'src.metrics',
    'src.plots'
]

for mod_name in modules:
    try:
        __import__(mod_name)
        print(f"✓ {mod_name}")
    except Exception as e:
        print(f"✗ {mod_name}: {e}")

# Summary
print("\n" + "=" * 60)
if missing:
    print(f"⚠ MISSING DEPENDENCIES: {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")
else:
    print("✅ All dependencies installed!")

print("\nNext steps:")
print("  1. Ensure datasets are in data/ directory")
print("  2. Run: python run_example.py")
print("=" * 60)
