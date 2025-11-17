# CUDA-Accelerated PyTorch LSTM Training Script
# This script activates the pyTorch_LSTM conda environment and runs the training

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "CUDA-Accelerated PyTorch LSTM Training Pipeline" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Activate conda environment
Write-Host "[1/3] Activating pytorch_gpu conda environment..." -ForegroundColor Yellow
try {
    conda activate pytorch_gpu
    Write-Host "Environment activated successfully!" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: Failed to activate pytorch_gpu environment" -ForegroundColor Red
    Write-Host "Please ensure the environment exists: conda env list" -ForegroundColor Red
    # Read-Host "Press Enter to exit"
    exit 1
}

# Step 2: Verify CUDA
Write-Host "[2/3] Verifying CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else `"N/A`"}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch or CUDA verification failed" -ForegroundColor Red
    # Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Step 3: Run training
Write-Host "[3/3] Starting LSTM training with CUDA acceleration..." -ForegroundColor Yellow
Write-Host ""
python evaluate_saved_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training script failed" -ForegroundColor Red
    # Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "Training completed successfully!" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "Check the results folder for outputs and plots" -ForegroundColor Cyan
Write-Host "Check the logs folder for training logs" -ForegroundColor Cyan
# Read-Host "Press Enter to exit"
