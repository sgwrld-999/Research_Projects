@echo off
REM CUDA-Accelerated PyTorch LSTM Training Script
REM This script activates the pyTorch_LSTM conda environment and runs the training

echo ===================================================================
echo CUDA-Accelerated PyTorch LSTM Training Pipeline
echo ===================================================================
echo.

echo [1/3] Activating pytorch_gpu conda environment...
call conda activate pytorch_gpu
if errorlevel 1 (
    echo ERROR: Failed to activate pytorch_gpu environment
    echo Please ensure the environment exists: conda env list
    pause
    exit /b 1
)
echo Environment activated successfully!
echo.

echo [2/3] Verifying CUDA availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo ERROR: PyTorch or CUDA verification failed
    pause
    exit /b 1
)
echo.

echo [3/3] Starting LSTM training with CUDA acceleration...
echo.
python scirpts\train_pytorch_cuda.py
if errorlevel 1 (
    echo.
    echo ERROR: Training script failed
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo Training completed successfully!
echo ===================================================================
echo Check the results folder for outputs and plots
echo Check the logs folder for training logs
pause
