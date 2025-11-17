# Activate environment and run evaluation
$logFile = "eval_output.txt"
Start-Transcript -Path $logFile -Append

try {
    Write-Host "Activating environment..."
    conda activate pytorch_gpu
    if ($?) {
        Write-Host "Environment activated. Running script..."
        python evaluate_saved_model.py
    } else {
        Write-Host "Failed to activate environment."
    }
} catch {
    Write-Host "Error: $_"
}

Stop-Transcript
