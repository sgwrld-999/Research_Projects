@echo off
echo Hello > batch_output.txt
python --version >> batch_output.txt 2>&1
conda --version >> batch_output.txt 2>&1
