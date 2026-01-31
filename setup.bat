@echo off
echo ========================================
echo MRAF-Net Setup Script for Windows
echo ========================================
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing PyTorch with CUDA support...
echo Detecting CUDA version...

REM Try to install PyTorch with CUDA 11.8 (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Step 5: Installing other requirements...
pip install -r requirements.txt

echo Step 6: Creating necessary directories...
if not exist "data\brats2020" mkdir data\brats2020
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs

echo Step 7: Verifying installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download BraTS 2020 dataset
echo 2. Extract to data\brats2020\MICCAI_BraTS2020_TrainingData\
echo 3. Run: python scripts\prepare_data.py
echo 4. Run: python scripts\train.py --config config\config.yaml
echo.
pause
