@echo off
REM ============================================
REM MRAF-Net Brain Tumor Segmentation GUI
REM Author: Anne Nidhusha Nithiyalan (w1985740)
REM ============================================

echo.
echo ============================================
echo  MRAF-Net Brain Tumor Segmentation GUI
echo  University of Westminster / IIT
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install requirements
echo Checking dependencies...
pip install -q -r requirements.txt

REM Run the GUI
echo.
echo Starting MRAF-Net GUI...
echo.
echo Opening in browser: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
