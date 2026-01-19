@echo off
echo ==========================================
echo   Customer Churn ML - Quick Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo [4/4] Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\sample" mkdir data\sample
if not exist "models\artifacts" mkdir models\artifacts
if not exist "mlruns" mkdir mlruns

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo To run the pipeline:
echo   1. run_pipeline.bat
echo.
echo To start services:
echo   - API: run_api.bat
echo   - Dashboard: run_dashboard.bat
echo   - MLflow: run_mlflow.bat
echo.
pause
