@echo off
echo ==========================================
echo   Starting MLflow UI
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting MLflow UI on http://localhost:5000
echo.
echo Press Ctrl+C to stop MLflow
echo.

mlflow ui --port 5000
