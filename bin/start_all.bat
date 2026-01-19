@echo off
REM ============================================
REM Start All Services
REM ============================================

echo.
echo ============================================
echo    STARTING ALL SERVICES
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

echo [1/3] Starting MLflow UI on port 5000...
start "MLflow UI" cmd /k "venv\Scripts\activate && mlflow ui --port 5000"

timeout /t 3 /nobreak > nul

echo [2/3] Starting FastAPI on port 8000...
start "FastAPI" cmd /k "venv\Scripts\activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [3/3] Starting Streamlit Dashboard on port 8501...
start "Streamlit Dashboard" cmd /k "venv\Scripts\activate && streamlit run dashboard/app.py --server.port 8501"

echo.
echo ============================================
echo    ALL SERVICES STARTED!
echo ============================================
echo.
echo Access the services at:
echo   - MLflow UI:     http://localhost:5000
echo   - FastAPI:       http://localhost:8000
echo   - API Docs:      http://localhost:8000/docs
echo   - Dashboard:     http://localhost:8501
echo.
echo Close this window when done.
echo.
pause
