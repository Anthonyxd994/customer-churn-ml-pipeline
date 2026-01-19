@echo off
echo ==========================================
echo   Starting FastAPI Server
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting API on http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
