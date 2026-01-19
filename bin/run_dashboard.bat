@echo off
echo ==========================================
echo   Starting Streamlit Dashboard
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting Dashboard on http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run dashboard/app.py --server.port 8501
