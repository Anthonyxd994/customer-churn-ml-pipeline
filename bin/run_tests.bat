@echo off
echo ==========================================
echo   Running Tests
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Running pytest with coverage...
echo.

pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

echo.
echo ==========================================
echo   Tests Complete!
echo ==========================================
pause
