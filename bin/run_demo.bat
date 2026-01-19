@echo off
REM ============================================
REM Run Demo - Complete ML Pipeline Demonstration
REM ============================================

echo.
echo ============================================
echo    CUSTOMER CHURN PREDICTION - DEMO
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Run the demo script
python scripts\demo.py

echo.
pause
