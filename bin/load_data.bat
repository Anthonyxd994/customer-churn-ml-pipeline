@echo off
REM ============================================
REM Load CSV Data to PostgreSQL
REM ============================================

echo.
echo ============================================
echo    LOAD DATA TO POSTGRESQL
echo ============================================
echo.

call venv\Scripts\activate
python scripts\load_to_postgres.py

echo.
pause
