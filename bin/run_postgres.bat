@echo off
REM ============================================
REM Start PostgreSQL Database (Docker)
REM ============================================

echo.
echo ============================================
echo    POSTGRESQL DATABASE (DOCKER)
echo ============================================
echo.

REM Check if Docker is running
docker info > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not running!
    echo         Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if container exists
docker ps -a --filter "name=churn-postgres" --format "{{.Names}}" | findstr /i "churn-postgres" > nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Container 'churn-postgres' exists
    
    REM Check if running
    docker ps --filter "name=churn-postgres" --format "{{.Names}}" | findstr /i "churn-postgres" > nul
    if %ERRORLEVEL% equ 0 (
        echo [OK] PostgreSQL is already running!
    ) else (
        echo [INFO] Starting existing container...
        docker start churn-postgres
        echo [OK] PostgreSQL started!
    )
) else (
    echo [INFO] Creating new PostgreSQL container...
    docker run --name churn-postgres -e POSTGRES_PASSWORD=postgres123 -e POSTGRES_DB=churn_db -p 5433:5432 -d postgres:15
    echo [OK] PostgreSQL created and started!
)

echo.
echo ============================================
echo    PostgreSQL is ready!
echo ============================================
echo    Host: localhost
echo    Port: 5433
echo    Database: churn_db
echo    User: postgres
echo    Password: postgres123
echo ============================================
echo.
echo To connect: psql -h localhost -p 5433 -U postgres -d churn_db
echo.
pause
