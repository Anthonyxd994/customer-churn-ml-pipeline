@echo off
echo ==========================================
echo   Running ML Pipeline
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/3] Generating customer data...
python -m src.data_generation.generate_data
if errorlevel 1 (
    echo [ERROR] Data generation failed
    pause
    exit /b 1
)

echo.
echo [2/3] Running feature engineering...
python -m src.feature_engineering.create_features
if errorlevel 1 (
    echo [ERROR] Feature engineering failed
    pause
    exit /b 1
)

echo.
echo [3/3] Training models...
python -m src.training.train_models
if errorlevel 1 (
    echo [ERROR] Model training failed
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   Pipeline Complete!
echo ==========================================
echo.
echo Generated files:
echo   - data/raw/customers.csv
echo   - data/processed/processed_features.csv
echo   - models/artifacts/best_model.joblib
echo.
echo Next steps:
echo   - run_api.bat (Start prediction API)
echo   - run_dashboard.bat (Start dashboard)
echo   - run_mlflow.bat (View experiments)
echo.
pause
