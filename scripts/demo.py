"""
Quick Demo Script - Run the complete ML pipeline and show results.

This script demonstrates the full workflow:
1. Generate synthetic customer data
2. Run feature engineering
3. Train models with MLflow tracking
4. Start the API and make predictions
"""

import subprocess
import sys
import time
import os
import requests
import json

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_step(name, command, cwd=None):
    """Run a step of the demo."""
    print(f"\n[RUNNING] {name}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[SUCCESS] {name} completed!")
            return True
        else:
            print(f"[ERROR] {name} failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_api():
    """Test the prediction API."""
    print("\n[TESTING] API endpoint...")
    
    test_customer = {
        "customer_id": "DEMO_001",
        "tenure_months": 6,
        "monthly_spend": 150.0,
        "total_orders": 5,
        "avg_order_value": 75.0,
        "days_since_last_order": 45,
        "login_frequency": 3,
        "products_viewed": 10,
        "cart_abandonment_rate": 0.4,
        "support_tickets": 3,
        "discount_usage_rate": 0.5,
        "satisfaction_score": 5,
        "complaint_count": 2
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_customer,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[PREDICTION RESULT]")
            print(f"  Customer ID: {result['customer_id']}")
            print(f"  Churn Probability: {result['churn_probability']:.1%}")
            print(f"  Risk Segment: {result['risk_segment'].upper()}")
            print(f"  Prediction: {'WILL CHURN' if result['churn_prediction'] else 'WILL STAY'}")
            return True
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("[WARNING] API not running. Start with: run_api.bat")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def main():
    """Run the complete demo."""
    print_header("CUSTOMER CHURN PREDICTION - DEMO")
    
    print("\nThis demo will:")
    print("  1. Generate synthetic customer data")
    print("  2. Run feature engineering pipeline")
    print("  3. Train ML models")
    print("  4. Test the prediction API (if running)")
    
    input("\nPress Enter to start...")
    
    # Step 1: Data Generation
    print_header("STEP 1: DATA GENERATION")
    run_step(
        "Data Generation",
        "python -m src.data_generation.generate_data"
    )
    
    # Step 2: Feature Engineering
    print_header("STEP 2: FEATURE ENGINEERING")
    run_step(
        "Feature Engineering",
        "python -m src.feature_engineering.create_features"
    )
    
    # Step 3: Model Training
    print_header("STEP 3: MODEL TRAINING")
    run_step(
        "Model Training",
        "python -m src.training.train_models"
    )
    
    # Step 4: Test API
    print_header("STEP 4: API TESTING")
    test_api()
    
    # Summary
    print_header("DEMO COMPLETE!")
    
    print("\n[NEXT STEPS]")
    print("  1. Start the API:        run_api.bat")
    print("  2. Start the Dashboard:  run_dashboard.bat")
    print("  3. View MLflow:          run_mlflow.bat")
    print("  4. Run Tests:            run_tests.bat")
    
    print("\n[URLS]")
    print("  - API:        http://localhost:8000")
    print("  - API Docs:   http://localhost:8000/docs")
    print("  - Dashboard:  http://localhost:8501")
    print("  - MLflow:     http://localhost:5000")
    
    print("\n")

if __name__ == "__main__":
    main()
