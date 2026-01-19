# 📚 Project Setup Guide

This guide will help you set up and run the Customer Churn Prediction System on your local machine.

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (Optional, for containerized deployment)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Open Command Prompt** in the project directory

2. **Run the setup script:**

   ```batch
   setup.bat
   ```

3. **Run the ML pipeline:**

   ```batch
   run_pipeline.bat
   ```

4. **Start the services:**
   ```batch
   run_api.bat           # Start FastAPI (in one terminal)
   run_dashboard.bat     # Start Streamlit (in another terminal)
   run_mlflow.bat        # Start MLflow UI (in another terminal)
   ```

### Option 2: Manual Setup

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data:**

   ```bash
   python -m src.data_generation.generate_data
   ```

4. **Run feature engineering:**

   ```bash
   python -m src.feature_engineering.create_features
   ```

5. **Train models:**

   ```bash
   python -m src.training.train_models
   ```

6. **Start API:**

   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

7. **Start Dashboard (new terminal):**
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Accessing the Services

Once running, access the services at:

| Service       | URL                        | Description           |
| ------------- | -------------------------- | --------------------- |
| **FastAPI**   | http://localhost:8000      | Prediction API        |
| **API Docs**  | http://localhost:8000/docs | Swagger documentation |
| **Dashboard** | http://localhost:8501      | Streamlit UI          |
| **MLflow**    | http://localhost:5000      | Experiment tracking   |

## 🐳 Docker Deployment

To run with Docker:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📁 Project Structure

```
churn_e2e_ml/
├── api/                 # FastAPI prediction service
├── dashboard/           # Streamlit dashboard
├── data/
│   ├── raw/            # Generated customer data
│   ├── processed/      # Feature-engineered data
│   └── sample/         # Sample datasets
├── docker/             # Dockerfiles
├── models/
│   └── artifacts/      # Trained models
├── mlruns/             # MLflow experiments
├── src/
│   ├── data_generation/    # Data generation code
│   ├── feature_engineering/# Feature creation
│   ├── training/           # Model training
│   └── utils/              # Helper functions
├── tests/              # Unit tests
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
└── docker-compose.yml  # Docker orchestration
```

## 🔮 Making Predictions

### Using the API

```python
import requests

customer = {
    "customer_id": "CUST_001",
    "tenure_months": 18,
    "monthly_spend": 250.50,
    "total_orders": 24,
    "avg_order_value": 85.00,
    "days_since_last_order": 15,
    "login_frequency": 12,
    "products_viewed": 25,
    "cart_abandonment_rate": 0.15,
    "support_tickets": 2,
    "discount_usage_rate": 0.3,
    "satisfaction_score": 7,
    "complaint_count": 1
}

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
```

### Using the Dashboard

1. Navigate to http://localhost:8501
2. Click on "🎯 Predictions" in the sidebar
3. Fill in the customer information
4. Click "🔮 Predict Churn"

## ❓ Troubleshooting

### Model not loading

Make sure you've run the training pipeline first:

```bash
run_pipeline.bat
```

### API returns 503

The model file doesn't exist. Run:

```bash
python -m src.training.train_models
```

### Port already in use

Kill the process using the port or use a different port:

```bash
uvicorn api.main:app --port 8001
```

## 📧 Support

If you encounter any issues, please open an issue on GitHub.
