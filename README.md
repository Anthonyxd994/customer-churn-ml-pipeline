# 🎯 RetentionIQ - Customer Intelligence Platform

<div align="center">

<!-- Logo/Banner -->
<img src="https://img.shields.io/badge/🎯_RetentionIQ-Customer_Intelligence_Platform-6366F1?style=for-the-badge&labelColor=1E1E2E" alt="RetentionIQ Banner"/>

<br/>

<!-- Typing Animation Badge -->

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6366F1&center=true&vCenter=true&random=false&width=600&lines=Predict+Customer+Churn+Before+It+Happens;67%25+Model+Accuracy+%7C+%3C100ms+Predictions;Production-Ready+ML+Pipeline+with+MLOps)](https://git.io/typing-svg)

<br/>

<!-- Main Badges -->

![CI/CD](https://img.shields.io/github/actions/workflow/status/mohamednoorulnaseem/customer-churn-ml-pipeline/ci.yml?branch=main&style=for-the-badge&label=CI%2FCD&logo=github-actions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<!-- Status Badges -->

![Tests](https://img.shields.io/badge/Tests-21%20Passed-success?style=for-the-badge&logo=pytest&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-85%25-green?style=for-the-badge&logo=codecov&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)
![Version](https://img.shields.io/badge/Version-2.0-6366F1?style=for-the-badge&logo=semver&logoColor=white)

<br/>

<!-- Quick Action Buttons -->

[<img src="https://img.shields.io/badge/🚀_Quick_Start-Get_Running_in_5_Min-6366F1?style=for-the-badge" alt="Quick Start"/>](#-quick-start)
[<img src="https://img.shields.io/badge/📖_Documentation-Setup_Guide-4CAF50?style=for-the-badge" alt="Documentation"/>](docs/SETUP_GUIDE.md)
[<img src="https://img.shields.io/badge/🔌_API_Reference-8_Endpoints-FF6B6B?style=for-the-badge" alt="API Reference"/>](#-api-endpoints)
[<img src="https://img.shields.io/badge/📊_Dashboard-Live_Demo-00D4FF?style=for-the-badge" alt="Dashboard"/>](#-dashboard)

</div>

<br/>

<!-- Demo Preview -->
<div align="center">
<h3>📸 Dashboard Preview</h3>

> _Premium light-theme dashboard with real-time analytics, export functionality, and Userpilot-inspired design_

|   Dashboard Overview   |    Risk Analysis    | Real-time Predictions |
| :--------------------: | :-----------------: | :-------------------: |
|     📊 KPI Metrics     |  🎯 Risk Segments   |  ⚡ <100ms Response   |
|    📈 Churn Trends     | 🔴 High-Risk Alerts |   🤖 ML Predictions   |
| 📉 Distribution Charts | 📊 Customer Health  | 💡 SHAP Explanations  |

</div>

---

## 🌟 Highlights

<table>
<tr>
<td width="50%">

### 🤖 Machine Learning

- **3 Models**: XGBoost, Random Forest, Logistic Regression
- **68 Features**: RFM, engagement, risk indicators
- **SMOTE**: Handles class imbalance
- **SHAP**: Model explainability

</td>
<td width="50%">

### ⚡ Production Ready

- **FastAPI**: 8 REST endpoints
- **Real-time**: <100ms predictions
- **Docker**: One-command deployment
- **PostgreSQL**: Enterprise database

</td>
</tr>
<tr>
<td width="50%">

### 📊 Monitoring & Tracking

- **MLflow**: Experiment tracking
- **Model Registry**: Version control
- **6-Page Dashboard**: Business intelligence
- **Health Checks**: System monitoring

</td>
<td width="50%">

### ✅ Quality Assured

- **21 Unit Tests**: 100% passing
- **Pydantic**: Input validation
- **Logging**: Comprehensive logs
- **Documentation**: API docs included

</td>
</tr>
</table>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [API Endpoints](#-api-endpoints)
- [Dashboard](#-dashboard)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Problem Statement

### The Business Challenge

> **"Acquiring a new customer costs 5-7x more than retaining an existing one"** — Harvard Business Review

E-commerce companies lose **$1.6 trillion annually** due to customer churn. The challenge is identifying at-risk customers **before** they leave, enabling proactive retention strategies.

### Our Solution

This system predicts customer churn probability using machine learning, providing:

| Capability                | Business Value                                 |
| ------------------------- | ---------------------------------------------- |
| **Early Warning**         | Identify churners 30-90 days before they leave |
| **Risk Segmentation**     | Prioritize high-value, high-risk customers     |
| **Actionable Insights**   | Understand WHY customers churn (SHAP)          |
| **Real-time Predictions** | Instant risk assessment via API                |
| **Retention ROI**         | Target campaigns, reduce CAC                   |

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CUSTOMER CHURN PREDICTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │    DATA     │────▶│  FEATURE    │────▶│   MODEL     │────▶│   MODEL     │  │
│   │  INGESTION  │     │ ENGINEERING │     │  TRAINING   │     │  REGISTRY   │  │
│   │             │     │             │     │             │     │             │  │
│   │ • CSV/SQL   │     │ • 68 Features│    │ • XGBoost   │     │ • MLflow    │  │
│   │ • Postgres  │     │ • RFM       │     │ • RF, LR    │     │ • Versioning│  │
│   │ • Streaming │     │ • SMOTE     │     │ • Cross-val │     │ • Artifacts │  │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘  │
│          │                   │                   │                   │          │
│          └───────────────────┴───────────────────┴───────────────────┘          │
│                                      │                                           │
│                    ┌─────────────────┴─────────────────┐                        │
│                    │         MLFLOW TRACKING           │                        │
│                    │   Experiments • Metrics • Models  │                        │
│                    └─────────────────┬─────────────────┘                        │
│                                      │                                           │
│   ┌──────────────────────────────────┴──────────────────────────────────────┐   │
│   │                        SERVING LAYER                                     │   │
│   │                                                                          │   │
│   │  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │   │
│   │  │   FASTAPI      │   │   STREAMLIT    │   │     SHAP       │          │   │
│   │  │   REST API     │   │   DASHBOARD    │   │  EXPLANATIONS  │          │   │
│   │  │                │   │                │   │                │          │   │
│   │  │ • /predict     │   │ • Overview     │   │ • Feature      │          │   │
│   │  │ • /batch       │   │ • Predictions  │   │   Importance   │          │   │
│   │  │ • /health      │   │ • Analysis     │   │ • Local        │          │   │
│   │  │ • /explain     │   │ • Performance  │   │   Explanations │          │   │
│   │  └────────────────┘   └────────────────┘   └────────────────┘          │   │
│   │                                                                          │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)

```bash
# Clone and setup
git clone https://github.com/yourusername/churn_e2e_ml.git
cd churn_e2e_ml

# Run setup script (creates venv, installs deps)
setup.bat

# Run the complete pipeline
run_pipeline.bat

# Start all services
start_all.bat
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data
python -m src.data_generation.generate_data

# Feature engineering
python -m src.feature_engineering.create_features

# Train models
python -m src.training.train_models

# Start services
python -m api.main                    # API on http://localhost:8000
streamlit run dashboard/app.py        # Dashboard on http://localhost:8501
mlflow ui --port 5000                 # MLflow on http://localhost:5000
```

### Option 3: Docker Compose

```bash
docker-compose up -d

# Access:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

---

## ✨ Features

### 🔮 Prediction Engine

| Feature               | Description                                   |
| --------------------- | --------------------------------------------- |
| **Single Prediction** | Real-time churn probability for one customer  |
| **Batch Prediction**  | Process thousands of customers simultaneously |
| **Risk Segmentation** | Automatic Low/Medium/High risk classification |
| **Confidence Scores** | Prediction reliability metrics                |
| **SHAP Explanations** | Top factors driving each prediction           |

### 📊 Feature Engineering (68 Features)

<details>
<summary><b>Click to expand feature categories</b></summary>

| Category            | Features                                               | Purpose                     |
| ------------------- | ------------------------------------------------------ | --------------------------- |
| **RFM Metrics**     | Recency, Frequency, Monetary, RFM Score                | Customer value segmentation |
| **Engagement**      | Login frequency, Session duration, Page views          | Platform usage patterns     |
| **Transactional**   | Avg order value, Total orders, Spend trends            | Purchase behavior           |
| **Risk Indicators** | Days since last order, Complaint rate, Support tickets | Early warning signals       |
| **Temporal**        | Customer age, Tenure, Seasonality                      | Time-based patterns         |
| **Behavioral**      | Cart abandonment, Discount usage, Browse-to-buy ratio  | User behavior               |
| **Value Metrics**   | CLV proxy, Profit margin, Retention value              | Business impact             |

</details>

### 🤖 Model Training

- **Algorithms**: XGBoost, Random Forest, Logistic Regression
- **Cross-Validation**: 5-fold stratified CV
- **Hyperparameter Tuning**: Grid search with MLflow logging
- **Class Imbalance**: SMOTE oversampling
- **Model Selection**: Best model by ROC-AUC

---

## 📈 Model Performance

### Training Results

| Model                      | Accuracy | Precision | Recall | F1     | AUC-ROC    |
| -------------------------- | -------- | --------- | ------ | ------ | ---------- |
| **Logistic Regression** ⭐ | 71.45%   | 49.10%    | 24.20% | 32.43% | **66.76%** |
| Random Forest              | 68.45%   | 42.89%    | 34.63% | 38.32% | 65.82%     |
| XGBoost                    | 63.25%   | 39.34%    | 55.12% | 45.92% | 65.53%     |

### Key Insights

- **Best for Production**: Logistic Regression (highest AUC-ROC, interpretable)
- **Best for Recall**: XGBoost (catches more churners, higher false positive rate)
- **Churn Rate**: 28.3% (class imbalance addressed with SMOTE)

### Confusion Matrix (Logistic Regression)

```
                Predicted
              No Churn  Churn
Actual  No    1286      143
Churn   Yes    428      143
```

---

## 🌐 API Endpoints

### Base URL: `http://localhost:8000`

| Method | Endpoint         | Description                   |
| ------ | ---------------- | ----------------------------- |
| `GET`  | `/`              | API information               |
| `GET`  | `/health`        | Health check & model status   |
| `POST` | `/predict`       | Single customer prediction    |
| `POST` | `/predict/batch` | Batch predictions             |
| `GET`  | `/features`      | List of required features     |
| `GET`  | `/model/info`    | Model metadata                |
| `GET`  | `/docs`          | Interactive API documentation |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "tenure_months": 12,
    "monthly_spend": 150.0,
    "total_orders": 15,
    "days_since_last_order": 45,
    "satisfaction_score": 6
  }'
```

### Example Response

```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.67,
  "churn_prediction": true,
  "risk_segment": "high",
  "confidence": 0.67,
  "top_churn_factors": [
    {
      "feature": "days_since_last_order",
      "importance": 0.35,
      "direction": "increases"
    },
    {
      "feature": "satisfaction_score",
      "importance": 0.22,
      "direction": "increases"
    }
  ],
  "recommendations": [
    "Personal outreach from customer success team",
    "Offer loyalty discount or exclusive deal"
  ]
}
```

---

## 📊 Dashboard

### 6 Interactive Pages

| Page                     | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| **📊 Overview**          | KPIs, churn distribution, risk segmentation          |
| **🎯 Predictions**       | Real-time prediction form with recommendations       |
| **🔍 Customer Analysis** | RFM scatter plots, segment analysis, high-risk table |
| **📈 Model Performance** | Metrics, confusion matrix, model comparison          |
| **🔬 Feature Analysis**  | Feature importance, category breakdown               |
| **📋 Data Explorer**     | Filter, search, and download customer data           |

### Dashboard Features

- 📊 Real-time metrics and KPIs
- 📈 Interactive Plotly visualizations
- 🔮 Live predictions with explanations
- 📥 CSV export functionality
- 🔗 Direct links to API docs and MLflow

---

## 📁 Project Structure

```
churn_e2e_ml/
│
├── 📂 api/                         # FastAPI Application
│   ├── main.py                     # API endpoints
│   └── schemas.py                  # Pydantic models
│
├── 📂 bin/                         # Executable Scripts
│   ├── setup.bat                   # Environment setup
│   ├── run_pipeline.bat            # ML pipeline
│   ├── run_api.bat                 # Start API server
│   ├── run_dashboard.bat           # Start Streamlit
│   ├── run_mlflow.bat              # Start MLflow UI
│   ├── run_tests.bat               # Run test suite
│   └── start_all.bat               # Launch all services
│
├── 📂 dashboard/                   # Streamlit Dashboard
│   └── app.py                      # Premium RetentionIQ UI
│
├── 📂 docs/                        # Documentation
│   ├── SETUP_GUIDE.md              # Installation guide
│   ├── PROJECT_SHOWCASE.md         # Demo & screenshots
│   └── CONTRIBUTING.md             # Contribution guide
│
├── 📂 src/                         # Source Code
│   ├── 📂 data_generation/         # Synthetic data generator
│   ├── 📂 feature_engineering/     # Feature pipeline
│   ├── 📂 training/                # Model training
│   ├── 📂 data/                    # Database utilities
│   └── 📂 utils/                   # Helper functions
│
├── 📂 models/                      # Model Artifacts
│   └── 📂 artifacts/               # Saved models & scalers
│
├── 📂 data/                        # Data Storage
│   ├── 📂 raw/                     # Raw customer data
│   ├── 📂 processed/               # Processed features
│   └── 📂 sample/                  # Sample datasets
│
├── 📂 scripts/                     # Python Utilities
│   ├── demo.py                     # Demo script
│   └── load_to_postgres.py         # Database loader
│
├── 📂 tests/                       # Unit Tests (21 tests)
│
├── 📂 docker/                      # Docker Configurations
│
├── 📂 .github/                     # CI/CD Workflows
│   └── workflows/ci.yml            # GitHub Actions
│
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Dependencies
├── config.yaml                     # Configuration
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🛠️ Technology Stack

<table>
<tr>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" height="48" alt="Python" />
<br>Python
</td>
<td align="center" width="100">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="48" height="48" alt="scikit-learn" />
<br>scikit-learn
</td>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="48" height="48" alt="FastAPI" />
<br>FastAPI
</td>
<td align="center" width="100">
<img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" width="48" height="48" alt="Streamlit" />
<br>Streamlit
</td>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="48" height="48" alt="Docker" />
<br>Docker
</td>
</tr>
<tr>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="48" height="48" alt="PostgreSQL" />
<br>PostgreSQL
</td>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="48" height="48" alt="Pandas" />
<br>Pandas
</td>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="48" height="48" alt="NumPy" />
<br>NumPy
</td>
<td align="center" width="100">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytest/pytest-original.svg" width="48" height="48" alt="Pytest" />
<br>Pytest
</td>
<td align="center" width="100">
<img src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="48" height="48" alt="MLflow" />
<br>MLflow
</td>
</tr>
</table>

---

## 🔮 Future Enhancements

- [ ] **Model Monitoring**: Drift detection with Evidently AI
- [ ] **A/B Testing**: Compare model versions in production
- [ ] **Feature Store**: Feast for feature management
- [ ] **CI/CD Pipeline**: GitHub Actions for automation
- [ ] **Kubernetes**: K8s deployment manifests
- [ ] **Real-time Streaming**: Kafka integration
- [ ] **Email Alerts**: Automated alerts for high-risk customers
- [ ] **Model Retraining**: Scheduled retraining pipeline

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

<div align="center">

<img src="https://img.shields.io/badge/Built_by-Mohamed_Noorul_Naseem-6366F1?style=for-the-badge" alt="Author Badge"/>

### **Mohamed Noorul Naseem**

_Data Scientist & ML Engineer_

Passionate about building production-ready ML systems that drive real business value.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamednoorulnaseem)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohamednoorulnaseem)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:noorulnaseem11@gmail.com)

</div>

---

## 💖 Support

<div align="center">

If you found this project helpful, please consider:

[![Star](https://img.shields.io/badge/⭐_Star_this_repo-6366F1?style=for-the-badge)](https://github.com/mohamednoorulnaseem/customer-churn-ml-pipeline)
[![Fork](https://img.shields.io/badge/🍴_Fork_it-4CAF50?style=for-the-badge)](https://github.com/mohamednoorulnaseem/customer-churn-ml-pipeline/fork)
[![Share](https://img.shields.io/badge/📤_Share_it-00D4FF?style=for-the-badge)](https://twitter.com/intent/tweet?text=Check%20out%20this%20End-to-End%20Customer%20Churn%20Prediction%20ML%20Pipeline!%20https://github.com/mohamednoorulnaseem/customer-churn-ml-pipeline)

<br/>

**Made with ❤️ and lots of ☕**

<br/>

[![Back to Top](https://img.shields.io/badge/⬆️_Back_to_Top-6366F1?style=flat-square)](#-retentioniq---customer-intelligence-platform)

</div>
