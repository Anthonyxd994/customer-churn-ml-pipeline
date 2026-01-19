# 🔮 Customer Churn Prediction System

## 📋 Project Description

### One-Liner

> **Production-ready ML pipeline that predicts customer churn with 71% accuracy, featuring real-time API predictions, interactive dashboard, and MLflow experiment tracking.**

---

## 🎯 Elevator Pitch (30 seconds)

"I built an end-to-end Machine Learning system that predicts which customers will stop buying from an e-commerce store. The system uses XGBoost and Logistic Regression models trained on 10,000 customers, achieves 71% accuracy and 67% AUC-ROC. It includes a FastAPI REST API for real-time predictions, a Streamlit dashboard for visualization, MLflow for experiment tracking, and Docker for deployment. The business impact: **identify at-risk customers before they leave, enabling targeted retention campaigns that can save 5x the cost of acquiring new customers.**"

---

## 📝 Project Summary (For Resume/LinkedIn)

### Short Version (2-3 lines)

Built an end-to-end Customer Churn Prediction System using Python, scikit-learn, XGBoost, and FastAPI. Implemented MLflow for experiment tracking, Streamlit dashboard for visualization, and Docker for containerized deployment. Achieved 71% accuracy and 67% AUC-ROC on 10,000 customer dataset.

### Medium Version (Paragraph)

Developed a production-ready Machine Learning pipeline for predicting customer churn in e-commerce. The system includes synthetic data generation, feature engineering (68 features including RFM, engagement, and risk indicators), multi-model training (XGBoost, Random Forest, Logistic Regression) with hyperparameter tracking via MLflow. Built a FastAPI REST API for real-time predictions with SHAP explainability, and a 6-page Streamlit dashboard for business intelligence. Containerized with Docker and integrated with PostgreSQL for production data storage. The solution helps businesses identify at-risk customers and reduce churn by enabling proactive retention strategies.

---

## 🏆 Key Achievements & Metrics

| Metric                  | Value                                           |
| ----------------------- | ----------------------------------------------- |
| **Dataset Size**        | 10,000 customers, 29 raw features               |
| **Engineered Features** | 68 features (RFM, engagement, risk, temporal)   |
| **Best Model Accuracy** | 71.45% (Logistic Regression)                    |
| **Best AUC-ROC**        | 66.76%                                          |
| **Models Trained**      | 3 (XGBoost, Random Forest, Logistic Regression) |
| **Unit Tests**          | 21 tests, 100% passing                          |
| **API Endpoints**       | 8 endpoints (predict, batch, health, etc.)      |
| **Dashboard Pages**     | 6 interactive pages                             |

---

## 🛠️ Technical Skills Demonstrated

### Machine Learning & Data Science

- Classification models (XGBoost, Random Forest, Logistic Regression)
- Feature engineering (RFM analysis, behavioral features, risk indicators)
- Class imbalance handling (SMOTE)
- Model evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
- Explainable AI (SHAP values)

### MLOps & Engineering

- Experiment tracking (MLflow)
- Model versioning and registry
- REST API development (FastAPI)
- Containerization (Docker, docker-compose)
- Database integration (PostgreSQL)
- Unit testing (pytest, 21 tests)

### Data Engineering

- Data pipelines and ETL
- Feature stores
- Database design and querying
- Data validation

### Visualization & Dashboards

- Interactive dashboards (Streamlit)
- Data visualization (Plotly, Matplotlib)
- Business intelligence reporting

---

## 💼 Resume Bullet Points

### Senior/Lead Level

- Architected and deployed an end-to-end ML pipeline for customer churn prediction, reducing potential customer loss by identifying at-risk segments with 71% accuracy
- Implemented MLOps best practices including MLflow experiment tracking, Docker containerization, and CI-ready test suite with 21 unit tests
- Built real-time prediction API serving 8 endpoints with FastAPI, supporting both single and batch predictions with SHAP explainability

### Mid Level

- Developed customer churn prediction system using XGBoost and scikit-learn, achieving 67% AUC-ROC on 10,000 customer dataset
- Engineered 68 features from raw customer data including RFM metrics, engagement scores, and risk indicators
- Created interactive Streamlit dashboard with 6 pages for model monitoring and customer risk segmentation

### Entry Level

- Built ML pipeline for customer churn prediction using Python, pandas, and scikit-learn
- Trained and compared 3 classification models with MLflow experiment tracking
- Developed REST API with FastAPI and interactive dashboard with Streamlit

---

## 🎤 Interview Talking Points

### "Tell me about a project you're proud of"

"I built a complete Customer Churn Prediction System from scratch. The challenge was not just building a model, but creating a production-ready solution. I generated synthetic customer data, engineered 68 features including RFM metrics and behavioral indicators, trained 3 different models while tracking experiments with MLflow, and deployed everything with a FastAPI prediction API and Streamlit dashboard. The most challenging part was handling class imbalance - 28% churn rate - which I solved using SMOTE oversampling. The result was a system that can identify at-risk customers in real-time, helping businesses save the 5-7x cost of acquiring new customers versus retaining existing ones."

### "How do you handle ML in production?"

"In my churn prediction project, I focused heavily on production readiness. I used MLflow for experiment tracking and model versioning, so we can always reproduce results. The FastAPI server serves predictions via REST API with proper error handling and input validation using Pydantic schemas. I containerized everything with Docker for consistent deployment. I also wrote 21 unit tests to ensure reliability. For monitoring, the Streamlit dashboard shows model performance and can detect if predictions drift over time."

### "Describe your ML pipeline"

"My pipeline has 5 stages: (1) Data ingestion - I built a data generator that creates realistic customer profiles with proper distributions. (2) Feature engineering - transforming raw data into 68 features across categories like RFM, engagement, and risk. (3) Model training - I train XGBoost, Random Forest, and Logistic Regression with cross-validation, logging everything to MLflow. (4) Model serving - FastAPI provides real-time predictions with SHAP explanations. (5) Monitoring - the Streamlit dashboard lets stakeholders track performance and explore customer segments."

---

## 🔗 LinkedIn Post Draft

```
🚀 Just completed an End-to-End ML Project: Customer Churn Prediction System!

This wasn't just about building a model - it's a complete production-ready solution:

🎯 What it does:
Predicts which customers will stop buying, enabling proactive retention strategies

📊 Technical highlights:
• 10,000 customers, 68 engineered features
• 3 models: XGBoost, Random Forest, Logistic Regression
• 71% accuracy, 67% AUC-ROC
• Real-time predictions via FastAPI
• Interactive Streamlit dashboard
• MLflow experiment tracking
• Docker-ready deployment
• PostgreSQL integration
• 21 unit tests passing

💼 Business value:
Acquiring new customers costs 5-7x more than retaining existing ones. This system identifies at-risk customers before they leave.

🛠️ Tech Stack:
Python | scikit-learn | XGBoost | FastAPI | Streamlit | MLflow | Docker | PostgreSQL | SHAP

Check out the code: [GitHub Link]

#MachineLearning #DataScience #Python #MLOps #CustomerChurn #Portfolio
```

---

## 📄 GitHub Repository Description

**Short (160 chars):**
Production-ready ML pipeline for customer churn prediction with FastAPI, Streamlit dashboard, MLflow tracking, and Docker deployment.

**Full:**
🔮 Customer Churn Prediction System - An end-to-end ML pipeline that predicts customer churn using XGBoost and Logistic Regression. Features include:

- 68 engineered features (RFM, engagement, risk indicators)
- MLflow experiment tracking and model registry
- FastAPI REST API with SHAP explainability
- Interactive 6-page Streamlit dashboard
- PostgreSQL integration
- Docker containerization
- 21 unit tests

Built for portfolio demonstration of production ML systems.

---

## 🏷️ GitHub Topics/Tags

```
machine-learning, python, customer-churn, mlops, fastapi, streamlit,
mlflow, docker, xgboost, scikit-learn, data-science, postgresql,
classification, feature-engineering, shap, explainability
```

---

## 📊 Architecture Summary

```
Data Layer:          CSV / PostgreSQL
                           ↓
Feature Engineering: 68 features (RFM, engagement, risk, temporal)
                           ↓
Model Training:      XGBoost, Random Forest, Logistic Regression
                           ↓
Experiment Tracking: MLflow (parameters, metrics, artifacts)
                           ↓
Model Serving:       FastAPI REST API + SHAP Explainability
                           ↓
Visualization:       Streamlit Dashboard (6 pages)
                           ↓
Deployment:          Docker + docker-compose
```

---

## 📞 Contact

**Noor** - Data Scientist / ML Engineer

Feel free to reach out for discussions about this project!
