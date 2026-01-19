"""
FastAPI Prediction Service for Customer Churn

This module provides a REST API for:
- Real-time churn predictions
- Batch predictions
- Model information
- SHAP explanations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import shap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    🔮 **Customer Churn Prediction Service**
    
    This API provides real-time predictions for customer churn probability
    using machine learning models trained on customer behavior data.
    
    ## Features
    - 📊 Single customer predictions
    - 📦 Batch predictions
    - 🔍 SHAP-based explanations
    - 📈 Model information and metrics
    
    ## Risk Segments
    - **Low**: < 30% churn probability
    - **Medium**: 30-60% churn probability  
    - **High**: > 60% churn probability
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and artifacts
model = None
scaler = None
feature_names = None
model_info = None
shap_explainer = None


# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Customer features for prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    monthly_spend: float = Field(..., ge=0, description="Average monthly spend")
    total_orders: int = Field(..., ge=0, description="Total number of orders")
    avg_order_value: float = Field(..., ge=0, description="Average order value")
    days_since_last_order: int = Field(..., ge=0, description="Days since last purchase")
    login_frequency: int = Field(..., ge=0, description="Monthly login count")
    products_viewed: int = Field(..., ge=0, description="Products viewed per session")
    cart_abandonment_rate: float = Field(..., ge=0, le=1, description="Cart abandonment rate")
    support_tickets: int = Field(..., ge=0, description="Support tickets raised")
    discount_usage_rate: float = Field(..., ge=0, le=1, description="Discount usage rate")
    satisfaction_score: int = Field(..., ge=1, le=10, description="Customer satisfaction (1-10)")
    complaint_count: int = Field(..., ge=0, description="Number of complaints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_000001",
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
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_segment: str
    confidence: float
    top_churn_factors: List[Dict[str, Any]]
    prediction_timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, Any]


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    metrics: Dict[str, float]
    n_features: int
    trained_at: str
    feature_names: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


def load_model_artifacts():
    """Load model and related artifacts."""
    global model, scaler, feature_names, model_info, shap_explainer
    
    artifacts_path = Path("models/artifacts")
    
    try:
        # Load model
        model_path = artifacts_path / "best_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
        else:
            logger.warning(f"Model not found at: {model_path}")
            return False
        
        # Load scaler
        scaler_path = artifacts_path / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
        
        # Load feature names
        features_path = artifacts_path / "feature_names.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        # Load model info
        info_path = artifacts_path / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            logger.info(f"Model info loaded: {model_info['model_name']}")
        
        # Initialize SHAP explainer
        try:
            shap_explainer = shap.TreeExplainer(model)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            shap_explainer = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        return False


def prepare_features(customer: CustomerFeatures) -> np.ndarray:
    """
    Prepare customer features for prediction.
    
    This creates a feature vector matching the training features.
    For simplicity, we use the core features and derive others.
    """
    # Create base features dictionary
    features = {
        'tenure_months': customer.tenure_months,
        'monthly_spend': customer.monthly_spend,
        'total_orders': customer.total_orders,
        'avg_order_value': customer.avg_order_value,
        'days_since_last_order': customer.days_since_last_order,
        'login_frequency': customer.login_frequency,
        'products_viewed': customer.products_viewed,
        'cart_abandonment_rate': customer.cart_abandonment_rate,
        'support_tickets': customer.support_tickets,
        'discount_usage_rate': customer.discount_usage_rate,
        'satisfaction_score': customer.satisfaction_score,
        'complaint_count': customer.complaint_count,
    }
    
    # Derive additional features
    tenure = max(customer.tenure_months, 1)
    
    # RFM features
    features['recency_score'] = 1 - (customer.days_since_last_order / 180)
    features['order_frequency'] = customer.total_orders / tenure
    features['monetary_score'] = min(customer.monthly_spend / 5000, 1)
    features['rfm_score'] = (
        features['recency_score'] * 0.35 +
        min(features['order_frequency'] / 3, 1) * 0.35 +
        features['monetary_score'] * 0.30
    )
    
    # Engagement features
    features['engagement_score'] = min(customer.login_frequency / 50, 1) * 0.5 + 0.5
    features['activity_ratio'] = (customer.login_frequency * tenure) / max(customer.total_orders, 1)
    features['browse_to_buy_ratio'] = customer.total_orders / max(customer.products_viewed * tenure, 1)
    
    # Risk features
    features['support_burden'] = (customer.support_tickets + customer.complaint_count) / tenure
    features['return_behavior'] = 0.1  # Default
    features['dependency_score'] = (tenure / 60) * 0.5 + (customer.total_orders / 200) * 0.5
    features['price_sensitivity'] = customer.discount_usage_rate
    features['inactivity_risk'] = (customer.days_since_last_order / 180) * 0.5 + (1 - customer.login_frequency / 50) * 0.5
    
    # Value features
    features['order_velocity'] = customer.total_orders / tenure
    features['clv_proxy'] = customer.monthly_spend * 36
    features['avg_transaction_value'] = customer.monthly_spend * tenure / max(customer.total_orders, 1)
    
    # Additional derived features (with defaults for missing)
    features['review_count'] = 2
    features['email_open_rate'] = 0.3
    features['wishlist_items'] = 5
    features['referral_count'] = 0
    features['return_rate'] = 0.1
    features['avg_delivery_rating'] = 4.0
    features['push_notification_enabled'] = 1
    features['newsletter_subscribed'] = 1
    features['app_installed'] = 1
    
    # Create feature vector in correct order
    if feature_names:
        feature_vector = []
        for fname in feature_names:
            if fname in features:
                feature_vector.append(features[fname])
            else:
                # Default values for categorical encoded features
                feature_vector.append(0)
        return np.array(feature_vector).reshape(1, -1)
    else:
        return np.array(list(features.values())).reshape(1, -1)


def get_risk_segment(probability: float) -> str:
    """Get risk segment based on churn probability."""
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"


def get_top_factors(customer: CustomerFeatures, shap_values: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Get top factors contributing to churn prediction."""
    # If SHAP values available, use them
    if shap_values is not None and feature_names:
        # Get absolute values and pair with feature names
        abs_shap = np.abs(shap_values[0])
        feature_importance = list(zip(feature_names, shap_values[0], abs_shap))
        feature_importance.sort(key=lambda x: x[2], reverse=True)
        
        top_factors = []
        for fname, shap_val, abs_val in feature_importance[:5]:
            direction = "increases" if shap_val > 0 else "decreases"
            top_factors.append({
                "feature": fname,
                "impact": round(float(shap_val), 4),
                "direction": direction,
                "importance": round(float(abs_val), 4)
            })
        return top_factors
    
    # Fallback to rule-based factors
    factors = []
    
    if customer.days_since_last_order > 60:
        factors.append({
            "feature": "days_since_last_order",
            "impact": 0.35,
            "direction": "increases",
            "importance": 0.35
        })
    
    if customer.support_tickets > 3:
        factors.append({
            "feature": "support_tickets",
            "impact": 0.25,
            "direction": "increases",
            "importance": 0.25
        })
    
    if customer.satisfaction_score < 5:
        factors.append({
            "feature": "satisfaction_score",
            "impact": 0.20,
            "direction": "increases",
            "importance": 0.20
        })
    
    if customer.login_frequency < 3:
        factors.append({
            "feature": "login_frequency",
            "impact": 0.15,
            "direction": "increases",
            "importance": 0.15
        })
    
    if customer.tenure_months < 6:
        factors.append({
            "feature": "tenure_months",
            "impact": 0.12,
            "direction": "increases",
            "importance": 0.12
        })
    
    return factors[:5] if factors else [{"feature": "overall_profile", "impact": 0.1, "direction": "neutral", "importance": 0.1}]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    logger.info("Starting API server...")
    success = load_model_artifacts()
    if not success:
        logger.warning("Model not loaded - predictions will not work until model is available")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_info.get('model_name', 'unknown'),
        version="1.0.0",
        metrics=model_info.get('metrics', {}),
        n_features=model_info.get('n_features', 0),
        trained_at=model_info.get('trained_at', 'unknown'),
        feature_names=feature_names or []
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    
    Returns:
    - Churn probability (0-1)
    - Binary prediction (True/False)
    - Risk segment (low/medium/high)
    - Top contributing factors
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(customer)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Get prediction
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = probability >= 0.5
        
        # Get SHAP values if available
        shap_values = None
        if shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(features_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
        
        # Get top factors
        top_factors = get_top_factors(customer, shap_values)
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(float(probability), 4),
            churn_prediction=bool(prediction),
            risk_segment=get_risk_segment(probability),
            confidence=round(float(max(probability, 1-probability)), 4),
            top_churn_factors=top_factors,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Returns predictions for all customers along with batch summary statistics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for customer in request.customers:
        try:
            pred = await predict_churn(customer)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error for {customer.customer_id}: {e}")
    
    # Calculate batch summary
    probabilities = [p.churn_probability for p in predictions]
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for p in predictions:
        risk_counts[p.risk_segment] += 1
    
    summary = {
        "total_customers": len(predictions),
        "avg_churn_probability": round(np.mean(probabilities), 4) if probabilities else 0,
        "predicted_churners": sum(1 for p in predictions if p.churn_prediction),
        "risk_distribution": risk_counts
    }
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_summary=summary
    )


@app.get("/features", tags=["Features"])
async def get_required_features():
    """Get list of required features for prediction."""
    return {
        "required_features": [
            {"name": "customer_id", "type": "string", "description": "Unique customer ID"},
            {"name": "tenure_months", "type": "integer", "description": "Months as customer"},
            {"name": "monthly_spend", "type": "float", "description": "Average monthly spend"},
            {"name": "total_orders", "type": "integer", "description": "Total number of orders"},
            {"name": "avg_order_value", "type": "float", "description": "Average order value"},
            {"name": "days_since_last_order", "type": "integer", "description": "Days since last purchase"},
            {"name": "login_frequency", "type": "integer", "description": "Monthly login count"},
            {"name": "products_viewed", "type": "integer", "description": "Products viewed per session"},
            {"name": "cart_abandonment_rate", "type": "float", "description": "Cart abandonment rate (0-1)"},
            {"name": "support_tickets", "type": "integer", "description": "Support tickets raised"},
            {"name": "discount_usage_rate", "type": "float", "description": "Discount usage rate (0-1)"},
            {"name": "satisfaction_score", "type": "integer", "description": "Customer satisfaction (1-10)"},
            {"name": "complaint_count", "type": "integer", "description": "Number of complaints"}
        ]
    }


# Run with: uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
