"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CustomerFeatures(BaseModel):
    """Single customer feature input for prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    monthly_spend: float = Field(..., ge=0, description="Average monthly spending")
    total_orders: int = Field(..., ge=0, description="Total number of orders")
    avg_order_value: float = Field(default=0, ge=0, description="Average order value")
    days_since_last_order: int = Field(..., ge=0, description="Days since last purchase")
    login_frequency: int = Field(default=0, ge=0, description="Monthly login count")
    products_viewed: int = Field(default=0, ge=0, description="Products viewed per session")
    cart_abandonment_rate: float = Field(default=0, ge=0, le=1, description="Cart abandonment rate")
    support_tickets: int = Field(default=0, ge=0, description="Support tickets raised")
    discount_usage_rate: float = Field(default=0, ge=0, le=1, description="Discount usage rate")
    satisfaction_score: int = Field(default=5, ge=1, le=10, description="Satisfaction score 1-10")
    complaint_count: int = Field(default=0, ge=0, description="Number of complaints")
    
    # Optional enhanced fields
    segment: Optional[str] = Field(default="Individual", description="Customer segment")
    region: Optional[str] = Field(default="North America", description="Customer region")
    favorite_category: Optional[str] = Field(default="Electronics", description="Favorite product category")
    category_diversity: Optional[int] = Field(default=3, ge=1, le=10, description="Number of categories purchased")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "tenure_months": 12,
                "monthly_spend": 250.0,
                "total_orders": 15,
                "avg_order_value": 75.0,
                "days_since_last_order": 30,
                "login_frequency": 8,
                "products_viewed": 15,
                "cart_abandonment_rate": 0.2,
                "support_tickets": 2,
                "discount_usage_rate": 0.3,
                "satisfaction_score": 7,
                "complaint_count": 1
            }
        }


class ChurnFactor(BaseModel):
    """Individual churn contributing factor."""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="Impact score")
    direction: str = Field(..., description="Direction: increases or decreases churn risk")
    importance: float = Field(..., description="Feature importance")


class ChurnPrediction(BaseModel):
    """Churn prediction output."""
    customer_id: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_segment: str = Field(..., description="Risk level: low, medium, or high")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    top_churn_factors: List[ChurnFactor] = Field(default=[], description="Top factors contributing to churn")
    prediction_timestamp: datetime = Field(default_factory=datetime.now, description="When prediction was made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "churn_probability": 0.73,
                "churn_prediction": True,
                "risk_segment": "high",
                "confidence": 0.73,
                "top_churn_factors": [
                    {"feature": "days_since_last_order", "impact": 0.35, "direction": "increases", "importance": 0.35}
                ],
                "prediction_timestamp": "2024-01-18T12:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    customers: List[CustomerFeatures] = Field(..., description="List of customers to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[ChurnPrediction] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total customers processed")
    high_risk_count: int = Field(..., description="Number of high-risk customers")
    medium_risk_count: int = Field(default=0, description="Number of medium-risk customers")
    low_risk_count: int = Field(default=0, description="Number of low-risk customers")
    average_churn_probability: float = Field(default=0, description="Average churn probability")


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str = Field(..., description="Name of the model")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    roc_auc: float = Field(..., ge=0, le=1, description="ROC AUC score")
    trained_at: Optional[str] = Field(default=None, description="Training timestamp")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(default=None, description="Loaded model name")
    api_version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: Optional[float] = Field(default=None, description="API uptime in seconds")


class FeatureImportance(BaseModel):
    """Feature importance response."""
    feature_name: str
    importance: float
    rank: int


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    n_features: int
    features: List[str]
    metrics: Optional[ModelMetrics] = None
    trained_at: Optional[str] = None
