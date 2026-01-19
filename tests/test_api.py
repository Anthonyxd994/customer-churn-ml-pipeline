"""
Unit Tests for API Endpoints

Tests the FastAPI prediction endpoints to ensure
correct request handling and response formats.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)


@pytest.fixture
def sample_customer():
    """Sample customer data for testing."""
    return {
        "customer_id": "CUST_TEST_001",
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


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_structure(self, client):
        """Test health response structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_info(self, client):
        """Test root response contains API info."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data


class TestFeaturesEndpoint:
    """Test features endpoint."""
    
    def test_features_returns_200(self, client):
        """Test that features endpoint returns 200."""
        response = client.get("/features")
        assert response.status_code == 200
    
    def test_features_list_structure(self, client):
        """Test features response structure."""
        response = client.get("/features")
        data = response.json()
        
        assert "required_features" in data
        assert isinstance(data["required_features"], list)
        assert len(data["required_features"]) > 0


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_valid_customer(self, client, sample_customer):
        """Test prediction with valid customer data."""
        response = client.post("/predict", json=sample_customer)
        
        # May return 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "customer_id" in data
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_segment" in data
    
    def test_predict_missing_field(self, client, sample_customer):
        """Test prediction with missing required field."""
        incomplete_customer = sample_customer.copy()
        del incomplete_customer["tenure_months"]
        
        response = client.post("/predict", json=incomplete_customer)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_predict_invalid_value(self, client, sample_customer):
        """Test prediction with invalid value."""
        invalid_customer = sample_customer.copy()
        invalid_customer["tenure_months"] = -10  # Invalid negative value
        
        response = client.post("/predict", json=invalid_customer)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_predict_response_structure(self, client, sample_customer):
        """Test prediction response structure."""
        response = client.post("/predict", json=sample_customer)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            assert "customer_id" in data
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_segment" in data
            assert "confidence" in data
            assert "top_churn_factors" in data
            assert "prediction_timestamp" in data
            
            # Check value types
            assert isinstance(data["churn_probability"], float)
            assert isinstance(data["churn_prediction"], bool)
            assert data["risk_segment"] in ["low", "medium", "high"]
            assert 0 <= data["churn_probability"] <= 1


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_multiple_customers(self, client, sample_customer):
        """Test batch prediction with multiple customers."""
        customers = [
            sample_customer.copy(),
            {**sample_customer.copy(), "customer_id": "CUST_TEST_002"}
        ]
        
        response = client.post("/predict/batch", json={"customers": customers})
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "batch_summary" in data
            assert len(data["predictions"]) == 2
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"customers": []})
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 0


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info_structure(self, client):
        """Test model info response structure."""
        response = client.get("/model/info")
        
        # May return 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "version" in data
            assert "metrics" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
