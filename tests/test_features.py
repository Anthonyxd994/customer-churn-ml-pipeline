"""
Unit Tests for Feature Engineering

Tests the feature engineering pipeline to ensure
correct transformations and data integrity.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.create_features import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer data for testing."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'customer_id': [f"CUST_{i:06d}" for i in range(n)],
            'gender': np.random.choice(['Male', 'Female'], n),
            'age': np.random.randint(18, 70, n),
            'location_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], n),
            'marital_status': np.random.choice(['Single', 'Married'], n),
            'tenure_months': np.random.randint(1, 60, n),
            'membership_type': np.random.choice(['Basic', 'Silver', 'Gold'], n),
            'preferred_payment': np.random.choice(['Credit Card', 'UPI'], n),
            'preferred_device': np.random.choice(['Mobile', 'Desktop'], n),
            'total_orders': np.random.randint(1, 100, n),
            'monthly_spend': np.random.uniform(50, 500, n),
            'avg_order_value': np.random.uniform(20, 200, n),
            'days_since_last_order': np.random.randint(0, 180, n),
            'login_frequency': np.random.randint(0, 30, n),
            'products_viewed': np.random.randint(1, 50, n),
            'cart_abandonment_rate': np.random.uniform(0, 0.8, n),
            'support_tickets': np.random.randint(0, 10, n),
            'discount_usage_rate': np.random.uniform(0, 0.5, n),
            'review_count': np.random.randint(0, 20, n),
            'email_open_rate': np.random.uniform(0, 1, n),
            'push_notification_enabled': np.random.choice([0, 1], n),
            'newsletter_subscribed': np.random.choice([0, 1], n),
            'app_installed': np.random.choice([0, 1], n),
            'wishlist_items': np.random.randint(0, 30, n),
            'referral_count': np.random.randint(0, 5, n),
            'satisfaction_score': np.random.randint(1, 11, n),
            'complaint_count': np.random.randint(0, 5, n),
            'return_rate': np.random.uniform(0, 0.3, n),
            'avg_delivery_rating': np.random.uniform(3, 5, n),
            'churn': np.random.choice([0, 1], n, p=[0.74, 0.26])
        })
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    def test_clean_data_removes_duplicates(self, feature_engineer, sample_data):
        """Test that duplicate records are removed."""
        # Add duplicate
        duplicate_row = sample_data.iloc[0:1].copy()
        data_with_dup = pd.concat([sample_data, duplicate_row], ignore_index=True)
        
        cleaned = feature_engineer.clean_data(data_with_dup)
        
        assert len(cleaned) == len(sample_data)
    
    def test_create_rfm_features(self, feature_engineer, sample_data):
        """Test RFM feature creation."""
        result = feature_engineer.create_rfm_features(sample_data)
        
        assert 'recency_score' in result.columns
        assert 'order_frequency' in result.columns
        assert 'monetary_score' in result.columns
        assert 'rfm_score' in result.columns
        
        # Check value ranges
        assert result['recency_score'].min() >= 0
        assert result['recency_score'].max() <= 1
        assert result['monetary_score'].min() >= 0
        assert result['monetary_score'].max() <= 1
    
    def test_create_engagement_features(self, feature_engineer, sample_data):
        """Test engagement feature creation."""
        result = feature_engineer.create_engagement_features(sample_data)
        
        assert 'engagement_score' in result.columns
        assert 'activity_ratio' in result.columns
        assert 'browse_to_buy_ratio' in result.columns
        
        # Check engagement score is bounded
        assert result['engagement_score'].min() >= 0
        assert result['engagement_score'].max() <= 1
    
    def test_create_risk_features(self, feature_engineer, sample_data):
        """Test risk feature creation."""
        result = feature_engineer.create_risk_features(sample_data)
        
        assert 'support_burden' in result.columns
        assert 'return_behavior' in result.columns
        assert 'dependency_score' in result.columns
        assert 'inactivity_risk' in result.columns
    
    def test_prepare_for_training_shape(self, feature_engineer, sample_data):
        """Test that prepare_for_training returns correct shapes."""
        # First engineer features
        df_features = feature_engineer.engineer_features(sample_data)
        
        X, y, feature_names = feature_engineer.prepare_for_training(df_features)
        
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert len(feature_names) == X.shape[1]
    
    def test_prepare_for_training_no_nulls(self, feature_engineer, sample_data):
        """Test that prepared data has no null values."""
        df_features = feature_engineer.engineer_features(sample_data)
        X, y, _ = feature_engineer.prepare_for_training(df_features)
        
        assert not pd.DataFrame(X).isna().any().any()
        assert not pd.Series(y).isna().any()
    
    def test_engineer_features_full_pipeline(self, feature_engineer, sample_data):
        """Test full feature engineering pipeline."""
        result = feature_engineer.engineer_features(sample_data)
        
        # Should have more columns than input
        assert len(result.columns) > len(sample_data.columns)
        
        # Should have same number of rows
        assert len(result) == len(sample_data)


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_missing_values_handled(self):
        """Test that missing values are handled properly."""
        fe = FeatureEngineer()
        
        data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'tenure_months': [12, np.nan, 24],
            'monthly_spend': [100, 200, np.nan],
            'total_orders': [5, 10, 15],
            'days_since_last_order': [10, 20, 30],
            'login_frequency': [5, 10, 15],
            'products_viewed': [20, 30, 40],
            'cart_abandonment_rate': [0.1, 0.2, 0.3],
            'support_tickets': [1, 2, 3],
            'discount_usage_rate': [0.1, 0.2, 0.3]
        })
        
        cleaned = fe.clean_data(data)
        
        assert not cleaned.isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
