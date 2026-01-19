"""
Feature Engineering Pipeline for Customer Churn Prediction

This module transforms raw customer data into ML-ready features through:
- Data cleaning and validation
- Feature transformations
- Feature creation (RFM, ratios, binning)
- Encoding categorical variables
- Scaling numerical features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import yaml
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for churn prediction.
    
    Transforms raw customer data into ML-ready features with:
    - RFM (Recency, Frequency, Monetary) features
    - Engagement metrics
    - Customer lifetime value indicators
    - Risk scores
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature engineer with configuration."""
        self.config = self._load_config(config_path)
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'features': {
                'numerical': [
                    'tenure_months', 'monthly_spend', 'total_orders',
                    'avg_order_value', 'days_since_last_order', 'login_frequency',
                    'support_tickets', 'products_viewed', 'cart_abandonment_rate',
                    'discount_usage_rate'
                ],
                'categorical': [
                    'gender', 'location_tier', 'preferred_payment',
                    'preferred_device', 'membership_type', 'marital_status'
                ],
                'target': 'churn'
            },
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed'
            }
        }
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load customer data from CSV file.
        
        Args:
            filepath: Path to CSV file. Uses config path if not provided.
            
        Returns:
            pd.DataFrame: Raw customer data
        """
        if filepath is None:
            filepath = Path(self.config['data']['raw_path']) / "customers.csv"
        
        logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.
        
        Steps:
        - Handle missing values
        - Remove duplicates
        - Fix data types
        - Handle outliers
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['customer_id'])
        removed = initial_rows - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        
        # Handle missing values
        numerical_cols = self.config['features']['numerical']
        categorical_cols = self.config['features']['categorical']
        
        # Fill numerical NAs with median
        for col in numerical_cols:
            if col in df_clean.columns and df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} NAs with median: {median_val}")
        
        # Fill categorical NAs with mode
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isna().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} NAs with mode: {mode_val}")
        
        # Cap outliers (IQR method for numerical columns)
        for col in numerical_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.01)
                Q3 = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(Q1, Q3)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        RFM is a proven technique for customer segmentation:
        - Recency: How recently a customer purchased
        - Frequency: How often they purchase
        - Monetary: How much they spend
        """
        logger.info("Creating RFM features...")
        df_rfm = df.copy()
        
        # Recency score (inverse of days since last order)
        max_days = df_rfm['days_since_last_order'].max()
        df_rfm['recency_score'] = 1 - (df_rfm['days_since_last_order'] / max_days)
        
        # Frequency score (orders per month)
        df_rfm['order_frequency'] = df_rfm['total_orders'] / df_rfm['tenure_months'].replace(0, 1)
        
        # Monetary score (normalized monthly spend)
        max_spend = df_rfm['monthly_spend'].max()
        df_rfm['monetary_score'] = df_rfm['monthly_spend'] / max_spend
        
        # Combined RFM score
        df_rfm['rfm_score'] = (
            df_rfm['recency_score'] * 0.35 +
            df_rfm['order_frequency'] / df_rfm['order_frequency'].max() * 0.35 +
            df_rfm['monetary_score'] * 0.30
        )
        
        # RFM segments
        df_rfm['rfm_segment'] = pd.cut(
            df_rfm['rfm_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['At Risk', 'Needs Attention', 'Loyal', 'Champions']
        )
        
        return df_rfm
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer engagement features.
        
        Features:
        - Engagement score
        - Activity ratio
        - Channel usage patterns
        """
        logger.info("Creating engagement features...")
        df_eng = df.copy()
        
        # Overall engagement score
        df_eng['engagement_score'] = (
            df_eng['login_frequency'] / 50 * 0.25 +
            df_eng['email_open_rate'] * 0.20 +
            df_eng['push_notification_enabled'] * 0.15 +
            df_eng['newsletter_subscribed'] * 0.10 +
            df_eng['app_installed'] * 0.15 +
            (df_eng['wishlist_items'] / 50) * 0.10 +
            (df_eng['referral_count'] / 10) * 0.05
        )
        df_eng['engagement_score'] = df_eng['engagement_score'].clip(0, 1)
        
        # Activity ratio (logins per order)
        df_eng['activity_ratio'] = (
            df_eng['login_frequency'] * df_eng['tenure_months'] / 
            df_eng['total_orders'].replace(0, 1)
        )
        
        # Browse to buy ratio
        df_eng['browse_to_buy_ratio'] = (
            df_eng['total_orders'] / 
            (df_eng['products_viewed'] * df_eng['tenure_months']).replace(0, 1)
        )
        
        return df_eng
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create churn risk indicator features.
        
        Features based on factors that typically predict churn:
        - Support burden
        - Satisfaction trends
        - Activity decline indicators
        """
        logger.info("Creating risk features...")
        df_risk = df.copy()
        
        # Support burden ratio
        df_risk['support_burden'] = (
            df_risk['support_tickets'] + df_risk['complaint_count']
        ) / df_risk['tenure_months'].replace(0, 1)
        
        # Return behavior score
        df_risk['return_behavior'] = (
            df_risk['return_rate'] * 0.6 +
            (1 - df_risk['avg_delivery_rating'] / 5) * 0.4
        )
        
        # Dependency score (how "locked in" is the customer)
        df_risk['dependency_score'] = (
            (df_risk['tenure_months'] / 60) * 0.30 +
            (df_risk['total_orders'] / 200) * 0.25 +
            (df_risk['review_count'] / 30) * 0.15 +
            (df_risk['referral_count'] / 10) * 0.15 +
            (df_risk['wishlist_items'] / 50) * 0.15
        )
        df_risk['dependency_score'] = df_risk['dependency_score'].clip(0, 1)
        
        # Price sensitivity (high discount usage = price sensitive)
        df_risk['price_sensitivity'] = df_risk['discount_usage_rate']
        
        # Inactivity risk
        df_risk['inactivity_risk'] = (
            (df_risk['days_since_last_order'] / 180) * 0.50 +
            (1 - df_risk['login_frequency'] / 50) * 0.30 +
            (1 - df_risk['email_open_rate']) * 0.20
        )
        df_risk['inactivity_risk'] = df_risk['inactivity_risk'].clip(0, 1)
        
        return df_risk
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features:
        - Account age categories
        - Lifecycle stage
        - Tenure bins
        """
        logger.info("Creating temporal features...")
        df_temp = df.copy()
        
        # Tenure category
        df_temp['tenure_category'] = pd.cut(
            df_temp['tenure_months'],
            bins=[0, 3, 12, 24, 48, 60],
            labels=['New', 'Growing', 'Established', 'Mature', 'Veteran']
        )
        
        # Customer lifecycle stage
        conditions = [
            (df_temp['tenure_months'] <= 3) & (df_temp['total_orders'] <= 2),
            (df_temp['tenure_months'] <= 6) & (df_temp['total_orders'] <= 5),
            (df_temp['tenure_months'] <= 12) & (df_temp['total_orders'] <= 12),
            (df_temp['tenure_months'] > 12) & (df_temp['total_orders'] > 12)
        ]
        choices = ['Onboarding', 'Activation', 'Engagement', 'Retention']
        df_temp['lifecycle_stage'] = np.select(conditions, choices, default='At Risk')
        
        # Order velocity (orders per month recently)
        df_temp['order_velocity'] = df_temp['total_orders'] / df_temp['tenure_months'].replace(0, 1)
        
        return df_temp
    
    def create_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer value features.
        
        Features:
        - Customer Lifetime Value (CLV) proxy
        - Spend categories
        - Value segments
        """
        logger.info("Creating value features...")
        df_val = df.copy()
        
        # CLV proxy (simplified)
        avg_monthly_spend = df_val['monthly_spend']
        predicted_lifetime = np.where(
            df_val['tenure_months'] > 24,
            48,  # Long-term customers likely to stay 4+ years
            np.where(
                df_val['tenure_months'] > 12,
                36,
                24
            )
        )
        df_val['clv_proxy'] = avg_monthly_spend * predicted_lifetime
        
        # Spend category
        df_val['spend_category'] = pd.qcut(
            df_val['monthly_spend'],
            q=4,
            labels=['Budget', 'Mid-Range', 'Premium', 'VIP'],
            duplicates='drop'
        )
        
        # Value segment (combining spend and engagement)
        df_val['value_segment'] = pd.cut(
            df_val['clv_proxy'],
            bins=[0, 2000, 5000, 15000, df_val['clv_proxy'].max() + 1],
            labels=['Low', 'Medium', 'High', 'Strategic']
        )
        
        # Average transaction value trend proxy
        df_val['avg_transaction_value'] = (
            df_val['monthly_spend'] * df_val['tenure_months'] / 
            df_val['total_orders'].replace(0, 1)
        )
        
        return df_val
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Raw customer data
            
        Returns:
            pd.DataFrame: Feature-engineered dataset
        """
        logger.info("=" * 50)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Create feature groups
        df_features = self.create_rfm_features(df_clean)
        df_features = self.create_engagement_features(df_features)
        df_features = self.create_risk_features(df_features)
        df_features = self.create_temporal_features(df_features)
        df_features = self.create_value_features(df_features)
        
        logger.info("=" * 50)
        logger.info(f"Feature engineering complete!")
        logger.info(f"Final dataset shape: {df_features.shape}")
        logger.info(f"New features created: {len(df_features.columns) - len(df.columns)}")
        logger.info("=" * 50)
        
        return df_features
    
    def prepare_for_training(
        self, 
        df: pd.DataFrame,
        target_col: str = 'churn'
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare feature-engineered data for model training.
        
        Steps:
        - Select relevant features
        - Encode categorical variables
        - Scale numerical features
        
        Args:
            df: Feature-engineered DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing data for training...")
        
        # Define feature columns
        numerical_features = [
            'tenure_months', 'monthly_spend', 'total_orders', 'avg_order_value',
            'days_since_last_order', 'login_frequency', 'products_viewed',
            'cart_abandonment_rate', 'support_tickets', 'discount_usage_rate',
            'review_count', 'email_open_rate', 'wishlist_items', 'referral_count',
            'satisfaction_score', 'complaint_count', 'return_rate', 'avg_delivery_rating',
            # Engineered features
            'recency_score', 'order_frequency', 'monetary_score', 'rfm_score',
            'engagement_score', 'activity_ratio', 'browse_to_buy_ratio',
            'support_burden', 'return_behavior', 'dependency_score',
            'price_sensitivity', 'inactivity_risk', 'order_velocity',
            'clv_proxy', 'avg_transaction_value'
        ]
        
        binary_features = [
            'push_notification_enabled', 'newsletter_subscribed', 'app_installed'
        ]
        
        categorical_features = [
            'gender', 'location_tier', 'preferred_payment', 'preferred_device',
            'membership_type', 'marital_status', 'rfm_segment', 'tenure_category',
            'lifecycle_stage', 'spend_category', 'value_segment'
        ]
        
        # Filter to existing columns
        numerical_features = [f for f in numerical_features if f in df.columns]
        binary_features = [f for f in binary_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Create feature matrix
        X_num = df[numerical_features].copy()
        X_bin = df[binary_features].copy()
        
        # Encode categorical features
        X_cat_encoded = pd.get_dummies(df[categorical_features], drop_first=True)
        
        # Combine all features
        X = pd.concat([X_num, X_bin, X_cat_encoded], axis=1)
        y = df[target_col]
        
        # Get feature names
        self.feature_names = list(X.columns)
        
        # Fill any remaining NAs
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_names)} features")
        
        return X, y, self.feature_names
    
    def save_processed_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_names: List[str],
        filename_prefix: str = "processed"
    ) -> str:
        """Save processed data to files."""
        output_dir = Path(self.config['data']['processed_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and target
        X.to_csv(output_dir / f"{filename_prefix}_features.csv", index=False)
        y.to_csv(output_dir / f"{filename_prefix}_target.csv", index=False)
        
        # Save feature names
        with open(output_dir / f"{filename_prefix}_feature_names.txt", 'w') as f:
            f.write('\n'.join(feature_names))
        
        logger.info(f"Saved processed data to: {output_dir}")
        return str(output_dir)


def main():
    """Main function to run feature engineering pipeline."""
    print("=" * 60)
    print("[*] Feature Engineering Pipeline")
    print("=" * 60)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Load raw data
    print("\n[1/4] Loading raw data...")
    df = fe.load_data()
    
    # Run feature engineering
    print("\n[2/4] Running feature engineering pipeline...")
    df_features = fe.engineer_features(df)
    
    # Prepare for training
    print("\n[3/4] Preparing data for training...")
    X, y, feature_names = fe.prepare_for_training(df_features)
    
    # Display summary
    print("\n[INFO] Feature Summary:")
    print(f"  - Total samples: {len(X):,}")
    print(f"  - Total features: {len(feature_names)}")
    print(f"  - Target distribution:")
    print(f"    - Not Churned (0): {(y == 0).sum():,} ({(y == 0).mean():.1%})")
    print(f"    - Churned (1): {(y == 1).sum():,} ({(y == 1).mean():.1%})")
    
    # Save processed data
    print("\n[4/4] Saving processed data...")
    fe.save_processed_data(X, y, feature_names)
    
    # Also save the full feature-engineered dataset
    output_dir = Path("data/processed")
    df_features.to_csv(output_dir / "customers_featured.csv", index=False)
    
    print("\n[DONE] Feature engineering complete!")
    print(f"  - Features saved to: data/processed/processed_features.csv")
    print(f"  - Full dataset saved to: data/processed/customers_featured.csv")


if __name__ == "__main__":
    main()
