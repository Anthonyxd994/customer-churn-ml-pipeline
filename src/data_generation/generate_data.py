"""
Synthetic Customer Data Generator for Churn Prediction

This module generates realistic e-commerce customer data with various 
behavioral patterns that correlate with churn likelihood.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import yaml
from typing import Tuple, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerDataGenerator:
    """
    Generates synthetic customer data for churn prediction modeling.
    
    The generator creates realistic patterns where certain behaviors
    correlate with higher churn probability:
    - Low engagement (login frequency, time on site)
    - Declining purchase frequency
    - High support ticket volume
    - Short tenure
    - High cart abandonment
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the generator with configuration."""
        self.config = self._load_config(config_path)
        self.num_customers = self.config['data_generation']['num_customers']
        self.churn_rate = self.config['data_generation']['churn_rate']
        self.random_state = self.config['data']['random_state']
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'data_generation': {
                'num_customers': 10000,
                'churn_rate': 0.26,
                'start_date': '2023-01-01',
                'end_date': '2024-12-31'
            },
            'data': {
                'random_state': 42,
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'sample_path': 'data/sample'
            }
        }
    
    def generate_customer_ids(self) -> np.ndarray:
        """Generate unique customer IDs."""
        return np.array([f"CUST_{str(i).zfill(6)}" for i in range(1, self.num_customers + 1)])
    
    def generate_demographics(self, n: int) -> pd.DataFrame:
        """
        Generate demographic features.
        
        Features:
        - gender: Male/Female/Other
        - age: 18-75 (normally distributed around 35)
        - location_tier: Tier1/Tier2/Tier3 cities
        - marital_status: Single/Married/Divorced
        """
        demographics = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04]),
            'age': np.clip(np.random.normal(35, 12, n).astype(int), 18, 75),
            'location_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], n, p=[0.35, 0.40, 0.25]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.35, 0.55, 0.10])
        })
        return demographics
    
    def generate_account_features(self, n: int) -> pd.DataFrame:
        """
        Generate account-related features.
        
        Features:
        - tenure_months: Account age in months (1-60)
        - membership_type: Basic/Silver/Gold/Platinum
        - preferred_payment: Credit/Debit/UPI/COD/Wallet
        - preferred_device: Mobile/Desktop/Tablet
        """
        # Tenure follows exponential distribution (more newer customers)
        tenure = np.clip(np.random.exponential(18, n).astype(int), 1, 60)
        
        # Membership correlates with tenure
        membership_probs = []
        for t in tenure:
            if t < 6:
                membership_probs.append([0.70, 0.20, 0.08, 0.02])
            elif t < 18:
                membership_probs.append([0.40, 0.35, 0.20, 0.05])
            elif t < 36:
                membership_probs.append([0.20, 0.35, 0.30, 0.15])
            else:
                membership_probs.append([0.10, 0.25, 0.35, 0.30])
        
        memberships = [np.random.choice(['Basic', 'Silver', 'Gold', 'Platinum'], p=probs) 
                       for probs in membership_probs]
        
        account = pd.DataFrame({
            'tenure_months': tenure,
            'membership_type': memberships,
            'preferred_payment': np.random.choice(
                ['Credit Card', 'Debit Card', 'UPI', 'COD', 'Wallet'], 
                n, p=[0.25, 0.20, 0.30, 0.15, 0.10]
            ),
            'preferred_device': np.random.choice(
                ['Mobile', 'Desktop', 'Tablet'], 
                n, p=[0.65, 0.28, 0.07]
            )
        })
        return account
    
    def generate_behavioral_features(self, n: int, tenure: np.ndarray) -> pd.DataFrame:
        """
        Generate behavioral features that correlate with churn.
        
        Features:
        - total_orders: Total orders placed
        - monthly_spend: Average monthly spending
        - avg_order_value: Average value per order
        - days_since_last_order: Days since last purchase
        - login_frequency: Monthly logins
        - products_viewed: Products browsed per session
        - cart_abandonment_rate: % of abandoned carts
        - support_tickets: Support tickets raised
        - discount_usage_rate: % orders with discount
        - review_count: Number of reviews written
        """
        # Base behavioral patterns
        base_orders = np.clip(tenure * np.random.uniform(0.5, 2, n), 1, 200).astype(int)
        base_spend = np.clip(np.random.lognormal(6, 0.8, n), 50, 5000)
        
        behavioral = pd.DataFrame({
            'total_orders': base_orders,
            'monthly_spend': np.round(base_spend, 2),
            'avg_order_value': np.round(base_spend / np.clip(base_orders / tenure, 0.5, 10), 2),
            'days_since_last_order': np.clip(np.random.exponential(30, n).astype(int), 0, 180),
            'login_frequency': np.clip(np.random.poisson(8, n), 0, 50),
            'products_viewed': np.clip(np.random.poisson(15, n), 1, 100),
            'cart_abandonment_rate': np.clip(np.random.beta(2, 5, n), 0, 0.9),
            'support_tickets': np.clip(np.random.exponential(1.5, n).astype(int), 0, 20),
            'discount_usage_rate': np.clip(np.random.beta(3, 4, n), 0, 1),
            'review_count': np.clip(np.random.poisson(2, n), 0, 30)
        })
        
        # Round rates to 2 decimal places
        behavioral['cart_abandonment_rate'] = np.round(behavioral['cart_abandonment_rate'], 2)
        behavioral['discount_usage_rate'] = np.round(behavioral['discount_usage_rate'], 2)
        
        return behavioral
    
    def generate_engagement_features(self, n: int) -> pd.DataFrame:
        """
        Generate customer engagement features.
        
        Features:
        - email_open_rate: % of marketing emails opened
        - push_notification_enabled: Whether push notifications are on
        - newsletter_subscribed: Newsletter subscription status
        - app_installed: Mobile app installation status
        - wishlist_items: Number of items in wishlist
        - referral_count: Number of referrals made
        """
        engagement = pd.DataFrame({
            'email_open_rate': np.round(np.clip(np.random.beta(2, 3, n), 0, 1), 2),
            'push_notification_enabled': np.random.choice([0, 1], n, p=[0.35, 0.65]),
            'newsletter_subscribed': np.random.choice([0, 1], n, p=[0.40, 0.60]),
            'app_installed': np.random.choice([0, 1], n, p=[0.30, 0.70]),
            'wishlist_items': np.clip(np.random.poisson(5, n), 0, 50),
            'referral_count': np.clip(np.random.poisson(0.5, n), 0, 10)
        })
        return engagement
    
    def generate_satisfaction_features(self, n: int) -> pd.DataFrame:
        """
        Generate customer satisfaction features.
        
        Features:
        - satisfaction_score: Last NPS score (1-10)
        - complaint_count: Number of complaints filed
        - return_rate: % of orders returned
        - avg_delivery_rating: Average delivery rating
        """
        satisfaction = pd.DataFrame({
            'satisfaction_score': np.clip(np.random.normal(7.5, 1.5, n), 1, 10).astype(int),
            'complaint_count': np.clip(np.random.exponential(0.8, n).astype(int), 0, 10),
            'return_rate': np.round(np.clip(np.random.beta(1.5, 10, n), 0, 0.5), 2),
            'avg_delivery_rating': np.round(np.clip(np.random.normal(4.2, 0.6, n), 1, 5), 1)
        })
        return satisfaction
    
    def calculate_churn_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate churn probability based on customer features.
        
        Churn is more likely with:
        - Short tenure
        - Low login frequency
        - High days since last order
        - High cart abandonment
        - High support tickets
        - Low satisfaction score
        - High complaint count
        - Low email engagement
        """
        # Normalize features for probability calculation
        churn_score = np.zeros(len(df))
        
        # Tenure: shorter tenure → higher churn
        tenure_score = 1 - (df['tenure_months'] / 60)
        churn_score += tenure_score * 0.15
        
        # Days since last order: more days → higher churn
        recency_score = df['days_since_last_order'] / 180
        churn_score += recency_score * 0.20
        
        # Login frequency: lower logins → higher churn
        login_score = 1 - (df['login_frequency'] / 50)
        churn_score += login_score * 0.12
        
        # Cart abandonment: higher → higher churn
        cart_score = df['cart_abandonment_rate']
        churn_score += cart_score * 0.10
        
        # Support tickets: more tickets → higher churn
        support_score = df['support_tickets'] / 20
        churn_score += support_score * 0.15
        
        # Satisfaction: lower → higher churn
        satisfaction_score = 1 - (df['satisfaction_score'] / 10)
        churn_score += satisfaction_score * 0.15
        
        # Complaint count: more complaints → higher churn
        complaint_score = df['complaint_count'] / 10
        churn_score += complaint_score * 0.08
        
        # Email engagement: lower → higher churn
        email_score = 1 - df['email_open_rate']
        churn_score += email_score * 0.05
        
        # Add some randomness
        noise = np.random.normal(0, 0.1, len(df))
        churn_score = np.clip(churn_score + noise, 0, 1)
        
        return churn_score
    
    def generate_churn_labels(self, churn_probabilities: np.ndarray) -> np.ndarray:
        """
        Generate binary churn labels based on probabilities.
        
        Adjusts threshold to achieve target churn rate.
        """
        # Sort probabilities and find threshold for target churn rate
        sorted_probs = np.sort(churn_probabilities)[::-1]
        threshold_idx = int(len(sorted_probs) * self.churn_rate)
        threshold = sorted_probs[threshold_idx]
        
        # Generate labels
        labels = (churn_probabilities >= threshold).astype(int)
        
        # Add some randomness to avoid perfect correlation
        random_flip = np.random.random(len(labels)) < 0.05
        labels[random_flip] = 1 - labels[random_flip]
        
        return labels
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete customer dataset.
        
        Returns:
            pd.DataFrame: Complete customer dataset with all features and churn labels
        """
        logger.info(f"Generating dataset with {self.num_customers} customers...")
        
        n = self.num_customers
        
        # Generate customer IDs
        customer_ids = self.generate_customer_ids()
        
        # Generate all feature groups
        demographics = self.generate_demographics(n)
        account = self.generate_account_features(n)
        behavioral = self.generate_behavioral_features(n, account['tenure_months'].values)
        engagement = self.generate_engagement_features(n)
        satisfaction = self.generate_satisfaction_features(n)
        
        # Combine all features
        df = pd.DataFrame({'customer_id': customer_ids})
        df = pd.concat([df, demographics, account, behavioral, engagement, satisfaction], axis=1)
        
        # Calculate churn probability and generate labels
        churn_prob = self.calculate_churn_probability(df)
        df['churn_probability'] = np.round(churn_prob, 3)
        df['churn'] = self.generate_churn_labels(churn_prob)
        
        # Add timestamps
        df['created_at'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(
            np.random.randint(0, 730, n), unit='D'
        )
        df['last_updated'] = pd.to_datetime('2024-12-31')
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Churn rate: {df['churn'].mean():.2%}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "customers.csv") -> str:
        """
        Save generated dataset to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        # Create output directory
        output_dir = Path(self.config['data']['raw_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to: {output_path}")
        return str(output_path)
    
    def generate_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a smaller sample dataset for testing."""
        original_n = self.num_customers
        self.num_customers = n_samples
        sample_df = self.generate_dataset()
        self.num_customers = original_n
        return sample_df


def main():
    """Main function to generate and save customer data."""
    print("=" * 60)
    print("[*] Customer Churn Data Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = CustomerDataGenerator()
    
    # Generate full dataset
    print("\n[1/4] Generating customer dataset...")
    df = generator.generate_dataset()
    
    # Display sample
    print("\n[INFO] Sample Data (first 5 rows):")
    print(df.head().to_string())
    
    # Display statistics
    print("\n[INFO] Dataset Statistics:")
    print(f"  - Total Customers: {len(df):,}")
    print(f"  - Churned Customers: {df['churn'].sum():,} ({df['churn'].mean():.1%})")
    print(f"  - Active Customers: {len(df) - df['churn'].sum():,} ({1-df['churn'].mean():.1%})")
    print(f"  - Features: {len(df.columns) - 4}")  # Exclude ID, churn, timestamps
    
    # Save dataset
    print("\n[2/4] Saving dataset...")
    output_path = generator.save_dataset(df)
    
    # Generate and save sample dataset
    print("\n[3/4] Generating sample dataset (1000 customers)...")
    sample_df = generator.generate_sample_dataset(1000)
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_dir / "customers_sample.csv", index=False)
    
    print("\n[DONE] Data generation complete!")
    print(f"  - Full dataset: {output_path}")
    print(f"  - Sample dataset: data/sample/customers_sample.csv")


if __name__ == "__main__":
    main()
