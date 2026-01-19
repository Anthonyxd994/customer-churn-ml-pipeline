"""
Load CSV data into PostgreSQL database.

This script:
1. Connects to PostgreSQL
2. Creates tables if they don't exist
3. Loads customer data from CSV files
4. Verifies the data was loaded correctly
"""

import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


def get_connection_string():
    """Get PostgreSQL connection string from environment."""
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5433')
    database = os.getenv('POSTGRES_DB', 'churn_db')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres123')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def create_tables(engine):
    """Create database tables."""
    print("[INFO] Creating tables...")
    
    create_sql = """
    -- Drop existing tables
    DROP TABLE IF EXISTS churn_predictions CASCADE;
    DROP TABLE IF EXISTS customers CASCADE;
    
    -- Customers table
    CREATE TABLE customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        gender VARCHAR(20),
        age INTEGER,
        tenure_months INTEGER,
        monthly_spend DECIMAL(10,2),
        total_orders INTEGER,
        avg_order_value DECIMAL(10,2),
        days_since_last_order INTEGER,
        login_frequency INTEGER,
        products_viewed INTEGER,
        cart_abandonment_rate DECIMAL(5,4),
        support_tickets INTEGER,
        discount_usage_rate DECIMAL(5,4),
        satisfaction_score INTEGER,
        complaint_count INTEGER,
        churn INTEGER,
        churn_probability DECIMAL(5,4),
        location_tier VARCHAR(20),
        membership_type VARCHAR(20),
        preferred_payment VARCHAR(50),
        preferred_device VARCHAR(20),
        marital_status VARCHAR(20),
        referral_count INTEGER,
        coupon_used_count INTEGER,
        created_at TIMESTAMP,
        last_updated TIMESTAMP
    );
    
    -- Churn predictions table  
    CREATE TABLE churn_predictions (
        prediction_id SERIAL PRIMARY KEY,
        customer_id VARCHAR(50) REFERENCES customers(customer_id),
        churn_probability DECIMAL(5,4),
        churn_prediction BOOLEAN,
        risk_segment VARCHAR(20),
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes
    CREATE INDEX idx_customers_churn ON customers(churn);
    CREATE INDEX idx_customers_churn_prob ON customers(churn_probability);
    CREATE INDEX idx_predictions_customer ON churn_predictions(customer_id);
    """
    
    with engine.connect() as conn:
        for statement in create_sql.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print(f"  [WARN] {e}")
        conn.commit()
    
    print("[OK] Tables created!")


def load_customers(engine, csv_path):
    """Load customer data from CSV."""
    print(f"[INFO] Loading customers from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"       Loaded {len(df)} rows from CSV")
    
    # Map columns to database schema
    column_mapping = {
        'customer_id': 'customer_id',
        'gender': 'gender',
        'age': 'age',
        'tenure_months': 'tenure_months',
        'monthly_spend': 'monthly_spend',
        'total_orders': 'total_orders',
        'avg_order_value': 'avg_order_value',
        'days_since_last_order': 'days_since_last_order',
        'login_frequency': 'login_frequency',
        'products_viewed': 'products_viewed',
        'cart_abandonment_rate': 'cart_abandonment_rate',
        'support_tickets': 'support_tickets',
        'discount_usage_rate': 'discount_usage_rate',
        'satisfaction_score': 'satisfaction_score',
        'complaint_count': 'complaint_count',
        'churn': 'churn',
        'churn_probability': 'churn_probability',
        'location_tier': 'location_tier',
        'membership_type': 'membership_type',
        'preferred_payment': 'preferred_payment',
        'preferred_device': 'preferred_device',
        'marital_status': 'marital_status',
        'referral_count': 'referral_count',
        'coupon_used_count': 'coupon_used_count',
        'created_at': 'created_at',
        'last_updated': 'last_updated'
    }
    
    # Select only columns that exist in both
    available_cols = [c for c in column_mapping.keys() if c in df.columns]
    df_to_load = df[available_cols].copy()
    
    # Load to database
    df_to_load.to_sql('customers', engine, if_exists='append', index=False)
    print(f"[OK] Loaded {len(df_to_load)} customers to database!")
    
    return len(df_to_load)


def verify_data(engine):
    """Verify data was loaded correctly."""
    print("\n[INFO] Verifying data...")
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM customers"))
        count = result.scalar()
        print(f"       Total customers in database: {count}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM customers WHERE churn = 1"))
        churned = result.scalar()
        print(f"       Churned customers: {churned}")
        
        result = conn.execute(text("SELECT AVG(churn_probability) FROM customers"))
        avg_prob = result.scalar()
        if avg_prob:
            print(f"       Average churn probability: {avg_prob:.2%}")
    
    return count


def main():
    """Main function to load data."""
    print("=" * 60)
    print("  LOAD CSV DATA TO POSTGRESQL")
    print("=" * 60)
    
    # Check for CSV file
    csv_path = Path('data/raw/customers.csv')
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        print("        Run data generation first: python -m src.data_generation.generate_data")
        return
    
    # Connect to database
    print(f"\n[INFO] Connecting to PostgreSQL...")
    try:
        connection_string = get_connection_string()
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[OK] Connected to PostgreSQL!")
        
    except Exception as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("        Run: run_postgres.bat")
        return
    
    # Create tables
    create_tables(engine)
    
    # Load data
    load_customers(engine, csv_path)
    
    # Verify
    verify_data(engine)
    
    print("\n" + "=" * 60)
    print("  DONE! Data loaded to PostgreSQL")
    print("=" * 60)
    print("\nConnection Details:")
    print(f"  Host: {os.getenv('POSTGRES_HOST', 'localhost')}")
    print(f"  Port: {os.getenv('POSTGRES_PORT', '5433')}")
    print(f"  Database: {os.getenv('POSTGRES_DB', 'churn_db')}")
    print(f"  User: {os.getenv('POSTGRES_USER', 'postgres')}")
    print()


if __name__ == "__main__":
    main()
