"""
Database Configuration and Connection for PostgreSQL.

This module provides database connectivity for loading data from PostgreSQL
instead of CSV files.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv()


class DatabaseConfig:
    """Database configuration from environment variables."""
    
    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = os.getenv('POSTGRES_PORT', '5432')
        self.database = os.getenv('POSTGRES_DB', 'churn_db')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', 'postgres123')
    
    @property
    def connection_string(self):
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def __repr__(self):
        return f"DatabaseConfig(host={self.host}, port={self.port}, database={self.database})"


class DatabaseConnection:
    """Manage database connections."""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session = None
    
    def connect(self):
        """Create database connection."""
        try:
            self.engine = create_engine(self.config.connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            print(f"[OK] Connected to PostgreSQL: {self.config.database}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        print("[OK] Disconnected from PostgreSQL")
    
    def test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"[ERROR] Connection test failed: {e}")
            return False
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return pd.DataFrame()
    
    def load_customers(self) -> pd.DataFrame:
        """Load customer data from database."""
        query = """
        SELECT 
            customer_id,
            customer_name,
            gender,
            age,
            location_tier,
            marital_status,
            tenure_months,
            membership_type,
            preferred_payment,
            preferred_device,
            acquisition_date
        FROM customers
        """
        return self.execute_query(query)
    
    def load_transactions(self) -> pd.DataFrame:
        """Load transaction data from database."""
        query = """
        SELECT 
            customer_id,
            order_id,
            order_date,
            category,
            quantity,
            total_amount,
            profit
        FROM transactions
        """
        return self.execute_query(query)
    
    def save_predictions(self, predictions_df: pd.DataFrame, table_name: str = 'churn_predictions'):
        """Save predictions to database."""
        try:
            predictions_df.to_sql(
                table_name, 
                self.engine, 
                if_exists='replace', 
                index=False
            )
            print(f"[OK] Saved {len(predictions_df)} predictions to {table_name}")
        except Exception as e:
            print(f"[ERROR] Failed to save predictions: {e}")


class DataLoader:
    """
    Load data from CSV files or PostgreSQL database.
    
    Usage:
        # From CSV (default)
        loader = DataLoader(source='csv')
        customers, transactions = loader.load_data()
        
        # From PostgreSQL
        loader = DataLoader(source='postgres')
        customers, transactions = loader.load_data()
    """
    
    def __init__(self, source: str = 'csv'):
        """
        Initialize data loader.
        
        Args:
            source: 'csv' or 'postgres'
        """
        self.source = source
        self.db = None
        
        if source == 'postgres':
            self.db = DatabaseConnection()
            self.db.connect()
    
    def load_from_csv(self):
        """Load data from CSV files."""
        customers_path = Path('data/raw/customers.csv')
        
        if not customers_path.exists():
            raise FileNotFoundError(f"Data file not found: {customers_path}")
        
        customers = pd.read_csv(customers_path)
        
        # Parse date columns
        date_cols = ['created_at', 'last_updated', 'acquisition_date']
        for col in date_cols:
            if col in customers.columns:
                customers[col] = pd.to_datetime(customers[col])
        
        print(f"[OK] Loaded {len(customers)} customers from CSV")
        return customers
    
    def load_from_postgres(self):
        """Load data from PostgreSQL database."""
        if not self.db:
            raise ConnectionError("Database not connected")
        
        customers = self.db.load_customers()
        print(f"[OK] Loaded {len(customers)} customers from PostgreSQL")
        return customers
    
    def load_data(self):
        """Load data based on configured source."""
        if self.source == 'csv':
            return self.load_from_csv()
        elif self.source == 'postgres':
            return self.load_from_postgres()
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def create_churn_labels(self, df: pd.DataFrame, threshold_days: int = 90) -> pd.DataFrame:
        """
        Create churn labels based on recency.
        
        Churn = No purchase in last threshold_days
        """
        if 'days_since_last_order' in df.columns:
            df['is_churned'] = (df['days_since_last_order'] > threshold_days).astype(int)
        elif 'churn' in df.columns:
            df['is_churned'] = df['churn']
        else:
            # Default to random for demo
            import numpy as np
            df['is_churned'] = np.random.choice([0, 1], len(df), p=[0.74, 0.26])
        
        churn_rate = df['is_churned'].mean()
        print(f"[INFO] Churn Rate: {churn_rate:.2%}")
        print(f"       Churned: {df['is_churned'].sum()}")
        print(f"       Active: {(~df['is_churned'].astype(bool)).sum()}")
        
        return df
    
    def close(self):
        """Close database connection if open."""
        if self.db:
            self.db.disconnect()


# Initialize database tables
def init_database():
    """Initialize database tables."""
    db = DatabaseConnection()
    if not db.connect():
        return False
    
    create_tables_sql = """
    -- Customers table
    CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        customer_name VARCHAR(100),
        gender VARCHAR(20),
        age INTEGER,
        location_tier VARCHAR(20),
        marital_status VARCHAR(20),
        tenure_months INTEGER,
        membership_type VARCHAR(20),
        preferred_payment VARCHAR(50),
        preferred_device VARCHAR(20),
        acquisition_date DATE
    );
    
    -- Transactions table
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id SERIAL PRIMARY KEY,
        customer_id VARCHAR(50) REFERENCES customers(customer_id),
        order_id VARCHAR(50),
        order_date DATE,
        category VARCHAR(50),
        quantity INTEGER,
        total_amount DECIMAL(10,2),
        profit DECIMAL(10,2)
    );
    
    -- Churn predictions table
    CREATE TABLE IF NOT EXISTS churn_predictions (
        prediction_id SERIAL PRIMARY KEY,
        customer_id VARCHAR(50),
        churn_probability DECIMAL(5,4),
        churn_prediction BOOLEAN,
        risk_segment VARCHAR(20),
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        with db.engine.connect() as conn:
            for statement in create_tables_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        print("[OK] Database tables created")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create tables: {e}")
        return False
    finally:
        db.disconnect()


def load_csv_to_postgres():
    """Load CSV data into PostgreSQL database."""
    db = DatabaseConnection()
    if not db.connect():
        return False
    
    try:
        # Load customers
        customers = pd.read_csv('data/raw/customers.csv')
        
        # Select relevant columns
        customer_cols = ['customer_id', 'gender', 'age', 'location_tier', 
                        'marital_status', 'tenure_months', 'membership_type',
                        'preferred_payment', 'preferred_device']
        
        available_cols = [c for c in customer_cols if c in customers.columns]
        customers_db = customers[available_cols].copy()
        
        # Save to database
        customers_db.to_sql('customers', db.engine, if_exists='replace', index=False)
        print(f"[OK] Loaded {len(customers_db)} customers to PostgreSQL")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return False
    finally:
        db.disconnect()


if __name__ == "__main__":
    print("=" * 60)
    print("PostgreSQL Database Setup")
    print("=" * 60)
    
    # Test connection
    db = DatabaseConnection()
    if db.connect():
        if db.test_connection():
            print("[OK] Database connection successful!")
        db.disconnect()
    else:
        print("\n[INFO] To set up PostgreSQL:")
        print("  1. Install PostgreSQL or run with Docker:")
        print("     docker run --name churn-postgres -e POSTGRES_PASSWORD=postgres123 -e POSTGRES_DB=churn_db -p 5432:5432 -d postgres:15")
        print("  2. Update .env file with your credentials")
        print("  3. Run this script again")
