"""
Utility Functions for Churn Prediction System

This module provides common utilities for:
- Configuration loading
- Logging setup
- Data validation
- Metrics calculation
"""

import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )
    return logging.getLogger(__name__)


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> bool:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        raise_error: Whether to raise error on validation failure
        
    Returns:
        True if valid, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        if raise_error:
            raise ValueError(f"Missing required columns: {missing}")
        return False
    
    return True


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class DataValidator:
    """Validates customer data for prediction."""
    
    REQUIRED_FIELDS = [
        'customer_id', 'tenure_months', 'monthly_spend',
        'total_orders', 'days_since_last_order'
    ]
    
    NUMERICAL_BOUNDS = {
        'tenure_months': (0, 120),
        'monthly_spend': (0, 10000),
        'total_orders': (0, 1000),
        'days_since_last_order': (0, 365),
        'login_frequency': (0, 100),
        'satisfaction_score': (1, 10),
        'cart_abandonment_rate': (0, 1),
        'discount_usage_rate': (0, 1)
    }
    
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate customer data.
        
        Args:
            data: Customer data dictionary
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check numerical bounds
        for field, (min_val, max_val) in cls.NUMERICAL_BOUNDS.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric")
                elif value < min_val or value > max_val:
                    errors.append(f"{field} must be between {min_val} and {max_val}")
        
        return len(errors) == 0, errors


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$") -> str:
    """Format value as currency string."""
    return f"{symbol}{value:,.2f}"
