"""
Model Training Pipeline with MLflow Tracking

This module trains multiple ML models for churn prediction with:
- Hyperparameter tuning
- Cross-validation
- MLflow experiment tracking
- Model comparison and selection
- Best model registration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import logging
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Trains and evaluates multiple churn prediction models.
    
    Features:
    - Multiple model training (XGBoost, Random Forest, Logistic Regression)
    - Class imbalance handling with SMOTE
    - Cross-validation
    - MLflow experiment tracking
    - Model comparison and selection
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize MLflow
        self._setup_mlflow()
        
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
            'training': {
                'cv_folds': 5,
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'models': [
                    {
                        'name': 'xgboost',
                        'params': {
                            'n_estimators': 200,
                            'max_depth': 6,
                            'learning_rate': 0.1,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'scale_pos_weight': 3,
                            'random_state': 42
                        }
                    },
                    {
                        'name': 'random_forest',
                        'params': {
                            'n_estimators': 200,
                            'max_depth': 12,
                            'min_samples_split': 5,
                            'class_weight': 'balanced',
                            'random_state': 42
                        }
                    },
                    {
                        'name': 'logistic_regression',
                        'params': {
                            'C': 1.0,
                            'max_iter': 1000,
                            'class_weight': 'balanced',
                            'random_state': 42
                        }
                    }
                ]
            },
            'mlflow': {
                'experiment_name': 'churn_prediction',
                'tracking_uri': 'mlruns'
            },
            'data': {
                'processed_path': 'data/processed',
                'train_test_split': 0.2,
                'random_state': 42
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        tracking_uri = self.config['mlflow']['tracking_uri']
        experiment_name = self.config['mlflow']['experiment_name']
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed features and target.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        processed_path = Path(self.config['data']['processed_path'])
        
        X = pd.read_csv(processed_path / "processed_features.csv")
        y = pd.read_csv(processed_path / "processed_target.csv").squeeze()
        
        # Load feature names
        with open(processed_path / "processed_feature_names.txt", 'r') as f:
            self.feature_names = f.read().strip().split('\n')
        
        logger.info(f"Loaded {len(X):,} samples with {len(self.feature_names)} features")
        logger.info(f"Class distribution - Not Churned: {(y==0).sum():,}, Churned: {(y==1).sum():,}")
        
        return X, y
    
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        apply_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Steps:
        - Train/test split
        - Apply SMOTE for class imbalance
        - Scale features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            apply_smote: Whether to apply SMOTE oversampling
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.config['data']['train_test_split']
        random_state = self.config['data']['random_state']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Apply SMOTE for class imbalance
        if apply_smote:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set: {len(X_train):,} samples")
            logger.info(f"Class distribution after SMOTE: {np.bincount(y_train)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def get_model(self, model_name: str, params: Dict[str, Any]):
        """
        Get model instance by name.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Sklearn-compatible model instance
        """
        if model_name == 'xgboost':
            return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics
    
    def cross_validate(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model instance
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV metrics
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[f'cv_{metric}_mean'] = scores.mean()
            cv_results[f'cv_{metric}_std'] = scores.std()
        
        return cv_results
    
    def train_model(
        self,
        model_name: str,
        params: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model with MLflow tracking.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            X_train, X_test: Train/test features
            y_train, y_test: Train/test labels
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            
            # Get and train model
            model = self.get_model(model_name, params)
            
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            logger.info("Evaluating model...")
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_dict = dict(zip(self.feature_names, importance))
                
                # Log top features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info("\nTop 10 Features:")
                for feat, imp in top_features:
                    logger.info(f"  {feat}: {imp:.4f}")
                
                # Save feature importance
                mlflow.log_dict(importance_dict, "feature_importance.json")
            
            # Log model
            if model_name == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Log confusion matrix as artifact
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
            mlflow.log_dict(cm_dict, "confusion_matrix.json")
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
        return model, metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all configured models.
        
        Args:
            X_train, X_test: Train/test features
            y_train, y_test: Train/test labels
            
        Returns:
            Dictionary of model results
        """
        model_configs = self.config['training']['models']
        
        for model_config in model_configs:
            model_name = model_config['name']
            params = model_config['params']
            
            self.train_model(
                model_name, params,
                X_train, X_test, y_train, y_test
            )
        
        return self.results
    
    def select_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any]:
        """
        Select the best model based on a metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model name, model instance)
        """
        best_score = -1
        best_name = None
        
        for name, metrics in self.results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Best Model: {best_name.upper()}")
        logger.info(f"Best {metric}: {best_score:.4f}")
        logger.info(f"{'='*50}")
        
        return best_name, self.best_model
    
    def save_best_model(self, output_dir: str = "models/artifacts") -> str:
        """
        Save the best model and related artifacts.
        
        Args:
            output_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "best_model.joblib"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Saved model to: {model_path}")
        
        # Save scaler
        scaler_path = output_path / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to: {scaler_path}")
        
        # Save feature names
        features_path = output_path / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to: {features_path}")
        
        # Save model info
        info = {
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'n_features': len(self.feature_names),
            'trained_at': datetime.now().isoformat()
        }
        info_path = output_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved model info to: {info_path}")
        
        return str(model_path)
    
    def print_comparison(self):
        """Print comparison of all trained models."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.round(4)
        
        # Highlight best values
        print(comparison.to_string())
        
        print("\n" + "-" * 70)
        print(f"[BEST] Best Model: {self.best_model_name.upper()}")
        print("-" * 70)


def main():
    """Main function to run the training pipeline."""
    print("=" * 60)
    print("[*] Model Training Pipeline")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Load data
    print("\n[1/5] Loading processed data...")
    X, y = trainer.load_data()
    
    # Prepare data
    print("\n[2/5] Preparing data for training...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, apply_smote=True)
    
    # Train all models
    print("\n[3/5] Training models...")
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Select best model
    print("\n[4/5] Selecting best model...")
    best_name, best_model = trainer.select_best_model(metric='roc_auc')
    
    # Print comparison
    trainer.print_comparison()
    
    # Save best model
    print("\n[5/5] Saving best model...")
    model_path = trainer.save_best_model()
    
    print("\n[DONE] Training complete!")
    print(f"  - Best model: {best_name}")
    print(f"  - Model saved to: {model_path}")
    print(f"  - MLflow experiments: mlruns/")
    print("\n[INFO] To view MLflow UI, run:")
    print("  mlflow ui --port 5000")


if __name__ == "__main__":
    main()
