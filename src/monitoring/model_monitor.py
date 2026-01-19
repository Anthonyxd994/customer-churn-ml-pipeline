"""
Model Monitoring Module

This module provides:
- Prediction logging
- Model performance tracking
- Data drift detection
- Alerts for high-risk customers
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionLogger:
    """Log predictions for monitoring and analysis."""
    
    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []
        
    def log_prediction(self, 
                      customer_id: str,
                      features: Dict,
                      prediction: float,
                      risk_segment: str,
                      model_version: str = "v1.0"):
        """Log a single prediction."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "prediction": prediction,
            "risk_segment": risk_segment,
            "model_version": model_version,
            "features": features
        }
        
        self.predictions.append(log_entry)
        
        # Save every 100 predictions
        if len(self.predictions) >= 100:
            self.flush()
            
    def flush(self):
        """Write predictions to disk."""
        if not self.predictions:
            return
            
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            for entry in self.predictions:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Saved {len(self.predictions)} predictions to {filepath}")
        self.predictions = []
        
    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """Get predictions from the last N hours."""
        all_predictions = []
        
        for file in self.log_dir.glob("predictions_*.jsonl"):
            with open(file, 'r') as f:
                for line in f:
                    all_predictions.append(json.loads(line))
        
        if not all_predictions:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        return df[df['timestamp'] >= cutoff]


class ModelMonitor:
    """Monitor model performance and detect issues."""
    
    def __init__(self, baseline_metrics: Optional[Dict] = None):
        self.baseline_metrics = baseline_metrics or {
            'accuracy': 0.71,
            'precision': 0.49,
            'recall': 0.24,
            'f1': 0.32,
            'roc_auc': 0.67
        }
        self.alerts = []
        
    def check_prediction_distribution(self, 
                                      predictions: List[float],
                                      threshold: float = 0.1) -> Dict:
        """
        Check if prediction distribution has shifted.
        
        Args:
            predictions: List of recent predictions
            threshold: Alert if mean shift > threshold
        """
        if not predictions:
            return {"status": "no_data"}
            
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Expected baseline: ~35% churn probability
        baseline_mean = 0.35
        
        shift = abs(mean_pred - baseline_mean)
        
        result = {
            "status": "ok" if shift < threshold else "drift_detected",
            "current_mean": mean_pred,
            "baseline_mean": baseline_mean,
            "shift": shift,
            "std": std_pred,
            "sample_size": len(predictions)
        }
        
        if result["status"] == "drift_detected":
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "type": "distribution_drift",
                "details": result
            })
            logger.warning(f"Distribution drift detected: {shift:.2%}")
            
        return result
    
    def check_high_risk_rate(self, 
                            predictions: List[float],
                            threshold: float = 0.6,
                            alert_if_above: float = 0.3) -> Dict:
        """
        Check if unusually high number of predictions are high-risk.
        """
        if not predictions:
            return {"status": "no_data"}
            
        high_risk_rate = np.mean([p > threshold for p in predictions])
        
        result = {
            "status": "ok" if high_risk_rate < alert_if_above else "alert",
            "high_risk_rate": high_risk_rate,
            "threshold": threshold,
            "alert_threshold": alert_if_above,
            "sample_size": len(predictions)
        }
        
        if result["status"] == "alert":
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "type": "high_risk_spike",
                "details": result
            })
            logger.warning(f"High risk rate alert: {high_risk_rate:.2%}")
            
        return result
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        return [
            a for a in self.alerts 
            if datetime.fromisoformat(a["timestamp"]) >= cutoff
        ]
    
    def get_health_status(self, predictions: List[float]) -> Dict:
        """Get overall model health status."""
        dist_check = self.check_prediction_distribution(predictions)
        risk_check = self.check_high_risk_rate(predictions)
        
        issues = []
        if dist_check.get("status") == "drift_detected":
            issues.append("Distribution drift detected")
        if risk_check.get("status") == "alert":
            issues.append("Unusual high-risk rate")
            
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "distribution_check": dist_check,
            "high_risk_check": risk_check,
            "recent_alerts": len(self.get_alerts(24)),
            "timestamp": datetime.now().isoformat()
        }


class HighRiskAlertManager:
    """Manage alerts for high-risk customers."""
    
    def __init__(self, risk_threshold: float = 0.7):
        self.risk_threshold = risk_threshold
        self.pending_alerts = []
        
    def check_customer(self, 
                      customer_id: str,
                      churn_probability: float,
                      customer_value: float = 0) -> Optional[Dict]:
        """
        Check if customer should trigger an alert.
        
        Returns alert dict if high-risk, None otherwise.
        """
        if churn_probability < self.risk_threshold:
            return None
            
        alert = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "churn_probability": churn_probability,
            "customer_value": customer_value,
            "priority": self._calculate_priority(churn_probability, customer_value),
            "status": "pending",
            "recommended_actions": self._get_recommendations(churn_probability)
        }
        
        self.pending_alerts.append(alert)
        logger.info(f"High-risk alert for customer {customer_id}: {churn_probability:.1%}")
        
        return alert
    
    def _calculate_priority(self, probability: float, value: float) -> str:
        """Calculate alert priority based on risk and value."""
        if probability > 0.9 or (probability > 0.7 and value > 500):
            return "critical"
        elif probability > 0.8 or value > 300:
            return "high"
        else:
            return "medium"
    
    def _get_recommendations(self, probability: float) -> List[str]:
        """Get recommended actions based on churn probability."""
        actions = []
        
        if probability > 0.8:
            actions.extend([
                "Immediate personal outreach from account manager",
                "Offer exclusive loyalty discount (20%+)",
                "Schedule a feedback call"
            ])
        elif probability > 0.6:
            actions.extend([
                "Send personalized re-engagement email",
                "Offer targeted discount or free shipping",
                "Add to retention campaign"
            ])
        else:
            actions.extend([
                "Include in automated nurture sequence",
                "Send product recommendations",
                "Request feedback survey"
            ])
            
        return actions
    
    def get_pending_alerts(self, priority: Optional[str] = None) -> List[Dict]:
        """Get pending alerts, optionally filtered by priority."""
        if priority:
            return [a for a in self.pending_alerts if a["priority"] == priority]
        return self.pending_alerts
    
    def acknowledge_alert(self, customer_id: str):
        """Mark an alert as acknowledged."""
        for alert in self.pending_alerts:
            if alert["customer_id"] == customer_id:
                alert["status"] = "acknowledged"
                alert["acknowledged_at"] = datetime.now().isoformat()
                logger.info(f"Alert acknowledged for customer {customer_id}")
                break


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL MONITORING DEMO")
    print("=" * 60)
    
    # Initialize components
    pred_logger = PredictionLogger()
    monitor = ModelMonitor()
    alert_manager = HighRiskAlertManager()
    
    # Simulate some predictions
    np.random.seed(42)
    sample_predictions = np.random.beta(2, 4, 100).tolist()  # Skewed distribution
    
    print("\n[TEST] Prediction Distribution Check:")
    dist_result = monitor.check_prediction_distribution(sample_predictions)
    print(f"  Status: {dist_result['status']}")
    print(f"  Current Mean: {dist_result['current_mean']:.2%}")
    print(f"  Shift: {dist_result['shift']:.2%}")
    
    print("\n[TEST] High Risk Rate Check:")
    risk_result = monitor.check_high_risk_rate(sample_predictions)
    print(f"  Status: {risk_result['status']}")
    print(f"  High Risk Rate: {risk_result['high_risk_rate']:.2%}")
    
    print("\n[TEST] Overall Health Status:")
    health = monitor.get_health_status(sample_predictions)
    print(f"  Healthy: {health['healthy']}")
    print(f"  Issues: {health['issues']}")
    
    print("\n[TEST] High Risk Customer Alert:")
    alert = alert_manager.check_customer("CUST_001", 0.85, customer_value=250)
    if alert:
        print(f"  Customer: {alert['customer_id']}")
        print(f"  Priority: {alert['priority']}")
        print(f"  Recommendations:")
        for rec in alert['recommended_actions']:
            print(f"    - {rec}")
    
    print("\n" + "=" * 60)
    print("  MONITORING MODULE READY")
    print("=" * 60)
