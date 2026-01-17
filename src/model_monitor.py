#!/usr/bin/env python3
"""
Model Performance Monitor

Tracks model performance over time and detects:
- Concept drift (distribution shift)
- Performance degradation
- Model staleness

Alerts when models need retraining.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime
from scipy import stats
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects distribution drift using multiple statistical tests.
    """
    
    def __init__(self, reference_window: int = 200, test_window: int = 50):
        """
        Args:
            reference_window: Size of reference distribution
            test_window: Size of recent window to compare
        """
        self.reference_window = reference_window
        self.test_window = test_window
        self.reference_data: Optional[np.ndarray] = None
        self.alerts: List[Dict] = []
    
    def set_reference(self, data: np.ndarray):
        """Set reference distribution."""
        self.reference_data = data[-self.reference_window:] if len(data) > self.reference_window else data.copy()
        logger.info(f"Reference distribution set with {len(self.reference_data)} samples")
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                      n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        # Create bins from reference
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize
        ref_pct = (ref_hist + 0.001) / len(reference)
        cur_pct = (cur_hist + 0.001) / len(current)
        
        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return psi
    
    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """
        Kolmogorov-Smirnov test for distribution difference.
        
        Returns:
            Dict with statistic and p-value
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_different': p_value < 0.05
        }
    
    def detect_drift(self, current_data: np.ndarray) -> Dict:
        """
        Detect if drift has occurred.
        
        Args:
            current_data: Recent data to test
        
        Returns:
            Dict with drift detection results
        """
        if self.reference_data is None:
            return {'error': 'Reference not set'}
        
        current = current_data[-self.test_window:] if len(current_data) > self.test_window else current_data
        
        # Calculate metrics
        psi = self.calculate_psi(self.reference_data, current)
        ks = self.ks_test(self.reference_data, current)
        
        # Mean shift
        ref_mean = np.mean(self.reference_data)
        cur_mean = np.mean(current)
        mean_shift = (cur_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
        
        # Variance ratio
        ref_var = np.var(self.reference_data)
        cur_var = np.var(current)
        var_ratio = cur_var / ref_var if ref_var > 0 else 1
        
        # Overall drift decision
        drift_detected = psi > 0.25 or ks['is_different'] or abs(mean_shift) > 0.3
        severity = 'high' if psi > 0.25 else 'medium' if psi > 0.1 else 'low'
        
        result = {
            'drift_detected': drift_detected,
            'severity': severity,
            'psi': psi,
            'ks_test': ks,
            'mean_shift': mean_shift,
            'var_ratio': var_ratio,
            'reference_mean': ref_mean,
            'current_mean': cur_mean,
            'recommendation': self._get_recommendation(drift_detected, severity)
        }
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'drift',
                'severity': severity,
                'details': result
            }
            self.alerts.append(alert)
            logger.warning(f"Drift detected! Severity: {severity}, PSI: {psi:.4f}")
        
        return result
    
    def _get_recommendation(self, drift_detected: bool, severity: str) -> str:
        """Generate recommendation based on drift."""
        if not drift_detected:
            return "No action needed - distribution stable"
        elif severity == 'high':
            return "⚠️ RETRAIN REQUIRED - Significant distribution shift detected"
        else:
            return "📊 MONITOR - Moderate shift, consider retraining soon"


class PerformanceTracker:
    """
    Tracks rolling model performance metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.metrics_history: List[Dict] = []
    
    def record(self, prediction: float, actual: float, timestamp: datetime = None):
        """Record a prediction-actual pair."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
    
    def calculate_metrics(self) -> Dict:
        """Calculate current performance metrics."""
        if len(self.predictions) < 10:
            return {'error': 'Insufficient data'}
        
        preds = np.array(list(self.predictions))
        acts = np.array(list(self.actuals))
        
        # Error metrics
        errors = preds - acts
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(abs_errors / (np.abs(acts) + 1e-10)) * 100
        
        # Direction accuracy (for binary signals)
        if np.all(np.isin(preds, [0, 1])):
            accuracy = np.mean(preds == acts)
        else:
            # For continuous predictions, use correlation
            correlation = np.corrcoef(preds, acts)[0, 1] if len(preds) > 1 else 0
            accuracy = correlation
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy': accuracy,
            'n_samples': len(preds),
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def detect_degradation(self, baseline_mae: float = None) -> Dict:
        """
        Detect if performance has degraded.
        
        Args:
            baseline_mae: Expected MAE (uses first recorded if None)
        
        Returns:
            Degradation analysis
        """
        current = self.calculate_metrics()
        if 'error' in current:
            return current
        
        if baseline_mae is None and len(self.metrics_history) > 10:
            baseline_mae = np.mean([m['mae'] for m in self.metrics_history[:10]])
        
        if baseline_mae is None:
            return {'error': 'No baseline available'}
        
        current_mae = current['mae']
        degradation = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0
        
        return {
            'baseline_mae': baseline_mae,
            'current_mae': current_mae,
            'degradation_pct': degradation * 100,
            'degraded': degradation > 0.2,
            'recommendation': (
                "⚠️ Performance degraded >20% - consider retraining"
                if degradation > 0.2 else
                "✅ Performance within acceptable range"
            )
        }


class ModelMonitor:
    """
    Complete model monitoring system.
    
    Combines drift detection and performance tracking.
    """
    
    def __init__(self,
                 reference_data: np.ndarray = None,
                 drift_check_interval: int = 50,
                 performance_window: int = 100):
        """
        Args:
            reference_data: Initial reference distribution
            drift_check_interval: Check for drift every N observations
            performance_window: Rolling window for performance metrics
        """
        self.drift_detector = DriftDetector()
        self.performance_tracker = PerformanceTracker(performance_window)
        self.drift_check_interval = drift_check_interval
        self.observation_count = 0
        self.alerts: List[Dict] = []
        self._callbacks: List[Callable] = []
        
        if reference_data is not None:
            self.drift_detector.set_reference(reference_data)
    
    def on_alert(self, callback: Callable[[Dict], None]):
        """Register callback for alerts."""
        self._callbacks.append(callback)
    
    def _trigger_alert(self, alert: Dict):
        """Trigger alert callbacks."""
        self.alerts.append(alert)
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def observe(self, 
                value: float,
                prediction: float = None,
                actual: float = None):
        """
        Record new observation.
        
        Args:
            value: New feature value (for drift detection)
            prediction: Model prediction (if available)
            actual: Actual outcome (if available)
        """
        self.observation_count += 1
        
        # Record prediction if available
        if prediction is not None and actual is not None:
            self.performance_tracker.record(prediction, actual)
        
        # Periodic drift check
        if self.observation_count % self.drift_check_interval == 0:
            self._check_drift(value)
    
    def _check_drift(self, recent_value: float):
        """Check for distribution drift."""
        # Would need to accumulate recent values - simplified here
        pass
    
    def check_drift(self, current_data: np.ndarray) -> Dict:
        """
        Manually check for drift.
        
        Args:
            current_data: Recent data to compare against reference
        
        Returns:
            Drift detection results
        """
        result = self.drift_detector.detect_drift(current_data)
        
        if result.get('drift_detected', False):
            alert = {
                'type': 'drift',
                'timestamp': datetime.now().isoformat(),
                'severity': result['severity'],
                'message': result['recommendation'],
                'details': result
            }
            self._trigger_alert(alert)
        
        return result
    
    def get_performance(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_tracker.calculate_metrics()
    
    def get_status(self) -> Dict:
        """
        Get complete monitor status.
        
        Returns:
            Dict with all monitoring info
        """
        perf = self.performance_tracker.calculate_metrics()
        degradation = self.performance_tracker.detect_degradation()
        
        status = 'healthy'
        if degradation.get('degraded', False):
            status = 'degraded'
        if len(self.alerts) > 0 and self.alerts[-1].get('severity') == 'high':
            status = 'critical'
        
        return {
            'status': status,
            'observation_count': self.observation_count,
            'performance': perf,
            'degradation': degradation,
            'recent_alerts': self.alerts[-5:],
            'timestamp': datetime.now().isoformat()
        }
    
    def save_report(self, filepath: str):
        """Save monitoring report to JSON."""
        report = {
            'status': self.get_status(),
            'all_alerts': self.alerts,
            'metrics_history': self.performance_tracker.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {filepath}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MONITOR")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Split into reference and current
    split = len(data) // 2
    reference = data[:split]
    current = data[split:]
    
    # Create monitor
    monitor = ModelMonitor(reference_data=reference)
    
    # Check for drift
    print("\n📊 DRIFT DETECTION")
    print("-" * 40)
    drift = monitor.check_drift(current)
    
    print(f"   PSI: {drift['psi']:.4f}")
    print(f"   KS p-value: {drift['ks_test']['p_value']:.4f}")
    print(f"   Mean Shift: {drift['mean_shift']*100:.1f}%")
    print(f"   Drift Detected: {'YES' if drift['drift_detected'] else 'NO'}")
    print(f"   {drift['recommendation']}")
    
    # Simulate some predictions
    print("\n📈 PERFORMANCE TRACKING")
    print("-" * 40)
    
    # Simulate predictions (slightly noisy actuals)
    for i in range(50):
        actual = data[split + i]
        prediction = actual + np.random.normal(0, actual * 0.1)
        monitor.performance_tracker.record(prediction, actual)
    
    perf = monitor.get_performance()
    print(f"   MAE: {perf['mae']:.4f}")
    print(f"   RMSE: {perf['rmse']:.4f}")
    print(f"   Samples: {perf['n_samples']}")
    
    # Get overall status
    print("\n🔍 OVERALL STATUS")
    print("-" * 40)
    status = monitor.get_status()
    print(f"   Status: {status['status'].upper()}")
    print(f"   Observations: {status['observation_count']}")
    
    # Save report
    monitor.save_report('outputs/monitor_report.json')
    print("\n✅ Report saved to outputs/monitor_report.json")
