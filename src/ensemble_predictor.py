import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(self, models: Dict, weights: Optional[Dict] = None, method: str = 'weighted_average'):
        self.models = models
        self.weights = weights or {}
        self.method = method
        self.auto_weights = {}
        
    def calculate_model_performance(self, data: np.ndarray, train_size: float = 0.8):
        split_idx = int(len(data) * train_size)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        performances = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'train'):
                    model.train(train_data)
                
                predictions = []
                for i in range(len(test_data)):
                    recent_data = np.concatenate([train_data, test_data[:i]]) if i > 0 else train_data
                    pred = model.predict_next(recent_data, steps=1)[0]
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                mse = mean_squared_error(test_data, predictions)
                mae = mean_absolute_error(test_data, predictions)
                
                performances[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'score': 1 / (1 + mse)
                }
                
                logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                performances[model_name] = {'mse': float('inf'), 'mae': float('inf'), 'score': 0}
        
        return performances
    
    def calculate_auto_weights(self, data: np.ndarray):
        performances = self.calculate_model_performance(data)
        
        total_score = sum(p['score'] for p in performances.values())
        
        if total_score > 0:
            self.auto_weights = {
                name: perf['score'] / total_score 
                for name, perf in performances.items()
            }
        else:
            n_models = len(self.models)
            self.auto_weights = {name: 1/n_models for name in self.models.keys()}
        
        logger.info(f"Auto-calculated weights: {self.auto_weights}")
        return self.auto_weights
    
    def predict_next(self, data: np.ndarray, steps: int = 1, use_auto_weights: bool = True) -> Dict:
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_next(data, steps=steps)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                predictions[model_name] = np.array([data[-1]] * steps)
        
        if use_auto_weights and self.auto_weights:
            weights = self.auto_weights
        else:
            weights = self.weights if self.weights else {name: 1/len(self.models) for name in self.models.keys()}
        
        if self.method == 'weighted_average':
            ensemble_pred = self._weighted_average(predictions, weights)
        elif self.method == 'median':
            ensemble_pred = self._median(predictions)
        elif self.method == 'trimmed_mean':
            ensemble_pred = self._trimmed_mean(predictions)
        else:
            ensemble_pred = self._simple_average(predictions)
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions,
            'weights': weights
        }
    
    def _weighted_average(self, predictions: Dict, weights: Dict) -> np.ndarray:
        weighted_sum = None
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1/len(predictions))
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    def _simple_average(self, predictions: Dict) -> np.ndarray:
        return np.mean(list(predictions.values()), axis=0)
    
    def _median(self, predictions: Dict) -> np.ndarray:
        return np.median(list(predictions.values()), axis=0)
    
    def _trimmed_mean(self, predictions: Dict, trim_percent: float = 0.2) -> np.ndarray:
        pred_array = np.array(list(predictions.values()))
        from scipy import stats
        return stats.trim_mean(pred_array, trim_percent, axis=0)
    
    def get_prediction_statistics(self, predictions: Dict) -> Dict:
        pred_values = np.array(list(predictions.values()))
        
        return {
            'mean': np.mean(pred_values, axis=0),
            'median': np.median(pred_values, axis=0),
            'std': np.std(pred_values, axis=0),
            'min': np.min(pred_values, axis=0),
            'max': np.max(pred_values, axis=0),
            'q25': np.percentile(pred_values, 25, axis=0),
            'q75': np.percentile(pred_values, 75, axis=0)
        }
    
    def get_confidence_interval(self, data: np.ndarray, steps: int = 1, 
                               confidence: float = 0.95) -> Dict:
        result = self.predict_next(data, steps=steps)
        stats = self.get_prediction_statistics(result['individual'])
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        pred_values = np.array(list(result['individual'].values()))
        
        return {
            'prediction': result['ensemble'],
            'lower_bound': np.percentile(pred_values, lower_percentile, axis=0),
            'upper_bound': np.percentile(pred_values, upper_percentile, axis=0),
            'confidence': confidence
        }
