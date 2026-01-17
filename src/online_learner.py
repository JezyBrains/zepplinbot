#!/usr/bin/env python3
"""
Online Learning Module

Implements adaptive model updates for continuous learning:
- Incremental training after each round
- Exponential decay for older data
- Automatic weight recalibration
- Concept drift adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExponentialDecayBuffer:
    """
    Data buffer with exponential decay weighting.
    
    Recent observations have more weight than older ones.
    """
    
    def __init__(self, max_size: int = 1000, decay_rate: float = 0.995):
        """
        Args:
            max_size: Maximum buffer size
            decay_rate: Weight decay per observation (0.995 = recent 200 obs have 63% of total weight)
        """
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.data: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
    
    def add(self, value: float, timestamp: datetime = None):
        """Add new observation."""
        self.data.append(value)
        self.timestamps.append(timestamp or datetime.now())
    
    def get_weighted_data(self) -> tuple:
        """
        Get data with exponential decay weights.
        
        Returns:
            (data_array, weights_array)
        """
        data = np.array(list(self.data))
        n = len(data)
        
        # Weights: most recent = 1, older = decay^age
        weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
        weights = weights / np.sum(weights)  # Normalize
        
        return data, weights
    
    def weighted_mean(self) -> float:
        """Calculate weighted mean."""
        data, weights = self.get_weighted_data()
        return np.sum(data * weights)
    
    def weighted_std(self) -> float:
        """Calculate weighted standard deviation."""
        data, weights = self.get_weighted_data()
        mean = np.sum(data * weights)
        variance = np.sum(weights * (data - mean) ** 2)
        return np.sqrt(variance)
    
    def __len__(self) -> int:
        return len(self.data)


class OnlineStatistics:
    """
    Online (streaming) computation of statistics.
    
    Uses Welford's algorithm for numerical stability.
    """
    
    def __init__(self, decay_factor: float = 0.01):
        """
        Args:
            decay_factor: Learning rate for exponential moving average
        """
        self.decay_factor = decay_factor
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences
        self.ema = 0.0  # Exponential moving average
    
    def update(self, value: float):
        """Update statistics with new value (Welford's algorithm)."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Exponential moving average
        if self.count == 1:
            self.ema = value
        else:
            self.ema = self.decay_factor * value + (1 - self.decay_factor) * self.ema
    
    def variance(self) -> float:
        """Get current variance estimate."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)
    
    def std(self) -> float:
        """Get current standard deviation."""
        return np.sqrt(self.variance())
    
    def get_stats(self) -> Dict:
        """Get all statistics."""
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.variance(),
            'std': self.std(),
            'ema': self.ema
        }


class OnlineKellyOptimizer:
    """
    Online Kelly Criterion optimizer.
    
    Continuously updates probability estimates and optimal bet sizes.
    """
    
    def __init__(self,
                 targets: List[float] = None,
                 decay_rate: float = 0.99,
                 min_samples: int = 50,
                 fractional_kelly: float = 0.25):
        """
        Args:
            targets: Target multipliers to track
            decay_rate: Exponential decay for probability estimates
            min_samples: Minimum samples before betting
            fractional_kelly: Kelly fraction for safety
        """
        self.targets = targets or [1.5, 2.0, 2.5, 3.0]
        self.decay_rate = decay_rate
        self.min_samples = min_samples
        self.fractional_kelly = fractional_kelly
        
        # Track exponentially weighted win rates for each target
        self.win_ema = {t: 0.5 for t in self.targets}  # Start at 50%
        self.n_observations = 0
    
    def update(self, crash_value: float):
        """
        Update probability estimates with new crash observation.
        
        Args:
            crash_value: New crash point
        """
        self.n_observations += 1
        
        for target in self.targets:
            won = 1.0 if crash_value >= target else 0.0
            # Exponential moving average update
            self.win_ema[target] = self.decay_rate * self.win_ema[target] + (1 - self.decay_rate) * won
    
    def get_best_bet(self, bankroll: float) -> Dict:
        """
        Get optimal bet based on current estimates.
        
        Args:
            bankroll: Current bankroll
        
        Returns:
            Bet recommendation
        """
        if self.n_observations < self.min_samples:
            return {
                'action': 'wait',
                'reason': f'Need {self.min_samples - self.n_observations} more observations'
            }
        
        best_ev = -float('inf')
        best_target = None
        best_kelly = 0
        best_prob = 0
        
        for target in self.targets:
            p = self.win_ema[target] * 0.99  # 1% house edge adjustment
            q = 1 - p
            b = target - 1
            
            kelly = (b * p - q) / b if b > 0 else 0
            ev = (p * b) - q
            
            if ev > best_ev and kelly > 0:
                best_ev = ev
                best_target = target
                best_kelly = kelly
                best_prob = p
        
        if best_target is None or best_kelly < 0.01:
            return {
                'action': 'skip',
                'reason': 'No positive EV opportunity',
                'probabilities': self.win_ema.copy()
            }
        
        safe_kelly = min(best_kelly * self.fractional_kelly, 0.10)
        bet_amount = bankroll * safe_kelly
        
        return {
            'action': 'bet',
            'target': best_target,
            'bet_amount': bet_amount,
            'kelly_fraction': safe_kelly * 100,
            'win_probability': best_prob * 100,
            'expected_value': best_ev,
            'probabilities': self.win_ema.copy(),
            'n_observations': self.n_observations
        }


class OnlineLearner:
    """
    Complete online learning system.
    
    Combines:
    - Exponential decay buffer for data
    - Online statistics
    - Adaptive Kelly optimization
    - Automatic weight recalibration
    """
    
    def __init__(self,
                 buffer_size: int = 500,
                 decay_rate: float = 0.995,
                 recalibrate_interval: int = 50):
        """
        Args:
            buffer_size: Size of data buffer
            decay_rate: Exponential decay rate
            recalibrate_interval: Recalibrate weights every N observations
        """
        self.buffer = ExponentialDecayBuffer(buffer_size, decay_rate)
        self.stats = OnlineStatistics(decay_factor=0.02)
        self.kelly = OnlineKellyOptimizer(decay_rate=decay_rate)
        self.recalibrate_interval = recalibrate_interval
        self.observation_count = 0
        
        # Model weights (for ensemble)
        self.model_weights = {}
        self.model_errors: Dict[str, deque] = {}
    
    def observe(self, value: float, timestamp: datetime = None):
        """
        Process new observation.
        
        Args:
            value: New crash value
            timestamp: Observation time
        """
        self.buffer.add(value, timestamp)
        self.stats.update(value)
        self.kelly.update(value)
        self.observation_count += 1
        
        # Periodic recalibration
        if self.observation_count % self.recalibrate_interval == 0:
            self._recalibrate()
    
    def _recalibrate(self):
        """Recalibrate model weights based on recent performance."""
        if not self.model_errors:
            return
        
        # Calculate inverse error weights
        weights = {}
        for model, errors in self.model_errors.items():
            if len(errors) > 0:
                mae = np.mean([abs(e) for e in errors])
                weights[model] = 1 / (mae + 0.01)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            self.model_weights = {m: w / total for m, w in weights.items()}
            logger.info(f"Recalibrated weights: {self.model_weights}")
    
    def record_model_prediction(self, model_name: str, prediction: float, actual: float):
        """
        Record a model's prediction error.
        
        Args:
            model_name: Model identifier
            prediction: What model predicted
            actual: Actual outcome
        """
        if model_name not in self.model_errors:
            self.model_errors[model_name] = deque(maxlen=100)
        
        error = prediction - actual
        self.model_errors[model_name].append(error)
    
    def get_recommendation(self, bankroll: float) -> Dict:
        """Get current betting recommendation."""
        kelly_rec = self.kelly.get_best_bet(bankroll)
        
        # Add context
        kelly_rec['buffer_stats'] = {
            'weighted_mean': self.buffer.weighted_mean() if len(self.buffer) > 0 else 0,
            'weighted_std': self.buffer.weighted_std() if len(self.buffer) > 0 else 0,
            'buffer_size': len(self.buffer)
        }
        kelly_rec['online_stats'] = self.stats.get_stats()
        kelly_rec['model_weights'] = self.model_weights
        
        return kelly_rec
    
    def get_weighted_ensemble_prediction(self, 
                                         model_predictions: Dict[str, float]) -> float:
        """
        Get ensemble prediction using adaptive weights.
        
        Args:
            model_predictions: Dict of model_name -> prediction
        
        Returns:
            Weighted prediction
        """
        if not self.model_weights:
            # Equal weights if not calibrated
            return np.mean(list(model_predictions.values()))
        
        weighted_sum = 0
        total_weight = 0
        
        for model, pred in model_predictions.items():
            weight = self.model_weights.get(model, 0.1)
            weighted_sum += pred * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(model_predictions.values()))
    
    def save_state(self, filepath: str):
        """Save learner state."""
        state = {
            'buffer_data': list(self.buffer.data),
            'stats': self.stats.get_stats(),
            'kelly_win_ema': self.kelly.win_ema,
            'kelly_n_obs': self.kelly.n_observations,
            'model_weights': self.model_weights,
            'observation_count': self.observation_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load learner state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        for val in state['buffer_data']:
            self.buffer.data.append(val)
        
        self.kelly.win_ema = state['kelly_win_ema']
        self.kelly.n_observations = state['kelly_n_obs']
        self.model_weights = state['model_weights']
        self.observation_count = state['observation_count']
        
        logger.info(f"State loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("ONLINE LEARNING MODULE")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Create online learner
    learner = OnlineLearner(buffer_size=200, decay_rate=0.995)
    
    # Simulate streaming data
    print("\n📊 Processing crash data stream...")
    
    for i, value in enumerate(data):
        learner.observe(value)
        
        if (i + 1) % 100 == 0:
            rec = learner.get_recommendation(bankroll=100)
            print(f"\n[After {i+1} observations]")
            print(f"   Action: {rec['action'].upper()}")
            if rec['action'] == 'bet':
                print(f"   Target: {rec['target']}x")
                print(f"   Bet: ${rec['bet_amount']:.2f}")
                print(f"   Win Prob: {rec['win_probability']:.1f}%")
            print(f"   Buffer Mean: {rec['buffer_stats']['weighted_mean']:.2f}")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    
    final_rec = learner.get_recommendation(100)
    
    print(f"\n📈 Online Statistics:")
    stats = final_rec['online_stats']
    print(f"   Total Observations: {stats['count']}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std: {stats['std']:.2f}")
    print(f"   EMA: {stats['ema']:.2f}")
    
    print(f"\n🎯 Win Probabilities (EMA):")
    for target, prob in final_rec['probabilities'].items():
        print(f"   {target}x: {prob*100:.1f}%")
    
    print(f"\n💰 Recommendation:")
    print(f"   Action: {final_rec['action'].upper()}")
    if final_rec['action'] == 'bet':
        print(f"   Target: {final_rec['target']}x")
        print(f"   Bet Amount: ${final_rec['bet_amount']:.2f}")
        print(f"   Kelly Fraction: {final_rec['kelly_fraction']:.1f}%")
    
    # Save state
    learner.save_state('models/saved/online_learner_state.pkl')
    print("\n✅ State saved to models/saved/online_learner_state.pkl")
