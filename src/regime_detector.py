#!/usr/bin/env python3
"""
Regime Detection using Hidden Markov Models

Detects market regimes (hot/cold/neutral) using HMM.
Different regimes suggest different betting strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Using simplified regime detection.")


class RegimeDetector:
    """
    Hidden Markov Model for crash game regime detection.
    
    Detects three regimes:
    - Cold: Frequent low crashes (unfavorable)
    - Neutral: Normal distribution
    - Hot: Frequent high crashes (favorable)
    
    The regime affects optimal strategy selection.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: Number of hidden states (default: 3)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False
        self.regime_labels = {0: 'cold', 1: 'neutral', 2: 'hot'}
        self.regime_means = None
        self.transition_matrix = None
        
        # For simple implementation
        self._thresholds = [1.5, 3.0]
    
    def fit(self, data: np.ndarray) -> 'RegimeDetector':
        """
        Fit HMM to crash data.
        
        Args:
            data: Array of crash multipliers
        
        Returns:
            self
        """
        # Use log-transformed data for better Gaussian fit
        log_data = np.log(data[data > 0]).reshape(-1, 1)
        
        if HMM_AVAILABLE:
            self._fit_hmm(log_data)
        else:
            self._fit_simple(data)
        
        self.fitted = True
        return self
    
    def _fit_hmm(self, log_data: np.ndarray):
        """Fit using hmmlearn."""
        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
            self.model.fit(log_data)
            
            # Sort regimes by mean (cold=lowest, hot=highest)
            means = self.model.means_.flatten()
            sorted_indices = np.argsort(means)
            
            # Remap regime labels
            self.regime_labels = {
                sorted_indices[0]: 'cold',
                sorted_indices[1]: 'neutral',
                sorted_indices[2]: 'hot'
            }
            
            self.regime_means = {
                'cold': np.exp(means[sorted_indices[0]]),
                'neutral': np.exp(means[sorted_indices[1]]),
                'hot': np.exp(means[sorted_indices[2]])
            }
            
            self.transition_matrix = self.model.transmat_
            
            logger.info("HMM fit complete.")
            logger.info(f"Regime means: {self.regime_means}")
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}. Using simple detection.")
            self._fit_simple(np.exp(log_data.flatten()))
    
    def _fit_simple(self, data: np.ndarray):
        """Simple threshold-based regime detection."""
        # Calculate percentiles for thresholds
        self._thresholds = [
            np.percentile(data, 33),
            np.percentile(data, 67)
        ]
        
        self.regime_means = {
            'cold': np.mean(data[data < self._thresholds[0]]),
            'neutral': np.mean(data[(data >= self._thresholds[0]) & (data < self._thresholds[1])]),
            'hot': np.mean(data[data >= self._thresholds[1]])
        }
        
        logger.info(f"Simple regime detection with thresholds: {self._thresholds}")
    
    def predict_regime(self, data: np.ndarray) -> str:
        """
        Predict current regime based on recent data.
        
        Args:
            data: Array of recent crash values
        
        Returns:
            Regime label ('cold', 'neutral', 'hot')
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            log_data = np.log(data[data > 0]).reshape(-1, 1)
            state = self.model.predict(log_data)[-1]
            return self.regime_labels.get(state, 'neutral')
        else:
            # Simple threshold-based detection
            recent_mean = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            
            if recent_mean < self._thresholds[0]:
                return 'cold'
            elif recent_mean < self._thresholds[1]:
                return 'neutral'
            else:
                return 'hot'
    
    def predict_regime_probabilities(self, data: np.ndarray) -> Dict[str, float]:
        """
        Get probability of each regime.
        
        Args:
            data: Array of recent crash values
        
        Returns:
            Dict with regime probabilities
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            log_data = np.log(data[data > 0]).reshape(-1, 1)
            probs = self.model.predict_proba(log_data)[-1]
            
            return {
                self.regime_labels[i]: probs[i]
                for i in range(self.n_regimes)
            }
        else:
            # Simple probability based on recent mean position
            recent_mean = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            
            # Softmax-like approach
            distances = [
                self._thresholds[0] - recent_mean,
                0,  # neutral center
                recent_mean - self._thresholds[1]
            ]
            
            exp_neg = np.exp([-d/2 for d in distances])
            probs = exp_neg / np.sum(exp_neg)
            
            return {
                'cold': probs[0],
                'neutral': probs[1],
                'hot': probs[2]
            }
    
    def get_regime_sequence(self, data: np.ndarray) -> List[str]:
        """
        Get regime sequence for all data points.
        
        Useful for visualization and analysis.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            log_data = np.log(data[data > 0]).reshape(-1, 1)
            states = self.model.predict(log_data)
            return [self.regime_labels.get(s, 'neutral') for s in states]
        else:
            return [self._simple_classify(v) for v in data]
    
    def _simple_classify(self, value: float) -> str:
        """Simple threshold classification."""
        if value < self._thresholds[0]:
            return 'cold'
        elif value < self._thresholds[1]:
            return 'neutral'
        else:
            return 'hot'
    
    def get_transition_probabilities(self) -> Optional[Dict]:
        """
        Get regime transition probabilities.
        
        Shows likelihood of moving from one regime to another.
        """
        if not HMM_AVAILABLE or self.transition_matrix is None:
            return None
        
        result = {}
        for from_idx, from_label in self.regime_labels.items():
            result[from_label] = {}
            for to_idx, to_label in self.regime_labels.items():
                result[from_label][to_label] = self.transition_matrix[from_idx, to_idx]
        
        return result
    
    def get_regime_strategy(self, current_regime: str) -> Dict:
        """
        Get strategy recommendations based on current regime.
        
        Args:
            current_regime: 'cold', 'neutral', or 'hot'
        
        Returns:
            Strategy recommendations
        """
        strategies = {
            'cold': {
                'action': 'conservative',
                'kelly_multiplier': 0.5,
                'target_adjustment': -0.5,  # Lower targets
                'message': "🥶 Cold regime - Reduce bet sizes and targets",
                'recommended_targets': [1.3, 1.5],
                'skip_recommendation': True
            },
            'neutral': {
                'action': 'standard',
                'kelly_multiplier': 1.0,
                'target_adjustment': 0,
                'message': "📊 Neutral regime - Standard betting",
                'recommended_targets': [1.5, 2.0, 2.5],
                'skip_recommendation': False
            },
            'hot': {
                'action': 'aggressive',
                'kelly_multiplier': 1.25,
                'target_adjustment': 0.5,  # Higher targets
                'message': "🔥 Hot regime - Can be more aggressive",
                'recommended_targets': [2.0, 3.0, 4.0],
                'skip_recommendation': False
            }
        }
        
        return strategies.get(current_regime, strategies['neutral'])
    
    def get_analysis(self, data: np.ndarray) -> Dict:
        """
        Complete regime analysis.
        
        Args:
            data: Historical crash data
        
        Returns:
            Comprehensive analysis dict
        """
        current_regime = self.predict_regime(data)
        probs = self.predict_regime_probabilities(data)
        strategy = self.get_regime_strategy(current_regime)
        
        # Regime distribution in recent data
        recent_regimes = self.get_regime_sequence(data[-50:]) if len(data) >= 50 else self.get_regime_sequence(data)
        regime_dist = {
            'cold': recent_regimes.count('cold') / len(recent_regimes),
            'neutral': recent_regimes.count('neutral') / len(recent_regimes),
            'hot': recent_regimes.count('hot') / len(recent_regimes)
        }
        
        return {
            'current_regime': current_regime,
            'regime_probabilities': probs,
            'regime_distribution': regime_dist,
            'regime_means': self.regime_means,
            'strategy': strategy,
            'transition_matrix': self.get_transition_probabilities()
        }


class BettingWindowDetector:
    """
    V2 Regime Detection Engine using 2-state Hidden Markov Model.
    
    States:
    - ACCUMULATION: House recovery phase, high risk, frequent ≤1.2x crashes
    - DISTRIBUTION: Edge satisfied, high-probability betting window
    
    This is the core engine for the "Betting Windows" subscription service.
    """
    
    def __init__(self):
        self.model = None
        self.fitted = False
        self.state_labels = {0: 'ACCUMULATION', 1: 'DISTRIBUTION'}
        self.state_means = {'ACCUMULATION': 1.5, 'DISTRIBUTION': 3.0}
        self.transition_matrix = None
        self.window_open = False
        self.window_confidence = 0.0
        
        # Feature thresholds for simple detection
        self._pof_threshold = 1.5
        self._mv_anomaly_threshold = 1.8
        
    def fit(self, data: np.ndarray, pof_values: np.ndarray = None, mv_values: np.ndarray = None) -> 'BettingWindowDetector':
        """
        Fit 2-state HMM to crash data with optional advanced features.
        
        Args:
            data: Array of crash multipliers
            pof_values: Optional Pool Overload Factor values
            mv_values: Optional Multiplier Velocity values
        """
        log_crashes = np.log(data[data > 0]).reshape(-1, 1)
        
        # Build feature matrix
        if pof_values is not None and mv_values is not None:
            # Multi-feature training
            features = np.column_stack([
                log_crashes.flatten(),
                pof_values[:len(log_crashes)],
                mv_values[:len(log_crashes)]
            ])
        else:
            features = log_crashes
        
        if HMM_AVAILABLE:
            try:
                self.model = hmm.GaussianHMM(
                    n_components=2,
                    covariance_type='full',
                    n_iter=100,
                    random_state=42
                )
                self.model.fit(features)
                
                # Sort states by mean crash value (lower = accumulation)
                if features.ndim == 1:
                    means = self.model.means_.flatten()
                else:
                    means = self.model.means_[:, 0]  # Use crash value dimension
                    
                if means[0] > means[1]:
                    # Swap labels if needed
                    self.state_labels = {0: 'DISTRIBUTION', 1: 'ACCUMULATION'}
                
                self.state_means = {
                    'ACCUMULATION': np.exp(min(means)),
                    'DISTRIBUTION': np.exp(max(means))
                }
                
                self.transition_matrix = self.model.transmat_
                logger.info(f"BettingWindowDetector: 2-state HMM fit complete.")
                logger.info(f"State means: {self.state_means}")
                
            except Exception as e:
                logger.error(f"HMM fitting failed: {e}. Using simple detection.")
                self._fit_simple(data)
        else:
            self._fit_simple(data)
        
        self.fitted = True
        return self
    
    def _fit_simple(self, data: np.ndarray):
        """Fallback threshold-based detection."""
        median = np.median(data)
        self.state_means = {
            'ACCUMULATION': np.mean(data[data < median]),
            'DISTRIBUTION': np.mean(data[data >= median])
        }
        logger.info(f"BettingWindowDetector: Simple fit with median threshold: {median:.2f}")
    
    def predict_state(self, data: np.ndarray) -> str:
        """
        Predict current state (ACCUMULATION or DISTRIBUTION).
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            log_data = np.log(data[data > 0]).reshape(-1, 1)
            state = self.model.predict(log_data)[-1]
            return self.state_labels.get(state, 'ACCUMULATION')
        else:
            recent_mean = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            threshold = (self.state_means['ACCUMULATION'] + self.state_means['DISTRIBUTION']) / 2
            return 'DISTRIBUTION' if recent_mean >= threshold else 'ACCUMULATION'
    
    def get_betting_window_probability(self, data: np.ndarray) -> Dict:
        """
        Calculate the probability that a betting window is currently open.
        
        Returns:
            Dict with window status, confidence, and recommendation
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            log_data = np.log(data[data > 0]).reshape(-1, 1)
            probs = self.model.predict_proba(log_data)[-1]
            
            # Find which index corresponds to DISTRIBUTION
            dist_idx = [k for k, v in self.state_labels.items() if v == 'DISTRIBUTION'][0]
            window_prob = probs[dist_idx]
        else:
            recent_mean = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            # Sigmoid-like probability
            threshold = (self.state_means['ACCUMULATION'] + self.state_means['DISTRIBUTION']) / 2
            window_prob = 1 / (1 + np.exp(-(recent_mean - threshold)))
        
        self.window_confidence = window_prob
        self.window_open = window_prob >= 0.75
        
        # Generate recommendation
        if window_prob >= 0.85:
            recommendation = 'HIGH_CONVICTION_WINDOW'
            action = 'BET'
        elif window_prob >= 0.75:
            recommendation = 'WINDOW_OPEN'
            action = 'BET'
        elif window_prob >= 0.5:
            recommendation = 'TRANSITIONAL'
            action = 'WAIT'
        else:
            recommendation = 'ACCUMULATION_PHASE'
            action = 'SKIP'
        
        return {
            'window_open': self.window_open,
            'window_probability': round(window_prob, 3),
            'state': self.predict_state(data),
            'recommendation': recommendation,
            'action': action,
            'confidence': round(window_prob, 3)
        }
    
    def get_weighted_consensus_score(self, 
                                      data: np.ndarray, 
                                      pof: float = 1.0, 
                                      mv_anomaly: bool = False,
                                      wdi: float = 0.0) -> float:
        """
        Calculate the Weighted Consensus Score (S_w) for betting decisions.
        
        S_w = Σ(W_i × C_i) / Σ(W_i)
        
        Args:
            data: Recent crash data
            pof: Pool Overload Factor
            mv_anomaly: Whether velocity anomaly detected
            wdi: Whale Density Index
        
        Returns:
            Consensus score (0-1), where > 0.75 indicates high conviction
        """
        window_result = self.get_betting_window_probability(data)
        
        # Weighted factors
        weights = {
            'hmm_state': 0.35,
            'pof': 0.25,
            'mv': 0.20,
            'wdi': 0.20
        }
        
        scores = {
            'hmm_state': window_result['window_probability'],
            'pof': max(0, 1 - (pof / 2)),  # Lower POF = better
            'mv': 0.0 if mv_anomaly else 1.0,  # No anomaly = good
            'wdi': max(0, 1 - wdi)  # Lower WDI = better
        }
        
        total_weight = sum(weights.values())
        consensus = sum(weights[k] * scores[k] for k in weights) / total_weight
        
        return round(consensus, 3)
    
    def get_analysis(self, data: np.ndarray) -> Dict:
        """Complete betting window analysis."""
        current_state = self.predict_state(data)
        window_result = self.get_betting_window_probability(data)
        
        # Recent state distribution
        recent = data[-50:] if len(data) >= 50 else data
        states = []
        for i in range(10, len(recent)):
            chunk = recent[:i]
            if HMM_AVAILABLE and self.model is not None:
                log_chunk = np.log(chunk[chunk > 0]).reshape(-1, 1)
                state = self.model.predict(log_chunk)[-1]
                states.append(self.state_labels.get(state, 'ACCUMULATION'))
            else:
                states.append(self.predict_state(chunk))
        
        dist_pct = states.count('DISTRIBUTION') / len(states) if states else 0
        
        return {
            'current_state': current_state,
            'window_open': self.window_open,
            'window_probability': window_result['window_probability'],
            'recommendation': window_result['recommendation'],
            'action': window_result['action'],
            'confidence': window_result['confidence'],
            'state_means': self.state_means,
            'distribution_percentage': round(dist_pct * 100, 1),
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else None
        }


# Singleton instances for easy import
regime_detector = RegimeDetector()
betting_window_detector = BettingWindowDetector()

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("REGIME DETECTION (HMM)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Fit model
    detector = RegimeDetector(n_regimes=3)
    detector.fit(data)
    
    # Get analysis
    analysis = detector.get_analysis(data)
    
    print(f"\n🎯 Current Regime: {analysis['current_regime'].upper()}")
    print(f"\n📊 Regime Probabilities:")
    for regime, prob in analysis['regime_probabilities'].items():
        print(f"   {regime.capitalize()}: {prob*100:.1f}%")
    
    print(f"\n📈 Regime Distribution (last 50 crashes):")
    for regime, pct in analysis['regime_distribution'].items():
        bar = '█' * int(pct * 20)
        print(f"   {regime.capitalize():8s}: {bar} {pct*100:.1f}%")
    
    print(f"\n💰 Regime Mean Crash Points:")
    for regime, mean in analysis['regime_means'].items():
        print(f"   {regime.capitalize()}: {mean:.2f}x")
    
    strategy = analysis['strategy']
    print(f"\n🎮 Strategy Recommendation:")
    print(f"   {strategy['message']}")
    print(f"   Kelly Multiplier: {strategy['kelly_multiplier']:.2f}x")
    print(f"   Recommended Targets: {strategy['recommended_targets']}")
    
    if analysis['transition_matrix']:
        print(f"\n🔄 Transition Probabilities:")
        for from_regime, transitions in analysis['transition_matrix'].items():
            print(f"   From {from_regime}:")
            for to_regime, prob in transitions.items():
                print(f"      → {to_regime}: {prob*100:.1f}%")
