#!/usr/bin/env python3
"""
Risk Management & Bankroll Optimization System

Instead of predicting (impossible with SHA-256), this system:
1. Calculates real-time volatility index
2. Applies Kelly Criterion for optimal bet sizing
3. Identifies risk levels based on statistical distribution
4. Provides conservative auto-cashout recommendations

This is NOT prediction - it's probability-based risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Analyzes crash point volatility to assess current risk level.
    
    High volatility = Higher risk = Bet smaller or skip
    Low volatility = More stable = Can bet normally
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
    
    def calculate_volatility_index(self, recent_crashes: np.ndarray) -> Dict:
        """
        Calculate volatility metrics for recent crashes.
        
        Returns:
            - volatility_index: 0-100 (higher = more volatile)
            - risk_level: low/medium/high/extreme
            - std_dev: Standard deviation
            - coefficient_variation: CV ratio
        """
        if len(recent_crashes) < 10:
            return {
                'volatility_index': 50,
                'risk_level': 'unknown',
                'std_dev': 0,
                'coefficient_variation': 0,
                'message': 'Insufficient data'
            }
        
        mean = np.mean(recent_crashes)
        std_dev = np.std(recent_crashes)
        cv = (std_dev / mean) * 100 if mean > 0 else 0
        
        # Normalize to 0-100 scale
        # CV > 100% = extreme volatility
        volatility_index = min(cv, 100)
        
        # Risk levels
        if volatility_index < 40:
            risk_level = 'low'
        elif volatility_index < 60:
            risk_level = 'medium'
        elif volatility_index < 80:
            risk_level = 'high'
        else:
            risk_level = 'extreme'
        
        return {
            'volatility_index': volatility_index,
            'risk_level': risk_level,
            'std_dev': std_dev,
            'coefficient_variation': cv,
            'mean': mean,
            'median': np.median(recent_crashes)
        }
    
    def detect_streak(self, recent_crashes: np.ndarray, threshold: float = 2.0) -> Dict:
        """
        Detect if currently in a low/high streak.
        
        NOT for prediction - for risk awareness.
        """
        if len(recent_crashes) < 5:
            return {'streak_type': 'none', 'streak_length': 0}
        
        last_5 = recent_crashes[-5:]
        
        low_count = sum(1 for x in last_5 if x < threshold)
        high_count = sum(1 for x in last_5 if x >= threshold * 2)
        
        if low_count >= 4:
            return {
                'streak_type': 'low',
                'streak_length': low_count,
                'message': f'{low_count}/5 crashes below {threshold}x'
            }
        elif high_count >= 3:
            return {
                'streak_type': 'high',
                'streak_length': high_count,
                'message': f'{high_count}/5 crashes above {threshold*2}x'
            }
        else:
            return {
                'streak_type': 'mixed',
                'streak_length': 0,
                'message': 'No clear streak pattern'
            }


class KellyCriterionOptimizer:
    """
    Kelly Criterion for optimal bet sizing.
    
    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of bankroll to bet
    - b = odds received (payout - 1)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, house_edge: float = 0.01):
        """
        Args:
            house_edge: Typical 1-3% for crash games
        """
        self.house_edge = house_edge
    
    def calculate_win_probability(self, target_multiplier: float, 
                                  historical_data: np.ndarray) -> float:
        """
        Calculate probability of crash being >= target multiplier.
        
        Based on historical distribution, NOT prediction.
        """
        if len(historical_data) < 50:
            # Default conservative estimate
            return 0.5
        
        # Empirical probability
        wins = sum(1 for x in historical_data if x >= target_multiplier)
        probability = wins / len(historical_data)
        
        # Adjust for house edge
        adjusted_probability = probability * (1 - self.house_edge)
        
        return max(0.01, min(0.99, adjusted_probability))
    
    def optimal_bet_size(self, bankroll: float, target_multiplier: float,
                        historical_data: np.ndarray) -> Dict:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Returns:
            - optimal_bet: Dollar amount
            - kelly_fraction: Percentage of bankroll
            - win_probability: Estimated probability
            - expected_value: EV of the bet
        """
        p = self.calculate_win_probability(target_multiplier, historical_data)
        q = 1 - p
        b = target_multiplier - 1  # Net odds
        
        # Kelly formula
        kelly_fraction = (b * p - q) / b
        
        # Use fractional Kelly (more conservative)
        # Full Kelly can be aggressive, use 25-50% of Kelly
        conservative_fraction = kelly_fraction * 0.25
        
        # Clamp to reasonable range (1-10% of bankroll max)
        conservative_fraction = max(0.01, min(0.10, conservative_fraction))
        
        optimal_bet = bankroll * conservative_fraction
        
        # Expected value
        ev = (p * b * optimal_bet) - (q * optimal_bet)
        
        return {
            'optimal_bet': optimal_bet,
            'kelly_fraction': conservative_fraction * 100,
            'win_probability': p * 100,
            'expected_value': ev,
            'recommendation': self._get_recommendation(conservative_fraction, ev)
        }
    
    def _get_recommendation(self, fraction: float, ev: float) -> str:
        """Generate betting recommendation."""
        if ev < 0:
            return "SKIP - Negative expected value"
        elif fraction < 0.02:
            return "SKIP - Kelly suggests bet too small"
        elif fraction < 0.05:
            return "SMALL BET - Low Kelly fraction"
        elif fraction < 0.08:
            return "MODERATE BET - Reasonable Kelly fraction"
        else:
            return "LARGER BET - High Kelly fraction (verify calculation)"


class RiskBasedStrategy:
    """
    Combines volatility analysis and Kelly Criterion for complete strategy.
    """
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.kelly_optimizer = KellyCriterionOptimizer()
    
    def get_strategy(self, bankroll: float, historical_data: np.ndarray,
                    recent_window: int = 50) -> Dict:
        """
        Generate complete betting strategy based on current conditions.
        
        Args:
            bankroll: Current bankroll
            historical_data: All crash history
            recent_window: How many recent crashes to analyze
        
        Returns:
            Complete strategy with risk assessment and bet sizing
        """
        recent_crashes = historical_data[-recent_window:]
        
        # 1. Volatility analysis
        volatility = self.volatility_analyzer.calculate_volatility_index(recent_crashes)
        streak = self.volatility_analyzer.detect_streak(recent_crashes)
        
        # 2. Statistical distribution
        percentiles = {
            '25th': np.percentile(historical_data, 25),
            '50th': np.percentile(historical_data, 50),
            '75th': np.percentile(historical_data, 75),
            '90th': np.percentile(historical_data, 90)
        }
        
        # 3. Recommended cashout targets based on risk
        if volatility['risk_level'] == 'low':
            safe_target = percentiles['50th']
            moderate_target = percentiles['75th']
        elif volatility['risk_level'] == 'medium':
            safe_target = percentiles['25th']
            moderate_target = percentiles['50th']
        else:  # high or extreme
            safe_target = percentiles['25th'] * 0.8
            moderate_target = percentiles['25th']
        
        # 4. Kelly Criterion for each target
        safe_kelly_result = self.kelly_optimizer.optimal_bet_size(
            bankroll, safe_target, historical_data
        )
        moderate_kelly_result = self.kelly_optimizer.optimal_bet_size(
            bankroll, moderate_target, historical_data
        )
        
        # Combine target with kelly results
        safe_strategy = {'target': safe_target, **safe_kelly_result}
        moderate_strategy = {'target': moderate_target, **moderate_kelly_result}
        
        # 5. Overall recommendation
        overall_rec = self._generate_overall_recommendation(
            volatility, streak, safe_strategy, moderate_strategy
        )
        
        return {
            'volatility': volatility,
            'streak': streak,
            'percentiles': percentiles,
            'safe_strategy': safe_strategy,
            'moderate_strategy': moderate_strategy,
            'recommendation': overall_rec
        }
    
    def _generate_overall_recommendation(self, volatility: Dict, streak: Dict,
                                        safe_kelly: Dict, moderate_kelly: Dict) -> str:
        """Generate human-readable recommendation."""
        risk = volatility['risk_level']
        
        if risk == 'extreme':
            return (
                f"⚠️ EXTREME VOLATILITY ({volatility['volatility_index']:.0f}/100)\n"
                f"   Recommendation: SKIP or bet minimum\n"
                f"   Current conditions too unpredictable"
            )
        elif risk == 'high':
            return (
                f"⚠️ HIGH VOLATILITY ({volatility['volatility_index']:.0f}/100)\n"
                f"   Recommendation: Use SAFE strategy only\n"
                f"   Bet ${safe_kelly['optimal_bet']:.2f} "
                f"(Target: {safe_kelly['target']:.2f}x)"
            )
        elif risk == 'medium':
            return (
                f"📊 MODERATE VOLATILITY ({volatility['volatility_index']:.0f}/100)\n"
                f"   Recommendation: Balanced approach\n"
                f"   Safe: ${safe_kelly['optimal_bet']:.2f} @ {safe_kelly['target']:.2f}x\n"
                f"   Moderate: ${moderate_kelly['optimal_bet']:.2f} @ {moderate_kelly['target']:.2f}x"
            )
        else:  # low
            return (
                f"✅ LOW VOLATILITY ({volatility['volatility_index']:.0f}/100)\n"
                f"   Recommendation: Normal betting\n"
                f"   Moderate strategy: ${moderate_kelly['optimal_bet']:.2f} @ {moderate_kelly['target']:.2f}x"
            )


# Example usage
if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    strategy = RiskBasedStrategy()
    result = strategy.get_strategy(bankroll=100, historical_data=data)
    
    print("="*70)
    print("RISK-BASED STRATEGY ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Volatility: {result['volatility']['risk_level'].upper()}")
    print(f"   Index: {result['volatility']['volatility_index']:.1f}/100")
    print(f"   CV: {result['volatility']['coefficient_variation']:.1f}%")
    
    print(f"\n🎯 Current Streak: {result['streak']['streak_type']}")
    print(f"   {result['streak']['message']}")
    
    print(f"\n💰 Safe Strategy:")
    print(f"   Target: {result['safe_strategy']['target']:.2f}x")
    print(f"   Bet: ${result['safe_strategy']['kelly']['optimal_bet']:.2f}")
    print(f"   Win Probability: {result['safe_strategy']['kelly']['win_probability']:.1f}%")
    
    print(f"\n📈 Moderate Strategy:")
    print(f"   Target: {result['moderate_strategy']['target']:.2f}x")
    print(f"   Bet: ${result['moderate_strategy']['kelly']['optimal_bet']:.2f}")
    print(f"   Win Probability: {result['moderate_strategy']['kelly']['win_probability']:.1f}%")
    
    print(f"\n{result['recommendation']}")
