#!/usr/bin/env python3
"""
Kelly Criterion Betting Optimizer

Professional bankroll management using Kelly Criterion.
Calculates optimal bet size based on historical win probability.

This is NOT prediction - it's mathematical edge detection.
Only signals bets when historical data shows positive expected value.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellyBettingOptimizer:
    """
    Kelly Criterion implementation for crash games.
    
    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of bankroll to bet
    - b = net odds (multiplier - 1)
    - p = probability of winning (from historical data)
    - q = probability of losing (1 - p)
    
    Uses fractional Kelly (25%) for safety against volatility.
    """
    
    def __init__(self, fractional_kelly: float = 0.25, house_edge: float = 0.01):
        """
        Args:
            fractional_kelly: Fraction of Kelly bet to use (0.25 = 25% Kelly)
            house_edge: Estimated house edge (1-3% typical)
        """
        self.fractional_kelly = fractional_kelly
        self.house_edge = house_edge
    
    def calculate_win_probability(self, target_multiplier: float, 
                                  crash_history: np.ndarray) -> Dict:
        """
        Calculate empirical win probability from historical data.
        
        Args:
            target_multiplier: Target cashout (e.g., 2.0x)
            crash_history: Array of historical crash points
        
        Returns:
            Dict with probability metrics
        """
        if len(crash_history) < 20:
            return {
                'win_probability': 0.5,
                'sample_size': len(crash_history),
                'wins': 0,
                'losses': 0,
                'confidence': 'low',
                'message': 'Insufficient data (need 20+ crashes)'
            }
        
        # Count wins (crashes >= target)
        wins = sum(1 for x in crash_history if x >= target_multiplier)
        losses = len(crash_history) - wins
        
        # Raw probability
        p_raw = wins / len(crash_history)
        
        # Adjust for house edge (conservative)
        p_adjusted = p_raw * (1 - self.house_edge)
        
        # Confidence based on sample size
        if len(crash_history) < 50:
            confidence = 'low'
        elif len(crash_history) < 100:
            confidence = 'medium'
        else:
            confidence = 'high'
        
        return {
            'win_probability': p_adjusted,
            'raw_probability': p_raw,
            'sample_size': len(crash_history),
            'wins': wins,
            'losses': losses,
            'confidence': confidence,
            'win_rate': f"{p_adjusted*100:.1f}%"
        }
    
    def calculate_kelly_bet(self, bankroll: float, target_multiplier: float,
                           crash_history: np.ndarray) -> Dict:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Args:
            bankroll: Current bankroll
            target_multiplier: Target cashout multiplier
            crash_history: Historical crash data
        
        Returns:
            Dict with bet recommendation and analysis
        """
        # Get win probability
        prob_data = self.calculate_win_probability(target_multiplier, crash_history)
        p = prob_data['win_probability']
        q = 1 - p
        
        # Net odds
        b = target_multiplier - 1
        
        # Kelly formula: (bp - q) / b
        kelly_fraction = (b * p - q) / b
        
        # Expected value per unit bet
        ev_per_unit = (p * b) - q
        
        # Apply fractional Kelly for safety
        safe_kelly = kelly_fraction * self.fractional_kelly
        
        # Clamp to reasonable range (1-10% of bankroll max)
        safe_kelly = max(0, min(0.10, safe_kelly))
        
        # Calculate bet amount
        bet_amount = bankroll * safe_kelly
        
        # Expected value of this bet
        expected_value = bet_amount * ev_per_unit
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            kelly_fraction, safe_kelly, ev_per_unit, prob_data
        )
        
        return {
            'bet_amount': bet_amount,
            'kelly_fraction': safe_kelly * 100,
            'full_kelly': kelly_fraction * 100,
            'expected_value': expected_value,
            'ev_per_unit': ev_per_unit,
            'target_multiplier': target_multiplier,
            'win_probability': p * 100,
            'probability_data': prob_data,
            'recommendation': recommendation,
            'should_bet': ev_per_unit > 0 and safe_kelly > 0.01
        }
    
    def _generate_recommendation(self, full_kelly: float, safe_kelly: float,
                                ev_per_unit: float, prob_data: Dict) -> str:
        """Generate betting recommendation."""
        
        # Negative EV - don't bet
        if ev_per_unit <= 0:
            return (
                "🛑 SKIP - Negative Expected Value\n"
                f"   Historical data shows house has edge\n"
                f"   Win rate: {prob_data['win_rate']} (not enough)"
            )
        
        # Kelly too small - skip
        if safe_kelly < 0.01:
            return (
                "🛑 SKIP - Kelly fraction too small\n"
                f"   Edge exists but too small to exploit\n"
                f"   Win rate: {prob_data['win_rate']}"
            )
        
        # Low confidence - caution
        if prob_data['confidence'] == 'low':
            return (
                "⚠️ CAUTION - Low sample size\n"
                f"   Only {prob_data['sample_size']} crashes analyzed\n"
                f"   Collect more data for reliable edge detection"
            )
        
        # Positive EV and reasonable Kelly
        if safe_kelly < 0.03:
            return (
                "✅ SMALL BET - Slight edge detected\n"
                f"   Win rate: {prob_data['win_rate']}\n"
                f"   Bet {safe_kelly*100:.1f}% of bankroll"
            )
        elif safe_kelly < 0.06:
            return (
                "✅ MODERATE BET - Good edge detected\n"
                f"   Win rate: {prob_data['win_rate']}\n"
                f"   Bet {safe_kelly*100:.1f}% of bankroll"
            )
        else:
            return (
                "✅ STRONG BET - Significant edge detected\n"
                f"   Win rate: {prob_data['win_rate']}\n"
                f"   Bet {safe_kelly*100:.1f}% of bankroll"
            )
    
    def analyze_multiple_targets(self, bankroll: float, 
                                crash_history: np.ndarray,
                                targets: List[float] = None) -> Dict:
        """
        Analyze multiple cashout targets to find best opportunity.
        
        Args:
            bankroll: Current bankroll
            crash_history: Historical data
            targets: List of multipliers to analyze (default: [1.5, 2.0, 2.5, 3.0])
        
        Returns:
            Analysis for each target with best recommendation
        """
        if targets is None:
            targets = [1.5, 2.0, 2.5, 3.0, 4.0]
        
        results = {}
        best_ev = -float('inf')
        best_target = None
        
        for target in targets:
            analysis = self.calculate_kelly_bet(bankroll, target, crash_history)
            results[target] = analysis
            
            if analysis['expected_value'] > best_ev:
                best_ev = analysis['expected_value']
                best_target = target
        
        return {
            'targets': results,
            'best_target': best_target,
            'best_ev': best_ev,
            'any_positive_ev': best_ev > 0
        }


class BettingSignalSystem:
    """
    Generates betting signals based on Kelly Criterion analysis.
    
    Only signals bets when:
    1. Expected value is positive
    2. Kelly fraction is meaningful (>1%)
    3. Sample size is adequate
    """
    
    def __init__(self):
        self.kelly_optimizer = KellyBettingOptimizer(fractional_kelly=0.25)
    
    def get_betting_signal(self, bankroll: float, crash_history: np.ndarray) -> Dict:
        """
        Generate betting signal for current conditions.
        
        Returns:
            - signal: 'BET' or 'SKIP'
            - target: Recommended cashout multiplier
            - bet_amount: How much to bet
            - analysis: Full Kelly analysis
        """
        # Analyze multiple targets
        multi_analysis = self.kelly_optimizer.analyze_multiple_targets(
            bankroll, crash_history
        )
        
        if not multi_analysis['any_positive_ev']:
            return {
                'signal': 'SKIP',
                'reason': 'No positive expected value at any target',
                'target': None,
                'bet_amount': 0,
                'analysis': multi_analysis
            }
        
        # Get best target
        best_target = multi_analysis['best_target']
        best_analysis = multi_analysis['targets'][best_target]
        
        if not best_analysis['should_bet']:
            return {
                'signal': 'SKIP',
                'reason': best_analysis['recommendation'],
                'target': best_target,
                'bet_amount': 0,
                'analysis': multi_analysis
            }
        
        return {
            'signal': 'BET',
            'target': best_target,
            'bet_amount': best_analysis['bet_amount'],
            'expected_value': best_analysis['expected_value'],
            'win_probability': best_analysis['win_probability'],
            'kelly_fraction': best_analysis['kelly_fraction'],
            'recommendation': best_analysis['recommendation'],
            'analysis': multi_analysis
        }


# Example usage
if __name__ == "__main__":
    # Test with real data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    print("="*70)
    print("KELLY CRITERION BETTING OPTIMIZER")
    print("="*70)
    
    bankroll = 100
    
    # Test single target
    optimizer = KellyBettingOptimizer()
    result = optimizer.calculate_kelly_bet(bankroll, 2.0, data)
    
    print(f"\n📊 Analysis for 2.0x target:")
    print(f"   Win Probability: {result['win_probability']:.1f}%")
    print(f"   Kelly Fraction: {result['kelly_fraction']:.2f}%")
    print(f"   Bet Amount: ${result['bet_amount']:.2f}")
    print(f"   Expected Value: ${result['expected_value']:.2f}")
    print(f"   Should Bet: {'YES' if result['should_bet'] else 'NO'}")
    print(f"\n{result['recommendation']}")
    
    # Test signal system
    print("\n" + "="*70)
    print("BETTING SIGNAL SYSTEM")
    print("="*70)
    
    signal_system = BettingSignalSystem()
    signal = signal_system.get_betting_signal(bankroll, data)
    
    print(f"\n🚦 Signal: {signal['signal']}")
    if signal['signal'] == 'BET':
        print(f"   Target: {signal['target']}x")
        print(f"   Bet Amount: ${signal['bet_amount']:.2f}")
        print(f"   Expected Value: ${signal['expected_value']:.2f}")
        print(f"   Win Probability: {signal['win_probability']:.1f}%")
    else:
        print(f"   Reason: {signal['reason']}")
