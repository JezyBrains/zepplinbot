#!/usr/bin/env python3
"""
Kelly Criterion Betting Dashboard

Real-time betting signals based on mathematical edge detection.
Shows when to bet, how much, and at what target multiplier.

This is the FINAL system - no prediction, only probability-based edge detection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.kelly_optimizer import KellyBettingOptimizer, BettingSignalSystem
import argparse

def main():
    parser = argparse.ArgumentParser(description='Kelly Criterion Betting Dashboard')
    parser.add_argument('--bankroll', type=float, required=True, help='Current bankroll')
    parser.add_argument('--data-file', type=str, default='data/zeppelin_data.csv', help='Data file')
    parser.add_argument('--targets', type=str, default='1.5,2.0,2.5,3.0,4.0', 
                       help='Comma-separated target multipliers')
    args = parser.parse_args()
    
    print("="*70)
    print("KELLY CRITERION BETTING DASHBOARD")
    print("="*70)
    print("\n🎯 Mathematical Edge Detection System")
    print("   Based on 220+ historical crash points")
    print("   Uses 0.25 Fractional Kelly for safety\n")
    
    # Load data
    df = pd.read_csv(args.data_file)
    
    if 'value' in df.columns:
        data = df['value'].values
    elif 'coefficient' in df.columns:
        data = df['coefficient'].values
    else:
        print("❌ Error: No coefficient/value column found")
        return
    
    print(f"📊 Dataset: {len(data)} crashes")
    print(f"   Range: {data.min():.2f}x - {data.max():.2f}x")
    print(f"   Mean: {data.mean():.2f}x")
    print(f"   Median: {np.median(data):.2f}x")
    print(f"   Last 5: {data[-5:].tolist()}")
    
    # Parse targets
    targets = [float(x) for x in args.targets.split(',')]
    
    print("\n" + "="*70)
    print("ANALYZING ALL TARGETS")
    print("="*70)
    
    # Analyze each target
    optimizer = KellyBettingOptimizer(fractional_kelly=0.25)
    
    results = []
    for target in targets:
        analysis = optimizer.calculate_kelly_bet(args.bankroll, target, data)
        results.append((target, analysis))
        
        print(f"\n🎯 Target: {target}x")
        print(f"   Win Rate: {analysis['probability_data']['win_rate']} "
              f"({analysis['probability_data']['wins']}/{analysis['probability_data']['sample_size']})")
        print(f"   Expected Value: ${analysis['expected_value']:.2f}")
        print(f"   Kelly Bet: ${analysis['bet_amount']:.2f} ({analysis['kelly_fraction']:.2f}% of bankroll)")
        
        if analysis['should_bet']:
            print(f"   ✅ POSITIVE EDGE")
        else:
            print(f"   ❌ NEGATIVE EDGE")
    
    # Get best signal
    print("\n" + "="*70)
    print("BETTING SIGNAL")
    print("="*70)
    
    signal_system = BettingSignalSystem()
    signal = signal_system.get_betting_signal(args.bankroll, data)
    
    if signal['signal'] == 'BET':
        print(f"\n🟢 SIGNAL: BET")
        print(f"\n💰 Recommended Action:")
        print(f"   Target Cashout: {signal['target']}x")
        print(f"   Bet Amount: ${signal['bet_amount']:.2f}")
        print(f"   Kelly Fraction: {signal['kelly_fraction']:.2f}% of bankroll")
        print(f"   Win Probability: {signal['win_probability']:.1f}%")
        print(f"   Expected Value: ${signal['expected_value']:.2f}")
        print(f"\n{signal['recommendation']}")
        
        print(f"\n📋 Execution:")
        print(f"   1. Set auto-cashout to {signal['target']}x")
        print(f"   2. Bet ${signal['bet_amount']:.2f}")
        print(f"   3. Expected profit: ${signal['expected_value']:.2f} per bet")
        
    else:
        print(f"\n🔴 SIGNAL: SKIP")
        print(f"\n⚠️  Reason: {signal['reason']}")
        print(f"\n💡 Recommendation: Wait for better conditions")
        print(f"   - Current data shows no mathematical edge")
        print(f"   - House has advantage at all tested targets")
        print(f"   - Continue collecting data or skip this round")
    
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    # Show distribution
    percentiles = [25, 50, 75, 90, 95]
    print(f"\n📈 Crash Distribution:")
    for p in percentiles:
        val = np.percentile(data, p)
        print(f"   {p}th percentile: {val:.2f}x")
    
    # Recent volatility
    recent_50 = data[-50:]
    volatility = (np.std(recent_50) / np.mean(recent_50)) * 100
    print(f"\n📊 Recent Volatility (last 50):")
    print(f"   Coefficient of Variation: {volatility:.1f}%")
    if volatility > 70:
        print(f"   ⚠️  HIGH VOLATILITY - Use conservative targets")
    elif volatility > 50:
        print(f"   📊 MODERATE VOLATILITY - Normal betting")
    else:
        print(f"   ✅ LOW VOLATILITY - Stable conditions")
    
    print("\n" + "="*70)
    print("KEY PRINCIPLES")
    print("="*70)
    print("""
🎓 Kelly Criterion Formula: f* = (bp - q) / b

Where:
- f* = Fraction of bankroll to bet
- b = Net odds (multiplier - 1)
- p = Win probability (from YOUR data)
- q = Loss probability (1 - p)

🛡️ Safety Features:
- Uses 0.25 Fractional Kelly (25% of full Kelly)
- Only bets when Expected Value > 0
- Adjusts for house edge (1%)
- Requires minimum 20 crashes for analysis

⚠️  Important:
- This is NOT prediction
- This is probability-based edge detection
- SHA-256 ensures true randomness
- Kelly optimizes bet size, not outcomes
- No system guarantees wins

💡 Usage:
python3 kelly_dashboard.py --bankroll 100
python3 kelly_dashboard.py --bankroll 500 --targets 1.5,2.0,3.0
""")

if __name__ == "__main__":
    main()
