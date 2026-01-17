#!/usr/bin/env python3
"""
Real-Time Risk Management Dashboard

Pivot from prediction to probability-based risk management.
Uses Kelly Criterion and volatility analysis instead of trying to predict SHA-256.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.risk_manager import RiskBasedStrategy
import argparse

def main():
    parser = argparse.ArgumentParser(description='Risk-Based Betting Strategy')
    parser.add_argument('--bankroll', type=float, default=100, help='Current bankroll')
    parser.add_argument('--data-file', type=str, default='data/zeppelin_data.csv', help='Data file')
    args = parser.parse_args()
    
    print("="*70)
    print("RISK MANAGEMENT SYSTEM - Kelly Criterion & Volatility Analysis")
    print("="*70)
    print("\n⚠️  This is NOT prediction - it's probability-based risk management")
    print("   SHA-256 is unbreakable. We manage risk, not predict outcomes.\n")
    
    # Load data
    df = pd.read_csv(args.data_file)
    
    if 'value' in df.columns:
        data = df['value'].values
    elif 'coefficient' in df.columns:
        data = df['coefficient'].values
    else:
        print("❌ Error: No coefficient/value column found")
        return
    
    print(f"📊 Analyzing {len(data)} historical crashes...")
    print(f"   Range: {data.min():.2f}x - {data.max():.2f}x")
    print(f"   Mean: {data.mean():.2f}x")
    print(f"   Median: {np.median(data):.2f}x")
    print(f"   Last crash: {data[-1]:.2f}x")
    
    # Get strategy
    strategy_engine = RiskBasedStrategy()
    result = strategy_engine.get_strategy(
        bankroll=args.bankroll,
        historical_data=data,
        recent_window=50
    )
    
    # Display results
    print("\n" + "="*70)
    print("CURRENT MARKET CONDITIONS")
    print("="*70)
    
    vol = result['volatility']
    print(f"\n📈 Volatility Analysis:")
    print(f"   Risk Level: {vol['risk_level'].upper()}")
    print(f"   Volatility Index: {vol['volatility_index']:.1f}/100")
    print(f"   Coefficient of Variation: {vol['coefficient_variation']:.1f}%")
    print(f"   Standard Deviation: {vol['std_dev']:.2f}x")
    
    streak = result['streak']
    print(f"\n🎯 Recent Pattern:")
    print(f"   Type: {streak['streak_type'].upper()}")
    print(f"   {streak['message']}")
    
    print(f"\n📊 Historical Distribution:")
    for pct, val in result['percentiles'].items():
        print(f"   {pct} percentile: {val:.2f}x")
    
    print("\n" + "="*70)
    print("RECOMMENDED STRATEGIES")
    print("="*70)
    
    # Safe strategy
    safe = result['safe_strategy']
    print(f"\n✅ SAFE STRATEGY (Conservative)")
    print(f"   Target Cashout: {safe['target']:.2f}x")
    print(f"   Optimal Bet: ${safe['optimal_bet']:.2f} ({safe['kelly_fraction']:.1f}% of bankroll)")
    print(f"   Win Probability: {safe['win_probability']:.1f}%")
    print(f"   Expected Value: ${safe['expected_value']:.2f}")
    print(f"   {safe['recommendation']}")
    
    # Moderate strategy
    moderate = result['moderate_strategy']
    print(f"\n📊 MODERATE STRATEGY (Balanced)")
    print(f"   Target Cashout: {moderate['target']:.2f}x")
    print(f"   Optimal Bet: ${moderate['optimal_bet']:.2f} ({moderate['kelly_fraction']:.1f}% of bankroll)")
    print(f"   Win Probability: {moderate['win_probability']:.1f}%")
    print(f"   Expected Value: ${moderate['expected_value']:.2f}")
    print(f"   {moderate['recommendation']}")
    
    print("\n" + "="*70)
    print("OVERALL RECOMMENDATION")
    print("="*70)
    print(f"\n{result['recommendation']}")
    
    print("\n" + "="*70)
    print("KEY PRINCIPLES")
    print("="*70)
    print("""
1. This is NOT prediction - SHA-256 cannot be predicted
2. Kelly Criterion optimizes bet size for long-term growth
3. Volatility index helps avoid high-risk conditions
4. Always bet within your bankroll limits
5. No system beats true randomness - manage risk instead

📖 What this system does:
   ✅ Calculates optimal bet size (Kelly Criterion)
   ✅ Assesses current volatility/risk level
   ✅ Provides probability-based cashout targets
   ✅ Helps with bankroll management
   
   ❌ Does NOT predict next crash
   ❌ Does NOT guarantee wins
   ❌ Does NOT break SHA-256
""")
    
    print("\n💡 Usage:")
    print("   python3 risk_strategy.py --bankroll 100")
    print("   python3 risk_strategy.py --bankroll 500 --data-file data/zeppelin_data.csv")

if __name__ == "__main__":
    main()
