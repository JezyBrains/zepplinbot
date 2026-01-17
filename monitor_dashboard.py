#!/usr/bin/env python3
"""
Comprehensive Monitoring Dashboard

Combines:
1. Kelly Criterion (edge detection)
2. Streak Monitor (anomaly detection)
3. RTP Shift Detector (algorithm change detection)

This is the FINAL professional system for Zeppelin monitoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.kelly_optimizer import KellyBettingOptimizer
from src.streak_monitor import StreakMonitor, RTPShiftDetector, AnomalyBettingSignal
import argparse

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Monitoring Dashboard')
    parser.add_argument('--bankroll', type=float, default=100, help='Current bankroll')
    parser.add_argument('--data-file', type=str, default='data/zeppelin_data.csv', help='Data file')
    parser.add_argument('--target', type=float, default=2.0, help='Target multiplier')
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE MONITORING DASHBOARD")
    print("="*70)
    print("\n🎯 Professional Zeppelin Analysis System")
    print("   Kelly Criterion + Streak Monitor + RTP Detector\n")
    
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
    print(f"   Last crash: {data[-1]:.2f}x")
    
    # ========================================================================
    # 1. KELLY CRITERION ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("1. KELLY CRITERION - MATHEMATICAL EDGE DETECTION")
    print("="*70)
    
    kelly = KellyBettingOptimizer(fractional_kelly=0.25)
    kelly_result = kelly.calculate_kelly_bet(args.bankroll, args.target, data)
    
    print(f"\n🎯 Target: {args.target}x")
    print(f"   Win Rate: {kelly_result['probability_data']['win_rate']} "
          f"({kelly_result['probability_data']['wins']}/{kelly_result['probability_data']['sample_size']})")
    print(f"   Expected Value: ${kelly_result['expected_value']:.2f}")
    print(f"   Kelly Bet: ${kelly_result['bet_amount']:.2f} ({kelly_result['kelly_fraction']:.2f}% of bankroll)")
    
    if kelly_result['should_bet']:
        print(f"   ✅ POSITIVE EDGE DETECTED")
    else:
        print(f"   ❌ NEGATIVE EDGE (House advantage: ~{50 - kelly_result['win_probability']:.1f}%)")
    
    # Calculate effective house edge
    break_even_rate = 1 / args.target
    actual_rate = kelly_result['win_probability'] / 100
    house_edge = (break_even_rate - actual_rate) * 100
    
    print(f"\n📊 House Edge Analysis:")
    print(f"   Break-even win rate: {break_even_rate*100:.1f}%")
    print(f"   Actual win rate: {actual_rate*100:.1f}%")
    print(f"   Effective house edge: {house_edge:.1f}%")
    
    # ========================================================================
    # 2. STREAK MONITOR
    # ========================================================================
    print("\n" + "="*70)
    print("2. STREAK MONITOR - ANOMALY DETECTION")
    print("="*70)
    
    streak_monitor = StreakMonitor(low_threshold=1.2, streak_length=5)
    current_streak = streak_monitor.detect_current_streak(data[-20:])
    streak_patterns = streak_monitor.analyze_streak_patterns(data)
    
    print(f"\n📊 Current Conditions:")
    print(f"   In Low Streak: {current_streak['in_streak']}")
    print(f"   {current_streak['message']}")
    
    if current_streak['in_streak']:
        print(f"   Last {len(current_streak['last_n_crashes'])} crashes: {current_streak['last_n_crashes']}")
        print(f"   Historical low rate: {current_streak['historical_low_rate']:.1f}%")
        print(f"   Streak probability: {current_streak['expected_probability']:.2f}%")
    
    print(f"\n📈 Historical Streak Patterns:")
    print(f"   Total streaks (5+ lows): {streak_patterns['total_streaks']}")
    print(f"   Average streak length: {streak_patterns['average_length']:.1f}")
    print(f"   Max streak length: {streak_patterns['max_length']}")
    print(f"   Streak frequency: {streak_patterns['frequency']:.2f}% of rounds")
    
    # ========================================================================
    # 3. RTP SHIFT DETECTOR
    # ========================================================================
    print("\n" + "="*70)
    print("3. RTP SHIFT DETECTOR - ALGORITHM CHANGE MONITORING")
    print("="*70)
    
    rtp_detector = RTPShiftDetector(window_size=100)
    rtp_shift = rtp_detector.detect_rtp_shift(data, args.target)
    rtp_trend = rtp_detector.analyze_rtp_trend(data, args.target, num_windows=5)
    
    print(f"\n📊 RTP Shift Analysis (last 100 vs previous 100):")
    print(f"   Recent win rate: {rtp_shift['recent_win_rate']:.1f}%")
    print(f"   Older win rate: {rtp_shift['older_win_rate']:.1f}%")
    print(f"   Shift: {rtp_shift['shift_percentage']:+.1f}%")
    print(f"   Statistical significance: p={rtp_shift['p_value']:.4f}")
    print(f"   {rtp_shift['message']}")
    
    print(f"\n📈 RTP Trend (last 5 windows of 100):")
    print(f"   Trend: {rtp_trend['trend'].upper()}")
    if 'win_rates' in rtp_trend:
        print(f"   Win rates: {[f'{x:.1f}%' for x in rtp_trend['win_rates']]}")
        print(f"   Total change: {rtp_trend['total_change']:+.1f}%")
    else:
        print(f"   Need {500} crashes for full trend analysis")
    
    # ========================================================================
    # 4. COMBINED SIGNAL
    # ========================================================================
    print("\n" + "="*70)
    print("4. COMBINED BETTING SIGNAL")
    print("="*70)
    
    anomaly_signal = AnomalyBettingSignal()
    signal = anomaly_signal.get_anomaly_signal(data, args.target)
    
    print(f"\n🚦 Signal: {signal['signal']}")
    print(f"   {signal['recommendation']}")
    
    if signal['reasons']:
        print(f"\n   Factors:")
        for reason in signal['reasons']:
            print(f"   {reason}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    kelly_says_bet = kelly_result['should_bet']
    anomaly_detected = signal['signal'] in ['WATCH', 'CONSIDER']
    rtp_improved = rtp_shift['shift_detected'] and rtp_shift['direction'] == 'IMPROVED'
    
    print(f"\n📋 Decision Matrix:")
    print(f"   Kelly Criterion: {'✅ BET' if kelly_says_bet else '❌ SKIP'}")
    print(f"   Anomaly Detection: {signal['signal']}")
    print(f"   RTP Status: {rtp_shift['direction'] if rtp_shift['shift_detected'] else 'STABLE'}")
    
    if kelly_says_bet:
        print(f"\n🟢 RECOMMENDATION: BET")
        print(f"   Target: {args.target}x")
        print(f"   Bet Amount: ${kelly_result['bet_amount']:.2f}")
        print(f"   Expected Value: ${kelly_result['expected_value']:.2f}")
    elif anomaly_detected and rtp_improved:
        print(f"\n🟡 RECOMMENDATION: WATCH CLOSELY")
        print(f"   Kelly shows no edge, but anomalies detected")
        print(f"   Monitor next few rounds for RTP improvement")
        print(f"   Consider small test bet if conditions persist")
    else:
        print(f"\n🔴 RECOMMENDATION: SKIP")
        print(f"   No mathematical edge detected")
        print(f"   House edge: ~{house_edge:.1f}%")
        print(f"   Continue monitoring for changes")
    
    # ========================================================================
    # 5. MONITORING INSTRUCTIONS
    # ========================================================================
    print("\n" + "="*70)
    print("CONTINUOUS MONITORING PROTOCOL")
    print("="*70)
    
    print(f"""
📖 What to Monitor:

1. **Kelly Criterion** (Primary)
   - Only bet when Expected Value > 0
   - Current status: {'POSITIVE' if kelly_says_bet else 'NEGATIVE'}
   
2. **RTP Shifts** (Algorithm Changes)
   - Watch for win rate improvements
   - Current trend: {rtp_trend['trend'].upper()}
   - Alert if shift >5% and p<0.05
   
3. **Streak Anomalies** (Short-term)
   - Flag unusual low streaks
   - Current status: {'ANOMALY' if current_streak['is_anomaly'] else 'NORMAL'}

📊 Data Collection:
   - Current dataset: {len(data)} crashes
   - Target: 500+ for high confidence
   - Add new data: python3 manual_collect.py

🔄 Re-check Signal:
   python3 monitor_dashboard.py --bankroll {args.bankroll} --target {args.target}

⚠️  Key Principle:
   The system protects your capital by identifying when NO edge exists.
   A "SKIP" signal is SUCCESS - it's saving you from negative EV bets.
   
   Your 319-crash audit shows consistent ~{house_edge:.1f}% house edge.
   This is NORMAL for provably fair games.
   
   Keep monitoring. If RTP improves, Kelly will detect it first.
""")

if __name__ == "__main__":
    main()
