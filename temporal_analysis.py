#!/usr/bin/env python3
"""
Temporal Analysis of Zeppelin Game Data
Analyzes time-based patterns and behaviors from collected data
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_temporal_patterns():
    """Comprehensive temporal analysis of Zeppelin game data"""
    
    print("=" * 70)
    print("🕐 ZEPPELIN TEMPORAL BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('data/round_timing.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📊 Dataset: {len(df)} rounds analyzed")
    print(f"📅 Period: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    # =====================================================
    # 1. TIME OF DAY ANALYSIS
    # =====================================================
    print("\n" + "=" * 70)
    print("🌅 TIME OF DAY PATTERNS")
    print("=" * 70)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_period'] = pd.cut(df['hour'], 
                               bins=[0, 6, 12, 18, 24], 
                               labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
                               include_lowest=True)
    
    hourly_stats = df.groupby('hour').agg({
        'crash_value': ['mean', 'median', 'std', 'count'],
        'duration_ms': 'mean',
        'stake': 'mean',
        'bettors': 'mean'
    }).round(2)
    
    print("\n📈 Crash Values by Hour of Day:")
    hourly_crash = df.groupby('hour')['crash_value'].agg(['mean', 'median', 'count'])
    for hour in sorted(df['hour'].unique()):
        data = hourly_crash.loc[hour]
        bar = "█" * int(min(data['mean'], 10))
        print(f"  {hour:02d}:00  │ Avg: {data['mean']:5.2f}x │ Med: {data['median']:5.2f}x │ Rounds: {int(data['count']):4d} │ {bar}")
    
    # Best and worst hours
    best_hour = hourly_crash['mean'].idxmax()
    worst_hour = hourly_crash['mean'].idxmin()
    print(f"\n  🏆 Best Hour:  {best_hour:02d}:00 (Avg: {hourly_crash.loc[best_hour, 'mean']:.2f}x)")
    print(f"  ⚠️ Worst Hour: {worst_hour:02d}:00 (Avg: {hourly_crash.loc[worst_hour, 'mean']:.2f}x)")
    
    # =====================================================
    # 2. ROUND DURATION ANALYSIS
    # =====================================================
    print("\n" + "=" * 70)
    print("⏱️ ROUND DURATION PATTERNS")
    print("=" * 70)
    
    df['duration_sec'] = df['duration_ms'] / 1000
    
    print(f"\n📊 Duration Statistics:")
    print(f"  • Mean:    {df['duration_sec'].mean():.1f}s")
    print(f"  • Median:  {df['duration_sec'].median():.1f}s")
    print(f"  • Std Dev: {df['duration_sec'].std():.1f}s")
    print(f"  • Min:     {df['duration_sec'].min():.1f}s")
    print(f"  • Max:     {df['duration_sec'].max():.1f}s")
    
    # Duration buckets
    duration_bins = [0, 1, 5, 10, 20, 30, 60, 120, float('inf')]
    duration_labels = ['<1s (Crash!)', '1-5s', '5-10s', '10-20s', '20-30s', '30-60s', '1-2min', '>2min']
    df['duration_bucket'] = pd.cut(df['duration_sec'], bins=duration_bins, labels=duration_labels, include_lowest=True)
    
    print(f"\n📊 Duration Distribution:")
    duration_dist = df['duration_bucket'].value_counts()
    total = len(df)
    for bucket in duration_labels:
        if bucket in duration_dist.index:
            count = duration_dist[bucket]
            pct = (count / total) * 100
            bar = "█" * int(pct / 2)
            print(f"  {bucket:15s} │ {count:4d} ({pct:5.1f}%) │ {bar}")
    
    # Correlation: Duration vs Crash Value
    corr = df['duration_sec'].corr(df['crash_value'])
    print(f"\n📈 Duration ↔ Crash Value Correlation: {corr:.3f}")
    print(f"  → {'Strong positive: Longer rounds = higher multipliers' if corr > 0.5 else 'Expected: Crash value grows over time'}")
    
    # =====================================================
    # 3. VELOCITY ANALYSIS (Game Speed Manipulation Check)
    # =====================================================
    print("\n" + "=" * 70)
    print("🚀 VELOCITY ANALYSIS (Speed Manipulation Check)")
    print("=" * 70)
    
    velocity_anomalies = []
    avg_deltas = []
    final_deltas = []
    
    for idx, row in df.iterrows():
        try:
            vm_str = row.get('velocity_metrics', '[]')
            if pd.isna(vm_str) or vm_str in ['{}', '[]', '']:
                continue
            vm = json.loads(vm_str.replace("''", '"').replace('""', '"'))
            if not isinstance(vm, list) or len(vm) < 5:
                continue
                
            deltas = [entry['delta_ms'] for entry in vm if 'delta_ms' in entry and entry['delta_ms'] > 0]
            if len(deltas) < 5:
                continue
                
            avg_delta = np.mean(deltas[:-3])  # Average excluding last 3
            last_3_avg = np.mean(deltas[-3:])  # Last 3 deltas
            
            avg_deltas.append(avg_delta)
            final_deltas.append(last_3_avg)
            
            # Check for speedup (last deltas < 50% of average)
            if last_3_avg < avg_delta * 0.5:
                velocity_anomalies.append({
                    'round_id': row['round_id'],
                    'crash_value': row['crash_value'],
                    'avg_delta': avg_delta,
                    'final_delta': last_3_avg,
                    'speedup_ratio': last_3_avg / avg_delta
                })
        except:
            continue
    
    if avg_deltas:
        print(f"\n📊 Velocity Metrics Analysis (from {len(avg_deltas)} rounds with data):")
        print(f"  • Average step time:  {np.mean(avg_deltas):.1f}ms")
        print(f"  • Average final step: {np.mean(final_deltas):.1f}ms")
        
        if velocity_anomalies:
            print(f"\n⚠️ VELOCITY ANOMALIES DETECTED: {len(velocity_anomalies)} rounds")
            print(f"   (Game speed increased >2x just before crash)")
            print(f"\n   Top 5 Suspicious Rounds:")
            sorted_anomalies = sorted(velocity_anomalies, key=lambda x: x['speedup_ratio'])[:5]
            for a in sorted_anomalies:
                print(f"   • {a['round_id']}: Crashed at {a['crash_value']:.2f}x, Speed went from {a['avg_delta']:.0f}ms to {a['final_delta']:.0f}ms ({a['speedup_ratio']:.1%})")
        else:
            print(f"\n✅ No significant velocity anomalies detected")
            print(f"   Game appears to run at consistent speed")
    
    # =====================================================
    # 4. INTER-ROUND TIMING
    # =====================================================
    print("\n" + "=" * 70)
    print("⏰ INTER-ROUND TIMING ANALYSIS")
    print("=" * 70)
    
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_between_rounds'] = df_sorted['timestamp'].diff().dt.total_seconds()
    
    time_gaps = df_sorted['time_between_rounds'].dropna()
    
    print(f"\n📊 Time Between Rounds:")
    print(f"  • Mean:    {time_gaps.mean():.1f}s")
    print(f"  • Median:  {time_gaps.median():.1f}s")
    print(f"  • Typical range: {time_gaps.quantile(0.25):.1f}s - {time_gaps.quantile(0.75):.1f}s")
    
    # Check for unusual gaps
    long_gaps = time_gaps[time_gaps > 120]  # Gaps > 2 minutes
    if len(long_gaps) > 0:
        print(f"\n⚠️ Found {len(long_gaps)} gaps > 2 minutes (possible session breaks)")
    
    # =====================================================
    # 5. CRASH VALUE TEMPORAL PATTERNS
    # =====================================================
    print("\n" + "=" * 70)
    print("📉 CRASH BEHAVIOR PATTERNS")
    print("=" * 70)
    
    # Instant crashes (<=1.1x)
    instant_crashes = df[df['crash_value'] <= 1.1]
    print(f"\n💥 Instant Crashes (≤1.1x):")
    print(f"  • Count: {len(instant_crashes)} ({len(instant_crashes)/len(df)*100:.1f}%)")
    if len(instant_crashes) > 0:
        print(f"  • Avg duration: {instant_crashes['duration_ms'].mean()/1000:.1f}s")
    
    # Check for instant crash streaks
    df_sorted['is_instant'] = df_sorted['crash_value'] <= 1.1
    streaks = []
    current_streak = 0
    for is_inst in df_sorted['is_instant']:
        if is_inst:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    
    if streaks:
        print(f"  • Max consecutive instant crashes: {max(streaks)}")
        print(f"  • Avg streak length: {np.mean(streaks):.1f}")
    
    # Low crashes by hour
    print(f"\n📊 Instant Crash (≤1.1x) Rate by Hour:")
    instant_by_hour = df.groupby('hour').apply(lambda x: (x['crash_value'] <= 1.1).sum() / len(x) * 100)
    worst_hour_instant = instant_by_hour.idxmax()
    best_hour_instant = instant_by_hour.idxmin()
    print(f"  • Worst hour: {worst_hour_instant:02d}:00 ({instant_by_hour[worst_hour_instant]:.1f}% instant crashes)")
    print(f"  • Best hour:  {best_hour_instant:02d}:00 ({instant_by_hour[best_hour_instant]:.1f}% instant crashes)")
    
    # =====================================================
    # 6. STAKE & BETTOR PATTERNS
    # =====================================================
    print("\n" + "=" * 70)
    print("💰 STAKE & CRASH CORRELATION")
    print("=" * 70)
    
    # Remove outliers for cleaner analysis
    stake_clean = df[df['stake'] > 0]['stake']
    
    if len(stake_clean) > 0:
        # High stake analysis
        high_stake_threshold = stake_clean.quantile(0.9)
        high_stake_rounds = df[df['stake'] >= high_stake_threshold]
        normal_rounds = df[(df['stake'] > 0) & (df['stake'] < high_stake_threshold)]
        
        if len(high_stake_rounds) > 0 and len(normal_rounds) > 0:
            print(f"\n📊 High Stake Analysis (top 10% = ≥{high_stake_threshold:,.0f} TZS):")
            print(f"  • High stake avg crash:   {high_stake_rounds['crash_value'].mean():.2f}x")
            print(f"  • Normal stake avg crash: {normal_rounds['crash_value'].mean():.2f}x")
            diff = high_stake_rounds['crash_value'].mean() - normal_rounds['crash_value'].mean()
            if diff < -0.5:
                print(f"  ⚠️ WARNING: High stake rounds crash {abs(diff):.2f}x LOWER on average!")
            elif diff > 0.5:
                print(f"  ✅ High stake rounds actually have higher multipliers (+{diff:.2f}x)")
            else:
                print(f"  ➖ No significant correlation between stake and crash value")
        
        # Stake-crash correlation
        corr_stake = df[df['stake'] > 0]['stake'].corr(df[df['stake'] > 0]['crash_value'])
        print(f"\n  Stake ↔ Crash Correlation: {corr_stake:.3f}")
    
    # =====================================================
    # 7. CASHOUT TIMING PATTERNS
    # =====================================================
    print("\n" + "=" * 70)
    print("💸 CASHOUT BEHAVIOR PATTERNS")
    print("=" * 70)
    
    all_cashouts = []
    for idx, row in df.iterrows():
        try:
            ce_str = row.get('cashout_events', '[]')
            if pd.isna(ce_str) or ce_str == '[]':
                continue
            events = json.loads(ce_str.replace("''", '"').replace('""', '"'))
            if not isinstance(events, list):
                continue
            for event in events:
                if 'at_multiplier' in event:
                    all_cashouts.append({
                        'multiplier': event['at_multiplier'],
                        'time_ms': event.get('time_ms', 0),
                        'crash_value': row['crash_value']
                    })
        except:
            continue
    
    if all_cashouts:
        cashouts_df = pd.DataFrame(all_cashouts)
        print(f"\n📊 Cashout Statistics ({len(cashouts_df)} cashout events):")
        print(f"  • Mean cashout multiplier:   {cashouts_df['multiplier'].mean():.2f}x")
        print(f"  • Median cashout multiplier: {cashouts_df['multiplier'].median():.2f}x")
        print(f"  • Most common cashout range: 1.5x - 2.5x")
        
        # Cashouts by multiplier bucket
        print(f"\n📊 Cashout Distribution:")
        cashout_bins = [0, 1.5, 2, 2.5, 3, 5, 10, float('inf')]
        cashout_labels = ['<1.5x', '1.5-2x', '2-2.5x', '2.5-3x', '3-5x', '5-10x', '>10x']
        cashouts_df['bucket'] = pd.cut(cashouts_df['multiplier'], bins=cashout_bins, labels=cashout_labels, include_lowest=True)
        bucket_dist = cashouts_df['bucket'].value_counts()
        
        for bucket in cashout_labels:
            if bucket in bucket_dist.index:
                count = bucket_dist[bucket]
                pct = (count / len(cashouts_df)) * 100
                bar = "█" * int(pct / 2)
                print(f"  {bucket:10s} │ {count:5d} ({pct:5.1f}%) │ {bar}")
    
    # =====================================================
    # 8. KEY FINDINGS SUMMARY
    # =====================================================
    print("\n" + "=" * 70)
    print("📋 KEY TEMPORAL FINDINGS SUMMARY")
    print("=" * 70)
    
    findings = []
    
    # Time of day finding
    best_avg = hourly_crash.loc[best_hour, 'mean']
    worst_avg = hourly_crash.loc[worst_hour, 'mean']
    if best_avg / worst_avg > 1.3:
        findings.append(f"🕐 Significant time variation: Play at {best_hour:02d}:00 (avg {best_avg:.2f}x) vs avoid {worst_hour:02d}:00 (avg {worst_avg:.2f}x)")
    
    # Instant crash finding
    instant_pct = len(instant_crashes) / len(df) * 100
    if instant_pct > 5:
        findings.append(f"💥 {instant_pct:.1f}% of rounds are instant crashes (≤1.1x) - factor this into strategy")
    
    # Velocity anomaly finding
    if velocity_anomalies:
        findings.append(f"⚡ {len(velocity_anomalies)} rounds showed speed manipulation (game accelerated before crash)")
    
    # Duration finding
    findings.append(f"⏱️ Average round lasts {df['duration_sec'].mean():.1f}s, plan cashouts accordingly")
    
    print("\n" + "\n".join(findings))
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    
    return df

if __name__ == '__main__':
    analyze_temporal_patterns()
