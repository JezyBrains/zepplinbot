#!/usr/bin/env python3
"""
Add actual crash results and analyze prediction accuracy.
"""
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Actual crashes after 2.61x
actual_crashes = [1.48, 1.76, 1.32, 1.36, 1.04, 1.22, 2.35, 8.05, 1.00, 8.69]

# Predicted crashes were: 4.71, 4.87, 4.81, 4.87, 4.82
predicted_crashes = [4.71, 4.87, 4.81, 4.87, 4.82]

print("="*70)
print("PREDICTION ACCURACY ANALYSIS")
print("="*70)

print(f"\n📊 Predicted vs Actual (first 5):")
print(f"\n{'Step':<6} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'% Error'}")
print("-"*70)

errors = []
for i in range(min(5, len(actual_crashes))):
    pred = predicted_crashes[i]
    actual = actual_crashes[i]
    error = abs(pred - actual)
    pct_error = (error / actual) * 100
    errors.append(error)
    
    print(f"{i+1:<6} {pred:<12.2f} {actual:<12.2f} {error:<12.2f} {pct_error:.1f}%")

print("-"*70)
print(f"Mean Absolute Error: {np.mean(errors):.2f}x")
print(f"Average % Error: {np.mean([abs(predicted_crashes[i] - actual_crashes[i])/actual_crashes[i]*100 for i in range(5)]):.1f}%")

print(f"\n❌ PREDICTION FAILED")
print(f"   Model predicted: 4.81x average")
print(f"   Actual average: {np.mean(actual_crashes[:5]):.2f}x")
print(f"   Model was {predicted_crashes[0] - actual_crashes[0]:.2f}x too high on first prediction")

print(f"\n🔍 Pattern Analysis:")
print(f"   Predicted: High crashes (4.7-4.9x)")
print(f"   Actual: Low crashes (1.04-2.35x) with outliers (8.05x, 8.69x)")
print(f"   Issue: Model didn't predict the low streak")

print(f"\n📈 All 10 actual crashes:")
print(f"   {actual_crashes}")
print(f"   Mean: {np.mean(actual_crashes):.2f}x")
print(f"   Median: {np.median(actual_crashes):.2f}x")

# Add to dataset
df = pd.read_csv('data/zeppelin_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

last_timestamp = df['timestamp'].iloc[-1]
new_data = []

for i, crash in enumerate(actual_crashes):
    timestamp = last_timestamp + timedelta(minutes=i + 1)
    new_data.append({
        'timestamp': timestamp,
        'value': crash
    })

new_df = pd.DataFrame(new_data)
combined_df = pd.concat([df, new_df], ignore_index=True)
combined_df.to_csv('data/zeppelin_data.csv', index=False)

print(f"\n✅ Added {len(actual_crashes)} new crashes to dataset")
print(f"   Total dataset: {len(combined_df)} crashes")
print(f"   Last crash: {combined_df['value'].iloc[-1]}x")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\n💡 Key Findings:")
print("   1. Transformer predictions were significantly off")
print("   2. Model predicted high (4.8x), actual was low (1.5x avg)")
print("   3. The autocorrelation pattern may be weaker than detected")
print("   4. High volatility (8.05x, 8.69x spikes) makes prediction difficult")
print("\n🔄 Recommendation:")
print("   - Retrain model with new data")
print("   - Model needs more data to handle volatility")
print("   - Consider ensemble approach with conservative baseline")
