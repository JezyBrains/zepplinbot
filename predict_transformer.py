#!/usr/bin/env python3
"""
Use trained Transformer model to predict next Zeppelin crash points.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.models.transformer_model import TransformerPredictor
import argparse

def main():
    parser = argparse.ArgumentParser(description='Predict next crash points using Transformer')
    parser.add_argument('--steps', type=int, default=3, help='Number of future crashes to predict')
    parser.add_argument('--data-file', type=str, default='data/zeppelin_data.csv', help='Data file')
    args = parser.parse_args()
    
    print("="*70)
    print("TRANSFORMER PREDICTION - Pattern-Based Crash Forecasting")
    print("="*70)
    
    # Load model
    print("\n🤖 Loading Transformer model...")
    predictor = TransformerPredictor()
    
    try:
        predictor.load_model('models/saved/transformer')
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("\n   Run 'python3 train_transformer.py' first to train the model.")
        return
    
    # Load data
    print(f"\n📊 Loading crash data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    
    if 'value' in df.columns:
        data = df['value'].values
    elif 'coefficient' in df.columns:
        data = df['coefficient'].values
    else:
        print("❌ Error: No coefficient/value column found")
        return
    
    print(f"   Total crashes in history: {len(data)}")
    print(f"   Last crash: {data[-1]:.2f}x")
    
    # Make prediction
    print(f"\n🔮 Predicting next {args.steps} crash points...")
    print("   (Using attention mechanism to analyze sequence patterns)")
    
    recent_data = data[-predictor.sequence_length:]
    predictions = predictor.predict_next(recent_data, steps=args.steps)
    
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    for i, pred in enumerate(predictions, 1):
        # Clamp to reasonable range
        pred_clamped = max(1.0, min(pred, 50.0))
        
        print(f"\n   Step {i}: {pred_clamped:.2f}x")
        
        # Confidence based on historical distribution
        if pred_clamped < 2.0:
            confidence = "High (most common range)"
        elif pred_clamped < 5.0:
            confidence = "Medium (moderate range)"
        else:
            confidence = "Low (rare high multiplier)"
        
        print(f"           Confidence: {confidence}")
    
    # Show context
    print("\n" + "="*70)
    print("CONTEXT")
    print("="*70)
    
    print(f"\n   Last 10 actual crashes:")
    print(f"   {data[-10:].tolist()}")
    
    print(f"\n   Dataset statistics:")
    print(f"   - Mean: {data.mean():.2f}x")
    print(f"   - Median: {np.median(data):.2f}x")
    print(f"   - Std Dev: {data.std():.2f}x")
    
    # Calculate percentiles for predictions
    print(f"\n   Prediction analysis:")
    for i, pred in enumerate(predictions, 1):
        pred_clamped = max(1.0, min(pred, 50.0))
        percentile = (data < pred_clamped).sum() / len(data) * 100
        print(f"   - Step {i} ({pred_clamped:.2f}x) is at {percentile:.1f}th percentile")
    
    print("\n" + "="*70)
    print("BETTING STRATEGY")
    print("="*70)
    
    avg_pred = predictions.mean()
    
    print(f"\n   Average prediction: {avg_pred:.2f}x")
    print(f"\n   Recommended cash-out points:")
    print(f"   - Conservative: {avg_pred * 0.6:.2f}x (60% of prediction)")
    print(f"   - Moderate:     {avg_pred * 0.8:.2f}x (80% of prediction)")
    print(f"   - Aggressive:   {avg_pred:.2f}x (full prediction)")
    
    print("\n" + "="*70)
    print("\n⚠️  Remember: This exploits statistical patterns, not guarantees.")
    print("   Test predictions vs actual results to validate accuracy.")
    print("\n   To update with new crash data:")
    print("   1. python3 manual_collect.py")
    print("   2. python3 train_transformer.py")
    print("   3. python3 predict_transformer.py")

if __name__ == "__main__":
    main()
