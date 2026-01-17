#!/usr/bin/env python3
"""
Train Transformer model on Zeppelin crash data.

This script trains the Transformer architecture to exploit the
autocorrelation pattern detected in cryptanalysis (0.44 at lag 15).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.models.transformer_model import TransformerPredictor
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("TRANSFORMER MODEL TRAINING - Exploiting Sequence Patterns")
    print("="*70)
    
    # Load data
    print("\n📊 Loading crash data...")
    df = pd.read_csv('data/zeppelin_data.csv')
    
    if 'value' in df.columns:
        data = df['value'].values
    elif 'coefficient' in df.columns:
        data = df['coefficient'].values
    else:
        print("❌ Error: No coefficient/value column found")
        return
    
    print(f"   Total data points: {len(data)}")
    print(f"   Range: {data.min():.2f}x - {data.max():.2f}x")
    print(f"   Mean: {data.mean():.2f}x")
    
    # Check if enough data
    if len(data) < 60:
        print(f"\n⚠️  Warning: Only {len(data)} data points available.")
        print(f"   Transformer works best with 100+ points.")
        print(f"   Training anyway, but accuracy may be limited.")
    
    # Initialize Transformer
    print("\n🤖 Initializing Transformer model...")
    print("   Architecture:")
    print("   - Sequence length: 50 (looks at last 50 crashes)")
    print("   - Attention heads: 4 (finds patterns at different scales)")
    print("   - Transformer blocks: 2 (deep pattern recognition)")
    print("   - Embedding dim: 64")
    
    predictor = TransformerPredictor(
        sequence_length=min(50, len(data) - 10),  # Adjust if not enough data
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_blocks=2
    )
    
    # Train
    print("\n🔥 Training Transformer...")
    print("   This exploits the autocorrelation at lags: 15, 20, 35, 43, 48")
    print("   (Detected in cryptanalysis)")
    
    try:
        history = predictor.train(
            data,
            epochs=150,
            validation_split=0.2,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        # Save model
        os.makedirs('models/saved', exist_ok=True)
        predictor.save_model('models/saved/transformer')
        print("   Model saved to: models/saved/transformer")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Prediction Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/transformer_training.png', dpi=150)
        print("   Training plot saved to: outputs/transformer_training.png")
        
        # Test predictions
        print("\n🔮 Testing predictions on recent data...")
        
        # Use last 50 points to predict next 5
        recent_data = data[-predictor.sequence_length:]
        predictions = predictor.predict_next(recent_data, steps=5)
        
        print("\n   Predicted next 5 crashes:")
        for i, pred in enumerate(predictions, 1):
            print(f"      Step {i}: {pred:.2f}x")
        
        # Calculate baseline (mean prediction)
        baseline_pred = data.mean()
        print(f"\n   Baseline (mean): {baseline_pred:.2f}x")
        print(f"   Transformer predictions: {predictions.mean():.2f}x (avg)")
        
        # Show recent actual values for context
        print(f"\n   Recent actual crashes (last 10):")
        print(f"      {data[-10:].tolist()}")
        
        print("\n" + "="*70)
        print("✅ TRANSFORMER MODEL READY")
        print("="*70)
        print("\n📌 Next steps:")
        print("   1. Collect more crash data to improve accuracy")
        print("   2. Run: python3 predict_transformer.py")
        print("   3. Compare predictions with actual results")
        print("   4. Calculate your statistical advantage (ε)")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
