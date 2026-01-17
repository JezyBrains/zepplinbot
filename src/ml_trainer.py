import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib

# Paths
DATA_PATH = 'data/round_timing.csv'
MODEL_PATH = 'models/house_tolerance_model.joblib'


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation (Lopez de Prado Method).
    
    Prevents sequential data leakage by:
    1. Purging: Removing samples from training that are too close to test set boundaries
    2. Embargo: Adding a gap after each test set before training data resumes
    
    This is critical for time-series financial data where adjacent samples are correlated.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 5, embargo_pct: float = 0.01):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to remove before/after test boundaries
            embargo_pct: Percentage of training data to embargo after test set
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices for purged train/test splits.
        
        Yields:
            train_indices, test_indices for each fold
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Define test boundaries
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            # Calculate embargo size
            embargo_size = int(fold_size * self.embargo_pct)
            
            # Define purged training indices
            train_indices = []
            
            for i in range(n_samples):
                # Skip if in test set
                if test_start <= i < test_end:
                    continue
                
                # Skip if too close to test boundaries (purge zone)
                if abs(i - test_start) <= self.purge_gap or abs(i - test_end) <= self.purge_gap:
                    continue
                
                # Skip if in embargo zone (right after test set)
                if test_end <= i < test_end + embargo_size:
                    continue
                
                train_indices.append(i)
            
            test_indices = list(range(test_start, test_end))
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class HouseToleranceTrainer:
    def __init__(self, data_path=DATA_PATH, use_purged_cv: bool = True):
        self.data_path = data_path
        self.df = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.use_purged_cv = use_purged_cv

    def load_and_preprocess(self):
        if not os.path.exists(self.data_path):
            print(f"❌ Data file not found: {self.data_path}")
            return False
        
        # Read data, skipping malformed lines (headers/schema changes)
        self.df = pd.read_csv(self.data_path, on_bad_lines='skip')
        
        # Convert timestamps
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hour'] = self.df['timestamp'].dt.hour
        
        # Handle 'seeds' and 'velocity_metrics' if they exist
        # If missing (old data), fill with defaults
        for col in ['total_won', 'velocity_metrics', 'seeds']:
            if col not in self.df.columns:
                self.df[col] = 0 if col == 'total_won' else '{}'
        
        # Feature Engineering: 
        # 1. Total Stake
        # 2. Number of Bettors
        # 3. Time of Day (Hourly)
        # 4. Moving Average of last 5 crash values (Market Sentiment)
        self.df['crash_ma5'] = self.df['crash_value'].rolling(window=5).mean().fillna(2.0)
        
        # V2: Add Pool Overload Factor if stake data is available
        if 'stake' in self.df.columns:
            rolling_avg = self.df['stake'].rolling(window=50, min_periods=1).mean()
            self.df['pof'] = self.df['stake'] / rolling_avg.replace(0, 1)
        
        return True

    def train(self):
        if self.df is None or len(self.df) < 10:
            print("⚠️ Not enough data to train. Need at least 10 rounds.")
            return False
        
        # Select Features (V2: Include POF if available)
        base_features = ['stake', 'bettors', 'hour', 'crash_ma5']
        features = base_features + (['pof'] if 'pof' in self.df.columns else [])
        
        X = self.df[features].fillna(0)
        y = self.df['crash_value']
        
        if self.use_purged_cv:
            # Use Purged Cross-Validation
            print("🔒 Using Purged Cross-Validation (Lopez de Prado method)...")
            purged_cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.02)
            
            cv_scores = []
            for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                preds = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                cv_scores.append(mae)
                print(f"   Fold {fold+1}: MAE = {mae:.3f}")
            
            print(f"📊 Purged CV Average MAE: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
            
            # Final training on all data
            print(f"🧠 Final training on {len(X)} rounds...")
            self.model.fit(X, y)
        else:
            # Legacy split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            print(f"🧠 Training model on {len(X_train)} rounds...")
            self.model.fit(X_train, y_train)
            
            preds = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            print(f"✅ Training complete. MAE: {mae:.2f}")
        
        # Save
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("HOUSE TOLERANCE MODEL TRAINER (V2 with Purged CV)")
    print("=" * 60)
    
    trainer = HouseToleranceTrainer(use_purged_cv=True)
    if trainer.load_and_preprocess():
        trainer.train()
