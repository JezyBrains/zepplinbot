#!/usr/bin/env python3
"""
Feature Engineering Module for Zeppelin Pro V2

Implements advanced behavioral features for regime detection:
- Pool Overload Factor (POF)
- Multiplier Velocity (MV)
- Whale Density Index (WDI)
- Cashout Pressure (CP)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import json
import os

# Data file path
ROUND_DATA_FILE = 'data/round_timing.csv'


class FeatureEngine:
    """
    Centralized feature engineering for the Regime Detection Engine.
    Computes all advanced behavioral metrics from round telemetry.
    """
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.pool_history = deque(maxlen=500)
        self.velocity_history = deque(maxlen=100)
        self.whale_threshold_percentile = 90
        
    def load_round_data(self, limit: int = None) -> pd.DataFrame:
        """Load round timing data from CSV."""
        if not os.path.exists(ROUND_DATA_FILE):
            return pd.DataFrame()
        try:
            df = pd.read_csv(ROUND_DATA_FILE, on_bad_lines='skip', low_memory=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"Feature Engine: Error loading data: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # POOL OVERLOAD FACTOR (POF)
    # =========================================================================
    def calculate_pof(self, current_stake: float, df: pd.DataFrame = None) -> Dict:
        """
        Calculate Pool Overload Factor.
        
        POF = current_stake / rolling_24h_avg_stake
        
        POF > 1.5: High house exposure, potential early crash
        POF < 0.5: Low exposure, potential for higher multipliers
        """
        if df is None or len(df) == 0:
            df = self.load_round_data()
        
        if len(df) < 10 or 'stake' not in df.columns:
            return {'pof': 1.0, 'avg_stake': 0, 'signal': 'NEUTRAL', 'confidence': 0}
        
        # Filter to last 24 hours
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)
        recent = df[df['timestamp'] > cutoff]
        
        if len(recent) < 5:
            recent = df.tail(50)  # Fallback to last 50 rounds
        
        avg_stake = recent['stake'].mean()
        if avg_stake <= 0:
            return {'pof': 1.0, 'avg_stake': 0, 'signal': 'NEUTRAL', 'confidence': 0}
        
        pof = current_stake / avg_stake
        
        # Determine signal
        if pof >= 2.0:
            signal, confidence = 'SKIP', 0.85
        elif pof >= 1.5:
            signal, confidence = 'CAUTION', 0.65
        elif pof <= 0.4:
            signal, confidence = 'OPPORTUNITY', 0.7
        else:
            signal, confidence = 'NEUTRAL', 0.5
        
        return {
            'pof': round(pof, 3),
            'current_stake': current_stake,
            'avg_stake': round(avg_stake, 2),
            'signal': signal,
            'confidence': confidence
        }
    
    # =========================================================================
    # MULTIPLIER VELOCITY (MV)
    # =========================================================================
    def calculate_mv(self, velocity_metrics: List[Dict]) -> Dict:
        """
        Calculate Multiplier Velocity metrics.
        
        MV = Δcoef / Δt for each tick
        
        Detects:
        - Acceleration (speedup before crash)
        - Deceleration (potential for higher multiplier)
        - Stability (normal progression)
        """
        if not velocity_metrics or len(velocity_metrics) < 5:
            return {'mv_avg': 0, 'mv_trend': 'UNKNOWN', 'anomaly': False, 'speedup_ratio': 1.0}
        
        # Calculate velocity for each step
        velocities = []
        for i in range(1, len(velocity_metrics)):
            delta_coef = velocity_metrics[i].get('coef', 0) - velocity_metrics[i-1].get('coef', 0)
            delta_t = velocity_metrics[i].get('delta_ms', 100)
            if delta_t > 0:
                velocities.append(delta_coef / delta_t * 1000)  # Normalize to per-second
        
        if len(velocities) < 3:
            return {'mv_avg': 0, 'mv_trend': 'UNKNOWN', 'anomaly': False, 'speedup_ratio': 1.0}
        
        mv_avg = np.mean(velocities)
        mv_early = np.mean(velocities[:len(velocities)//2])
        mv_late = np.mean(velocities[len(velocities)//2:])
        
        # Speedup ratio (late vs early)
        speedup_ratio = mv_late / mv_early if mv_early > 0 else 1.0
        
        # Detect anomaly (2x speedup)
        anomaly = speedup_ratio > 2.0
        
        # Trend classification
        if speedup_ratio > 1.5:
            trend = 'ACCELERATING'
        elif speedup_ratio < 0.7:
            trend = 'DECELERATING'
        else:
            trend = 'STABLE'
        
        return {
            'mv_avg': round(mv_avg, 4),
            'mv_early': round(mv_early, 4),
            'mv_late': round(mv_late, 4),
            'speedup_ratio': round(speedup_ratio, 3),
            'mv_trend': trend,
            'anomaly': anomaly
        }
    
    # =========================================================================
    # WHALE DENSITY INDEX (WDI)
    # =========================================================================
    def calculate_wdi(self, df: pd.DataFrame = None, lookback_rounds: int = 50) -> Dict:
        """
        Calculate Whale Density Index.
        
        WDI = (HighStakeCount * HighStakeVolume) / TotalPool
        
        High WDI indicates concentration risk (whales may trigger house protection).
        """
        if df is None or len(df) == 0:
            df = self.load_round_data()
        
        if len(df) < 10 or 'stake' not in df.columns:
            return {'wdi': 0, 'whale_count': 0, 'whale_volume': 0, 'signal': 'NEUTRAL'}
        
        recent = df.tail(lookback_rounds)
        
        # Define whale threshold (top 10% of stakes)
        stake_values = recent['stake'].dropna()
        if len(stake_values) < 5:
            return {'wdi': 0, 'whale_count': 0, 'whale_volume': 0, 'signal': 'NEUTRAL'}
        
        whale_threshold = stake_values.quantile(self.whale_threshold_percentile / 100)
        
        # Calculate whale metrics
        whale_rounds = recent[recent['stake'] >= whale_threshold]
        whale_count = len(whale_rounds)
        whale_volume = whale_rounds['stake'].sum()
        total_pool = stake_values.sum()
        
        if total_pool <= 0:
            return {'wdi': 0, 'whale_count': 0, 'whale_volume': 0, 'signal': 'NEUTRAL'}
        
        wdi = (whale_count * whale_volume) / total_pool
        
        # Normalize WDI to 0-1 scale (empirical scaling)
        wdi_normalized = min(1.0, wdi / 100)
        
        # Signal determination
        if wdi_normalized > 0.7:
            signal = 'HIGH_RISK'
        elif wdi_normalized > 0.4:
            signal = 'ELEVATED'
        else:
            signal = 'NORMAL'
        
        return {
            'wdi': round(wdi_normalized, 3),
            'wdi_raw': round(wdi, 2),
            'whale_count': whale_count,
            'whale_volume': round(whale_volume, 2),
            'whale_threshold': round(whale_threshold, 2),
            'signal': signal
        }
    
    # =========================================================================
    # CASHOUT PRESSURE (CP)
    # =========================================================================
    def calculate_cp(self, cashout_events: List[Dict], total_bettors: int) -> Dict:
        """
        Calculate Cashout Pressure.
        
        CP = cumulative_cashouts / total_bettors at each multiplier level
        
        High early CP suggests crowd expects low crash.
        Low CP at high multipliers suggests confidence.
        """
        if not cashout_events or total_bettors <= 0:
            return {'cp_1_5x': 0, 'cp_2x': 0, 'cp_final': 0, 'crowd_sentiment': 'UNKNOWN'}
        
        # Calculate CP at key multiplier levels
        cp_at_1_5 = 0
        cp_at_2 = 0
        total_cashed = 0
        
        for event in cashout_events:
            multiplier = event.get('at_multiplier', 0)
            count = event.get('cashed_count', 0)
            total_cashed += count
            
            if multiplier <= 1.5:
                cp_at_1_5 = total_cashed / total_bettors
            if multiplier <= 2.0:
                cp_at_2 = total_cashed / total_bettors
        
        cp_final = total_cashed / total_bettors
        
        # Crowd sentiment analysis
        if cp_at_1_5 > 0.5:
            sentiment = 'FEARFUL'
        elif cp_at_2 > 0.7:
            sentiment = 'CAUTIOUS'
        elif cp_final < 0.3:
            sentiment = 'GREEDY'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'cp_1_5x': round(cp_at_1_5, 3),
            'cp_2x': round(cp_at_2, 3),
            'cp_final': round(cp_final, 3),
            'total_cashed': total_cashed,
            'total_bettors': total_bettors,
            'crowd_sentiment': sentiment
        }
    
    # =========================================================================
    # COMBINED FEATURE VECTOR
    # =========================================================================
    def get_feature_vector(self, 
                           current_stake: float = 0,
                           velocity_metrics: List[Dict] = None,
                           cashout_events: List[Dict] = None,
                           total_bettors: int = 0) -> Dict:
        """
        Get complete feature vector for regime detection.
        Returns all advanced features in a single dict.
        """
        df = self.load_round_data(limit=500)
        
        pof = self.calculate_pof(current_stake, df)
        mv = self.calculate_mv(velocity_metrics or [])
        wdi = self.calculate_wdi(df)
        cp = self.calculate_cp(cashout_events or [], total_bettors)
        
        # Aggregate risk score (0-1)
        risk_factors = [
            pof['pof'] / 2.0 if pof['pof'] > 1 else 0,  # POF contribution
            1.0 if mv['anomaly'] else 0,                 # MV anomaly
            wdi['wdi'],                                   # WDI contribution
            cp['cp_1_5x']                                 # Early cashout pressure
        ]
        aggregate_risk = min(1.0, sum(risk_factors) / 3)
        
        return {
            'pof': pof,
            'mv': mv,
            'wdi': wdi,
            'cp': cp,
            'aggregate_risk': round(aggregate_risk, 3),
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance for easy import
feature_engine = FeatureEngine()
