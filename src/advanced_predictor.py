#!/usr/bin/env python3
"""
Advanced Crash Predictor
Combines multiple analysis techniques for better predictions
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from datetime import datetime


# Import feature engineering module
try:
    from feature_engineering import feature_engine
    FEATURE_ENGINE_AVAILABLE = True
except ImportError:
    FEATURE_ENGINE_AVAILABLE = False


class BehavioralAnalysis:
    """
    Analyzes betting behavior patterns using advanced feature engineering:
    1. Pool Overload Factor (POF)
    2. Multiplier Velocity (MV)
    3. Whale Density Index (WDI)
    4. Cashout Pressure (CP)
    """
    def __init__(self):
        self.history = deque(maxlen=50)
        self.current = {}
        self.velocity_metrics = []
        self.cashout_events = []
        
    def add_data(self, data: Dict):
        """Update with latest behavioral snapshot"""
        self.current = data
        self.velocity_metrics = data.get('velocityMetrics', [])
        self.cashout_events = data.get('cashoutEvents', [])
        
        if data.get('poolSize', 0) > 0:
            self.history.append({
                'pool_size': data.get('poolSize', 0),
                'bettors': data.get('totalBettors', 0),
                'timestamp': data.get('timestamp')
            })
            
    def get_signal(self) -> Dict:
        """Get signal based on behavioral patterns with V2 advanced features"""
        if not self.current or len(self.history) < 5:
            return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Need more behavioral data'}

        pool_size = self.current.get('poolSize', 0)
        bettors = self.current.get('totalBettors', 0)
        
        # Calculate averages (legacy)
        pools = [b['pool_size'] for b in self.history]
        bettor_counts = [b['bettors'] for b in self.history]
        
        avg_pool = np.mean(pools) if pools else 0
        avg_bettors = np.mean(bettor_counts) if bettor_counts else 0
        
        signal = 'WAIT'
        confidence = 0
        reason = ''
        
        # V2: Use FeatureEngine if available
        pof_data = {}
        mv_data = {}
        wdi_data = {}
        
        if FEATURE_ENGINE_AVAILABLE:
            pof_data = feature_engine.calculate_pof(pool_size)
            mv_data = feature_engine.calculate_mv(self.velocity_metrics)
            wdi_data = feature_engine.calculate_wdi()
            
            # Advanced POF-based signal
            if pof_data.get('pof', 1.0) >= 2.0:
                signal = 'SKIP'
                confidence = 0.85
                reason = f"[POF] Massive Pool ({pof_data['pof']:.2f}x avg) - Crash Likely"
            elif pof_data.get('pof', 1.0) >= 1.5:
                signal = 'SKIP'
                confidence = 0.65
                reason = f"[POF] Pool Overload ({pof_data['pof']:.2f}x avg)"
            elif mv_data.get('anomaly', False):
                signal = 'SKIP'
                confidence = 0.8
                reason = f"[MV] Velocity Anomaly ({mv_data['speedup_ratio']:.2f}x speedup)"
            elif wdi_data.get('signal') == 'HIGH_RISK':
                signal = 'SKIP'
                confidence = 0.7
                reason = f"[WDI] Whale Concentration ({wdi_data['wdi']:.2f})"
            elif pof_data.get('pof', 1.0) <= 0.4 and bettors > 5:
                signal = 'BET'
                confidence = 0.7
                reason = f"[POF] Low Pool Load ({pof_data['pof']:.2f}x avg) - Safe Entry"
        else:
            # Legacy fallback
            if avg_pool > 0 and pool_size > avg_pool * 2.0:
                signal = 'SKIP'
                confidence = 0.85
                reason = f"MASSIVE POOL: {pool_size:,} (2x avg) - Crash Likely"
            elif avg_pool > 0 and pool_size > avg_pool * 1.5:
                signal = 'SKIP'
                confidence = 0.65
                reason = f"Pool Overload: {pool_size:,} (1.5x avg) - High Risk"
            elif avg_bettors > 0 and bettors > avg_bettors * 1.8 and pool_size > avg_pool:
                signal = 'SKIP'
                confidence = 0.6
                reason = f"Crowd FOMO: {bettors} players (1.8x avg) - Volatile"
            elif avg_pool > 0 and pool_size < avg_pool * 0.4 and bettors > 5:
                signal = 'BET'
                confidence = 0.6
                reason = f"Low Pool Load: {pool_size:,} (<0.4x avg) - Safe Entry"
            
        return {
            'signal': signal,
            'confidence': confidence,
            'pool_size': pool_size,
            'avg_pool': avg_pool,
            'bettors': bettors,
            'reason': reason,
            'pof': pof_data,
            'mv': mv_data,
            'wdi': wdi_data
        }


class MLBehavioralSignal:
    """
    Predicts crash values using the pre-trained House Tolerance model.
    """
    def __init__(self, model_path: str = 'models/house_tolerance_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"🤖 ML Signal: Loaded model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ ML Signal: Failed to load model: {e}")
        else:
            print(f"⚠️ ML Signal: Model not found at {self.model_path}")

    def predict(self, stake: float, bettors: int, crash_history: deque) -> Dict:
        if not self.model:
            return {'signal': 'WAIT', 'predicted_crash': 0, 'confidence': 0, 'reason': 'Model not loaded'}
            
        try:
            # Prepare features: [stake, bettors, hour, crash_ma5]
            hour = datetime.now().hour
            crash_list = list(crash_history)
            crash_ma5 = np.mean(crash_list[-5:]) if len(crash_list) >= 5 else 2.0
            
            features = np.array([[stake, bettors, hour, crash_ma5]])
            prediction = float(self.model.predict(features)[0])
            
            # Simple confidence based on how far we are from average prediction
            confidence = 0.7  
            
            signal = 'BET' if prediction >= 2.0 else 'SKIP'
            if prediction < 1.3: signal = 'SKIP' # High danger
            
            return {
                'signal': signal,
                'predicted_crash': prediction,
                'confidence': confidence,
                'reason': f"🤖 ML: Prediction {prediction:.2f}x based on load"
            }
        except Exception as e:
            return {'signal': 'WAIT', 'predicted_crash': 0, 'confidence': 0, 'reason': f"ML Error: {e}"}


class TemporalAnalysis:
    """
    Analyzes "Time Myths" and temporal patterns:
    1. Peak Hour Volatility (Evening spike in players)
    2. Deep Night Stability (Low player count, steady growth)
    3. Hourly Cyclic Patterns
    """
    def __init__(self):
        pass

    def get_signal(self) -> Dict:
        now = datetime.now()
        hour = now.hour
        
        # 0-6: Stability Regime (House often allows steady wins to keep liquidity)
        if 0 <= hour <= 6:
            return {'signal': 'BET', 'confidence': 0.65, 'reason': '🌕 Night Stability: Low player count regime'}
        
        # 18-23: High Volatility (Peak hours, house often wipes out large pools)
        if 18 <= hour <= 23:
            return {'signal': 'SKIP', 'confidence': 0.7, 'reason': '🔥 Peak Volatility: High player mass detected'}
            
        return {'signal': 'WAIT', 'confidence': 0.5, 'reason': '⏳ Normal Temporal state'}

    def get_hourly_bias(self, hour: int) -> float:
        # Static bias based on historical "Time Myths"
        biases = {
            0: 1.1, 1: 1.1, 2: 1.2, 3: 1.2, 4: 1.1, 5: 1.05,
            12: 0.9, 13: 0.9,
            18: 0.8, 19: 0.75, 20: 0.7, 21: 0.75, 22: 0.8, 23: 0.9
        }
        return biases.get(hour, 1.0)


class AdvancedCrashPredictor:
    """
    Multi-technique crash game analyzer combining 8 independent signals.
    """
    
    def __init__(self, target: float = 2.0, bankroll: float = 100000):
        self.target = target
        self.bankroll = bankroll
        self.history = deque(maxlen=500)
        self.behavioral = BehavioralAnalysis()
        self.ml_signal = MLBehavioralSignal()
        self.temporal = TemporalAnalysis()
        self.bait_rounds_left = 0 

    def add_crash(self, value: float):
        self.history.append(value)
        
    def add_crashes(self, values: List[float]):
        for v in values:
            self.history.append(v)

    def add_behavioral_data(self, data: Dict):
        self.behavioral.add_data(data)
    
    def get_behavioral_signal(self) -> Dict:
        return self.behavioral.get_signal()

    def get_trend_signal(self) -> Dict:
        if len(self.history) < 30:
            return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Calculating MAs'}
        
        data = np.array(self.history)
        ma10 = np.mean(data[-10:])
        ma30 = np.mean(data[-30:])
        
        if ma10 > ma30 * 1.1:
            return {'signal': 'BET', 'confidence': 0.6, 'reason': 'Uptrend (MA10 > MA30)'}
        elif ma10 < ma30 * 0.9:
            return {'signal': 'SKIP', 'confidence': 0.6, 'reason': 'Downtrend (MA10 < MA30)'}
        return {'signal': 'WAIT', 'confidence': 0, 'reason': 'No trend'}

    def get_pattern_signal(self) -> Dict:
        if len(self.history) < 10:
            return {'signal': 'WAIT', 'confidence': 0, 'reason': 'No patterns'}
            
        streak = 0
        last_val = self.history[-1]
        is_win = last_val >= self.target
        
        for v in reversed(list(self.history)[:-1]):
            if (v >= self.target) == is_win:
                streak += 1
            else:
                break
        
        current_streak = streak + 1 
        streak_type = 'win' if is_win else 'loss'
        
        if streak_type == 'loss':
            if current_streak >= 5:
                return {'signal': 'BET', 'confidence': 0.85, 'reason': f'{current_streak} Loss Streak - Reversal Likely', 'streak': current_streak, 'streak_type': 'loss'}
            elif current_streak >= 3:
                return {'signal': 'BET', 'confidence': 0.6, 'reason': f'{current_streak} Loss Streak', 'streak': current_streak, 'streak_type': 'loss'}
        
        return {'signal': 'WAIT', 'confidence': 0, 'reason': 'No strong pattern', 'streak': current_streak, 'streak_type': streak_type}

    def get_volatility_signal(self) -> Dict:
        if len(self.history) < 20:
             return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Calculating Volatility', 'regime': 'N/A'}
        
        data = np.array(self.history)
        vol = np.std(data[-20:])
        avg_vol = np.std(data) if len(data) > 50 else vol
        
        if vol > avg_vol * 1.5: 
            return {'signal': 'SKIP', 'confidence': 0.7, 'reason': 'High Volatility', 'regime': 'HIGH'}
        if vol < avg_vol * 0.5: 
            return {'signal': 'BET', 'confidence': 0.6, 'reason': 'Stable Market', 'regime': 'LOW'}
        
        return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Normal Volatility', 'regime': 'NORMAL'}

    def get_regime_signal(self) -> Dict:
         if len(self.history) < 20: return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Detecting Regime', 'regime': 'N/A'}
         
         data = np.array(self.history)
         wins = np.sum(data[-20:] >= self.target)
         win_rate = wins / 20.0
         
         if win_rate > 0.6: return {'signal': 'BET', 'confidence': 0.7, 'reason': 'Hot Regime', 'regime': 'HOT'}
         if win_rate < 0.3: return {'signal': 'SKIP', 'confidence': 0.7, 'reason': 'Cold Regime', 'regime': 'COLD'}
         return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Neutral Regime', 'regime': 'NEUTRAL'}

    def get_statistical_signal(self) -> Dict:
        if len(self.history) < 50: return {'signal': 'WAIT', 'confidence': 0}
        prob = np.mean(np.array(self.history)[-50:] >= self.target)
        if prob > 0.55: return {'signal': 'BET', 'confidence': 0.6, 'reason': f'High Win Rate ({prob:.0%})'}
        return {'signal': 'WAIT', 'confidence': 0, 'reason': 'Statistically Neutral'}

    def calculate_dynamic_target(self, cashout_ratio: float = 0.0, pool_size: float = 0.0) -> Dict:
        if len(self.history) < 10:
            return {'target': 1.5, 'reason': 'Default', 'confidence': 0.3, 'win_probability': 0.5}
        
        data = np.array(list(self.history))
        base_target = 1.5  
        adjustments = []
        confidence = 0.5
        
        if cashout_ratio > 0.8:
            base_target = max(1.3, base_target - 0.3)
            adjustments.append("🚨 High cashout = trap potential")
            confidence -= 0.1
        elif cashout_ratio < 0.3 and cashout_ratio > 0:
            base_target = min(2.0, base_target + 0.2)
            adjustments.append("💰 Low cashout = giveaway likely")
            confidence += 0.1
            
        loss_streak = 0
        for v in reversed(data):
            if v < 1.5: loss_streak += 1
            else: break
                
        if loss_streak >= 5:
            base_target = min(2.5, base_target + 0.5)
            adjustments.append(f"🔥 {loss_streak} streak reversal")
            confidence += 0.2
            
        last_crash = data[-1] if len(data) > 0 else 1.0
        if last_crash > 20:
            base_target = max(1.3, base_target - 0.3)
            adjustments.append("⚡ Post-spike caution")
            
        if last_crash > 20 and cashout_ratio < 0.1 and cashout_ratio > 0:
            self.bait_rounds_left = 3
            adjustments.append("🎣 BAIT DETECTED!")
            
        final_target = round(base_target, 1)
        if self.bait_rounds_left > 0:
            final_target = 1.2
            self.bait_rounds_left -= 1
            confidence += 0.3

        final_target = max(1.1, min(3.0, final_target))
        recent = data[-50:] if len(data) >= 50 else data
        win_prob = np.mean(recent >= final_target)
        
        return {
            'target': final_target,
            'win_probability': win_prob,
            'confidence': max(0.3, min(0.9, confidence)),
            'loss_streak': loss_streak,
            'reason': ' | '.join(adjustments[:2])
        }

    def get_kelly_bet(self, probability: float) -> float:
        b = self.target - 1
        q = 1 - probability
        f = (b * probability - q) / b if b > 0 else 0
        return self.bankroll * max(0, f * 0.25)

    def get_combined_signal(self) -> Dict:
        if len(self.history) < 20:
            return {'signal': 'COLLECT DATA', 'confidence': 0, 'bet_amount': 0, 'target': self.target, 'reason': 'Bootstrapping...', 'analysis': {}}
        
        # 1-6. Standard signals
        trend = self.get_trend_signal()
        pattern = self.get_pattern_signal()
        volatility = self.get_volatility_signal()
        regime = self.get_regime_signal()
        statistical = self.get_statistical_signal()
        behavioral = self.behavioral.get_signal()
        
        # 7. ML Signal
        pool_size = behavioral.get('pool_size', 0)
        bettors = behavioral.get('bettors', 0)
        ml_prediction = self.ml_signal.predict(pool_size, bettors, self.history)
        
        # 8. Temporal Signal
        temporal = self.temporal.get_signal()
        
        # Weighted Voting
        loss_streak = pattern.get('streak', 0)
        is_recovery = pattern.get('streak_type') == 'loss' and loss_streak >= 4
        
        if is_recovery:
            weights = {'trend': 0.1, 'pattern': 0.3, 'volatility': 0.1, 'regime': 0.1, 'statistical': 0.05, 'behavioral': 0.1, 'ml': 0.15, 'temporal': 0.1}
        else:
            weights = {'trend': 0.1, 'pattern': 0.1, 'volatility': 0.1, 'regime': 0.1, 'statistical': 0.05, 'behavioral': 0.15, 'ml': 0.25, 'temporal': 0.15}
            
        signals = {
            'trend': trend, 'pattern': pattern, 'volatility': volatility, 'regime': regime, 
            'statistical': statistical, 'behavioral': behavioral, 'ml': ml_prediction, 'temporal': temporal
        }

        
        bet_score = 0
        skip_score = 0
        total_w = 0
        reasons = []
        
        for name, res in signals.items():
            w = weights.get(name, 0.1)
            if res['signal'] == 'BET': bet_score += w * res['confidence']
            elif res['signal'] == 'SKIP': skip_score += w * res['confidence']
            total_w += w
            if res.get('reason'): reasons.append(f"{name.title()}: {res['reason']}")
            
        bet_score /= total_w
        skip_score /= total_w
        
        # Consensus
        major = 'BET' if bet_score > skip_score else 'SKIP'
        cons_count = sum(1 for r in signals.values() if r['signal'] == major)
        consensus_score = cons_count / len(signals)
        
        if bet_score > skip_score and bet_score > 0.35: signal = 'BET'
        elif skip_score > bet_score and skip_score > 0.35: signal = 'SKIP'
        else: signal = 'WAIT'
        
        prob = np.mean(np.array(self.history)[-50:] >= self.target)
        adj_prob = max(0.1, min(0.9, prob * (1 + (bet_score - 0.5) * 0.2)))
        
        return {
            'signal': signal, 'confidence': max(bet_score, skip_score), 'bet_amount': self.get_kelly_bet(adj_prob) if signal == 'BET' else 0,
            'target': self.target, 'win_probability': adj_prob, 'regime': regime['regime'], 'volatility': volatility['regime'],
            'streak': loss_streak, 'reason': ' | '.join(reasons[:2]), 'consensus_score': consensus_score,
            'analysis': signals
        }

    def get_dashboard_data(self) -> Dict:
        res = self.get_combined_signal()
        colors = {'BET': '#00d97e', 'SKIP': '#e63757', 'WAIT': '#ffc107', 'COLLECT DATA': '#888'}
        res['color'] = colors.get(res['signal'], '#888')
        res['bet'] = res['bet_amount']
        res['prob'] = res.get('win_probability', 0) * 100
        return res

predictor = AdvancedCrashPredictor()

def get_advanced_signal(crash_data: list, bankroll: float = 100000, target: float = 2.0) -> Dict:
    predictor.bankroll = bankroll
    predictor.target = target
    predictor.history.clear()
    predictor.add_crashes(crash_data)
    return predictor.get_dashboard_data()
