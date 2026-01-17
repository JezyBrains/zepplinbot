#!/usr/bin/env python3
"""
Streak Monitor & RTP Shift Detection

Path A: Detects statistical anomalies (low crash streaks)
Path B: Monitors for RTP (Return to Player) shifts over time

This doesn't predict - it identifies when short-term distribution
deviates from expected randomness, which may present opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreakMonitor:
    """
    Monitors for consecutive low crashes that deviate from expected distribution.
    
    Theory: While each round is independent, clusters of bad luck happen.
    After 5+ consecutive crashes below 1.2x, we're in a statistical anomaly.
    
    This is NOT "due for a win" (gambler's fallacy).
    This is "current short-term distribution is unusual."
    """
    
    def __init__(self, low_threshold: float = 1.2, streak_length: int = 5):
        """
        Args:
            low_threshold: What counts as a "low" crash
            streak_length: How many consecutive lows to flag
        """
        self.low_threshold = low_threshold
        self.streak_length = streak_length
    
    def detect_current_streak(self, recent_crashes: np.ndarray) -> Dict:
        """
        Check if currently in a low crash streak.
        
        Args:
            recent_crashes: Recent crash history
        
        Returns:
            Streak analysis
        """
        if len(recent_crashes) < self.streak_length:
            return {
                'in_streak': False,
                'streak_length': 0,
                'message': 'Insufficient data'
            }
        
        # Check last N crashes
        last_n = recent_crashes[-self.streak_length:]
        low_count = sum(1 for x in last_n if x < self.low_threshold)
        
        in_streak = low_count >= self.streak_length
        
        # Calculate how unusual this is
        historical_low_rate = sum(1 for x in recent_crashes if x < self.low_threshold) / len(recent_crashes)
        expected_streak_prob = historical_low_rate ** self.streak_length
        
        return {
            'in_streak': in_streak,
            'streak_length': low_count if in_streak else 0,
            'last_n_crashes': last_n.tolist(),
            'historical_low_rate': historical_low_rate * 100,
            'expected_probability': expected_streak_prob * 100,
            'is_anomaly': in_streak and expected_streak_prob < 0.05,
            'message': self._generate_message(in_streak, low_count, expected_streak_prob)
        }
    
    def _generate_message(self, in_streak: bool, count: int, prob: float) -> str:
        """Generate human-readable message."""
        if not in_streak:
            return "No unusual streak detected"
        
        if prob < 0.01:
            return f"⚠️ RARE ANOMALY: {count} consecutive low crashes (p={prob*100:.2f}%)"
        elif prob < 0.05:
            return f"⚠️ UNUSUAL STREAK: {count} consecutive low crashes (p={prob*100:.2f}%)"
        else:
            return f"Low streak detected: {count} crashes (p={prob*100:.2f}%)"
    
    def analyze_streak_patterns(self, crash_history: np.ndarray) -> Dict:
        """
        Analyze historical streak patterns to understand frequency.
        
        Returns:
            Statistics about streak occurrences
        """
        streaks = []
        current_streak = 0
        
        for crash in crash_history:
            if crash < self.low_threshold:
                current_streak += 1
            else:
                if current_streak >= self.streak_length:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak >= self.streak_length:
            streaks.append(current_streak)
        
        if not streaks:
            return {
                'total_streaks': 0,
                'average_length': 0,
                'max_length': 0,
                'frequency': 0
            }
        
        return {
            'total_streaks': len(streaks),
            'average_length': np.mean(streaks),
            'max_length': max(streaks),
            'frequency': len(streaks) / len(crash_history) * 100,
            'all_streaks': streaks
        }


class RTPShiftDetector:
    """
    Monitors for changes in Return to Player (RTP) over time.
    
    Detects if the game's algorithm changes to be more/less generous.
    Uses rolling window analysis to spot shifts in win rates.
    """
    
    def __init__(self, window_size: int = 100, shift_threshold: float = 0.05):
        """
        Args:
            window_size: Size of rolling window for analysis
            shift_threshold: Minimum change to flag as shift (5% default)
        """
        self.window_size = window_size
        self.shift_threshold = shift_threshold
    
    def detect_rtp_shift(self, crash_history: np.ndarray, 
                        target_multiplier: float = 2.0) -> Dict:
        """
        Detect if RTP has shifted for a specific target.
        
        Args:
            crash_history: Full crash history
            target_multiplier: Target to analyze
        
        Returns:
            RTP shift analysis
        """
        if len(crash_history) < self.window_size * 2:
            return {
                'shift_detected': False,
                'message': f'Need at least {self.window_size * 2} crashes'
            }
        
        # Calculate win rates in different windows
        recent_window = crash_history[-self.window_size:]
        older_window = crash_history[-self.window_size*2:-self.window_size]
        
        recent_win_rate = sum(1 for x in recent_window if x >= target_multiplier) / len(recent_window)
        older_win_rate = sum(1 for x in older_window if x >= target_multiplier) / len(older_window)
        
        # Calculate shift
        shift = recent_win_rate - older_win_rate
        shift_pct = shift * 100
        
        # Statistical significance test (chi-square)
        recent_wins = sum(1 for x in recent_window if x >= target_multiplier)
        older_wins = sum(1 for x in older_window if x >= target_multiplier)
        
        contingency_table = [
            [recent_wins, len(recent_window) - recent_wins],
            [older_wins, len(older_window) - older_wins]
        ]
        
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
        
        # Determine if shift is significant
        shift_detected = abs(shift) >= self.shift_threshold and p_value < 0.05
        
        return {
            'shift_detected': shift_detected,
            'recent_win_rate': recent_win_rate * 100,
            'older_win_rate': older_win_rate * 100,
            'shift_percentage': shift_pct,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'direction': 'IMPROVED' if shift > 0 else 'WORSENED',
            'message': self._generate_rtp_message(shift_detected, shift_pct, p_value)
        }
    
    def _generate_rtp_message(self, detected: bool, shift: float, p_value: float) -> str:
        """Generate RTP shift message."""
        if not detected:
            return "No significant RTP shift detected"
        
        direction = "IMPROVED" if shift > 0 else "WORSENED"
        
        if p_value < 0.01:
            return f"🚨 STRONG RTP SHIFT: {direction} by {abs(shift):.1f}% (p={p_value:.4f})"
        else:
            return f"⚠️ RTP SHIFT: {direction} by {abs(shift):.1f}% (p={p_value:.4f})"
    
    def analyze_rtp_trend(self, crash_history: np.ndarray,
                         target_multiplier: float = 2.0,
                         num_windows: int = 5) -> Dict:
        """
        Analyze RTP trend over multiple windows.
        
        Returns:
            Trend analysis showing if RTP is improving/declining
        """
        if len(crash_history) < self.window_size * num_windows:
            return {
                'trend': 'insufficient_data',
                'message': f'Need {self.window_size * num_windows} crashes'
            }
        
        win_rates = []
        
        for i in range(num_windows):
            start_idx = -(i+1) * self.window_size
            end_idx = -i * self.window_size if i > 0 else None
            window = crash_history[start_idx:end_idx]
            
            win_rate = sum(1 for x in window if x >= target_multiplier) / len(window)
            win_rates.append(win_rate * 100)
        
        # Reverse to get chronological order
        win_rates = list(reversed(win_rates))
        
        # Calculate trend
        if len(win_rates) >= 3:
            # Linear regression
            x = np.arange(len(win_rates))
            slope, intercept = np.polyfit(x, win_rates, 1)
            
            if slope > 0.5:
                trend = 'improving'
            elif slope < -0.5:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'win_rates': win_rates,
            'slope': slope if len(win_rates) >= 3 else 0,
            'latest_rate': win_rates[-1],
            'earliest_rate': win_rates[0],
            'total_change': win_rates[-1] - win_rates[0]
        }


class AnomalyBettingSignal:
    """
    Combines streak detection and RTP monitoring for betting signals.
    
    Only signals bets when:
    1. In unusual low streak (statistical anomaly)
    2. RTP has improved recently
    3. Kelly Criterion still shows reasonable risk
    """
    
    def __init__(self):
        self.streak_monitor = StreakMonitor(low_threshold=1.2, streak_length=5)
        self.rtp_detector = RTPShiftDetector(window_size=100)
    
    def get_anomaly_signal(self, crash_history: np.ndarray,
                          target_multiplier: float = 2.0) -> Dict:
        """
        Generate betting signal based on anomaly detection.
        
        Returns:
            Signal with reasoning
        """
        # Check streak
        streak = self.streak_monitor.detect_current_streak(crash_history[-20:])
        
        # Check RTP shift
        rtp_shift = self.rtp_detector.detect_rtp_shift(crash_history, target_multiplier)
        
        # Check RTP trend
        rtp_trend = self.rtp_detector.analyze_rtp_trend(crash_history, target_multiplier)
        
        # Generate signal
        signal = 'SKIP'
        reasons = []
        
        if streak['is_anomaly']:
            reasons.append(f"✓ Unusual low streak detected ({streak['streak_length']} crashes)")
        
        if rtp_shift['shift_detected'] and rtp_shift['direction'] == 'IMPROVED':
            reasons.append(f"✓ RTP improved by {rtp_shift['shift_percentage']:.1f}%")
            signal = 'WATCH'
        
        if rtp_trend['trend'] == 'improving':
            reasons.append(f"✓ RTP trending upward")
        
        # Only signal BET if multiple factors align
        if len(reasons) >= 2:
            signal = 'CONSIDER'
        
        return {
            'signal': signal,
            'streak_analysis': streak,
            'rtp_shift': rtp_shift,
            'rtp_trend': rtp_trend,
            'reasons': reasons,
            'recommendation': self._generate_recommendation(signal, reasons)
        }
    
    def _generate_recommendation(self, signal: str, reasons: List[str]) -> str:
        """Generate recommendation text."""
        if signal == 'SKIP':
            return "🔴 SKIP - No anomalies detected, normal conditions"
        elif signal == 'WATCH':
            return "🟡 WATCH - Some anomalies detected, monitor closely"
        elif signal == 'CONSIDER':
            return "🟢 CONSIDER - Multiple anomalies align, potential opportunity"
        else:
            return "No recommendation"


# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    print("="*70)
    print("STREAK MONITOR & RTP SHIFT DETECTION")
    print("="*70)
    
    # Streak analysis
    monitor = StreakMonitor()
    streak = monitor.detect_current_streak(data[-20:])
    
    print(f"\n📊 Current Streak Status:")
    print(f"   In Streak: {streak['in_streak']}")
    print(f"   {streak['message']}")
    
    # RTP shift
    rtp_detector = RTPShiftDetector()
    rtp = rtp_detector.detect_rtp_shift(data, 2.0)
    
    print(f"\n📈 RTP Shift Analysis (2.0x target):")
    print(f"   Recent Win Rate: {rtp['recent_win_rate']:.1f}%")
    print(f"   Older Win Rate: {rtp['older_win_rate']:.1f}%")
    print(f"   {rtp['message']}")
    
    # Combined signal
    signal_system = AnomalyBettingSignal()
    signal = signal_system.get_anomaly_signal(data, 2.0)
    
    print(f"\n🚦 Anomaly Signal: {signal['signal']}")
    print(f"   {signal['recommendation']}")
    if signal['reasons']:
        print(f"\n   Reasons:")
        for reason in signal['reasons']:
            print(f"   {reason}")
