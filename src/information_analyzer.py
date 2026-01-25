#!/usr/bin/env python3
"""
Information Theory Analyzer

Applies information theory concepts to crash game analysis:
- Shannon entropy of crash distribution
- Mutual information between consecutive crashes
- Autocorrelation analysis
- Pattern existence hypothesis testing

Uses these to mathematically analyze if patterns exist.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.special import rel_entr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationAnalyzer:
    """
    Information-theoretic analysis of crash sequences.
    
    Key concepts:
    - Entropy: Maximum entropy = no predictability (random)
    - Mutual Information: How much knowing X tells you about Y
    - If MI ≈ 0, crashes are independent (unpredictable)
    """
    
    def __init__(self, n_bins: int = 20):
        """
        Args:
            n_bins: Number of bins for discretization
        """
        self.n_bins = n_bins
    
    def calculate_entropy(self, data: np.ndarray) -> Dict:
        """
        Calculate Shannon entropy of crash distribution.
        
        Higher entropy = more uniform = less predictable.
        
        Args:
            data: Array of crash values
        
        Returns:
            Dict with entropy metrics
        """
        # Discretize into bins
        log_data = np.log(data[data > 0])
        hist, bin_edges = np.histogram(log_data, bins=self.n_bins, density=True)
        
        # Remove zeros and normalize
        hist = hist[hist > 0]
        hist = hist / np.sum(hist)
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Maximum possible entropy for this number of bins
        max_entropy = np.log2(self.n_bins)
        
        # Normalized entropy (0-1 scale)
        normalized_entropy = entropy / max_entropy
        
        # Interpretation
        if normalized_entropy > 0.9:
            interpretation = "Very high entropy - nearly uniform/random distribution"
        elif normalized_entropy > 0.7:
            interpretation = "High entropy - mostly unpredictable"
        elif normalized_entropy > 0.5:
            interpretation = "Moderate entropy - some structure exists"
        else:
            interpretation = "Low entropy - significant patterns may exist"
        
        return {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'interpretation': interpretation,
            'n_bins': self.n_bins
        }
    
    def calculate_mutual_information(self, data: np.ndarray, lag: int = 1) -> Dict:
        """
        Calculate mutual information between crashes at different lags.
        
        MI(X_t, X_{t+lag}) measures dependency between crashes.
        
        Args:
            data: Array of crash values
            lag: Time lag between observations
        
        Returns:
            Dict with MI metrics
        """
        log_data = np.log(data[data > 0])
        
        x = log_data[:-lag]
        y = log_data[lag:]
        
        # Create 2D histogram for joint distribution
        joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=self.n_bins)
        joint_hist = joint_hist / np.sum(joint_hist)  # Normalize
        
        # Marginal distributions
        p_x = np.sum(joint_hist, axis=1)
        p_y = np.sum(joint_hist, axis=0)
        
        # Calculate mutual information
        mi = 0
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                if joint_hist[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint_hist[i, j] * np.log2(
                        joint_hist[i, j] / (p_x[i] * p_y[j] + 1e-10)
                    )
        
        # Normalized MI (0-1)
        h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0] + 1e-10))
        h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0] + 1e-10))
        nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0
        
        # Interpretation
        if nmi < 0.05:
            interpretation = "Very low MI - crashes appear independent"
        elif nmi < 0.1:
            interpretation = "Low MI - weak dependency"
        elif nmi < 0.2:
            interpretation = "Moderate MI - some dependency exists"
        else:
            interpretation = "High MI - significant dependency detected"
        
        return {
            'mutual_information': mi,
            'normalized_mi': nmi,
            'lag': lag,
            'interpretation': interpretation,
            'entropy_x': h_x,
            'entropy_y': h_y
        }
    
    def analyze_autocorrelation(self, data: np.ndarray, max_lag: int = 20) -> Dict:
        """
        Analyze autocorrelation at different lags.
        
        Tests if past crashes predict future crashes.
        
        Args:
            data: Array of crash values
            max_lag: Maximum lag to test
        
        Returns:
            Dict with autocorrelation analysis
        """
        log_data = np.log(data[data > 0])
        n = len(log_data)
        mean = np.mean(log_data)
        var = np.var(log_data)
        
        autocorr = []
        significant_lags = []
        
        # Critical value for significance (95% CI)
        critical_value = 1.96 / np.sqrt(n)
        
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            
            # Calculate autocorrelation
            cov = np.sum((log_data[:-lag] - mean) * (log_data[lag:] - mean)) / n
            ac = cov / var if var > 0 else 0
            autocorr.append(ac)
            
            if abs(ac) > critical_value:
                significant_lags.append(lag)
        
        # Ljung-Box test for overall randomness
        # Use available lags or max 10
        h = min(len(autocorr), 10)
        if h > 0:
            q_stat = n * (n + 2) * np.sum((np.array(autocorr[:h])**2) / (n - np.arange(1, h + 1)))
            p_value = 1 - stats.chi2.cdf(q_stat, df=h)
        else:
            q_stat = 0
            p_value = 1.0

        return {
            'autocorrelations': {i+1: ac for i, ac in enumerate(autocorr)},
            'significant_lags': significant_lags,
            'critical_value': critical_value,
            'ljung_box_statistic': q_stat,
            'ljung_box_p_value': p_value,
            'is_random': p_value > 0.05,
            'interpretation': (
                "Sequence appears random (no significant autocorrelation)"
                if p_value > 0.05 else
                f"Significant autocorrelation detected at lags: {significant_lags}"
            )
        }
    
    def runs_test(self, data: np.ndarray, threshold: float = None) -> Dict:
        """
        Runs test for randomness.
        
        Tests if the sequence has too many or too few runs
        (consecutive values above/below threshold).
        
        Args:
            data: Array of crash values
            threshold: Threshold for above/below (default: median)
        
        Returns:
            Dict with runs test results
        """
        if threshold is None:
            threshold = np.median(data)
        
        # Convert to binary (above/below threshold)
        binary = (data >= threshold).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected runs and variance under null hypothesis
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        n = len(binary)
        
        if n1 == 0 or n2 == 0:
            return {'error': 'All values on one side of threshold'}
        
        expected_runs = (2 * n1 * n2) / n + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        if variance <= 0:
            return {'error': 'Cannot compute variance'}
        
        # Z-score
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'n_runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z,
            'p_value': p_value,
            'is_random': p_value > 0.05,
            'interpretation': (
                "Sequence appears random (runs test passed)"
                if p_value > 0.05 else
                f"Non-random pattern detected (p={p_value:.4f})"
            )
        }
    
    def conditional_entropy(self, data: np.ndarray, n_prev: int = 1) -> Dict:
        """
        Calculate conditional entropy H(X_t | X_{t-1}, ..., X_{t-n}).
        
        If conditional entropy ≈ unconditional entropy,
        past values don't help predict future.
        
        Args:
            data: Array of crash values
            n_prev: Number of previous values to condition on
        
        Returns:
            Dict with conditional entropy analysis
        """
        log_data = np.log(data[data > 0])
        
        # Discretize
        bins = np.linspace(log_data.min(), log_data.max(), self.n_bins + 1)
        discrete = np.digitize(log_data, bins) - 1
        discrete = np.clip(discrete, 0, self.n_bins - 1)
        
        # Build conditional distribution
        from collections import Counter
        
        # Count (history -> next) pairs
        transitions = Counter()
        history_counts = Counter()
        
        for i in range(n_prev, len(discrete)):
            history = tuple(discrete[i-n_prev:i])
            next_val = discrete[i]
            transitions[(history, next_val)] += 1
            history_counts[history] += 1
        
        # Calculate conditional entropy
        total = sum(history_counts.values())
        cond_entropy = 0
        
        for history, count in history_counts.items():
            p_history = count / total
            h_given_history = 0
            
            for next_val in range(self.n_bins):
                p_next_given_history = transitions.get((history, next_val), 0) / count
                if p_next_given_history > 0:
                    h_given_history -= p_next_given_history * np.log2(p_next_given_history)
            
            cond_entropy += p_history * h_given_history
        
        # Unconditional entropy
        uncond = self.calculate_entropy(data)
        uncond_entropy = uncond['entropy']
        
        # Information gain from knowing history
        info_gain = uncond_entropy - cond_entropy
        relative_gain = info_gain / uncond_entropy if uncond_entropy > 0 else 0
        
        return {
            'conditional_entropy': cond_entropy,
            'unconditional_entropy': uncond_entropy,
            'information_gain': info_gain,
            'relative_gain': relative_gain,
            'n_previous': n_prev,
            'interpretation': (
                f"Knowing {n_prev} previous values provides {relative_gain*100:.1f}% "
                f"information gain. "
                + ("Minimal predictive value." if relative_gain < 0.05 else "Some predictive value exists.")
            )
        }
    
    def full_analysis(self, data: np.ndarray) -> Dict:
        """
        Complete information-theoretic analysis.
        
        Args:
            data: Array of crash values
        
        Returns:
            Comprehensive analysis dict
        """
        entropy = self.calculate_entropy(data)
        mi_lag1 = self.calculate_mutual_information(data, lag=1)
        mi_lag5 = self.calculate_mutual_information(data, lag=5)
        autocorr = self.analyze_autocorrelation(data)
        runs = self.runs_test(data)
        cond_ent = self.conditional_entropy(data, n_prev=3)
        
        # Overall verdict
        randomness_score = 0
        if entropy['normalized_entropy'] > 0.8:
            randomness_score += 1
        if mi_lag1['normalized_mi'] < 0.1:
            randomness_score += 1
        if autocorr['is_random']:
            randomness_score += 1
        if runs['is_random']:
            randomness_score += 1
        if cond_ent['relative_gain'] < 0.05:
            randomness_score += 1
        
        if randomness_score >= 4:
            verdict = "RANDOM: Strong evidence that crashes are unpredictable"
        elif randomness_score >= 3:
            verdict = "LIKELY RANDOM: Most tests suggest randomness"
        elif randomness_score >= 2:
            verdict = "INCONCLUSIVE: Mixed evidence"
        else:
            verdict = "PATTERNS DETECTED: Some structure may exist"
        
        return {
            'entropy': entropy,
            'mutual_information': {
                'lag_1': mi_lag1,
                'lag_5': mi_lag5
            },
            'autocorrelation': autocorr,
            'runs_test': runs,
            'conditional_entropy': cond_ent,
            'randomness_score': f"{randomness_score}/5",
            'verdict': verdict
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("INFORMATION THEORY ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    analyzer = InformationAnalyzer(n_bins=20)
    analysis = analyzer.full_analysis(data)
    
    print(f"\n📊 ENTROPY ANALYSIS")
    print(f"   Shannon Entropy: {analysis['entropy']['entropy']:.3f} bits")
    print(f"   Normalized: {analysis['entropy']['normalized_entropy']:.3f}")
    print(f"   {analysis['entropy']['interpretation']}")
    
    print(f"\n🔗 MUTUAL INFORMATION (lag=1)")
    mi = analysis['mutual_information']['lag_1']
    print(f"   MI: {mi['mutual_information']:.4f} bits")
    print(f"   Normalized MI: {mi['normalized_mi']:.4f}")
    print(f"   {mi['interpretation']}")
    
    print(f"\n📈 AUTOCORRELATION")
    ac = analysis['autocorrelation']
    print(f"   Significant lags: {ac['significant_lags'] or 'None'}")
    print(f"   Ljung-Box p-value: {ac['ljung_box_p_value']:.4f}")
    print(f"   {ac['interpretation']}")
    
    print(f"\n🏃 RUNS TEST")
    rt = analysis['runs_test']
    print(f"   Runs: {rt['n_runs']} (expected: {rt['expected_runs']:.1f})")
    print(f"   Z-statistic: {rt['z_statistic']:.3f}")
    print(f"   {rt['interpretation']}")
    
    print(f"\n🎯 CONDITIONAL ENTROPY")
    ce = analysis['conditional_entropy']
    print(f"   H(X|history): {ce['conditional_entropy']:.3f} bits")
    print(f"   Info gain: {ce['relative_gain']*100:.1f}%")
    print(f"   {ce['interpretation']}")
    
    print(f"\n" + "=" * 60)
    print(f"🏆 VERDICT: {analysis['verdict']}")
    print(f"   Randomness Score: {analysis['randomness_score']}")
    print("=" * 60)
