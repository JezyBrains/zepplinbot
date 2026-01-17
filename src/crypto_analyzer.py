#!/usr/bin/env python3
"""
Cryptographic Analysis Module for Provably Fair Systems

This module analyzes SHA-256 based provably fair gaming systems to detect:
1. Seed biases (time-based, sequence-based)
2. UUID generation patterns
3. Coefficient distribution anomalies
4. Temporal correlations

Note: This does NOT attempt to reverse SHA-256 (mathematically infeasible).
Instead, it looks for implementation flaws in the randomness source.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProvenFairnessVerifier:
    """
    Verifies the integrity of provably fair game rounds.
    """
    
    @staticmethod
    def verify_round(round_number: str, coefficient: float, uuid: str, 
                     provided_hash: str) -> Tuple[bool, str]:
        """
        Verify that a game round's hash matches the claimed parameters.
        
        Args:
            round_number: Round identifier
            coefficient: Crash multiplier
            uuid: Unique round identifier
            provided_hash: Server-provided SHA-256 hash
            
        Returns:
            (is_valid, calculated_hash)
        """
        round_key = f"{round_number}_{coefficient}_{uuid}"
        calculated_hash = hashlib.sha256(round_key.encode()).hexdigest()
        
        is_valid = calculated_hash == provided_hash
        
        if not is_valid:
            logger.warning(f"Hash mismatch for round {round_number}")
        
        return is_valid, calculated_hash
    
    @staticmethod
    def batch_verify(rounds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify multiple rounds at once.
        
        Expected columns: round_number, coefficient, uuid, hash
        """
        results = []
        
        for _, row in rounds_df.iterrows():
            is_valid, calc_hash = ProvenFairnessVerifier.verify_round(
                str(row['round_number']),
                row['coefficient'],
                row['uuid'],
                row['hash']
            )
            results.append({
                'round': row['round_number'],
                'valid': is_valid,
                'calculated_hash': calc_hash
            })
        
        return pd.DataFrame(results)


class SeedBiasAnalyzer:
    """
    Analyzes potential biases in the seed/UUID generation.
    
    Target: Find correlations between:
    - Server timestamp and coefficient
    - UUID patterns and outcomes
    - Sequential dependencies
    """
    
    def __init__(self):
        self.entropy_threshold = 7.5  # Shannon entropy for good randomness
    
    def analyze_uuid_entropy(self, uuids: List[str]) -> Dict:
        """
        Calculate Shannon entropy of UUID generation.
        Low entropy suggests predictable patterns.
        """
        # Convert UUIDs to binary representation
        binary_data = ''.join([format(int(c, 16), '04b') 
                               for uuid in uuids 
                               for c in uuid.replace('-', '')])
        
        # Calculate bit frequency
        bit_counts = [binary_data.count('0'), binary_data.count('1')]
        total_bits = len(binary_data)
        
        # Shannon entropy
        entropy = -sum((count/total_bits) * np.log2(count/total_bits) 
                      for count in bit_counts if count > 0)
        
        return {
            'entropy': entropy,
            'is_random': entropy > self.entropy_threshold,
            'total_bits': total_bits,
            'bit_balance': bit_counts[1] / total_bits
        }
    
    def detect_temporal_bias(self, df: pd.DataFrame) -> Dict:
        """
        Check if coefficients correlate with time of day.
        
        Args:
            df: DataFrame with 'timestamp' and 'coefficient' columns
        """
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
        
        # Group by hour and calculate mean coefficient
        hourly_avg = df.groupby('hour')['coefficient'].mean()
        
        # Calculate variance - high variance suggests time-based bias
        temporal_variance = hourly_avg.var()
        
        # Chi-square test for uniform distribution
        expected_mean = df['coefficient'].mean()
        chi_square = sum((hourly_avg - expected_mean)**2 / expected_mean)
        
        return {
            'temporal_variance': temporal_variance,
            'chi_square': chi_square,
            'hourly_averages': hourly_avg.to_dict(),
            'bias_detected': chi_square > 23.685  # 95% confidence, 23 df
        }
    
    def analyze_sequence_autocorrelation(self, coefficients: List[float], 
                                        max_lag: int = 50) -> Dict:
        """
        Detect if current coefficient depends on previous ones.
        
        This is the key test: if autocorrelation exists, the PRNG is weak.
        """
        coeffs = np.array(coefficients)
        
        autocorr = []
        for lag in range(1, min(max_lag, len(coeffs))):
            corr = np.corrcoef(coeffs[:-lag], coeffs[lag:])[0, 1]
            autocorr.append(corr)
        
        # Find significant correlations (|r| > 0.1)
        significant_lags = [i+1 for i, r in enumerate(autocorr) if abs(r) > 0.1]
        
        return {
            'autocorrelations': autocorr,
            'max_autocorr': max(autocorr, key=abs) if autocorr else 0,
            'significant_lags': significant_lags,
            'pattern_detected': len(significant_lags) > 0
        }


class BinaryFeatureExtractor:
    """
    Converts game data into binary representations for neural network training.
    
    This allows the model to "think" at the bit level, similar to cryptographic functions.
    """
    
    @staticmethod
    def coefficient_to_binary(coefficient: float, bits: int = 32) -> List[int]:
        """
        Convert coefficient to binary representation.
        """
        # Scale to integer range
        scaled = int(coefficient * 1000)
        binary = format(scaled, f'0{bits}b')
        return [int(b) for b in binary]
    
    @staticmethod
    def uuid_to_binary(uuid: str) -> List[int]:
        """
        Convert UUID to binary representation (128 bits).
        """
        uuid_clean = uuid.replace('-', '')
        binary = ''.join([format(int(c, 16), '04b') for c in uuid_clean])
        return [int(b) for b in binary]
    
    @staticmethod
    def create_binary_features(df: pd.DataFrame) -> np.ndarray:
        """
        Create full binary feature matrix for training.
        
        Returns: (n_samples, n_features) array
        """
        features = []
        
        for _, row in df.iterrows():
            coeff_bits = BinaryFeatureExtractor.coefficient_to_binary(row['coefficient'])
            
            if 'uuid' in row:
                uuid_bits = BinaryFeatureExtractor.uuid_to_binary(row['uuid'])
                features.append(coeff_bits + uuid_bits)
            else:
                features.append(coeff_bits)
        
        return np.array(features)


class CryptoPatternDetector:
    """
    Main analyzer that combines all detection methods.
    """
    
    def __init__(self):
        self.verifier = ProvenFairnessVerifier()
        self.bias_analyzer = SeedBiasAnalyzer()
        self.feature_extractor = BinaryFeatureExtractor()
    
    def full_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete cryptographic analysis on game data.
        
        Args:
            df: DataFrame with columns: timestamp, coefficient, [uuid, hash, round_number]
        
        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting cryptographic pattern analysis...")
        
        results = {
            'data_summary': {
                'total_rounds': len(df),
                'coefficient_range': (df['coefficient'].min(), df['coefficient'].max()),
                'mean_coefficient': df['coefficient'].mean()
            }
        }
        
        # 1. Sequence autocorrelation (most important)
        logger.info("Analyzing sequence autocorrelation...")
        autocorr_results = self.bias_analyzer.analyze_sequence_autocorrelation(
            df['coefficient'].tolist()
        )
        results['autocorrelation'] = autocorr_results
        
        # 2. Temporal bias detection
        if 'timestamp' in df.columns:
            logger.info("Checking for temporal biases...")
            temporal_results = self.bias_analyzer.detect_temporal_bias(df)
            results['temporal_bias'] = temporal_results
        
        # 3. UUID entropy analysis
        if 'uuid' in df.columns:
            logger.info("Analyzing UUID entropy...")
            uuid_results = self.bias_analyzer.analyze_uuid_entropy(
                df['uuid'].tolist()
            )
            results['uuid_entropy'] = uuid_results
        
        # 4. Hash verification
        if all(col in df.columns for col in ['round_number', 'uuid', 'hash']):
            logger.info("Verifying hash integrity...")
            verification = self.verifier.batch_verify(df)
            results['hash_verification'] = {
                'total_verified': len(verification),
                'all_valid': verification['valid'].all(),
                'invalid_count': (~verification['valid']).sum()
            }
        
        # 5. Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        """
        recommendations = []
        
        # Check autocorrelation
        if analysis.get('autocorrelation', {}).get('pattern_detected'):
            recommendations.append(
                "⚠️ PATTERN DETECTED: Sequence shows autocorrelation. "
                "A Transformer model may find exploitable patterns."
            )
        else:
            recommendations.append(
                "✅ No autocorrelation detected. System appears to use strong randomness."
            )
        
        # Check temporal bias
        if analysis.get('temporal_bias', {}).get('bias_detected'):
            recommendations.append(
                "⚠️ TEMPORAL BIAS: Coefficients correlate with time of day. "
                "Consider time-based features in your model."
            )
        
        # Check UUID entropy
        uuid_entropy = analysis.get('uuid_entropy', {})
        if uuid_entropy and not uuid_entropy.get('is_random'):
            recommendations.append(
                "⚠️ LOW UUID ENTROPY: Seed generation may be predictable. "
                "Focus on UUID pattern analysis."
            )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'coefficient': np.random.exponential(2.5, 100)
    })
    
    detector = CryptoPatternDetector()
    results = detector.full_analysis(sample_data)
    
    print("\n" + "="*60)
    print("CRYPTOGRAPHIC ANALYSIS REPORT")
    print("="*60)
    
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        print(value)
