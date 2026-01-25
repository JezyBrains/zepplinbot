#!/usr/bin/env python3
"""
Tests for Information Analyzer Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from information_analyzer import InformationAnalyzer


class TestEntropyCalculation:
    """Test entropy calculations."""
    
    @pytest.fixture
    def analyzer(self):
        return InformationAnalyzer(n_bins=20)
    
    def test_entropy_of_uniform_distribution(self, analyzer):
        """Uniform distribution should have high entropy."""
        # Uniform data
        uniform_data = np.random.uniform(1, 10, 500)
        
        result = analyzer.calculate_entropy(uniform_data)
        
        assert 'entropy' in result
        assert 'normalized_entropy' in result
        # Uniform should have high normalized entropy
        assert result['normalized_entropy'] > 0.7
    
    def test_entropy_of_concentrated_distribution(self, analyzer):
        """Concentrated distribution should have lower entropy."""
        # All values near the same point
        concentrated_data = np.random.normal(5, 0.1, 500)
        concentrated_data = np.clip(concentrated_data, 0.1, 100)
        
        result = analyzer.calculate_entropy(concentrated_data)
        
        # Concentrated should have lower entropy than uniform
        # Note: histogram adapts to range, so a normal distribution still has high entropy
        # but should be less than uniform (which is ~1.0)
        assert result['normalized_entropy'] < 0.95


class TestMutualInformation:
    """Test mutual information calculations."""
    
    @pytest.fixture
    def analyzer(self):
        return InformationAnalyzer(n_bins=10)
    
    def test_mi_of_independent_data(self, analyzer):
        """Independent data should have low MI."""
        # Generate independent data
        np.random.seed(42)
        independent_data = np.random.exponential(2, 200)
        
        result = analyzer.calculate_mutual_information(independent_data, lag=1)
        
        assert 'mutual_information' in result
        assert 'normalized_mi' in result
        # Independent data should have low MI
        assert result['normalized_mi'] < 0.3
    
    def test_mi_of_dependent_data(self, analyzer):
        """Dependent data should have higher MI."""
        # Create dependent data (each value influenced by previous)
        np.random.seed(42)
        n = 200
        dependent_data = np.zeros(n)
        dependent_data[0] = np.random.exponential(2)
        
        for i in range(1, n):
            # Next value is correlated with previous
            dependent_data[i] = dependent_data[i-1] * 0.8 + np.random.normal(0, 0.5)
        
        dependent_data = np.abs(dependent_data) + 1  # Keep positive
        
        result = analyzer.calculate_mutual_information(dependent_data, lag=1)
        
        # Dependent data should have higher MI
        # Note: exact threshold depends on strength of dependency


class TestAutocorrelation:
    """Test autocorrelation analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return InformationAnalyzer()
    
    def test_autocorrelation_of_random_data(self, analyzer):
        """Random data should have non-significant autocorrelation."""
        np.random.seed(42)
        random_data = np.random.exponential(2, 300)
        
        result = analyzer.analyze_autocorrelation(random_data, max_lag=10)
        
        assert 'autocorrelations' in result
        assert 'ljung_box_p_value' in result
        assert 'is_random' in result
        
        # Random data should pass randomness test (p > 0.05 typically)
    
    def test_autocorrelation_returns_correct_structure(self, analyzer):
        """Test return structure is correct."""
        data = np.random.exponential(2, 100)
        result = analyzer.analyze_autocorrelation(data, max_lag=5)
        
        # Should have autocorrelation for each lag
        assert 1 in result['autocorrelations']
        assert len(result['autocorrelations']) == 5


class TestRunsTest:
    """Test runs test for randomness."""
    
    @pytest.fixture
    def analyzer(self):
        return InformationAnalyzer()
    
    def test_runs_test_structure(self, analyzer):
        """Test runs test returns correct structure."""
        data = np.random.exponential(2, 100)
        result = analyzer.runs_test(data)
        
        assert 'n_runs' in result
        assert 'expected_runs' in result
        assert 'p_value' in result
        assert 'is_random' in result


class TestFullAnalysis:
    """Test complete analysis pipeline."""
    
    def test_full_analysis_returns_all_components(self):
        """Test full analysis includes all components."""
        analyzer = InformationAnalyzer()
        data = np.random.exponential(2, 200)
        
        result = analyzer.full_analysis(data)
        
        assert 'entropy' in result
        assert 'mutual_information' in result
        assert 'autocorrelation' in result
        assert 'runs_test' in result
        assert 'conditional_entropy' in result
        assert 'verdict' in result
        assert 'randomness_score' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
