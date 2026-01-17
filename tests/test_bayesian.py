#!/usr/bin/env python3
"""
Tests for Bayesian Model Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from bayesian_model import BayesianCrashPredictor, BayesianKellyOptimizer


class TestBayesianCrashPredictor:
    """Test Bayesian prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample crash data."""
        np.random.seed(42)
        return np.exp(np.random.normal(0.5, 0.8, 200))
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return BayesianCrashPredictor(n_samples=500, n_chains=1)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.prior_mu == 1.0
        assert model.prior_sigma == 1.0
        assert not model.fitted
    
    def test_model_fit(self, model, sample_data):
        """Test model fits without error."""
        model.fit(sample_data)
        
        assert model.fitted
        assert model.posterior_mu is not None
        assert model.posterior_sigma > 0
    
    def test_predict_distribution(self, model, sample_data):
        """Test distribution sampling."""
        model.fit(sample_data)
        
        samples = model.predict_distribution(n_samples=100)
        
        assert len(samples) == 100
        assert all(s > 0 for s in samples)  # Crash values are positive
    
    def test_predict_point(self, model, sample_data):
        """Test point prediction."""
        model.fit(sample_data)
        
        point = model.predict_point()
        
        assert point > 0
        assert isinstance(point, float)
    
    def test_credible_interval(self, model, sample_data):
        """Test credible interval calculation."""
        model.fit(sample_data)
        
        ci = model.get_credible_interval(confidence=0.95)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower < Upper
    
    def test_probability_above(self, model, sample_data):
        """Test probability calculation."""
        model.fit(sample_data)
        
        p_above_2 = model.probability_above(2.0)
        
        assert 0 <= p_above_2 <= 1
    
    def test_probability_decreases_with_threshold(self, model, sample_data):
        """Test that P(X > t) decreases as t increases."""
        model.fit(sample_data)
        
        p1 = model.probability_above(1.5)
        p2 = model.probability_above(2.0)
        p3 = model.probability_above(3.0)
        
        # Probabilities should decrease
        assert p1 >= p2 >= p3
    
    def test_get_summary(self, model, sample_data):
        """Test summary generation."""
        model.fit(sample_data)
        
        summary = model.get_summary()
        
        assert 'point_prediction' in summary
        assert 'credible_interval_95' in summary
        assert 'p_above_2.0x' in summary
        assert 'uncertainty' in summary
    
    def test_unfitted_model_raises_error(self, model):
        """Test that unfitted model raises error."""
        with pytest.raises(ValueError):
            model.predict_distribution()


class TestBayesianKellyOptimizer:
    """Test Bayesian Kelly optimizer."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.exp(np.random.normal(0.5, 0.8, 200))
    
    @pytest.fixture
    def fitted_model(self, sample_data):
        model = BayesianCrashPredictor()
        model.fit(sample_data)
        return model
    
    @pytest.fixture
    def optimizer(self, fitted_model):
        return BayesianKellyOptimizer(fitted_model, fractional_kelly=0.25)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer.fractional_kelly == 0.25
        assert optimizer.house_edge == 0.01
    
    def test_calculate_bet(self, optimizer):
        """Test bet calculation."""
        result = optimizer.calculate_bet(bankroll=100.0, target_multiplier=2.0)
        
        assert 'bet_amount' in result
        assert 'kelly_fraction' in result
        assert 'win_probability' in result
        assert 'should_bet' in result
        assert 'bayesian_probability' in result
    
    def test_bet_amount_reasonable(self, optimizer):
        """Test bet amount is reasonable."""
        result = optimizer.calculate_bet(bankroll=100.0, target_multiplier=2.0)
        
        # Bet should not exceed 10% of bankroll (capped Kelly)
        assert result['bet_amount'] <= 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
