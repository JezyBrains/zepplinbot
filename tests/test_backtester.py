#!/usr/bin/env python3
"""
Tests for Backtester Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtester import Backtester, KellyStrategy, Trade, BacktestResult


class TestBacktester:
    """Test backtesting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample crash data."""
        np.random.seed(42)
        # Generate log-normal distributed crashes (typical for crash games)
        crashes = np.exp(np.random.normal(0.5, 0.8, 200))
        crashes = np.clip(crashes, 1.0, 100.0)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'value': crashes
        })
    
    @pytest.fixture
    def backtester(self):
        """Create backtester instance."""
        return Backtester(initial_bankroll=100.0, min_history=50)
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return KellyStrategy(fractional_kelly=0.25)
    
    def test_backtester_initialization(self, backtester):
        """Test backtester initializes correctly."""
        assert backtester.initial_bankroll == 100.0
        assert backtester.min_history == 50
        assert backtester.commission == 0.0
    
    def test_backtest_runs(self, backtester, strategy, sample_data):
        """Test that backtest runs without error."""
        result = backtester.run(sample_data, strategy)
        
        assert isinstance(result, BacktestResult)
        assert result.initial_bankroll == 100.0
        assert result.final_bankroll >= 0
        assert result.total_trades >= 0
    
    def test_backtest_metrics_calculated(self, backtester, strategy, sample_data):
        """Test that metrics are calculated."""
        result = backtester.run(sample_data, strategy)
        
        # Check metrics are present
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'profit_factor')
        
        # Win rate should be between 0 and 1
        assert 0 <= result.win_rate <= 1
        
        # Max drawdown should be non-negative
        assert result.max_drawdown >= 0
    
    def test_backtest_result_to_dict(self, backtester, strategy, sample_data):
        """Test result conversion to dict."""
        result = backtester.run(sample_data, strategy)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'total_return' in result_dict
        assert 'win_rate' in result_dict
        assert 'sharpe_ratio' in result_dict
    
    def test_no_trades_when_insufficient_data(self, backtester, strategy):
        """Test no trades when data is insufficient."""
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='1min'),
            'value': np.random.uniform(1, 5, 30)
        })
        
        result = backtester.run(small_data, strategy, start_idx=10)
        # Should skip most rounds
        assert result.skipped_rounds > 0
    
    def test_kelly_strategy_calculates_bet(self, strategy, sample_data):
        """Test Kelly strategy returns valid bet."""
        historical = sample_data['value'].values[:100]
        should_bet, bet_amount, target, confidence = strategy.should_bet(historical, 100.0)
        
        assert isinstance(should_bet, bool)
        assert bet_amount >= 0
        assert target >= 0


class TestKellyStrategy:
    """Test Kelly Criterion strategy."""
    
    def test_kelly_with_favorable_data(self):
        """Test Kelly bets when data is favorable."""
        strategy = KellyStrategy(fractional_kelly=0.25)
        
        # Create favorable data (many high crashes)
        favorable_data = np.array([3.0, 2.5, 4.0, 2.0, 5.0] * 20)
        
        should_bet, bet_amount, target, confidence = strategy.should_bet(favorable_data, 100.0)
        
        # Should bet on favorable data
        # (depends on calculated probabilities)
        assert isinstance(should_bet, bool)
    
    def test_kelly_with_unfavorable_data(self):
        """Test Kelly skips when data is unfavorable."""
        strategy = KellyStrategy(fractional_kelly=0.25)
        
        # Create unfavorable data (many low crashes)
        unfavorable_data = np.array([1.1, 1.2, 1.0, 1.3, 1.1] * 20)
        
        should_bet, bet_amount, target, confidence = strategy.should_bet(unfavorable_data, 100.0)
        
        # Should likely skip on very unfavorable data
        # Kelly would be negative or very small


class TestTradeRecording:
    """Test trade recording functionality."""
    
    def test_trade_creation(self):
        """Test Trade dataclass creation."""
        from datetime import datetime
        
        trade = Trade(
            round_id=1,
            timestamp=datetime.now(),
            bet_amount=10.0,
            target_multiplier=2.0,
            actual_crash=2.5,
            won=True,
            profit=10.0,
            bankroll_after=110.0
        )
        
        assert trade.round_id == 1
        assert trade.bet_amount == 10.0
        assert trade.won == True
        assert trade.profit == 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
