#!/usr/bin/env python3
"""
Backtesting Framework for Crash Game Strategies

Provides historical simulation with realistic timing, trade-by-trade 
analysis, and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    round_id: int
    timestamp: datetime
    bet_amount: float
    target_multiplier: float
    actual_crash: float
    won: bool
    profit: float
    bankroll_after: float
    signal_confidence: float = 0.0
    strategy_name: str = ""


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[Trade] = field(default_factory=list)
    initial_bankroll: float = 0.0
    final_bankroll: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    skipped_rounds: int = 0
    
    # Performance metrics
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.final_bankroll,
            'total_return': f"{self.total_return:.2%}",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'sharpe_ratio': f"{self.sharpe_ratio:.3f}",
            'sortino_ratio': f"{self.sortino_ratio:.3f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_win': f"{self.avg_win:.2f}",
            'avg_loss': f"{self.avg_loss:.2f}",
            'expectancy': f"{self.expectancy:.4f}"
        }


class Strategy:
    """Base strategy class to be subclassed."""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    def should_bet(self, historical_data: np.ndarray, bankroll: float) -> Tuple[bool, float, float, float]:
        """
        Decide whether to bet.
        
        Returns:
            (should_bet, bet_amount, target_multiplier, confidence)
        """
        raise NotImplementedError
    
    def on_result(self, won: bool, profit: float):
        """Called after each round with the result."""
        pass


class KellyStrategy(Strategy):
    """Kelly Criterion-based betting strategy."""
    
    def __init__(self, fractional_kelly: float = 0.25, min_bet_fraction: float = 0.01):
        super().__init__("KellyStrategy")
        self.fractional_kelly = fractional_kelly
        self.min_bet_fraction = min_bet_fraction
        self.default_targets = [1.5, 2.0, 2.5, 3.0]
    
    def should_bet(self, historical_data: np.ndarray, bankroll: float) -> Tuple[bool, float, float, float]:
        if len(historical_data) < 50:
            return (False, 0, 0, 0)
        
        best_ev = -float('inf')
        best_target = None
        best_kelly = 0
        
        for target in self.default_targets:
            wins = sum(1 for x in historical_data if x >= target)
            p = wins / len(historical_data) * 0.99  # 1% house edge
            q = 1 - p
            b = target - 1
            
            kelly = (b * p - q) / b if b > 0 else 0
            ev = (p * b) - q
            
            if ev > best_ev and kelly > 0:
                best_ev = ev
                best_target = target
                best_kelly = kelly
        
        if best_target is None or best_kelly < self.min_bet_fraction:
            return (False, 0, 0, 0)
        
        safe_kelly = min(best_kelly * self.fractional_kelly, 0.10)
        bet_amount = bankroll * safe_kelly
        confidence = min(best_ev * 10, 1.0)  # Normalize to 0-1
        
        return (True, bet_amount, best_target, confidence)


class Backtester:
    """
    Historical backtesting engine for crash game strategies.
    
    Features:
    - Walk-forward testing
    - Realistic trade simulation
    - Comprehensive metrics calculation
    - Support for multiple strategies
    """
    
    def __init__(self, 
                 initial_bankroll: float = 100.0,
                 min_history: int = 50,
                 commission: float = 0.0):
        """
        Args:
            initial_bankroll: Starting bankroll
            min_history: Minimum data points before trading
            commission: Per-trade commission (fraction)
        """
        self.initial_bankroll = initial_bankroll
        self.min_history = min_history
        self.commission = commission
    
    def run(self, 
            data: pd.DataFrame,
            strategy: Strategy,
            start_idx: Optional[int] = None,
            end_idx: Optional[int] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
            strategy: Strategy instance
            start_idx: Starting index (default: min_history)
            end_idx: Ending index (default: len(data))
        
        Returns:
            BacktestResult with all metrics
        """
        if 'value' not in data.columns:
            raise ValueError("Data must have 'value' column")
        
        crash_values = data['value'].values
        timestamps = data['timestamp'].values if 'timestamp' in data.columns else range(len(data))
        
        start_idx = start_idx or self.min_history
        end_idx = end_idx or len(crash_values)
        
        bankroll = self.initial_bankroll
        trades: List[Trade] = []
        bankroll_history = [bankroll]
        skipped = 0
        
        for i in range(start_idx, end_idx):
            historical = crash_values[:i]
            actual_crash = crash_values[i]
            
            should_bet, bet_amount, target, confidence = strategy.should_bet(historical, bankroll)
            
            if not should_bet or bet_amount <= 0:
                skipped += 1
                bankroll_history.append(bankroll)
                continue
            
            # Ensure we don't bet more than we have
            bet_amount = min(bet_amount, bankroll)
            
            # Apply commission
            bet_amount_after_commission = bet_amount * (1 - self.commission)
            
            # Determine outcome
            won = actual_crash >= target
            if won:
                profit = bet_amount_after_commission * (target - 1)
            else:
                profit = -bet_amount
            
            bankroll += profit
            
            # Record trade
            trade = Trade(
                round_id=i,
                timestamp=timestamps[i] if hasattr(timestamps[i], 'isoformat') else datetime.now(),
                bet_amount=bet_amount,
                target_multiplier=target,
                actual_crash=actual_crash,
                won=won,
                profit=profit,
                bankroll_after=bankroll,
                signal_confidence=confidence,
                strategy_name=strategy.name
            )
            trades.append(trade)
            bankroll_history.append(bankroll)
            
            # Notify strategy
            strategy.on_result(won, profit)
            
            # Stop if bankrupt
            if bankroll <= 0:
                logger.warning(f"Bankrupt at round {i}")
                break
        
        # Calculate metrics
        result = self._calculate_metrics(trades, bankroll_history, skipped)
        return result
    
    def _calculate_metrics(self, 
                          trades: List[Trade], 
                          bankroll_history: List[float],
                          skipped: int) -> BacktestResult:
        """Calculate all performance metrics."""
        result = BacktestResult()
        result.trades = trades
        result.initial_bankroll = self.initial_bankroll
        result.final_bankroll = bankroll_history[-1] if bankroll_history else self.initial_bankroll
        result.total_trades = len(trades)
        result.skipped_rounds = skipped
        
        if not trades:
            return result
        
        # Basic stats
        profits = [t.profit for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) if trades else 0
        
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        
        # Total return
        result.total_return = (result.final_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy (average profit per trade)
        result.expectancy = np.mean(profits)
        
        # Sharpe Ratio (annualized, assuming ~1000 rounds/day)
        if len(profits) > 1:
            returns = np.array(profits) / self.initial_bankroll
            result.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(1000)
        
        # Sortino Ratio (only downside deviation)
        if losses:
            downside_returns = np.array(losses) / self.initial_bankroll
            downside_std = np.std(downside_returns)
            result.sortino_ratio = np.mean(profits) / self.initial_bankroll / (downside_std + 1e-10) * np.sqrt(1000)
        
        # Max Drawdown
        peak = bankroll_history[0]
        max_dd = 0
        dd_start = 0
        max_dd_duration = 0
        current_dd_start = 0
        
        for i, val in enumerate(bankroll_history):
            if val > peak:
                peak = val
                if i - current_dd_start > max_dd_duration:
                    max_dd_duration = i - current_dd_start
                current_dd_start = i
            
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        result.max_drawdown = max_dd
        result.max_drawdown_duration = max_dd_duration
        
        return result
    
    def run_monte_carlo(self,
                        data: pd.DataFrame,
                        strategy: Strategy,
                        n_simulations: int = 1000,
                        sample_size: int = 200) -> Dict:
        """
        Run Monte Carlo simulation by resampling historical data.
        
        Returns distribution of outcomes.
        """
        crash_values = data['value'].values
        results = []
        
        for _ in range(n_simulations):
            # Resample with replacement
            sampled_indices = np.random.choice(len(crash_values), size=sample_size, replace=True)
            sampled_data = pd.DataFrame({
                'value': crash_values[sampled_indices],
                'timestamp': range(sample_size)
            })
            
            result = self.run(sampled_data, strategy, start_idx=50)
            results.append(result.total_return)
        
        return {
            'mean_return': np.mean(results),
            'std_return': np.std(results),
            'median_return': np.median(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_95': np.percentile(results, 95),
            'prob_profit': sum(1 for r in results if r > 0) / len(results),
            'prob_loss_50pct': sum(1 for r in results if r < -0.5) / len(results)
        }
    
    def generate_report(self, result: BacktestResult, filepath: str = None) -> str:
        """Generate text report of backtest results."""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        report.append(f"\nStrategy: {result.trades[0].strategy_name if result.trades else 'N/A'}")
        report.append(f"Total Rounds: {result.total_trades + result.skipped_rounds}")
        report.append(f"Traded Rounds: {result.total_trades}")
        report.append(f"Skipped Rounds: {result.skipped_rounds}")
        report.append("")
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Initial Bankroll:  ${result.initial_bankroll:.2f}")
        report.append(f"Final Bankroll:    ${result.final_bankroll:.2f}")
        report.append(f"Total Return:      {result.total_return:.2%}")
        report.append("")
        report.append(f"Win Rate:          {result.win_rate:.2%}")
        report.append(f"Profit Factor:     {result.profit_factor:.2f}")
        report.append(f"Expectancy:        ${result.expectancy:.4f}")
        report.append("")
        report.append(f"Sharpe Ratio:      {result.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio:     {result.sortino_ratio:.3f}")
        report.append(f"Max Drawdown:      {result.max_drawdown:.2%}")
        report.append(f"Max DD Duration:   {result.max_drawdown_duration} rounds")
        report.append("")
        report.append(f"Avg Win:           ${result.avg_win:.2f}")
        report.append(f"Avg Loss:          ${result.avg_loss:.2f}")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {filepath}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    
    print("=" * 60)
    print("BACKTESTING FRAMEWORK")
    print("=" * 60)
    
    # Create strategy and backtester
    strategy = KellyStrategy(fractional_kelly=0.25)
    backtester = Backtester(initial_bankroll=100.0)
    
    # Run backtest
    result = backtester.run(df, strategy)
    
    # Print report
    print(backtester.generate_report(result))
    
    # Save detailed results
    result_dict = result.to_dict()
    print("\nJSON Results:")
    print(json.dumps(result_dict, indent=2))
    
    # Monte Carlo simulation
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION (1000 runs)")
    print("=" * 60)
    
    mc_results = backtester.run_monte_carlo(df, KellyStrategy(), n_simulations=100)
    print(f"\nMean Return: {mc_results['mean_return']:.2%}")
    print(f"Std Dev: {mc_results['std_return']:.2%}")
    print(f"5th Percentile: {mc_results['percentile_5']:.2%}")
    print(f"95th Percentile: {mc_results['percentile_95']:.2%}")
    print(f"Probability of Profit: {mc_results['prob_profit']:.2%}")
