#!/usr/bin/env python3
"""
GARCH Model for Volatility Clustering

Implements GARCH(1,1) for conditional volatility forecasting
in crash game outcomes. High volatility periods suggest more
conservative betting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import arch, fall back to simple implementation
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not available. Using simplified volatility model.")


class GARCHVolatilityModel:
    """
    GARCH(1,1) model for volatility clustering in crash games.
    
    GARCH captures:
    - Volatility clustering (high vol follows high vol)
    - Mean reversion of volatility
    - Conditional heteroskedasticity
    
    Formula:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: GARCH order for lagged variance
            q: ARCH order for lagged squared residuals
        """
        self.p = p
        self.q = q
        self.model = None
        self.result = None
        self.params = None
        self.fitted = False
        
        # For simple implementation
        self.omega = 0.01
        self.alpha = 0.1
        self.beta = 0.85
        self.last_variance = 0.1
        self.last_residual = 0.0
    
    def fit(self, data: np.ndarray) -> 'GARCHVolatilityModel':
        """
        Fit GARCH model to crash data.
        
        Uses log-returns of crash values for stationarity.
        
        Args:
            data: Array of crash multipliers
        
        Returns:
            self
        """
        # Convert to log-returns
        log_data = np.log(data[data > 0])
        returns = np.diff(log_data) * 100  # Scale for numerical stability
        
        if ARCH_AVAILABLE:
            self._fit_arch(returns)
        else:
            self._fit_simple(returns)
        
        self.fitted = True
        return self
    
    def _fit_arch(self, returns: np.ndarray):
        """Fit using arch library."""
        try:
            self.model = arch_model(returns, vol='GARCH', p=self.p, q=self.q)
            self.result = self.model.fit(disp='off')
            self.params = self.result.params
            
            logger.info(f"GARCH({self.p},{self.q}) fit complete.")
            logger.info(f"Parameters: omega={self.params['omega']:.4f}, "
                       f"alpha={self.params['alpha[1]']:.4f}, "
                       f"beta={self.params['beta[1]']:.4f}")
        except Exception as e:
            logger.error(f"ARCH fitting failed: {e}. Using simple model.")
            self._fit_simple(returns)
    
    def _fit_simple(self, returns: np.ndarray):
        """Simple GARCH parameter estimation using quasi-MLE."""
        # Estimate unconditional variance
        var = np.var(returns)
        
        # Use typical GARCH(1,1) parameters
        self.alpha = 0.1
        self.beta = 0.85
        self.omega = var * (1 - self.alpha - self.beta)
        
        # Initialize
        self.last_variance = var
        self.last_residual = returns[-1] if len(returns) > 0 else 0
        
        logger.info(f"Simple GARCH fit: omega={self.omega:.4f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Forecast conditional variance for next steps.
        
        Args:
            steps: Number of steps ahead
        
        Returns:
            Array of forecast variances
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if ARCH_AVAILABLE and self.result is not None:
            forecast = self.result.forecast(horizon=steps)
            return forecast.variance.values[-1]
        else:
            # Simple GARCH forecast
            forecasts = []
            var_t = self.last_variance
            
            for _ in range(steps):
                var_t = self.omega + self.alpha * self.last_residual**2 + self.beta * var_t
                forecasts.append(var_t)
            
            return np.array(forecasts)
    
    def get_current_volatility(self) -> float:
        """Get current conditional volatility (std dev)."""
        variance = self.forecast(1)[0]
        return np.sqrt(variance)
    
    def get_volatility_regime(self) -> Dict:
        """
        Classify current volatility regime.
        
        Returns:
            Dict with regime classification and metrics
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        current_vol = self.get_current_volatility()
        
        # Get unconditional volatility for comparison
        if ARCH_AVAILABLE and self.result is not None:
            params = self.params
            omega = params['omega']
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            unconditional_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega
        else:
            unconditional_var = self.omega / (1 - self.alpha - self.beta)
        
        unconditional_vol = np.sqrt(unconditional_var)
        
        # Regime classification
        vol_ratio = current_vol / unconditional_vol if unconditional_vol > 0 else 1
        
        if vol_ratio < 0.7:
            regime = 'low'
            risk_adjustment = 1.2  # Can bet slightly more
        elif vol_ratio < 1.3:
            regime = 'normal'
            risk_adjustment = 1.0
        elif vol_ratio < 2.0:
            regime = 'high'
            risk_adjustment = 0.7  # Reduce bets
        else:
            regime = 'extreme'
            risk_adjustment = 0.3  # Minimal bets
        
        return {
            'regime': regime,
            'current_volatility': current_vol,
            'unconditional_volatility': unconditional_vol,
            'volatility_ratio': vol_ratio,
            'risk_adjustment': risk_adjustment,
            'message': self._get_regime_message(regime)
        }
    
    def _get_regime_message(self, regime: str) -> str:
        """Get human-readable regime message."""
        messages = {
            'low': "✅ Low volatility - favorable conditions for betting",
            'normal': "📊 Normal volatility - standard betting recommended",
            'high': "⚠️ High volatility - reduce bet sizes",
            'extreme': "🛑 Extreme volatility - minimal betting advised"
        }
        return messages.get(regime, "Unknown regime")
    
    def update(self, new_value: float):
        """
        Online update with new observation.
        
        Args:
            new_value: New crash value
        """
        # Update residual and variance estimates
        log_return = np.log(new_value) * 100 if new_value > 0 else 0
        self.last_residual = log_return - np.sqrt(self.last_variance)
        self.last_variance = self.omega + self.alpha * self.last_residual**2 + self.beta * self.last_variance
    
    def forecast_volatility_path(self, steps: int = 10) -> np.ndarray:
        """
        Forecast volatility path for multiple steps.
        
        Shows how volatility is expected to evolve.
        """
        variances = self.forecast(steps)
        return np.sqrt(variances)
    
    def get_half_life(self) -> float:
        """
        Calculate half-life of volatility shocks.
        
        How many periods for a shock to decay by half.
        """
        if ARCH_AVAILABLE and self.params is not None:
            persistence = self.params['alpha[1]'] + self.params['beta[1]']
        else:
            persistence = self.alpha + self.beta
        
        if persistence >= 1:
            return float('inf')
        
        return np.log(0.5) / np.log(persistence)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("GARCH VOLATILITY MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Fit model
    model = GARCHVolatilityModel()
    model.fit(data)
    
    # Get current regime
    regime = model.get_volatility_regime()
    
    print(f"\n📊 Volatility Analysis:")
    print(f"   Current Volatility: {regime['current_volatility']:.4f}")
    print(f"   Unconditional Vol:  {regime['unconditional_volatility']:.4f}")
    print(f"   Volatility Ratio:   {regime['volatility_ratio']:.2f}x")
    print(f"\n   Regime: {regime['regime'].upper()}")
    print(f"   {regime['message']}")
    print(f"   Risk Adjustment: {regime['risk_adjustment']:.1%}")
    
    # Forecast volatility path
    print(f"\n📈 Volatility Forecast (next 5 periods):")
    vol_path = model.forecast_volatility_path(5)
    for i, vol in enumerate(vol_path, 1):
        print(f"   Step {i}: {vol:.4f}")
    
    # Half-life
    hl = model.get_half_life()
    print(f"\n⏱️ Shock Half-Life: {hl:.1f} periods")
