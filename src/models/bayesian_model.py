#!/usr/bin/env python3
"""
Bayesian Crash Predictor

Probabilistic modeling using Bayesian inference for:
- Posterior distributions of crash points
- Credible intervals (not just confidence)
- Uncertainty-aware predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PyMC, fall back to simple implementation if not available
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning("PyMC not available. Using simplified Bayesian inference.")


class BayesianCrashPredictor:
    """
    Bayesian model for crash point prediction.
    
    Uses a log-normal prior (crash points are positive, right-skewed)
    with empirical Bayes updating.
    """
    
    def __init__(self, 
                 prior_mu: float = 1.0,
                 prior_sigma: float = 1.0,
                 n_samples: int = 2000,
                 n_chains: int = 2):
        """
        Args:
            prior_mu: Prior log-mean
            prior_sigma: Prior log-std
            n_samples: Number of MCMC samples
            n_chains: Number of MCMC chains
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.n_samples = n_samples
        self.n_chains = n_chains
        
        self.posterior_mu = prior_mu
        self.posterior_sigma = prior_sigma
        self.trace = None
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> 'BayesianCrashPredictor':
        """
        Fit Bayesian model to historical crash data.
        
        Args:
            data: Array of crash multipliers
        
        Returns:
            self
        """
        # Filter valid data (crashes > 1.0)
        data = data[data >= 1.0]
        log_data = np.log(data)
        
        if PYMC_AVAILABLE:
            self._fit_pymc(log_data)
        else:
            self._fit_simple(log_data)
        
        self.fitted = True
        return self
    
    def _fit_pymc(self, log_data: np.ndarray):
        """Fit using PyMC."""
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=self.prior_mu, sigma=1.0)
            sigma = pm.HalfNormal('sigma', sigma=self.prior_sigma)
            
            # Likelihood
            likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=log_data)
            
            # Sample
            self.trace = pm.sample(
                self.n_samples, 
                chains=self.n_chains,
                return_inferencedata=True,
                progressbar=False
            )
        
        # Extract posterior parameters
        self.posterior_mu = float(self.trace.posterior['mu'].mean())
        self.posterior_sigma = float(self.trace.posterior['sigma'].mean())
        
        logger.info(f"PyMC fit complete: mu={self.posterior_mu:.3f}, sigma={self.posterior_sigma:.3f}")
    
    def _fit_simple(self, log_data: np.ndarray):
        """Simple conjugate prior update (Normal-Normal)."""
        n = len(log_data)
        data_mean = np.mean(log_data)
        data_var = np.var(log_data)
        
        # Posterior mean (weighted average of prior and data)
        prior_precision = 1 / (self.prior_sigma ** 2)
        data_precision = n / data_var if data_var > 0 else n
        
        posterior_precision = prior_precision + data_precision
        self.posterior_mu = (prior_precision * self.prior_mu + data_precision * data_mean) / posterior_precision
        self.posterior_sigma = np.sqrt(1 / posterior_precision + data_var / n)
        
        logger.info(f"Simple Bayesian fit: mu={self.posterior_mu:.3f}, sigma={self.posterior_sigma:.3f}")
    
    def predict_distribution(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from the posterior predictive distribution.
        
        Returns:
            Array of sampled crash point predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Sample from log-normal posterior
        log_samples = np.random.normal(self.posterior_mu, self.posterior_sigma, n_samples)
        crash_samples = np.exp(log_samples)
        
        return crash_samples
    
    def predict_point(self) -> float:
        """Get point prediction (posterior mean)."""
        return np.exp(self.posterior_mu + self.posterior_sigma**2 / 2)
    
    def get_credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get Bayesian credible interval.
        
        Unlike frequentist CI, this has a direct probability interpretation:
        "There is a 95% probability the next crash is in this range."
        
        Args:
            confidence: Credible level (0-1)
        
        Returns:
            (lower, upper) bounds
        """
        samples = self.predict_distribution(10000)
        alpha = (1 - confidence) / 2
        
        lower = np.percentile(samples, alpha * 100)
        upper = np.percentile(samples, (1 - alpha) * 100)
        
        return (lower, upper)
    
    def probability_above(self, threshold: float) -> float:
        """
        Calculate probability that next crash is >= threshold.
        
        Useful for Kelly Criterion calculations.
        
        Args:
            threshold: Crash multiplier threshold
        
        Returns:
            Probability (0-1)
        """
        samples = self.predict_distribution(10000)
        return np.mean(samples >= threshold)
    
    def probability_below(self, threshold: float) -> float:
        """Calculate probability that next crash is < threshold."""
        return 1 - self.probability_above(threshold)
    
    def get_uncertainty(self) -> Dict:
        """
        Get uncertainty metrics.
        
        Returns:
            Dict with various uncertainty measures
        """
        samples = self.predict_distribution(10000)
        
        return {
            'posterior_mu': self.posterior_mu,
            'posterior_sigma': self.posterior_sigma,
            'prediction_mean': np.mean(samples),
            'prediction_std': np.std(samples),
            'prediction_median': np.median(samples),
            'entropy': self._calculate_entropy(samples),
            'coefficient_of_variation': np.std(samples) / np.mean(samples)
        }
    
    def _calculate_entropy(self, samples: np.ndarray) -> float:
        """Calculate entropy of the predictive distribution."""
        # Use histogram-based entropy estimation
        hist, _ = np.histogram(samples, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def update(self, new_data: np.ndarray):
        """
        Online update with new data (incremental Bayesian learning).
        
        Args:
            new_data: New crash observations
        """
        # Use current posterior as new prior
        self.prior_mu = self.posterior_mu
        self.prior_sigma = self.posterior_sigma
        
        # Fit on new data
        self.fit(new_data)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        ci_90 = self.get_credible_interval(0.90)
        ci_95 = self.get_credible_interval(0.95)
        
        return {
            'point_prediction': self.predict_point(),
            'credible_interval_90': ci_90,
            'credible_interval_95': ci_95,
            'p_above_1.5x': self.probability_above(1.5),
            'p_above_2.0x': self.probability_above(2.0),
            'p_above_3.0x': self.probability_above(3.0),
            'p_above_5.0x': self.probability_above(5.0),
            'uncertainty': self.get_uncertainty()
        }


class BayesianKellyOptimizer:
    """
    Kelly Criterion using Bayesian probability estimates.
    
    Uses posterior predictive probabilities instead of frequentist estimates.
    """
    
    def __init__(self, 
                 bayesian_model: BayesianCrashPredictor,
                 fractional_kelly: float = 0.25,
                 house_edge: float = 0.01):
        self.model = bayesian_model
        self.fractional_kelly = fractional_kelly
        self.house_edge = house_edge
    
    def calculate_bet(self, bankroll: float, target_multiplier: float) -> Dict:
        """
        Calculate optimal bet using Bayesian probabilities.
        
        Args:
            bankroll: Current bankroll
            target_multiplier: Target cashout
        
        Returns:
            Bet recommendation with uncertainty
        """
        # Get Bayesian probability
        p_raw = self.model.probability_above(target_multiplier)
        p = p_raw * (1 - self.house_edge)
        q = 1 - p
        b = target_multiplier - 1
        
        # Kelly formula
        kelly = (b * p - q) / b if b > 0 else 0
        safe_kelly = max(0, min(0.10, kelly * self.fractional_kelly))
        
        bet_amount = bankroll * safe_kelly
        ev = (p * b - q) * bet_amount
        
        # Get uncertainty from Bayesian model
        uncertainty = self.model.get_uncertainty()
        
        return {
            'bet_amount': bet_amount,
            'kelly_fraction': safe_kelly * 100,
            'win_probability': p * 100,
            'expected_value': ev,
            'should_bet': ev > 0 and safe_kelly > 0.01,
            'bayesian_probability': p_raw,
            'uncertainty': uncertainty['coefficient_of_variation'],
            'confidence': 'high' if uncertainty['coefficient_of_variation'] < 0.5 else 'medium'
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("BAYESIAN CRASH PREDICTOR")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Create and fit model
    model = BayesianCrashPredictor()
    model.fit(data)
    
    # Print summary
    summary = model.get_summary()
    print(f"\n📊 Point Prediction: {summary['point_prediction']:.2f}x")
    print(f"📈 90% Credible Interval: [{summary['credible_interval_90'][0]:.2f}, {summary['credible_interval_90'][1]:.2f}]")
    print(f"📉 95% Credible Interval: [{summary['credible_interval_95'][0]:.2f}, {summary['credible_interval_95'][1]:.2f}]")
    
    print(f"\n🎯 Probability Analysis:")
    print(f"   P(crash >= 1.5x): {summary['p_above_1.5x']*100:.1f}%")
    print(f"   P(crash >= 2.0x): {summary['p_above_2.0x']*100:.1f}%")
    print(f"   P(crash >= 3.0x): {summary['p_above_3.0x']*100:.1f}%")
    print(f"   P(crash >= 5.0x): {summary['p_above_5.0x']*100:.1f}%")
    
    # Bayesian Kelly
    print("\n" + "=" * 60)
    print("BAYESIAN KELLY OPTIMIZER")
    print("=" * 60)
    
    kelly = BayesianKellyOptimizer(model)
    bet_info = kelly.calculate_bet(100, 2.0)
    
    print(f"\n💰 For 2.0x target with $100 bankroll:")
    print(f"   Bayesian Win Probability: {bet_info['bayesian_probability']*100:.1f}%")
    print(f"   Recommended Bet: ${bet_info['bet_amount']:.2f}")
    print(f"   Kelly Fraction: {bet_info['kelly_fraction']:.2f}%")
    print(f"   Expected Value: ${bet_info['expected_value']:.2f}")
    print(f"   Should Bet: {'YES' if bet_info['should_bet'] else 'NO'}")
    print(f"   Confidence: {bet_info['confidence']}")
