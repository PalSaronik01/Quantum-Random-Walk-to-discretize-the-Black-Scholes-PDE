"""
tests/test_pricer.py
Unit tests for option pricer
"""

import pytest
import numpy as np
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call, monte_carlo_option_price


class TestBlackScholes:
    """Test Black-Scholes pricing"""
    
    def test_call_price_atm(self):
        """Test ATM (at-the-money) call price"""
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(S0, K, T, r, sigma)
        assert price > 0, "Call price must be positive"
        assert price < S0, "Call price should be less than spot price"
    
    def test_call_price_itm(self):
        """Test ITM (in-the-money) call price"""
        S0, K, T, r, sigma = 110, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(S0, K, T, r, sigma)
        intrinsic = S0 - K
        assert price > intrinsic, "Option price > intrinsic value"
    
    def test_call_price_otm(self):
        """Test OTM (out-of-the-money) call price"""
        S0, K, T, r, sigma = 90, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(S0, K, T, r, sigma)
        assert price > 0, "OTM call still has value"
    
    def test_zero_time_value(self):
        """Test expiration (T=0) case"""
        S0, K, T, r, sigma = 110, 100, 0.001, 0.05, 0.2
        price = black_scholes_call(S0, K, T, r, sigma)
        intrinsic = max(S0 - K, 0)
        assert np.isclose(price, intrinsic, atol=0.1), "Near expiration, price ~intrinsic value"


class TestMonteCarlo:
    """Test Monte Carlo pricing"""
    
    def test_mc_price_reasonable(self):
        """Test MC price is reasonable"""
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bs_price = black_scholes_call(S0, K, T, r, sigma)
        mc_price, _ = monte_carlo_option_price(S0, K, T, r, sigma, 50000)
        error = abs(mc_price - bs_price) / bs_price * 100
        assert error < 1, "MC error should be < 1%"
    
    def test_mc_convergence(self):
        """Test MC convergence with increasing simulations"""
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        bs_price = black_scholes_call(S0, K, T, r, sigma)
        
        prices = []
        for n_sims in [1000, 5000, 10000]:
            mc_price, _ = monte_carlo_option_price(S0, K, T, r, sigma, n_sims)
            prices.append(mc_price)
        
        # Check that errors decrease
        errors = [abs(p - bs_price) for p in prices]
        assert errors[1] <= errors[0], "Error should decrease with more simulations"


class TestQuantumWalk:
    """Test quantum walk pricer"""
    
    def test_qw_initialization(self):
        """Test pricer initialization"""
        pricer = ImprovedQuantumWalkOptionPricer(12, 20)
        assert pricer.num_path_qubits == 12
        assert pricer.num_walk_steps == 20
        assert pricer.num_positions == 4096
    
    def test_qw_pricing(self):
        """Test quantum walk pricing"""
        pricer = ImprovedQuantumWalkOptionPricer(10, 15)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        price, details = pricer.price_option(S0, K, T, r, sigma, 5000, verbose=False)
        
        assert price > 0, "QW price must be positive"
        assert "terminal_prices" in details
        assert "payoffs" in details
        assert "runtime" in details
    
    def test_qw_vs_bs(self):
        """Test QW vs Black-Scholes"""
        pricer = ImprovedQuantumWalkOptionPricer(12, 20)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs_price = black_scholes_call(S0, K, T, r, sigma)
        qw_price, _ = pricer.price_option(S0, K, T, r, sigma, 10000, verbose=False)
        
        error = abs(qw_price - bs_price) / bs_price * 100
        assert error < 1, "QW error should be < 1%"


class TestDistributions:
    """Test quantum walk distribution"""
    
    def test_distribution_sums_to_one(self):
        """Test probability distribution sums to 1"""
        pricer = ImprovedQuantumWalkOptionPricer(8, 15)
        probs = pricer.quantum_walk_distribution()
        assert np.isclose(np.sum(probs), 1.0), "Probabilities must sum to 1"
    
    def test_distribution_positive(self):
        """Test all probabilities are non-negative"""
        pricer = ImprovedQuantumWalkOptionPricer(8, 15)
        probs = pricer.quantum_walk_distribution()
        assert np.all(probs >= 0), "All probabilities must be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])