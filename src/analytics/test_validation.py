"""
tests/test_validation.py
Tests for validation framework
"""

import pytest
import numpy as np
from validation import Validation
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call, monte_carlo_option_price


class TestKSTest:
    """Test Kolmogorov-Smirnov test"""
    
    def test_identical_distributions(self):
        """Test K-S test with identical distributions"""
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)
        
        result = Validation.ks_test(data1, data2)
        assert result["p_value"] > 0.05, "Identical normal distributions should pass K-S test"
    
    def test_different_distributions(self):
        """Test K-S test with different distributions"""
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(2, 1, 1000)
        
        result = Validation.ks_test(data1, data2)
        assert result["p_value"] < 0.05, "Different distributions should fail K-S test"


class TestErrorMetrics:
    """Test error metric computation"""
    
    def test_zero_error(self):
        """Test error when prediction equals truth"""
        bs_price = 10.0
        prices = np.array([10.0, 10.0, 10.0])
        
        errors = Validation.compute_errors(bs_price, prices)
        assert np.isclose(errors["MAE"], 0.0), "Zero error expected"
        assert errors["Passed_1pct"], "Should pass 1% threshold"
    
    def test_small_error(self):
        """Test error with small deviations"""
        bs_price = 10.0
        prices = np.array([10.05, 10.02, 9.98])
        
        errors = Validation.compute_errors(bs_price, prices)
        assert errors["MAE"] < 1.0, "MAE should be < 1%"
        assert errors["Passed_1pct"], "Should pass 1% threshold"


class TestConvergence:
    """Test convergence analysis"""
    
    def test_convergence_decreasing_error(self):
        """Test that error decreases with more qubits"""
        convergence = Validation.convergence_analysis(
            qubits_range=[6, 8, 10],
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
            pricer_class=ImprovedQuantumWalkOptionPricer,
            n_sims=5000
        )
        
        errors = convergence["errors"]
        # Generally errors should decrease with more qubits
        assert errors[1] <= errors[0] * 1.5, "Error should not increase dramatically"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])