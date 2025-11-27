"""
src/black_scholes.py
Black-Scholes analytics and Greeks computation
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


class BlackScholesAnalytics:
    """Black-Scholes option pricing and Greeks computation"""
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European call option price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return float(call)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European put option price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return float(put)
    
    @staticmethod
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Compute d1 and d2 for Greeks calculations
        
        Args:
            S, K, T, r, sigma: Option parameters
            
        Returns:
            Tuple of (d1, d2)
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Option delta (price sensitivity to spot price)
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Delta value
        """
        d1, _ = BlackScholesAnalytics.d1_d2(S, K, T, r, sigma)
        if option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Option gamma (delta sensitivity to spot price)
        
        Returns:
            Gamma value
        """
        d1, _ = BlackScholesAnalytics.d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return float(gamma)
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Option vega (price sensitivity to volatility)
        
        Returns:
            Vega value (per 1% change in volatility)
        """
        d1, _ = BlackScholesAnalytics.d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return float(vega)
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Option theta (time decay)
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Theta value (per day)
        """
        d1, d2 = BlackScholesAnalytics.d1_d2(S, K, T, r, sigma)
        
        if option_type == "call":
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
        
        return float(theta)
    
    @staticmethod
    def all_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict:
        """
        Compute all Greeks at once
        
        Args:
            option_type: "call" or "put"
            
        Returns:
            Dictionary with all Greeks
        """
        return {
            "delta": BlackScholesAnalytics.delta(S, K, T, r, sigma, option_type),
            "gamma": BlackScholesAnalytics.gamma(S, K, T, r, sigma),
            "vega": BlackScholesAnalytics.vega(S, K, T, r, sigma),
            "theta": BlackScholesAnalytics.theta(S, K, T, r, sigma, option_type),
            "call_price": BlackScholesAnalytics.call_price(S, K, T, r, sigma),
            "put_price": BlackScholesAnalytics.put_price(S, K, T, r, sigma),
        }