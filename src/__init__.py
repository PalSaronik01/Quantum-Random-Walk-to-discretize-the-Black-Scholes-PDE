"""
src/__init__.py
Package initialization for quantum-walk-black-scholes
"""

__version__ = "1.0.0"
__author__ = "Advanced Quantum Finance"
__all__ = [
    "ImprovedQuantumWalkOptionPricer",
    "black_scholes_call",
    "monte_carlo_option_price",
    "Validation",
]

from .pricer import (
    ImprovedQuantumWalkOptionPricer,
    black_scholes_call,
    monte_carlo_option_price,
)
from .validation import Validation