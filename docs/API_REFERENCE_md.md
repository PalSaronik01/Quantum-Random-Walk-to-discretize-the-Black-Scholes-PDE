# API Reference

Complete API documentation for the Quantum Walk Black-Scholes Option Pricer.

## Main Classes

### ImprovedQuantumWalkOptionPricer

**Location**: `pricer.py`

Main quantum walk option pricer with all 4 bug fixes.

```python
class ImprovedQuantumWalkOptionPricer:
    def __init__(self, num_path_qubits: int = 12, num_walk_steps: int = 20)
    def quantum_walk_distribution(self, theta: float = np.pi/4) -> np.ndarray
    def price_option(self, S0: float, K: float, T: float, r: float, sigma: float, 
                     n_sims: int = 50000, verbose: bool = True) -> Tuple[float, Dict]
```

**Parameters**:
- `num_path_qubits`: Number of position qubits (default: 12 for 4,096 positions)
- `num_walk_steps`: Number of quantum walk steps (default: 20)

**Methods**:
- `quantum_walk_distribution(theta)`: Generate probability distribution
- `price_option(...)`: Price European call option

**Returns**: Tuple of (price: float, details: dict)

### BlackScholesAnalytics

**Location**: `black_scholes.py`

Black-Scholes option pricing and Greeks computation.

```python
class BlackScholesAnalytics:
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type: str = "call") -> float
    
    @staticmethod
    def gamma(S, K, T, r, sigma) -> float
    
    @staticmethod
    def vega(S, K, T, r, sigma) -> float
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type: str = "call") -> float
    
    @staticmethod
    def all_greeks(S, K, T, r, sigma, option_type: str = "call") -> Dict
```

### QuantumWalkSimulator

**Location**: `quantum_walk.py`

Discrete-time quantum walk simulator.

```python
class QuantumWalkSimulator:
    def __init__(self, num_qubits: int = 12, num_steps: int = 20)
    def generate_distribution(self, theta: float = np.pi/4) -> np.ndarray
    def get_statistics(self, theta: float = np.pi/4) -> Dict
    def sample(self, n_samples: int, theta: float = np.pi/4) -> np.ndarray
```

### Validation

**Location**: `validation.py`

Statistical validation framework.

```python
class Validation:
    @staticmethod
    def ks_test(mc_prices: np.ndarray, qw_prices: np.ndarray) -> Dict
    
    @staticmethod
    def compute_errors(bs_price: float, qw_prices: np.ndarray) -> Dict
    
    @staticmethod
    def convergence_analysis(qubits_range, S0, K, T, r, sigma, 
                            pricer_class, n_sims: int = 10000) -> Dict
    
    @staticmethod
    def sensitivity_analysis(base_params, param_name, param_range, 
                           pricer_class, n_sims: int = 5000) -> Dict
```

## Utility Functions

### ConfigManager

**Location**: `utils.py`

```python
class ConfigManager:
    @staticmethod
    def load_yaml(filepath: str) -> Dict
    @staticmethod
    def save_yaml(config: Dict, filepath: str) -> None
    @staticmethod
    def load_json(filepath: str) -> Dict
    @staticmethod
    def save_json(data: Dict, filepath: str) -> None
```

### DataProcessor

**Location**: `utils.py`

```python
class DataProcessor:
    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray
    @staticmethod
    def standardize_array(arr: np.ndarray) -> np.ndarray
    @staticmethod
    def compute_percentiles(arr, percentiles) -> Dict
    @staticmethod
    def remove_outliers(arr, n_std: float = 3.0) -> np.ndarray
```

### ResultsWriter

**Location**: `utils.py`

```python
class ResultsWriter:
    @staticmethod
    def write_csv(data: Dict, filepath: str, delimiter: str = ',') -> None
    @staticmethod
    def write_results(results: Dict, filepath: str) -> None
```

### Logger

**Location**: `utils.py`

```python
class Logger:
    def __init__(self, name: str = "QuantumWalk", verbose: bool = True)
    def info(self, message: str) -> None
    def warning(self, message: str) -> None
    def error(self, message: str) -> None
    def success(self, message: str) -> None
```

## Module-Level Functions

### pricer.py

```python
def black_scholes_call(S, K, T, r, sigma) -> float
    """Analytical Black-Scholes call option price"""

def monte_carlo_option_price(S0, K, T, r, sigma, n_sims=50000) -> Tuple[float, np.ndarray]
    """Monte Carlo option pricing baseline"""
```

### validation.py

```python
def compute_greeks(pricer, S0, K, T, r, sigma, dS=0.01, dsigma=0.001) -> Dict
    """Compute option Greeks using finite differences"""
```

## Data Structures

### Option Details (returned from price_option)

```python
details = {
    "terminal_prices": np.ndarray,      # S(T) values
    "payoffs": np.ndarray,              # max(S(T) - K, 0)
    "itm_probability": float,           # P(S(T) > K)
    "runtime": float,                   # Execution time in seconds
    "n_simulations": int,               # Number of simulations
    "parameters": {                     # Option parameters
        "S0": float,
        "K": float,
        "T": float,
        "r": float,
        "sigma": float,
        "num_path_qubits": int,
        "num_walk_steps": int
    }
}
```

### K-S Test Result

```python
ks_result = {
    "statistic": float,                 # K-S statistic
    "p_value": float,                   # p-value
    "passed": bool,                     # Distributions identical?
    "interpretation": str               # Human-readable result
}
```

### Error Metrics

```python
errors = {
    "MAE": float,                       # Mean Absolute Error %
    "RMSE": float,                      # Root Mean Square Error %
    "Max": float,                       # Maximum Error %
    "Min": float,                       # Minimum Error %
    "Std": float,                       # Standard Deviation %
    "Passed_1pct": bool                 # Below 1% threshold?
}
```

## Configuration Files

All configuration files located in `config/`:

- `default_config.yaml`: Standard configuration
- `production_config.yaml`: Production-grade settings
- `test_config.yaml`: Fast testing configuration

### Configuration Schema

```yaml
default_config:
  pricer:
    num_path_qubits: int               # 6-14 recommended
    num_walk_steps: int                # 10-20 recommended
    n_simulations: int                 # 5000-100000 recommended
  
  option:
    S0: float                          # Spot price
    K: float                           # Strike price
    T: float                           # Time to maturity (years)
    r: float                           # Risk-free rate
    sigma: float                       # Volatility
  
  validation:
    ks_alpha: float                    # K-S test significance
    error_threshold: float             # Max acceptable error
```

## Type Hints

Common type patterns used throughout:

```python
from typing import Dict, List, Tuple, Any

# Option parameters
S0: float                              # Spot price
K: float                               # Strike price
T: float                               # Time to maturity
r: float                               # Risk-free rate
sigma: float                           # Volatility

# Results
prices: np.ndarray                     # Array of prices
option_type: str                       # "call" or "put"
n_sims: int                            # Number of simulations
```

## Error Handling

### Common Exceptions

```python
ValueError                             # Invalid parameters
TypeError                              # Wrong type
RuntimeError                           # Computation error
FileNotFoundError                      # Config file not found
```

### Example Error Handling

```python
try:
    pricer = ImprovedQuantumWalkOptionPricer(12, 20)
    price, details = pricer.price_option(100, 100, 1.0, 0.05, 0.2)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Computation error: {e}")
```

## Performance Notes

- **Qubits 6-8**: Fast (< 1s), ~0.5% error
- **Qubits 10-12**: Balanced (1-5s), ~0.1% error
- **Qubits 14+**: Slow (> 10s), minimal improvement

## Reproducibility

Set random seed for reproducible results:

```python
import numpy as np
np.random.seed(42)

# Now all random operations are reproducible
pricer = ImprovedQuantumWalkOptionPricer(12, 20)
price, _ = pricer.price_option(100, 100, 1.0, 0.05, 0.2)
```
