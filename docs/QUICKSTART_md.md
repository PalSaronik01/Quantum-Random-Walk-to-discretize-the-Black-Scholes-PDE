# Quick Start Guide

Get started with the Quantum Walk Black-Scholes Option Pricer in 5 minutes!

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## First Run: Basic Pricing

```python
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call

# Option parameters
S0 = 100      # Spot price
K = 100       # Strike price
T = 1.0       # 1 year to expiration
r = 0.05      # 5% risk-free rate
sigma = 0.2   # 20% volatility

# Black-Scholes (analytical)
bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes: ${bs_price:.6f}")

# Quantum Walk pricing
pricer = ImprovedQuantumWalkOptionPricer(num_path_qubits=12, num_walk_steps=20)
qw_price, details = pricer.price_option(S0, K, T, r, sigma, n_sims=50000)
print(f"Quantum Walk:  ${qw_price:.6f}")
print(f"Error: {abs(qw_price - bs_price) / bs_price * 100:.4f}%")
```

## Run Examples

```bash
# Basic pricing
python examples/basic_pricing.py

# Batch pricing
python examples/batch_pricing.py

# Greeks calculation
python examples/greeks_calculation.py

# Noise analysis
python examples/noise_analysis.py

# Method comparison
python examples/comparison.py
```

## Calculate Greeks

```python
from black_scholes import BlackScholesAnalytics

# Get all Greeks
greeks = BlackScholesAnalytics.all_greeks(S0, K, T, r, sigma)

print(f"Call Price: ${greeks['call_price']:.6f}")
print(f"Delta:      {greeks['delta']:.6f}")
print(f"Gamma:      {greeks['gamma']:.6f}")
print(f"Vega:       {greeks['vega']:.6f}")
print(f"Theta:      {greeks['theta']:.6f}")
```

## Statistical Validation

```python
from validation import Validation
from pricer import monte_carlo_option_price

# Generate sample prices
mc_price, mc_paths = monte_carlo_option_price(S0, K, T, r, sigma, 50000)

# K-S test
ks_result = Validation.ks_test(mc_paths, qw_details['terminal_prices'])
print(f"K-S p-value: {ks_result['p_value']:.4f}")
print(f"Distributions match: {ks_result['passed']}")

# Error metrics
errors = Validation.compute_errors(bs_price, [qw_price]*100)
print(f"MAE: {errors['MAE']:.4f}%")
```

## Convergence Analysis

```python
# Test convergence with increasing qubits
convergence = Validation.convergence_analysis(
    qubits_range=[6, 8, 10, 12, 14],
    S0=S0, K=K, T=T, r=r, sigma=sigma,
    pricer_class=ImprovedQuantumWalkOptionPricer,
    n_sims=10000
)

for q, err in zip(convergence['qubits'], convergence['errors']):
    print(f"{q} qubits: {err:.4f}% error")
```

## Configuration

```python
from utils import ConfigManager

# Load configuration
config = ConfigManager.load_yaml("config/default_config.yaml")

# Use in pricer
pricer_config = config['default_config']['pricer']
pricer = ImprovedQuantumWalkOptionPricer(
    pricer_config['num_path_qubits'],
    pricer_config['num_walk_steps']
)
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_pricer.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Save Results

```python
from utils import ResultsWriter

results = {
    "method": ["BS", "MC", "QW"],
    "price": [bs_price, mc_price, qw_price],
    "error": [0.0, mc_error, qw_error]
}

ResultsWriter.write_csv(results, "results.csv")
ResultsWriter.write_results(results, "results.txt")
```

## Common Tasks

### Price multiple options
```python
options = [
    {"S0": 95, "K": 100, "T": 0.5},
    {"S0": 105, "K": 100, "T": 1.5},
]

for opt in options:
    price = black_scholes_call(opt["S0"], opt["K"], opt["T"], r, sigma)
    print(f"S={opt['S0']}, K={opt['K']}, T={opt['T']}: ${price:.6f}")
```

### Compare methods
```python
from pricer import monte_carlo_option_price

bs = black_scholes_call(S0, K, T, r, sigma)
mc, _ = monte_carlo_option_price(S0, K, T, r, sigma, 50000)
qw, _ = pricer.price_option(S0, K, T, r, sigma, 50000, verbose=False)

print(f"BS: ${bs:.6f}, MC: ${mc:.6f}, QW: ${qw:.6f}")
```

### Sensitivity analysis
```python
# Price sensitivity to volatility
for vol in [0.10, 0.15, 0.20, 0.25, 0.30]:
    price = black_scholes_call(S0, K, T, r, vol)
    print(f"σ={vol*100:.0f}%: ${price:.6f}")
```

## Next Steps

- Read [API_REFERENCE.md](API_REFERENCE.md) for complete API
- Study [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md)
- Check [EXAMPLES.md](EXAMPLES.md) for advanced usage
- Review [BUG_FIXES.md](BUG_FIXES.md) for implementation details

## Troubleshooting

**ImportError**: Run `pip install -e .` from root directory
**Runtime slow**: Use fewer qubits or simulations for testing
**Results vary**: Random sampling causes variation; use seed for reproducibility

## Support

- Issues: Check [GitHub Issues](https://github.com/yourusername/quantum-walk-black-scholes/issues)
- Questions: Create [GitHub Discussion](https://github.com/yourusername/quantum-walk-black-scholes/discussions)
