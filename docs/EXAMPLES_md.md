# Examples and Usage Patterns

Complete guide to using the Quantum Walk Black-Scholes Option Pricer.

## Basic Examples

### Example 1: Simple Option Pricing

```python
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call

# Create pricer instance
pricer = ImprovedQuantumWalkOptionPricer(num_path_qubits=12, num_walk_steps=20)

# Price a single option
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
price, details = pricer.price_option(S0, K, T, r, sigma, n_sims=50000)

# Compare with Black-Scholes
bs_price = black_scholes_call(S0, K, T, r, sigma)
error = abs(price - bs_price) / bs_price * 100

print(f"Quantum Walk: ${price:.6f}")
print(f"Black-Scholes: ${bs_price:.6f}")
print(f"Error: {error:.4f}%")
```

### Example 2: Greeks Calculation

```python
from black_scholes import BlackScholesAnalytics

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

# Get all Greeks
greeks = BlackScholesAnalytics.all_greeks(S0, K, T, r, sigma)

print(f"Call Price: ${greeks['call_price']:.6f}")
print(f"Delta:      {greeks['delta']:.6f}")
print(f"Gamma:      {greeks['gamma']:.6f}")
print(f"Vega:       {greeks['vega']:.6f}")
print(f"Theta:      {greeks['theta']:.6f}")
```

### Example 3: Batch Processing

```python
import pandas as pd
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call

# Define multiple options
options_df = pd.DataFrame({
    'S': [95, 100, 105],
    'K': [100, 100, 100],
    'T': [0.5, 1.0, 1.5],
    'sigma': [0.15, 0.20, 0.25]
})

pricer = ImprovedQuantumWalkOptionPricer(12, 20)
results = []

for _, opt in options_df.iterrows():
    bs = black_scholes_call(opt['S'], opt['K'], opt['T'], 0.05, opt['sigma'])
    qw, _ = pricer.price_option(opt['S'], opt['K'], opt['T'], 0.05, opt['sigma'], 
                               n_sims=10000, verbose=False)
    results.append({
        'S': opt['S'], 'K': opt['K'], 'T': opt['T'],
        'BS': bs, 'QW': qw, 'Error%': abs(qw-bs)/bs*100
    })

results_df = pd.DataFrame(results)
print(results_df)
```

## Advanced Examples

### Example 4: Sensitivity Analysis

```python
from validation import Validation
from pricer import ImprovedQuantumWalkOptionPricer

pricer = ImprovedQuantumWalkOptionPricer(12, 20)

# Analyze sensitivity to volatility
sensitivity = Validation.sensitivity_analysis(
    base_params={'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.2},
    param_name='sigma',
    param_range=[0.10, 0.15, 0.20, 0.25, 0.30],
    pricer_class=ImprovedQuantumWalkOptionPricer,
    n_sims=10000
)

print("Volatility Sensitivity:")
for vol, price in zip(sensitivity['range'], sensitivity['prices']):
    print(f"σ = {vol:.2f}: ${price:.6f}")
```

### Example 5: Convergence Testing

```python
from validation import Validation
from pricer import ImprovedQuantumWalkOptionPricer

# Test convergence with increasing qubits
convergence = Validation.convergence_analysis(
    qubits_range=[6, 8, 10, 12, 14],
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    pricer_class=ImprovedQuantumWalkOptionPricer,
    n_sims=5000
)

print("Convergence Analysis:")
print(f"{'Qubits':<10} {'Positions':<15} {'Error %':<12} {'Time (s)':<12}")
print("-"*50)
for q, p, err, t in zip(convergence['qubits'], convergence['positions'],
                        convergence['errors'], convergence['times']):
    print(f"{q:<10} {p:<15} {err:<11.4f}% {t:<11.2f}")
```

### Example 6: Statistical Validation

```python
from validation import Validation
from pricer import ImprovedQuantumWalkOptionPricer, monte_carlo_option_price, black_scholes_call

# Generate distributions
mc_price, mc_paths = monte_carlo_option_price(100, 100, 1.0, 0.05, 0.2, 50000)
pricer = ImprovedQuantumWalkOptionPricer(12, 20)
qw_price, qw_details = pricer.price_option(100, 100, 1.0, 0.05, 0.2, 50000, verbose=False)

# K-S Test
ks_result = Validation.ks_test(mc_paths, qw_details['terminal_prices'])
print(f"K-S Statistic: {ks_result['statistic']:.6f}")
print(f"p-value: {ks_result['p_value']:.6f}")
print(f"Distributions match: {ks_result['passed']}")

# Error metrics
bs_price = black_scholes_call(100, 100, 1.0, 0.05, 0.2)
errors = Validation.compute_errors(bs_price, [qw_price]*100)
print(f"MAE: {errors['MAE']:.4f}%")
print(f"RMSE: {errors['RMSE']:.4f}%")
```

## Configuration Examples

### Example 7: Custom Configuration

```python
from utils import ConfigManager
from pricer import ImprovedQuantumWalkOptionPricer

# Create custom config
config = {
    'pricer': {
        'num_path_qubits': 10,
        'num_walk_steps': 15,
        'n_simulations': 25000
    },
    'option': {
        'S0': 100, 'K': 100, 'T': 1.0,
        'r': 0.05, 'sigma': 0.2
    }
}

# Save configuration
ConfigManager.save_yaml(config, 'my_config.yaml')

# Load and use
loaded_config = ConfigManager.load_yaml('my_config.yaml')
p_cfg = loaded_config['pricer']
o_cfg = loaded_config['option']

pricer = ImprovedQuantumWalkOptionPricer(p_cfg['num_path_qubits'], p_cfg['num_walk_steps'])
price, _ = pricer.price_option(o_cfg['S0'], o_cfg['K'], o_cfg['T'], 
                               o_cfg['r'], o_cfg['sigma'], 
                               n_sims=p_cfg['n_simulations'], verbose=False)
print(f"Price: ${price:.6f}")
```

### Example 8: Results Export

```python
from utils import ResultsWriter
import numpy as np

# Generate results
results = {
    'method': ['BS', 'MC', 'QW'],
    'price': [10.450609, 10.455234, 10.463843],
    'error': [0.0, 0.04, 0.127],
    'runtime': [0.001, 2.34, 5.67]
}

# Save to CSV
ResultsWriter.write_csv(results, 'results.csv')

# Save to text file
ResultsWriter.write_results(results, 'results.txt')

# Manually create and save
data = {'Option': 'ATM', 'Price': 10.45, 'Error%': 0.127}
ResultsWriter.write_results(data, 'summary.txt')
```

## Performance Examples

### Example 9: Speed Comparison

```python
import time
from pricer import ImprovedQuantumWalkOptionPricer, black_scholes_call, monte_carlo_option_price

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

# Black-Scholes
start = time.time()
bs = black_scholes_call(S0, K, T, r, sigma)
bs_time = time.time() - start

# Monte Carlo
start = time.time()
mc, _ = monte_carlo_option_price(S0, K, T, r, sigma, 50000)
mc_time = time.time() - start

# Quantum Walk
pricer = ImprovedQuantumWalkOptionPricer(12, 20)
start = time.time()
qw, _ = pricer.price_option(S0, K, T, r, sigma, 50000, verbose=False)
qw_time = time.time() - start

print(f"Black-Scholes: {bs_time*1000:.3f} ms")
print(f"Monte Carlo:   {mc_time:.3f} s")
print(f"Quantum Walk:  {qw_time:.3f} s")
```

### Example 10: Memory Efficiency

```python
from pricer import ImprovedQuantumWalkOptionPricer
import sys

# Small configuration (memory efficient)
pricer_small = ImprovedQuantumWalkOptionPricer(8, 10)  # 256 positions
price_small, details_small = pricer_small.price_option(100, 100, 1.0, 0.05, 0.2, 5000, verbose=False)

# Large configuration (more accurate)
pricer_large = ImprovedQuantumWalkOptionPricer(12, 20)  # 4,096 positions
price_large, details_large = pricer_large.price_option(100, 100, 1.0, 0.05, 0.2, 50000, verbose=False)

print(f"Small config memory: {sys.getsizeof(details_small)} bytes")
print(f"Large config memory: {sys.getsizeof(details_large)} bytes")
```

## Real-World Use Cases

### Portfolio Greeks Calculation

```python
# Calculate Greeks for a portfolio of options
portfolio = [
    {'S': 100, 'K': 95, 'T': 0.25},
    {'S': 100, 'K': 100, 'T': 0.5},
    {'S': 100, 'K': 105, 'T': 1.0},
]

total_delta = 0
total_vega = 0

for opt in portfolio:
    greeks = BlackScholesAnalytics.all_greeks(opt['S'], opt['K'], opt['T'], 0.05, 0.2)
    total_delta += greeks['delta']
    total_vega += greeks['vega']

print(f"Portfolio Delta: {total_delta:.4f}")
print(f"Portfolio Vega:  {total_vega:.4f}")
```

### Volatility Surface Pricing

```python
# Price options across spot and volatility ranges
spots = np.linspace(90, 110, 5)
vols = np.linspace(0.1, 0.3, 5)

prices = np.zeros((len(spots), len(vols)))

for i, S in enumerate(spots):
    for j, sigma in enumerate(vols):
        prices[i, j] = black_scholes_call(S, 100, 1.0, 0.05, sigma)

print("Volatility Surface:")
print(prices)
```

## Error Handling Examples

```python
from pricer import ImprovedQuantumWalkOptionPricer

try:
    pricer = ImprovedQuantumWalkOptionPricer(15, 25)  # Too many qubits
    price, _ = pricer.price_option(100, 100, 1.0, 0.05, 0.2)
except ValueError as e:
    print(f"Configuration error: {e}")
    pricer = ImprovedQuantumWalkOptionPricer(12, 20)  # Use defaults

try:
    price, _ = pricer.price_option(-100, 100, 1.0, 0.05, 0.2)  # Invalid spot
except ValueError as e:
    print(f"Parameter error: {e}")
```

These examples demonstrate the flexibility and power of the quantum walk option pricer framework!
