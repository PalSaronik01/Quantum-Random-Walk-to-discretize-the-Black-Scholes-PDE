# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Quick Install

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/quantum-walk-black-scholes.git
cd quantum-walk-black-scholes
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n quantum-bs python=3.8
conda activate quantum-bs
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Package
```bash
# Development installation (editable)
pip install -e .

# Or standard installation
pip install .
```

## Verification

Verify installation by running:
```bash
python -c "from src import ImprovedQuantumWalkOptionPricer; print('✓ Installation successful!')"
```

Or run quick start:
```bash
python quickstart.py
```

## Optional Dependencies

For additional features:
```bash
# Development tools
pip install pytest pytest-cov black flake8

# Jupyter notebooks
pip install jupyter jupyterlab

# Advanced visualization
pip install matplotlib seaborn plotly
```

## Troubleshooting

### ImportError: No module named 'pricer'
- Ensure you've run `pip install -e .` from the root directory
- Check that you're in the correct virtual environment

### YAML parsing errors
- Ensure pyyaml is installed: `pip install pyyaml>=5.3`

### Scipy/Numpy issues
- Upgrade: `pip install --upgrade numpy scipy`

## Docker Installation (Optional)

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python", "quickstart.py"]
```

Build and run:
```bash
docker build -t quantum-walk-bs .
docker run quantum-walk-bs
```

## Uninstall

```bash
pip uninstall quantum-walk-black-scholes
```

## System Requirements

- **Minimum**: 2GB RAM, 500MB disk space
- **Recommended**: 4GB RAM, 1GB disk space
- **For large simulations**: 8GB+ RAM

## Next Steps

After installation:
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run examples in `examples/`
3. Explore [API_REFERENCE.md](API_REFERENCE.md)
4. Check [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md)
