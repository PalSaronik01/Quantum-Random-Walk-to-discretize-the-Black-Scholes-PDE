from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-walk-black-scholes",
    version="1.0.0",
    author="Saronik Pal",
    author_email="saronik.pal2004@gmail.com",
    description="Quantum Walk Black-Scholes Option Pricer ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PalSaronik01/Quantum-Random-Walk-to-discretize-the-Black-Scholes-PDE.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "qiskit>=0.24.0",
        "pyyaml>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "jupyter>=1.0.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-walk-pricer=src.cli:main",
        ],
    },
)