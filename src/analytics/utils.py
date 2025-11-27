"""
src/utils.py
Utility functions for quantum walk option pricer
"""

import numpy as np
import json
import yaml
from typing import Dict, Any


class ConfigManager:
    """Configuration management"""
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(config: Dict, filepath: str) -> None:
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> None:
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]"""
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    @staticmethod
    def standardize_array(arr: np.ndarray) -> np.ndarray:
        """Standardize array (mean=0, std=1)"""
        return (arr - np.mean(arr)) / np.std(arr)
    
    @staticmethod
    def compute_percentiles(arr: np.ndarray, percentiles: list = [5, 25, 50, 75, 95]) -> Dict:
        """Compute percentiles"""
        return {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}
    
    @staticmethod
    def remove_outliers(arr: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """Remove outliers beyond n_std standard deviations"""
        mean = np.mean(arr)
        std = np.std(arr)
        return arr[np.abs(arr - mean) <= n_std * std]


class ResultsWriter:
    """Write results to files"""
    
    @staticmethod
    def write_csv(data: Dict[str, list], filepath: str, delimiter: str = ',') -> None:
        """Write dictionary to CSV"""
        if not data:
            return
        
        with open(filepath, 'w') as f:
            # Write header
            header = delimiter.join(data.keys())
            f.write(header + '\n')
            
            # Write rows
            n_rows = len(next(iter(data.values())))
            for i in range(n_rows):
                row = [str(data[key][i]) for key in data.keys()]
                f.write(delimiter.join(row) + '\n')
    
    @staticmethod
    def write_results(results: Dict[str, Any], filepath: str) -> None:
        """Write results dictionary to text file"""
        with open(filepath, 'w') as f:
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")


class Logger:
    """Simple logging utility"""
    
    def __init__(self, name: str = "QuantumWalk", verbose: bool = True):
        self.name = name
        self.verbose = verbose
    
    def info(self, message: str) -> None:
        """Log info message"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """Log error message"""
        if self.verbose:
            print(f"[ERROR] {message}")
    
    def success(self, message: str) -> None:
        """Log success message"""
        if self.verbose:
            print(f"[SUCCESS] {message}")


def print_banner(title: str, width: int = 70) -> None:
    """Print formatted banner"""
    print("\n" + "=" * width)
    print(f" {title.center(width-2)}")
    print("=" * width + "\n")


def print_section(title: str, width: int = 70) -> None:
    """Print section header"""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def format_table(headers: list, rows: list, width: int = 70) -> str:
    """Format data as table"""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    
    result = ""
    # Header
    result += " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + "\n"
    result += "-" * (sum(col_widths) + len(headers) * 3 - 1) + "\n"
    
    # Rows
    for row in rows:
        result += " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) + "\n"
    
    return result