"""
src/quantum_walk.py
Quantum walk implementation for option pricing
"""

import numpy as np
from typing import Tuple


class QuantumWalkSimulator:
    """Discrete-time quantum walk simulator"""
    
    def __init__(self, num_qubits: int = 12, num_steps: int = 20):
        """
        Initialize quantum walk simulator
        
        Args:
            num_qubits: Number of position qubits (default: 12 for 4096 positions)
            num_steps: Number of walk steps (default: 20 is optimal)
        """
        self.num_qubits = num_qubits
        self.num_steps = num_steps
        self.num_positions = 2 ** num_qubits
    
    def coin_operator(self, theta: float = np.pi/4) -> np.ndarray:
        """
        Generate coin operator as rotation matrix
        
        Args:
            theta: Rotation angle (pi/4 for unbiased walk)
            
        Returns:
            2x2 unitary coin matrix
        """
        return np.array([
            [np.cos(theta), 1j*np.sin(theta)],
            [1j*np.sin(theta), np.cos(theta)]
        ], dtype=complex)
    
    def get_probabilities(self, theta: float = np.pi/4) -> Tuple[float, float]:
        """
        Get right and left movement probabilities from coin
        
        Args:
            theta: Coin rotation angle
            
        Returns:
            Tuple of (p_right, p_left)
        """
        p_right = np.sin(theta) ** 2
        p_left = np.cos(theta) ** 2
        return float(p_right), float(p_left)
    
    def generate_distribution(self, theta: float = np.pi/4) -> np.ndarray:
        """
        Generate probability distribution from quantum walk
        
        Implements:
        1. Initial uniform superposition
        2. Coin rotation at each step
        3. Shift operator with cyclic boundaries
        4. Renormalization
        
        Args:
            theta: Coin angle (pi/4 for unbiased)
            
        Returns:
            Probability distribution over all positions
        """
        probs = np.ones(self.num_positions, dtype=np.float64) / self.num_positions
        p_right = np.sin(theta) ** 2
        p_left = np.cos(theta) ** 2
        
        for step in range(self.num_steps):
            new_probs = np.zeros(self.num_positions, dtype=np.float64)
            
            for pos in range(self.num_positions):
                # Cyclic boundaries
                right_pos = (pos + 1) % self.num_positions
                left_pos = (pos - 1) % self.num_positions
                
                new_probs[right_pos] += probs[pos] * p_right
                new_probs[left_pos] += probs[pos] * p_left
            
            total = np.sum(new_probs)
            if total > 0:
                probs = new_probs / total
        
        return probs / np.sum(probs)
    
    def get_statistics(self, theta: float = np.pi/4) -> dict:
        """
        Compute statistics of the walk distribution
        
        Args:
            theta: Coin angle
            
        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        probs = self.generate_distribution(theta)
        positions = np.arange(self.num_positions)
        
        mean = np.sum(positions * probs)
        variance = np.sum((positions - mean)**2 * probs)
        std = np.sqrt(variance)
        
        skewness = np.sum((positions - mean)**3 * probs) / (std**3) if std > 0 else 0
        kurtosis = np.sum((positions - mean)**4 * probs) / (std**4) if std > 0 else 0
        
        return {
            "mean": float(mean),
            "variance": float(variance),
            "std": float(std),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "min": float(np.min(positions[probs > 0])),
            "max": float(np.max(positions[probs > 0])),
        }
    
    def sample(self, n_samples: int, theta: float = np.pi/4) -> np.ndarray:
        """
        Generate samples from quantum walk distribution
        
        Args:
            n_samples: Number of samples
            theta: Coin angle
            
        Returns:
            Array of position samples
        """
        probs = self.generate_distribution(theta)
        samples = np.random.choice(self.num_positions, size=n_samples, p=probs, replace=True)
        return samples