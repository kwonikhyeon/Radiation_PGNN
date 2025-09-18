# ──────────────────────────────────────────────────────────────
# test/modules/safety_layer.py - Safety Layer Calculation Module
# ──────────────────────────────────────────────────────────────
"""
Safety layer calculation module for radiation field analysis.
Computes safety scores based on radiation field predictions to assess
potential safety levels for path planning and navigation.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import gaussian_filter

__all__ = [
    "SafetyCalculator", "calculate_safety_layer", "SafetyParameters"
]


class SafetyParameters:
    """Parameters for safety layer calculation."""
    
    def __init__(
        self,
        danger_threshold: float = 0.3,          # Radiation level above which safety decreases
        safety_threshold: float = 0.1,          # Radiation level below which safety is high
        safety_exponent: float = 2.0,           # Exponential scaling factor for safety
        smoothing_sigma: float = 1.0,           # Gaussian smoothing for safety field
        max_safety_value: float = 1.0,          # Maximum safety value (normalization)
        cumulative_exposure: bool = True,       # Consider cumulative exposure effects
        distance_decay: bool = True,            # Apply distance-based safety calculation
        gradient_penalty: float = 0.1           # Penalty for high gradient areas (uncertainty)
    ):
        self.danger_threshold = danger_threshold
        self.safety_threshold = safety_threshold
        self.safety_exponent = safety_exponent
        self.smoothing_sigma = smoothing_sigma
        self.max_safety_value = max_safety_value
        self.cumulative_exposure = cumulative_exposure
        self.distance_decay = distance_decay
        self.gradient_penalty = gradient_penalty


class SafetyCalculator:
    """Safety layer calculator for radiation field predictions."""
    
    def __init__(self, parameters: Optional[SafetyParameters] = None):
        self.params = parameters or SafetyParameters()
    
    def calculate_basic_safety(self, radiation_field: np.ndarray) -> np.ndarray:
        """
        Calculate basic safety from radiation field using inverse relationship.
        High radiation = Low safety, Low radiation = High safety
        
        Args:
            radiation_field: (H, W) predicted radiation field [0, 1]
            
        Returns:
            safety_field: (H, W) safety values [0, 1]
        """
        # Ensure values are in [0, 1] range
        field = np.clip(radiation_field, 0, 1)
        
        # Inverse relationship: safety = 1 - radiation
        # Higher radiation (closer to 1) means lower safety (closer to 0)
        # Lower radiation (closer to 0) means higher safety (closer to 1)
        safety = 1.0 - field
        
        return np.clip(safety, 0, self.params.max_safety_value)
    
    def calculate_gradient_penalty(self, radiation_field: np.ndarray) -> np.ndarray:
        """
        Calculate safety penalty based on field gradients (uncertainty regions).
        Higher gradients mean lower safety due to uncertainty.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            
        Returns:
            gradient_penalty: (H, W) penalty values [0, 1]
        """
        # Calculate gradients
        grad_y, grad_x = np.gradient(radiation_field)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Normalize gradient magnitude
        if gradient_magnitude.max() > 0:
            gradient_magnitude /= gradient_magnitude.max()
        
        # Apply penalty scaling
        penalty = gradient_magnitude * self.params.gradient_penalty
        
        return penalty
    
    def calculate_cumulative_exposure_safety(self, radiation_field: np.ndarray) -> np.ndarray:
        """
        Calculate safety considering cumulative exposure effects.
        Uses convolution to model exposure accumulation over time/movement.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            
        Returns:
            cumulative_safety: (H, W) cumulative exposure safety
        """
        if not self.params.cumulative_exposure:
            return np.zeros_like(radiation_field)
        
        # Model cumulative exposure using Gaussian kernel
        # Simulates radiation exposure over time in nearby areas
        kernel_size = max(3, int(self.params.smoothing_sigma * 3))
        cumulative_field = gaussian_filter(radiation_field, sigma=self.params.smoothing_sigma)
        
        # Convert cumulative exposure to safety
        cumulative_safety = self.calculate_basic_safety(cumulative_field) * 0.3  # Weighted contribution
        
        return cumulative_safety
    
    def calculate_distance_based_safety(self, radiation_field: np.ndarray, 
                                      measurement_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate safety with distance-based enhancement from low radiation areas.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            measurement_mask: (H, W) optional mask of measurement locations
            
        Returns:
            distance_safety: (H, W) distance-based safety
        """
        if not self.params.distance_decay:
            return np.zeros_like(radiation_field)
        
        # Find high radiation areas (above danger threshold)
        high_rad_mask = radiation_field > self.params.danger_threshold
        
        if not np.any(high_rad_mask):
            return np.ones_like(radiation_field) * 0.2  # Default safety bonus
        
        # Calculate distance from high radiation areas
        from scipy.ndimage import distance_transform_edt
        distance_from_danger = distance_transform_edt(~high_rad_mask)
        
        # Normalize distance (farther = higher safety)
        if distance_from_danger.max() > 0:
            distance_safety = distance_from_danger / distance_from_danger.max()
            distance_safety = distance_safety * 0.2  # Weighted contribution
        else:
            distance_safety = np.zeros_like(radiation_field)
        
        return distance_safety
    
    def calculate_safety_layer(self, radiation_field: np.ndarray, 
                             measurement_mask: Optional[np.ndarray] = None,
                             uncertainty_field: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive safety layer from radiation field prediction.
        
        Args:
            radiation_field: (H, W) predicted radiation field [0, 1]
            measurement_mask: (H, W) optional mask of measurement locations
            uncertainty_field: (H, W) optional uncertainty field
            
        Returns:
            dict containing:
                - 'total_safety': (H, W) combined safety field
                - 'basic_safety': (H, W) basic radiation safety
                - 'gradient_penalty': (H, W) gradient-based penalty
                - 'cumulative_safety': (H, W) cumulative exposure safety
                - 'distance_safety': (H, W) distance-based safety
                - 'metadata': dict with calculation parameters
        """
        # Calculate individual safety components
        basic_safety = self.calculate_basic_safety(radiation_field)
        gradient_penalty = self.calculate_gradient_penalty(radiation_field)
        cumulative_safety = self.calculate_cumulative_exposure_safety(radiation_field)
        distance_safety = self.calculate_distance_based_safety(radiation_field, measurement_mask)
        
        # Add uncertainty penalty if available (reduces safety)
        uncertainty_penalty = np.zeros_like(radiation_field)
        if uncertainty_field is not None:
            uncertainty_penalty = uncertainty_field * 0.2  # Weighted contribution
        
        # Combine all safety components (penalties reduce safety)
        total_safety = (basic_safety + 
                       cumulative_safety + 
                       distance_safety - 
                       gradient_penalty - 
                       uncertainty_penalty)
        
        # Apply smoothing to total safety
        if self.params.smoothing_sigma > 0:
            total_safety = gaussian_filter(total_safety, sigma=self.params.smoothing_sigma)
        
        # Normalize to [0, 1] range
        total_safety = np.clip(total_safety, 0, None)
        if total_safety.max() > 0:
            total_safety = total_safety / total_safety.max() * self.params.max_safety_value
        
        # Prepare metadata
        metadata = {
            'parameters': {
                'danger_threshold': self.params.danger_threshold,
                'safety_threshold': self.params.safety_threshold,
                'safety_exponent': self.params.safety_exponent,
                'smoothing_sigma': self.params.smoothing_sigma
            },
            'statistics': {
                'max_safety': float(total_safety.max()),
                'mean_safety': float(total_safety.mean()),
                'high_safety_area': float(np.sum(total_safety > 0.7) / total_safety.size),
                'low_safety_area': float(np.sum(total_safety < 0.3) / total_safety.size)
            }
        }
        
        return {
            'total_safety': total_safety,
            'basic_safety': basic_safety,
            'gradient_penalty': gradient_penalty,
            'cumulative_safety': cumulative_safety,
            'distance_safety': distance_safety,
            'uncertainty_penalty': uncertainty_penalty,
            'metadata': metadata
        }


# Convenience functions for standalone use
def calculate_safety_layer(radiation_field: np.ndarray, 
                          measurement_mask: Optional[np.ndarray] = None,
                          uncertainty_field: Optional[np.ndarray] = None,
                          parameters: Optional[SafetyParameters] = None) -> Dict[str, np.ndarray]:
    """
    Standalone function to calculate safety layer from radiation field.
    
    Args:
        radiation_field: (H, W) predicted radiation field [0, 1]
        measurement_mask: (H, W) optional mask of measurement locations
        uncertainty_field: (H, W) optional uncertainty field
        parameters: optional safety calculation parameters
        
    Returns:
        dict containing safety components and metadata
    """
    calculator = SafetyCalculator(parameters)
    return calculator.calculate_safety_layer(radiation_field, measurement_mask, uncertainty_field)


if __name__ == "__main__":
    # Test the module
    print("Testing Safety Layer Calculation...")
    
    # Create dummy radiation field
    field = np.random.exponential(0.3, (256, 256)).astype(np.float32)
    field = np.clip(field, 0, 1)
    
    # Create dummy measurement mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    for i in range(20):
        y, x = np.random.randint(0, 256, 2)
        mask[y, x] = 1
    
    # Calculate safety layer
    result = calculate_safety_layer(field, mask)
    
    print(f"Safety calculation completed:")
    print(f"  Total safety range: [{result['total_safety'].min():.4f}, {result['total_safety'].max():.4f}]")
    print(f"  Mean safety: {result['metadata']['statistics']['mean_safety']:.4f}")
    print(f"  High safety area: {result['metadata']['statistics']['high_safety_area']*100:.1f}%")
    print(f"  Low safety area: {result['metadata']['statistics']['low_safety_area']*100:.1f}%")
    print("Safety layer calculation test complete!")