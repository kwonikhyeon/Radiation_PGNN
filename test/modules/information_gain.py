# ──────────────────────────────────────────────────────────────
# test/modules/information_gain.py - Information Gain Layer Calculation Module
# ──────────────────────────────────────────────────────────────
"""
Information gain calculation module for radiation field analysis.
Computes uncertainty and information gain values to guide optimal measurement
point selection for path planning and exploration.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from scipy.ndimage import distance_transform_edt, gaussian_filter

__all__ = [
    "UncertaintyCalculator", "calculate_uncertainty", "InformationGainParameters", "calculate_information_gain_layer"
]


class InformationGainParameters:
    """Parameters for information gain and uncertainty calculation."""
    
    def __init__(
        self,
        # Uncertainty parameters
        distance_weight: float = 0.8,           # Weight for distance-based uncertainty (increased)
        distance_sigma: float = 20.0,           # Distance scaling factor (reduced for shorter range)
        smoothing_sigma: float = 1.0,           # Gaussian smoothing for final maps
        max_uncertainty_value: float = 1.0,     # Maximum uncertainty value (normalization)
        exploration_bonus: float = 0.3,         # Bonus for unexplored areas
        measurement_decay: float = 0.8,         # Decay factor for measurement influence
        # Gradient parameters
        gradient_threshold: float = 0.2,        # Threshold for selecting gradient prediction
        gradient_weight: float = 0.6,           # Weight for gradient component in final gain (increased for priority)
        uncertainty_weight: float = 0.5,        # Weight for uncertainty component in final gain (decreased)
        gradient_sigma: float = 1.5,            # Gaussian sigma for gradient smoothing
        # Measurement exclusion parameters
        measurement_penalty: float = 0.9,       # Penalty factor for measured points (0-1)
        exclusion_radius: int = 3,              # Radius around measurements to reduce gain
        exclusion_strength: float = 0.7         # Strength of exclusion zone penalty (0-1)
    ):
        # Uncertainty parameters
        self.distance_weight = distance_weight
        self.distance_sigma = distance_sigma
        self.smoothing_sigma = smoothing_sigma
        self.max_uncertainty_value = max_uncertainty_value
        self.exploration_bonus = exploration_bonus
        self.measurement_decay = measurement_decay
        # Gradient parameters  
        self.gradient_threshold = gradient_threshold
        self.gradient_weight = gradient_weight
        self.uncertainty_weight = uncertainty_weight
        self.gradient_sigma = gradient_sigma
        # Measurement exclusion parameters
        self.measurement_penalty = measurement_penalty
        self.exclusion_radius = exclusion_radius
        self.exclusion_strength = exclusion_strength


class UncertaintyCalculator:
    """Uncertainty and information gain calculator for radiation field predictions."""
    
    def __init__(self, parameters: Optional[InformationGainParameters] = None):
        self.params = parameters or InformationGainParameters()
    
    def calculate_model_uncertainty_from_predictions(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate model uncertainty from multiple predictions provided by the app.
        
        Args:
            predictions: (n_samples, H, W) array of multiple model predictions
            
        Returns:
            tuple containing:
                - mean_prediction: (H, W) mean prediction across samples
                - uncertainty_map: (H, W) prediction uncertainty (std deviation)
        """
        if len(predictions.shape) != 3:
            raise ValueError("Predictions should be (n_samples, H, W) shaped array")
        
        # Calculate mean and standard deviation across samples
        mean_prediction = predictions.mean(axis=0)
        uncertainty_map = predictions.std(axis=0)
        
        return mean_prediction, uncertainty_map
    
    def calculate_distance_uncertainty(self, measurement_mask: np.ndarray) -> np.ndarray:
        """
        Calculate distance-based uncertainty from measurement points.
        Areas farther from measurements have higher uncertainty.
        
        Args:
            measurement_mask: (H, W) binary mask of measurement locations
            
        Returns:
            distance_uncertainty: (H, W) distance-based uncertainty map
        """
        # Calculate distance from measurement points
        if np.any(measurement_mask > 0):
            # Distance transform from measurement points
            distance_map = distance_transform_edt(measurement_mask == 0).astype(np.float32)
            
            # Normalize distance
            if distance_map.max() > 0:
                distance_map = distance_map / distance_map.max()
        else:
            # No measurements - uniform high uncertainty
            distance_map = np.ones_like(measurement_mask, dtype=np.float32)
        
        # Apply exponential weighting
        # Higher distance = higher uncertainty
        weight = np.exp(distance_map * self.params.distance_sigma / 100.0)
        
        # Normalize to [0, 1] range
        if weight.max() > weight.min():
            weight = (weight - weight.min()) / (weight.max() - weight.min())
        
        return weight
    
    def select_widest_range_prediction(self, predictions: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Select the prediction with widest range above threshold from multiple predictions.
        
        Args:
            predictions: (n_samples, H, W) array of multiple model predictions
            threshold: minimum pixel value threshold (uses params.gradient_threshold if None)
            
        Returns:
            selected_prediction: (H, W) prediction with widest range above threshold
        """
        if threshold is None:
            threshold = self.params.gradient_threshold
        
        max_coverage = -1
        selected_prediction = predictions[0]  # Default to first prediction
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            # Count pixels above threshold
            above_threshold = np.sum(pred > threshold)
            
            if above_threshold > max_coverage:
                max_coverage = above_threshold
                selected_prediction = pred
        
        print(f"📊 Selected prediction with {max_coverage} pixels above {threshold:.3f}")
        return selected_prediction
    
    def calculate_gradient_information(self, prediction: np.ndarray) -> np.ndarray:
        """
        Calculate gradient-based information gain from a prediction.
        High gradient areas contain more information for source boundary detection.
        
        Args:
            prediction: (H, W) single prediction field
            
        Returns:
            gradient_info: (H, W) gradient-based information map
        """
        # Calculate spatial gradients
        grad_y, grad_x = np.gradient(prediction)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Apply smoothing to reduce noise
        if self.params.gradient_sigma > 0:
            gradient_magnitude = gaussian_filter(gradient_magnitude, sigma=self.params.gradient_sigma)
        
        # Normalize to [0, 1] range
        if gradient_magnitude.max() > gradient_magnitude.min():
            gradient_info = (gradient_magnitude - gradient_magnitude.min()) / \
                          (gradient_magnitude.max() - gradient_magnitude.min())
        else:
            gradient_info = np.zeros_like(gradient_magnitude)
        
        return gradient_info
    
    def create_measurement_exclusion_mask(self, measurement_mask: np.ndarray) -> np.ndarray:
        """
        Create exclusion mask around measurement points to prevent redundant measurements.
        
        Args:
            measurement_mask: (H, W) binary mask of measurement locations
            
        Returns:
            exclusion_mask: (H, W) penalty mask [0, 1] where 1 = full penalty
        """
        from scipy.ndimage import binary_dilation
        
        if not np.any(measurement_mask > 0):
            return np.zeros_like(measurement_mask, dtype=np.float32)
        
        # Create exclusion zone around measurement points
        if self.params.exclusion_radius > 0:
            # Create circular exclusion zones
            exclusion_zone = binary_dilation(
                measurement_mask > 0, 
                iterations=self.params.exclusion_radius
            ).astype(np.float32)
            
            # Apply different penalties: direct measurements vs exclusion zones
            exclusion_mask = np.zeros_like(measurement_mask, dtype=np.float32)
            
            # Direct measurement points get full penalty
            exclusion_mask[measurement_mask > 0] = self.params.measurement_penalty
            
            # Exclusion zones get partial penalty
            exclusion_only = exclusion_zone - (measurement_mask > 0).astype(np.float32)
            exclusion_mask[exclusion_only > 0] = self.params.exclusion_strength
            
        else:
            # Only penalize direct measurement points
            exclusion_mask = (measurement_mask > 0).astype(np.float32) * self.params.measurement_penalty
        
        return exclusion_mask
    
    def calculate_uncertainty(self, predictions: Optional[np.ndarray] = None, 
                            measurement_mask: Optional[np.ndarray] = None,
                            single_prediction: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive uncertainty map combining model and distance uncertainties.
        
        Args:
            predictions: (n_samples, H, W) multiple model predictions for uncertainty calculation
            measurement_mask: (H, W) binary mask of measurement locations  
            single_prediction: (H, W) optional single prediction (fallback if no multiple predictions)
            
        Returns:
            dict containing:
                - 'total_uncertainty': (H, W) combined uncertainty map
                - 'model_uncertainty': (H, W) model prediction uncertainty
                - 'distance_uncertainty': (H, W) distance-based uncertainty
                - 'mean_prediction': (H, W) mean model prediction
                - 'metadata': dict with calculation parameters
        """
        # Initialize uncertainty components (determine size from inputs)
        if predictions is not None:
            grid_size = predictions.shape[-1]  # Assume square grid
        elif single_prediction is not None:
            grid_size = single_prediction.shape[-1]
        else:
            grid_size = 256  # Default size
        
        model_uncertainty = np.zeros((grid_size, grid_size), dtype=np.float32)
        distance_uncertainty = np.zeros((grid_size, grid_size), dtype=np.float32)
        mean_prediction = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Calculate model uncertainty from multiple predictions if available
        if predictions is not None and len(predictions.shape) == 3:
            try:
                mean_prediction, model_uncertainty = self.calculate_model_uncertainty_from_predictions(predictions)
                print(f"✅ Model uncertainty calculated from {predictions.shape[0]} predictions")
            except Exception as e:
                print(f"❌ Model uncertainty calculation failed: {e}")
                model_uncertainty = np.zeros_like(mean_prediction)
        elif single_prediction is not None:
            # Use provided single prediction
            mean_prediction = single_prediction.copy()
            print(f"📊 Using single prediction, no model uncertainty available")
        
        # Calculate distance-based uncertainty
        if measurement_mask is not None:
            distance_uncertainty = self.calculate_distance_uncertainty(measurement_mask)
            print(f"✅ Distance uncertainty calculated from {measurement_mask.sum()} measurements")
        else:
            # No measurements - uniform high uncertainty
            distance_uncertainty = np.ones_like(model_uncertainty)
            print(f"⚠️  No measurement mask provided, using uniform uncertainty")
        
        # Normalize model uncertainty to [0, 1]
        if model_uncertainty.max() > model_uncertainty.min():
            model_uncertainty_norm = (model_uncertainty - model_uncertainty.min()) / \
                                   (model_uncertainty.max() - model_uncertainty.min())
        else:
            model_uncertainty_norm = model_uncertainty
        
        # Combine uncertainties using weighted sum
        total_uncertainty = ((1 - self.params.distance_weight) * model_uncertainty_norm + 
                           self.params.distance_weight * distance_uncertainty)
        
        # Apply smoothing if specified
        if self.params.smoothing_sigma > 0:
            total_uncertainty = gaussian_filter(total_uncertainty, sigma=self.params.smoothing_sigma)
        
        # Normalize to final range
        total_uncertainty = np.clip(total_uncertainty, 0, self.params.max_uncertainty_value)
        
        # Prepare metadata
        metadata = {
            'parameters': {
                'distance_weight': self.params.distance_weight,
                'distance_sigma': self.params.distance_sigma,
                'smoothing_sigma': self.params.smoothing_sigma
            },
            'statistics': {
                'max_uncertainty': float(total_uncertainty.max()),
                'mean_uncertainty': float(total_uncertainty.mean()),
                'high_uncertainty_area': float(np.sum(total_uncertainty > 0.7) / total_uncertainty.size),
                'low_uncertainty_area': float(np.sum(total_uncertainty < 0.3) / total_uncertainty.size)
            }
        }
        
        return {
            'total_uncertainty': total_uncertainty,
            'model_uncertainty': model_uncertainty,
            'distance_uncertainty': distance_uncertainty,
            'mean_prediction': mean_prediction,
            'metadata': metadata
        }
    
    def calculate_information_gain(self, uncertainty_map: np.ndarray, 
                                 measurement_mask: np.ndarray,
                                 current_position: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate information gain map for exploration guidance.
        Higher values indicate areas where new measurements would be most valuable.
        
        Args:
            uncertainty_map: (H, W) uncertainty values
            measurement_mask: (H, W) binary mask of existing measurements
            current_position: (y, x) current robot position for proximity weighting
            
        Returns:
            dict containing:
                - 'information_gain': (H, W) information gain map
                - 'exploration_bonus': (H, W) bonus for unexplored areas
                - 'accessibility_weight': (H, W) accessibility weighting
                - 'metadata': dict with calculation info
        """
        # Start with uncertainty as base information gain
        information_gain = uncertainty_map.copy()
        
        # Add exploration bonus for areas far from existing measurements
        exploration_bonus = np.zeros_like(uncertainty_map)
        if np.any(measurement_mask > 0):
            # Distance from any measurement point
            measurement_distance = distance_transform_edt(measurement_mask == 0)
            if measurement_distance.max() > 0:
                exploration_bonus = (measurement_distance / measurement_distance.max()) * self.params.exploration_bonus
        else:
            exploration_bonus = np.ones_like(uncertainty_map) * self.params.exploration_bonus
        
        # Calculate accessibility weight (closer to current position = higher weight)
        accessibility_weight = np.ones_like(uncertainty_map)
        if current_position is not None:
            y_pos, x_pos = current_position
            y_coords, x_coords = np.ogrid[:uncertainty_map.shape[0], :uncertainty_map.shape[1]]
            position_distance = np.sqrt((y_coords - y_pos)**2 + (x_coords - x_pos)**2)
            
            # Exponential decay with distance
            if position_distance.max() > 0:
                accessibility_weight = np.exp(-position_distance / (uncertainty_map.shape[0] * 0.3))
        
        # Combine information gain components
        total_information_gain = (information_gain + 
                                exploration_bonus) * accessibility_weight
        
        # Reduce information gain at existing measurement points
        measurement_penalty = measurement_mask * self.params.measurement_decay
        total_information_gain = total_information_gain * (1 - measurement_penalty)
        
        # Apply smoothing
        if self.params.smoothing_sigma > 0:
            total_information_gain = gaussian_filter(total_information_gain, 
                                                   sigma=self.params.smoothing_sigma)
        
        # Normalize to [0, 1]
        if total_information_gain.max() > 0:
            total_information_gain = total_information_gain / total_information_gain.max()
        
        metadata = {
            'max_gain': float(total_information_gain.max()),
            'mean_gain': float(total_information_gain.mean()),
            'high_gain_area': float(np.sum(total_information_gain > 0.7) / total_information_gain.size)
        }
        
        return {
            'information_gain': total_information_gain,
            'exploration_bonus': exploration_bonus,
            'accessibility_weight': accessibility_weight,
            'metadata': metadata
        }


# Convenience functions for standalone use
def calculate_uncertainty(predictions: Optional[np.ndarray] = None,
                        measurement_mask: Optional[np.ndarray] = None, 
                        single_prediction: Optional[np.ndarray] = None,
                        parameters: Optional[InformationGainParameters] = None) -> Dict[str, np.ndarray]:
    """
    Standalone function to calculate uncertainty map.
    
    Args:
        predictions: (n_samples, H, W) multiple model predictions for uncertainty calculation
        measurement_mask: (H, W) binary mask of measurement locations
        single_prediction: (H, W) single prediction as fallback
        parameters: optional uncertainty calculation parameters
        
    Returns:
        dict containing uncertainty components and metadata
    """
    calculator = UncertaintyCalculator(parameters)
    return calculator.calculate_uncertainty(predictions, measurement_mask, single_prediction)


def calculate_information_gain_layer(predictions: Optional[np.ndarray] = None,
                                   measurement_mask: Optional[np.ndarray] = None,
                                   single_prediction: Optional[np.ndarray] = None,
                                   parameters: Optional[InformationGainParameters] = None) -> Dict[str, np.ndarray]:
    """
    Calculate advanced information gain layer combining uncertainty and gradient information.
    
    Args:
        predictions: (n_samples, H, W) multiple model predictions for uncertainty calculation
        measurement_mask: (H, W) binary mask of measurement locations
        single_prediction: (H, W) single prediction as fallback
        parameters: optional calculation parameters
        
    Returns:
        dict containing:
            - 'information_gain': (H, W) combined information gain layer
            - 'uncertainty_component': (H, W) uncertainty component
            - 'gradient_component': (H, W) gradient component  
            - 'metadata': dict with calculation parameters
    """
    # Initialize calculator
    calculator = UncertaintyCalculator(parameters)
    params = parameters or InformationGainParameters()
    
    # Calculate uncertainty component
    uncertainty_result = calculator.calculate_uncertainty(
        predictions=predictions,
        measurement_mask=measurement_mask, 
        single_prediction=single_prediction
    )
    uncertainty_component = uncertainty_result['total_uncertainty']
    
    # Calculate gradient component from widest range prediction
    gradient_component = np.zeros_like(uncertainty_component)
    selected_prediction = None
    
    if predictions is not None and len(predictions.shape) == 3:
        # Select prediction with widest range above threshold
        selected_prediction = calculator.select_widest_range_prediction(predictions)
        gradient_component = calculator.calculate_gradient_information(selected_prediction)
        print(f"✅ Gradient information calculated from selected prediction")
    elif single_prediction is not None:
        # Use single prediction for gradient
        selected_prediction = single_prediction
        gradient_component = calculator.calculate_gradient_information(single_prediction)
        print(f"📊 Gradient information calculated from single prediction")
    else:
        print(f"⚠️  No predictions available, skipping gradient calculation")
    
    # Combine uncertainty and gradient components
    information_gain = (params.uncertainty_weight * uncertainty_component + 
                       params.gradient_weight * gradient_component)
    
    # Apply measurement exclusion to prevent redundant measurements
    exclusion_mask = np.zeros_like(information_gain)
    if measurement_mask is not None:
        exclusion_mask = calculator.create_measurement_exclusion_mask(measurement_mask)
        # Apply exclusion penalty: information_gain * (1 - exclusion_mask)
        information_gain = information_gain * (1 - exclusion_mask)
        
        excluded_points = np.sum(exclusion_mask > 0)
        print(f"🚫 Applied exclusion to {excluded_points} points (measurements + zones)")
    
    # Normalize final result to [0, 1]
    if information_gain.max() > 0:
        information_gain = information_gain / information_gain.max()
    
    # Prepare result with same structure as safety layer
    metadata = {
        'parameters': uncertainty_result['metadata']['parameters'],
        'statistics': {
            'max_information_gain': float(information_gain.max()),
            'mean_information_gain': float(information_gain.mean()),
            'high_gain_area': float(np.sum(information_gain > 0.7) / information_gain.size),
            'low_gain_area': float(np.sum(information_gain < 0.3) / information_gain.size)
        }
    }
    
    # Update metadata with gradient and exclusion information
    metadata.update({
        'gradient_info': {
            'uncertainty_weight': params.uncertainty_weight,
            'gradient_weight': params.gradient_weight,
            'gradient_threshold': params.gradient_threshold,
            'selected_prediction_available': selected_prediction is not None
        },
        'exclusion_info': {
            'measurement_penalty': params.measurement_penalty,
            'exclusion_radius': params.exclusion_radius,
            'exclusion_strength': params.exclusion_strength,
            'excluded_points': int(np.sum(exclusion_mask > 0)) if measurement_mask is not None else 0
        }
    })
    
    return {
        'information_gain': information_gain,
        'uncertainty_component': uncertainty_component,
        'gradient_component': gradient_component,
        'exclusion_mask': exclusion_mask,
        'total_uncertainty': uncertainty_result['total_uncertainty'],
        'model_uncertainty': uncertainty_result['model_uncertainty'], 
        'distance_uncertainty': uncertainty_result['distance_uncertainty'],
        'metadata': metadata
    }


if __name__ == "__main__":
    # Test the module
    print("Testing Information Gain Calculation...")
    
    # Create dummy data
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some random measurement points
    for _ in range(15):
        y, x = np.random.randint(50, 206, 2)
        measurement_mask[y, x] = 1
    
    # Create dummy single prediction and multiple predictions
    single_prediction = np.random.exponential(0.3, (256, 256)).astype(np.float32)
    single_prediction = np.clip(single_prediction, 0, 1)
    
    # Create multiple predictions for uncertainty calculation
    n_predictions = 8
    predictions = np.stack([
        np.clip(np.random.exponential(0.3, (256, 256)), 0, 1) 
        for _ in range(n_predictions)
    ]).astype(np.float32)
    
    # Calculate advanced information gain layer
    result = calculate_information_gain_layer(
        predictions=predictions,
        measurement_mask=measurement_mask
    )
    
    print(f"Advanced Information Gain calculation completed:")
    print(f"  Information Gain range: [{result['information_gain'].min():.4f}, {result['information_gain'].max():.4f}]")
    print(f"  Mean Information Gain: {result['metadata']['statistics']['mean_information_gain']:.4f}")
    print(f"  High gain area: {result['metadata']['statistics']['high_gain_area']*100:.1f}%")
    print(f"  Low gain area: {result['metadata']['statistics']['low_gain_area']*100:.1f}%")
    print(f"  Uncertainty weight: {result['metadata']['gradient_info']['uncertainty_weight']}")
    print(f"  Gradient weight: {result['metadata']['gradient_info']['gradient_weight']}")
    print(f"  Excluded points: {result['metadata']['exclusion_info']['excluded_points']}")
    print(f"  Exclusion radius: {result['metadata']['exclusion_info']['exclusion_radius']}")
    
    # Test exclusion effectiveness
    measurement_points = np.where(measurement_mask > 0)
    if len(measurement_points[0]) > 0:
        gain_at_measurements = result['information_gain'][measurement_points]
        print(f"  Info gain at measurement points: [{gain_at_measurements.min():.4f}, {gain_at_measurements.max():.4f}]")
    
    print("Advanced information gain calculation test complete!")