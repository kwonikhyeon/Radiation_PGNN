# ──────────────────────────────────────────────────────────────
# test/modules/traversability_layer.py - Traversability Layer Calculation Module
# ──────────────────────────────────────────────────────────────
"""
Traversability layer calculation module for radiation field analysis.
Computes traversability scores to assess path feasibility and navigation
safety based on radiation field predictions and source proximity.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.ndimage import gaussian_filter, distance_transform_edt
try:
    from .source_detection import detect_radiation_sources, SourceDetectionParameters
except ImportError:
    # For standalone mode
    from source_detection import detect_radiation_sources, SourceDetectionParameters

__all__ = [
    "TraversabilityCalculator", "calculate_traversability_layer", "TraversabilityParameters"
]


class TraversabilityParameters:
    """Parameters for traversability layer calculation."""
    
    def __init__(
        self,
        # Source proximity parameters
        proximity_weight: float = 0.6,              # Weight for source proximity component
        proximity_decay: float = 0.05,              # Exponential decay rate for proximity effect
        max_proximity_distance: float = 50.0,       # Maximum distance for proximity calculation
        
        # Source detection parameters (will be passed to SourceDetector)
        source_detection_params: Optional[SourceDetectionParameters] = None,
        
        # Traversability enhancement parameters
        low_radiation_bonus: float = 0.3,           # Bonus for low radiation areas
        path_smoothing_sigma: float = 2.0,          # Smoothing for traversability paths
        max_traversability_value: float = 1.0,      # Maximum traversability value
        
        # Robot navigation parameters  
        robot_directional_weight: float = 0.4,      # Weight for robot directional navigation (reduced)
        path_spread_sigma: float = 10.0,             # Gaussian spread around shortest paths (pixels)
        max_source_distance: float = 100.0,         # Maximum distance to consider sources
        distance_decay_rate: float = 0.02,          # Decay rate for distant sources
        
        # Fan-shaped navigation parameters
        base_fan_angle: float = np.pi / 6,           # Base fan angle in radians (30 degrees)
        max_fan_angle: float = np.pi / 2,            # Maximum fan angle in radians (90 degrees)
        min_fan_angle: float = np.pi / 8,            # Minimum fan angle in radians (22.5 degrees)
        fan_size_scaling: float = 2.0,              # How much source size affects fan angle
        fan_distance_decay: float = 0.015,          # Distance decay rate within fan
        source_boost_sigma: float = 8.0,            # Source boost area size
        
        # Future extension parameters (placeholders)
        obstacle_penalty: float = 0.0,              # Penalty for obstacles (future use)
        terrain_difficulty: float = 0.0,            # Terrain difficulty factor (future use)
        energy_efficiency: float = 0.0              # Energy efficiency consideration (future use)
    ):
        # Source proximity parameters
        self.proximity_weight = proximity_weight
        self.proximity_decay = proximity_decay
        self.max_proximity_distance = max_proximity_distance
        self.source_detection_params = source_detection_params
        
        # Traversability enhancement parameters
        self.low_radiation_bonus = low_radiation_bonus
        self.path_smoothing_sigma = path_smoothing_sigma
        self.max_traversability_value = max_traversability_value
        
        # Robot navigation parameters
        self.robot_directional_weight = robot_directional_weight
        self.path_spread_sigma = path_spread_sigma
        self.max_source_distance = max_source_distance
        self.distance_decay_rate = distance_decay_rate
        
        # Fan-shaped navigation parameters
        self.base_fan_angle = base_fan_angle
        self.max_fan_angle = max_fan_angle
        self.min_fan_angle = min_fan_angle
        self.fan_size_scaling = fan_size_scaling
        self.fan_distance_decay = fan_distance_decay
        self.source_boost_sigma = source_boost_sigma
        
        # Future extension parameters
        self.obstacle_penalty = obstacle_penalty
        self.terrain_difficulty = terrain_difficulty
        self.energy_efficiency = energy_efficiency


class TraversabilityCalculator:
    """Traversability layer calculator for radiation field navigation."""
    
    def __init__(self, parameters: Optional[TraversabilityParameters] = None):
        self.params = parameters or TraversabilityParameters()
    
    def detect_radiation_sources(self, radiation_field: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """
        Detect potential radiation source locations using dedicated source detection module.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            
        Returns:
            tuple containing:
                - source_locations: List of (y, x) coordinates of detected sources
                - detection_metadata: Metadata from source detection
        """
        # Use dedicated source detection module
        detection_result = detect_radiation_sources(
            radiation_field, 
            parameters=self.params.source_detection_params
        )
        
        source_locations = detection_result['source_locations']
        detection_metadata = detection_result['metadata']
        
        print(f"🎯 Detected {len(source_locations)} radiation sources using SourceDetector")
        
        return source_locations, detection_metadata
    
    def calculate_source_proximity_map(self, radiation_field: np.ndarray, 
                                     source_locations: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate proximity map showing distance to nearest radiation source.
        Closer to sources = higher traversability (for source investigation).
        
        Args:
            radiation_field: (H, W) radiation field for shape reference
            source_locations: List of (y, x) source coordinates
            
        Returns:
            proximity_map: (H, W) proximity-based traversability values
        """
        if not source_locations:
            # No sources detected - uniform low proximity
            return np.zeros_like(radiation_field, dtype=np.float32)
        
        H, W = radiation_field.shape
        y_coords, x_coords = np.ogrid[:H, :W]
        
        # Calculate distance to nearest source for each pixel
        min_distance_map = np.full((H, W), np.inf, dtype=np.float32)
        
        for source_y, source_x in source_locations:
            # Calculate Euclidean distance to this source
            distance_to_source = np.sqrt((y_coords - source_y)**2 + (x_coords - source_x)**2)
            
            # Keep minimum distance to any source
            min_distance_map = np.minimum(min_distance_map, distance_to_source)
        
        # Convert distance to proximity (closer = higher value)
        # Apply exponential decay: exp(-decay_rate * distance)
        proximity_map = np.exp(-self.params.proximity_decay * min_distance_map)
        
        # Limit maximum distance effect
        distance_mask = min_distance_map <= self.params.max_proximity_distance
        proximity_map = proximity_map * distance_mask
        
        # Normalize to [0, 1] range
        if proximity_map.max() > 0:
            proximity_map = proximity_map / proximity_map.max()
        
        print(f"📏 Proximity map calculated: max distance effect = {self.params.max_proximity_distance} pixels")
        return proximity_map
    
    def calculate_low_radiation_bonus(self, radiation_field: np.ndarray) -> np.ndarray:
        """
        Calculate traversability bonus for low radiation areas.
        Lower radiation = higher traversability for safety.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            
        Returns:
            low_radiation_bonus: (H, W) bonus values for low radiation areas
        """
        # Inverse relationship: low radiation = high traversability bonus
        bonus_map = 1.0 - np.clip(radiation_field, 0, 1)
        
        # Apply bonus scaling
        bonus_map = bonus_map * self.params.low_radiation_bonus
        
        return bonus_map
    
    def calculate_line_distance(self, point_y: np.ndarray, point_x: np.ndarray, 
                               line_start_y: float, line_start_x: float,
                               line_end_y: float, line_end_x: float) -> np.ndarray:
        """
        Calculate distance from points to a line segment.
        
        Args:
            point_y, point_x: Arrays of point coordinates
            line_start_y, line_start_x: Line start coordinates
            line_end_y, line_end_x: Line end coordinates
            
        Returns:
            distances: Array of distances from points to line
        """
        # Vector from line start to end
        line_vec_y = line_end_y - line_start_y
        line_vec_x = line_end_x - line_start_x
        line_length = np.sqrt(line_vec_y**2 + line_vec_x**2)
        
        if line_length == 0:
            # Degenerate case: start and end are the same
            return np.sqrt((point_y - line_start_y)**2 + (point_x - line_start_x)**2)
        
        # Normalize line vector
        line_unit_y = line_vec_y / line_length
        line_unit_x = line_vec_x / line_length
        
        # Vector from line start to points
        point_vec_y = point_y - line_start_y
        point_vec_x = point_x - line_start_x
        
        # Project point vectors onto line vector
        projection = point_vec_y * line_unit_y + point_vec_x * line_unit_x
        
        # Clamp projection to line segment
        projection = np.clip(projection, 0, line_length)
        
        # Find closest point on line segment
        closest_y = line_start_y + projection * line_unit_y
        closest_x = line_start_x + projection * line_unit_x
        
        # Calculate distance from points to closest point on line
        distances = np.sqrt((point_y - closest_y)**2 + (point_x - closest_x)**2)
        
        return distances
    
    def estimate_source_size(self, radiation_field: np.ndarray, 
                           source_position: Tuple[int, int]) -> float:
        """
        Estimate the effective size of a radiation source based on its spread.
        
        Args:
            radiation_field: (H, W) radiation field
            source_position: (y, x) source center position
            
        Returns:
            estimated_size: Estimated radius of the source in pixels
        """
        source_y, source_x = source_position
        H, W = radiation_field.shape
        
        # Get radiation value at source center
        center_intensity = radiation_field[source_y, source_x]
        
        # Define intensity thresholds for size estimation
        high_threshold = center_intensity * 0.7  # 70% of center intensity
        medium_threshold = center_intensity * 0.5  # 50% of center intensity
        low_threshold = center_intensity * 0.3   # 30% of center intensity
        
        # Search in expanding circles to find where intensity drops
        max_radius = min(30, min(H, W) // 4)  # Reasonable maximum radius
        
        high_radius = 0
        medium_radius = 0
        low_radius = 0
        
        for radius in range(1, max_radius + 1):
            # Sample points on the circle
            angles = np.linspace(0, 2*np.pi, max(8, radius * 2), endpoint=False)
            circle_y = source_y + radius * np.sin(angles)
            circle_x = source_x + radius * np.cos(angles)
            
            # Keep points within bounds
            valid_mask = ((circle_y >= 0) & (circle_y < H) & 
                         (circle_x >= 0) & (circle_x < W))
            
            if not valid_mask.any():
                break
                
            # Sample intensities at circle points
            circle_y_int = np.round(circle_y[valid_mask]).astype(int)
            circle_x_int = np.round(circle_x[valid_mask]).astype(int)
            circle_intensities = radiation_field[circle_y_int, circle_x_int]
            
            mean_intensity = circle_intensities.mean()
            
            # Check threshold crossings
            if mean_intensity >= high_threshold and high_radius == 0:
                high_radius = radius
            if mean_intensity >= medium_threshold and medium_radius == 0:
                medium_radius = radius
            if mean_intensity >= low_threshold and low_radius == 0:
                low_radius = radius
        
        # Use medium threshold radius as primary size indicator
        # Fall back to other thresholds if medium is not found
        if medium_radius > 0:
            estimated_size = medium_radius
        elif high_radius > 0:
            estimated_size = high_radius * 1.5  # Extrapolate
        elif low_radius > 0:
            estimated_size = low_radius * 0.7   # Conservative estimate
        else:
            estimated_size = 5.0  # Default minimum size
        
        # Clamp to reasonable range
        estimated_size = np.clip(estimated_size, 3.0, 25.0)
        
        return estimated_size
    
    def calculate_adaptive_fan_angle(self, source_size: float) -> float:
        """
        Calculate fan angle based on source size.
        
        Args:
            source_size: Estimated source radius in pixels
            
        Returns:
            fan_angle: Adaptive fan angle in radians
        """
        # Normalize source size (typical range: 3-25 pixels)
        size_normalized = (source_size - 3.0) / (25.0 - 3.0)
        size_normalized = np.clip(size_normalized, 0.0, 1.0)
        
        # Calculate adaptive angle: larger sources get wider fans
        # Use exponential scaling for more pronounced effect
        size_factor = (size_normalized ** (1.0 / self.params.fan_size_scaling))
        
        adaptive_angle = (self.params.min_fan_angle + 
                         size_factor * (self.params.max_fan_angle - self.params.min_fan_angle))
        
        return adaptive_angle
    
    def calculate_source_path_weight(self, radiation_field: np.ndarray,
                                   robot_position: Tuple[int, int],
                                   source_position: Tuple[int, int]) -> np.ndarray:
        """
        Calculate adaptive fan-shaped weight map for path to a single source.
        Fan angle adapts to the estimated size of the source.
        
        Args:
            radiation_field: (H, W) radiation field 
            robot_position: (y, x) robot position
            source_position: (y, x) source position
            
        Returns:
            weight_map: (H, W) weight values for this source path
        """
        H, W = radiation_field.shape
        robot_y, robot_x = robot_position
        source_y, source_x = source_position
        
        # Estimate source size and calculate adaptive fan angle
        source_size = self.estimate_source_size(radiation_field, source_position)
        adaptive_fan_angle = self.calculate_adaptive_fan_angle(source_size)
        
        print(f"🎯 Source at ({source_y}, {source_x}): size={source_size:.1f}px, fan_angle={adaptive_fan_angle*180/np.pi:.1f}°")
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:H, :W]
        
        # Calculate direction from robot to source
        direction_y = source_y - robot_y
        direction_x = source_x - robot_x
        target_angle = np.arctan2(direction_y, direction_x)
        
        # Calculate distance from robot to source
        source_distance = np.sqrt(direction_y**2 + direction_x**2)
        
        # Calculate angles from robot to all pixels
        pixel_y = y_coords - robot_y
        pixel_x = x_coords - robot_x
        pixel_angles = np.arctan2(pixel_y, pixel_x)
        
        # Calculate distance from robot to all pixels
        pixel_distances = np.sqrt(pixel_y**2 + pixel_x**2)
        
        # Calculate angular difference from target direction
        angle_diff = pixel_angles - target_angle
        # Normalize angle difference to [-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Create fan-shaped sector using adaptive angle
        fan_half_angle = adaptive_fan_angle / 2.0  # Half angle of the adaptive fan
        within_fan = np.abs(angle_diff) <= fan_half_angle
        
        # Calculate angular weight (higher at center, lower at edges)
        # Use smoother transition for larger fans
        angular_weight = np.cos(angle_diff / fan_half_angle * np.pi / 2) ** 2
        angular_weight = np.where(within_fan, angular_weight, 0.0)
        
        # Calculate distance weight (decays with distance from robot)
        distance_weight = np.exp(-pixel_distances * self.params.fan_distance_decay)
        
        # Boost weight near the source itself - scale boost with source size
        source_boost_sigma = max(self.params.source_boost_sigma, source_size * 0.8)
        source_boost = np.exp(-((y_coords - source_y)**2 + (x_coords - source_x)**2) / 
                             (2 * source_boost_sigma**2))
        
        # Combine all weights
        path_weights = angular_weight * distance_weight + source_boost * 0.3
        
        return path_weights
    
    def calculate_directional_navigation_map(self, radiation_field: np.ndarray,
                                           source_locations: List[Tuple[int, int]],
                                           robot_position: Tuple[int, int],
                                           robot_heading: float) -> np.ndarray:
        """
        Calculate directional navigation map with weighted paths to all sources.
        Creates weighted shortest paths to all sources, with closer sources having higher weights.
        
        Args:
            radiation_field: (H, W) radiation field for shape reference
            source_locations: List of (y, x) source coordinates
            robot_position: (y, x) current robot position
            robot_heading: Robot heading in radians (not used in this version)
            
        Returns:
            directional_map: (H, W) directional navigation values
        """
        if not source_locations:
            # No sources detected - provide exploration incentive
            print("🔍 No sources detected, providing exploration weights")
            return self.calculate_exploration_weights(radiation_field, robot_position)
        
        H, W = radiation_field.shape
        robot_y, robot_x = robot_position
        
        # Initialize combined weight map
        combined_weights = np.zeros((H, W), dtype=np.float32)
        
        # Calculate weights for each source
        for source_y, source_x in source_locations:
            # Calculate distance from robot to this source
            source_distance = np.sqrt((robot_y - source_y)**2 + (robot_x - source_x)**2)
            
            # Skip sources that are too far away
            if source_distance > self.params.max_source_distance:
                continue
            
            # Calculate distance-based weight for this source
            # Closer sources get higher weights
            distance_weight = np.exp(-source_distance * self.params.distance_decay_rate)
            
            # Calculate path weights for this source
            source_path_weights = self.calculate_source_path_weight(
                radiation_field, robot_position, (source_y, source_x)
            )
            
            # Apply distance-based weighting
            weighted_source_map = source_path_weights * distance_weight
            
            # Add to combined weights
            combined_weights += weighted_source_map
            
            print(f"📍 Added path to source at ({source_y}, {source_x}), "
                  f"distance: {source_distance:.1f}, weight: {distance_weight:.3f}")
        
        # Add small baseline to prevent pure zero areas (helps visibility)
        baseline = 0.05  # Reduced from 0.1 to 0.05
        combined_weights = combined_weights + baseline
        
        # Normalize to [0, 1] range
        if combined_weights.max() > 0:
            combined_weights = combined_weights / combined_weights.max()
        
        print(f"🗺️ Multi-source navigation map created: robot at ({robot_y}, {robot_x}), "
              f"{len(source_locations)} sources processed")
        
        return combined_weights
    
    def calculate_exploration_weights(self, radiation_field: np.ndarray,
                                    robot_position: Tuple[int, int]) -> np.ndarray:
        """
        Calculate exploration weights when no sources are detected.
        Provides incentive to explore areas away from robot position.
        
        Args:
            radiation_field: (H, W) radiation field for shape reference
            robot_position: (y, x) robot position
            
        Returns:
            exploration_map: (H, W) exploration weight values
        """
        H, W = radiation_field.shape
        robot_y, robot_x = robot_position
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:H, :W]
        
        # Calculate distance from robot
        distances = np.sqrt((y_coords - robot_y)**2 + (x_coords - robot_x)**2)
        
        # Exploration weight: moderate distance from robot is preferred
        # Too close = already explored, too far = inefficient
        optimal_exploration_distance = min(H, W) * 0.25  # Reduced from 30% to 25%
        
        # Gaussian around optimal exploration distance
        exploration_weights = np.exp(-((distances - optimal_exploration_distance)**2) / 
                                   (2 * (optimal_exploration_distance * 0.4)**2))  # Narrower spread
        
        # Reduce base exploration intensity to prevent saturation
        exploration_weights = exploration_weights * 0.4  # Scale down overall intensity (reduced from 0.6)
        
        # Add smaller randomness for exploration diversity
        noise = np.random.normal(0, 0.05, (H, W))  # Reduced noise
        exploration_weights += noise
        
        # Normalize to [0, 1] range
        exploration_weights = np.clip(exploration_weights, 0, 1)
        if exploration_weights.max() > 0:
            exploration_weights = exploration_weights / exploration_weights.max()
        
        print(f"🔍 Exploration weights generated: optimal distance {optimal_exploration_distance:.1f} pixels")
        
        return exploration_weights
    
    def calculate_traversability_layer(self, radiation_field: np.ndarray,
                                     measurement_mask: Optional[np.ndarray] = None,
                                     robot_position: Optional[Tuple[int, int]] = None,
                                     robot_heading: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive traversability layer for path planning with robot navigation.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            measurement_mask: (H, W) optional mask of measurement locations
            robot_position: (y, x) optional current robot position
            robot_heading: optional robot heading in radians (0 = positive x direction)
            
        Returns:
            dict containing:
                - 'total_traversability': (H, W) combined traversability layer
                - 'source_proximity': (H, W) source proximity component
                - 'low_radiation_bonus': (H, W) low radiation bonus component
                - 'directional_navigation': (H, W) directional navigation component (if robot info provided)
                - 'source_locations': List of detected source coordinates
                - 'metadata': dict with calculation parameters
        """
        # Detect radiation sources from prediction using dedicated module
        source_locations, detection_metadata = self.detect_radiation_sources(radiation_field)
        
        # Calculate source proximity component
        source_proximity = self.calculate_source_proximity_map(radiation_field, source_locations)
        
        # Calculate low radiation bonus
        low_radiation_bonus = self.calculate_low_radiation_bonus(radiation_field)
        
        # Calculate directional navigation component if robot info is provided
        directional_navigation = None
        if robot_position is not None and robot_heading is not None:
            directional_navigation = self.calculate_directional_navigation_map(
                radiation_field, source_locations, robot_position, robot_heading
            )
        
        # Combine components with weights using weighted average approach to prevent saturation
        if directional_navigation is not None:
            # Use weighted average instead of weighted sum to ensure total <= 1.0
            # Normalize weights so they sum to 1.0
            w_dir = self.params.robot_directional_weight
            w_prox = (1 - w_dir) * self.params.proximity_weight
            w_low_rad = (1 - w_dir) * (1 - self.params.proximity_weight)
            
            # Ensure weights sum to 1.0
            total_weight = w_dir + w_prox + w_low_rad
            w_dir /= total_weight
            w_prox /= total_weight  
            w_low_rad /= total_weight
            
            total_traversability = (
                w_prox * source_proximity + 
                w_low_rad * low_radiation_bonus +
                w_dir * directional_navigation
            )
        else:
            # Original calculation without directional navigation (already normalized)
            total_traversability = (self.params.proximity_weight * source_proximity + 
                                  (1 - self.params.proximity_weight) * low_radiation_bonus)
        
        # Future extension point: Add more components here
        # Example placeholders:
        # if self.params.obstacle_penalty > 0:
        #     obstacle_map = self.calculate_obstacle_penalty(...)
        #     total_traversability -= self.params.obstacle_penalty * obstacle_map
        #
        # if self.params.terrain_difficulty > 0:
        #     terrain_map = self.calculate_terrain_difficulty(...)
        #     total_traversability *= (1 - self.params.terrain_difficulty * terrain_map)
        
        # Apply smoothing for realistic path planning
        if self.params.path_smoothing_sigma > 0:
            total_traversability = gaussian_filter(total_traversability, 
                                                 sigma=self.params.path_smoothing_sigma)
        
        # Normalize to final range
        total_traversability = np.clip(total_traversability, 0, self.params.max_traversability_value)
        
        # Final normalization to [0, 1]
        if total_traversability.max() > 0:
            total_traversability = total_traversability / total_traversability.max()
        
        # Prepare metadata including source detection details
        metadata = {
            'parameters': {
                'proximity_weight': self.params.proximity_weight,
                'proximity_decay': self.params.proximity_decay,
                'max_proximity_distance': self.params.max_proximity_distance,
                'low_radiation_bonus': self.params.low_radiation_bonus,
                'robot_directional_weight': self.params.robot_directional_weight,
                'path_spread_sigma': self.params.path_spread_sigma,
                'max_source_distance': self.params.max_source_distance,
                'distance_decay_rate': self.params.distance_decay_rate
            },
            'robot_info': {
                'position': robot_position,
                'heading_rad': robot_heading,
                'directional_navigation_enabled': directional_navigation is not None
            },
            'statistics': {
                'num_sources_detected': len(source_locations),
                'max_traversability': float(total_traversability.max()),
                'mean_traversability': float(total_traversability.mean()),
                'high_traversability_area': float(np.sum(total_traversability > 0.7) / total_traversability.size),
                'low_traversability_area': float(np.sum(total_traversability < 0.3) / total_traversability.size)
            },
            'source_info': {
                'detected_sources': len(source_locations),
                'source_coordinates': source_locations
            },
            'source_detection': detection_metadata  # Include detailed source detection metadata
        }
        
        result = {
            'total_traversability': total_traversability,
            'source_proximity': source_proximity,
            'low_radiation_bonus': low_radiation_bonus,
            'source_locations': source_locations,
            'metadata': metadata
        }
        
        # Add directional navigation component if calculated
        if directional_navigation is not None:
            result['directional_navigation'] = directional_navigation
        
        return result


# Convenience functions for standalone use
def calculate_traversability_layer(radiation_field: np.ndarray,
                                 measurement_mask: Optional[np.ndarray] = None,
                                 robot_position: Optional[Tuple[int, int]] = None,
                                 robot_heading: Optional[float] = None,
                                 parameters: Optional[TraversabilityParameters] = None) -> Dict[str, np.ndarray]:
    """
    Standalone function to calculate traversability layer.
    
    Args:
        radiation_field: (H, W) predicted radiation field
        measurement_mask: (H, W) optional mask of measurement locations
        robot_position: (y, x) optional current robot position
        robot_heading: optional robot heading in radians (0 = positive x direction)
        parameters: optional traversability calculation parameters
        
    Returns:
        dict containing traversability components and metadata
    """
    calculator = TraversabilityCalculator(parameters)
    return calculator.calculate_traversability_layer(radiation_field, measurement_mask, robot_position, robot_heading)


if __name__ == "__main__":
    # Test the module
    print("Testing Traversability Layer Calculation...")
    
    # Create dummy radiation field with multiple sources
    field = np.zeros((256, 256), dtype=np.float32)
    
    # Add multiple Gaussian sources
    sources = [(80, 80), (180, 120), (120, 200)]
    for y, x in sources:
        Y, X = np.ogrid[:256, :256]
        source_field = 0.8 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 15**2))
        field += source_field
    
    # Clip to [0, 1] range
    field = np.clip(field, 0, 1)
    
    # Create dummy measurement mask
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    for i in range(10):
        y, x = np.random.randint(50, 206, 2)
        measurement_mask[y, x] = 1
    
    # Calculate traversability layer
    result = calculate_traversability_layer(field, measurement_mask)
    
    print(f"Traversability calculation completed:")
    print(f"  Total traversability range: [{result['total_traversability'].min():.4f}, {result['total_traversability'].max():.4f}]")
    print(f"  Mean traversability: {result['metadata']['statistics']['mean_traversability']:.4f}")
    print(f"  Sources detected: {result['metadata']['statistics']['num_sources_detected']}")
    print(f"  High traversability area: {result['metadata']['statistics']['high_traversability_area']*100:.1f}%")
    print(f"  Low traversability area: {result['metadata']['statistics']['low_traversability_area']*100:.1f}%")
    print(f"  Source locations: {result['source_locations']}")
    print("Traversability layer calculation test complete!")