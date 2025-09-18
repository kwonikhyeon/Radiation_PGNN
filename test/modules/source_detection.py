# ──────────────────────────────────────────────────────────────
# test/modules/source_detection.py - Radiation Source Detection Module
# ──────────────────────────────────────────────────────────────
"""
Radiation source detection module for identifying and localizing
radiation sources from predicted radiation fields. Provides robust
source detection with configurable parameters for different scenarios.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.ndimage import maximum_filter, binary_dilation
from scipy.signal import find_peaks

__all__ = [
    "SourceDetector", "detect_radiation_sources", "SourceDetectionParameters"
]


class SourceDetectionParameters:
    """Parameters for radiation source detection."""
    
    def __init__(
        self,
        # Basic detection parameters
        detection_threshold: float = 0.3,           # Minimum radiation level for source detection
        min_source_area: int = 9,                   # Minimum area (pixels) for a valid source
        max_sources: int = 10,                      # Maximum number of sources to detect
        
        # Peak detection parameters
        peak_prominence: float = 0.1,               # Minimum peak prominence for local maxima
        peak_distance: int = 15,                    # Minimum distance between peaks (pixels)
        
        # Filtering and validation parameters
        gaussian_sigma: float = 1.0,                # Pre-smoothing sigma for noise reduction
        merge_distance: int = 20,                   # Distance threshold for merging nearby sources
        intensity_weight: float = 0.7,             # Weight for intensity in source scoring
        
        # Advanced detection parameters (for future enhancement)
        use_watershed: bool = False,                # Use watershed algorithm for separation
        adaptive_threshold: bool = False,           # Use adaptive thresholding
        confidence_threshold: float = 0.5,          # Minimum confidence for source validation
        
        # Circular pattern analysis parameters
        use_circular_analysis: bool = True,         # Enable circular pattern analysis
        circular_radii: List[int] = None,           # Radii to test for circular patterns (default: [5,8,12,15])
        circularity_weight: float = 0.4,           # Weight for circular pattern score
        intensity_weight_enhanced: float = 0.6      # Enhanced weight for peak intensity
    ):
        # Basic parameters
        self.detection_threshold = detection_threshold
        self.min_source_area = min_source_area
        self.max_sources = max_sources
        
        # Peak detection parameters
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
        
        # Filtering parameters
        self.gaussian_sigma = gaussian_sigma
        self.merge_distance = merge_distance
        self.intensity_weight = intensity_weight
        
        # Advanced parameters
        self.use_watershed = use_watershed
        self.adaptive_threshold = adaptive_threshold
        self.confidence_threshold = confidence_threshold
        
        # Circular analysis parameters
        self.use_circular_analysis = use_circular_analysis
        self.circular_radii = circular_radii or [5, 8, 12, 15]
        self.circularity_weight = circularity_weight
        self.intensity_weight_enhanced = intensity_weight_enhanced


class SourceDetector:
    """Radiation source detection and localization."""
    
    def __init__(self, parameters: Optional[SourceDetectionParameters] = None):
        self.params = parameters or SourceDetectionParameters()
    
    def preprocess_field(self, radiation_field: np.ndarray) -> np.ndarray:
        """
        Preprocess radiation field for better source detection.
        
        Args:
            radiation_field: (H, W) raw radiation field
            
        Returns:
            processed_field: (H, W) processed radiation field
        """
        # Apply Gaussian smoothing to reduce noise
        if self.params.gaussian_sigma > 0:
            processed_field = gaussian_filter(radiation_field, sigma=self.params.gaussian_sigma)
        else:
            processed_field = radiation_field.copy()
        
        # Ensure values are in [0, 1] range
        processed_field = np.clip(processed_field, 0, 1)
        
        return processed_field
    
    def find_local_maxima(self, radiation_field: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Find local maxima in radiation field as potential source candidates.
        
        Args:
            radiation_field: (H, W) processed radiation field
            
        Returns:
            maxima: List of (y, x, intensity) tuples for local maxima
        """
        # Apply maximum filter to find local maxima
        local_maxima = maximum_filter(radiation_field, size=self.params.peak_distance) == radiation_field
        
        # Apply intensity threshold
        intensity_mask = radiation_field > self.params.detection_threshold
        
        # Combine conditions
        candidate_mask = local_maxima & intensity_mask
        
        # Get coordinates and intensities
        y_coords, x_coords = np.where(candidate_mask)
        intensities = radiation_field[y_coords, x_coords]
        
        # Create list of candidates
        candidates = [(int(y), int(x), float(intensity)) 
                     for y, x, intensity in zip(y_coords, x_coords, intensities)]
        
        # Sort by intensity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🔍 Found {len(candidates)} local maxima candidates")
        return candidates
    
    def calculate_circularity_score(self, radiation_field: np.ndarray, center_y: int, center_x: int) -> float:
        """
        Calculate circularity score for a potential source center.
        Higher score indicates better circular radiation pattern.
        
        Args:
            radiation_field: (H, W) radiation field
            center_y, center_x: Potential source center coordinates
            
        Returns:
            circularity_score: Score [0, 1] indicating circular pattern strength
        """
        H, W = radiation_field.shape
        
        # Ensure center is within bounds with margin
        if (center_y < max(self.params.circular_radii) or center_y >= H - max(self.params.circular_radii) or
            center_x < max(self.params.circular_radii) or center_x >= W - max(self.params.circular_radii)):
            return 0.0
        
        center_intensity = radiation_field[center_y, center_x]
        if center_intensity <= self.params.detection_threshold:
            return 0.0
        
        circularity_scores = []
        
        # Test multiple radii for circular patterns
        for radius in self.params.circular_radii:
            if radius <= 0:
                continue
                
            # Generate circle points at this radius
            circle_points = self._generate_circle_points(center_y, center_x, radius)
            
            # Filter points within bounds
            valid_points = [(y, x) for y, x in circle_points 
                           if 0 <= y < H and 0 <= x < W]
            
            if len(valid_points) < 8:  # Need minimum points for reliable analysis
                continue
            
            # Get intensities at circle points
            circle_intensities = [radiation_field[y, x] for y, x in valid_points]
            
            if not circle_intensities:
                continue
            
            # Calculate circular pattern metrics
            mean_circle_intensity = np.mean(circle_intensities)
            std_circle_intensity = np.std(circle_intensities)
            
            # Score based on: center > circle average, low variation around circle
            intensity_ratio = mean_circle_intensity / (center_intensity + 1e-8)
            uniformity_score = 1.0 / (1.0 + std_circle_intensity)
            
            # Good circular source: center >> circle, uniform around circle
            radius_score = (1.0 - intensity_ratio) * uniformity_score
            circularity_scores.append(max(0.0, radius_score))
        
        # Return best circularity score across all radii
        return max(circularity_scores) if circularity_scores else 0.0
    
    def _generate_circle_points(self, center_y: int, center_x: int, radius: int) -> List[Tuple[int, int]]:
        """
        Generate points on a circle using Bresenham's circle algorithm.
        
        Args:
            center_y, center_x: Circle center
            radius: Circle radius
            
        Returns:
            circle_points: List of (y, x) coordinates on circle
        """
        points = []
        x = 0
        y = radius
        d = 3 - 2 * radius
        
        while y >= x:
            # Add 8 symmetric points
            for dx, dy in [(x, y), (-x, y), (x, -y), (-x, -y), 
                          (y, x), (-y, x), (y, -x), (-y, -x)]:
                points.append((center_y + dy, center_x + dx))
            
            x += 1
            
            if d > 0:
                y -= 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
        
        return points
    
    def calculate_local_peak_score(self, radiation_field: np.ndarray, y: int, x: int, window_size: int = 5) -> float:
        """
        Calculate how much a point stands out as a local peak.
        
        Args:
            radiation_field: (H, W) radiation field
            y, x: Point coordinates  
            window_size: Size of local window for comparison
            
        Returns:
            peak_score: Score [0, 1] indicating peak strength
        """
        H, W = radiation_field.shape
        half_window = window_size // 2
        
        # Define local window bounds
        y_start = max(0, y - half_window)
        y_end = min(H, y + half_window + 1)
        x_start = max(0, x - half_window)
        x_end = min(W, x + half_window + 1)
        
        # Extract local window
        local_window = radiation_field[y_start:y_end, x_start:x_end]
        center_value = radiation_field[y, x]
        
        if local_window.size <= 1:
            return 0.0
        
        # Calculate how much center exceeds local average
        local_mean = local_window.mean()
        local_max = local_window.max()
        
        if local_max <= local_mean:
            return 0.0
        
        # Peak score: how much center exceeds local mean, normalized by local range
        peak_score = (center_value - local_mean) / (local_max - local_mean + 1e-8)
        
        return max(0.0, min(1.0, peak_score))
    
    def score_source_candidates(self, candidates: List[Tuple[int, int, float]], 
                               radiation_field: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """
        Score source candidates using intensity and circularity analysis.
        
        Args:
            candidates: List of (y, x, intensity) candidates
            radiation_field: (H, W) radiation field
            
        Returns:
            scored_candidates: List of (y, x, intensity, composite_score) tuples
        """
        scored = []
        
        for y, x, intensity in candidates:
            # Calculate intensity score (normalized)
            intensity_score = min(1.0, intensity / 1.0)  # Assume max intensity is 1.0
            
            # Calculate peak score
            peak_score = self.calculate_local_peak_score(radiation_field, y, x)
            
            # Calculate circularity score if enabled
            if self.params.use_circular_analysis:
                circularity_score = self.calculate_circularity_score(radiation_field, y, x)
            else:
                circularity_score = 0.0
            
            # Composite score combining intensity, peak, and circularity
            composite_score = (self.params.intensity_weight_enhanced * intensity_score + 
                             (1 - self.params.intensity_weight_enhanced - self.params.circularity_weight) * peak_score +
                             self.params.circularity_weight * circularity_score)
            
            scored.append((y, x, intensity, composite_score))
        
        # Sort by composite score (highest first)
        scored.sort(key=lambda x: x[3], reverse=True)
        
        print(f"🎯 Scored {len(scored)} candidates with enhanced circular analysis")
        return scored
    
    def merge_nearby_sources(self, candidates: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Merge nearby source candidates to avoid duplicate detections.
        
        Args:
            candidates: List of (y, x, intensity) candidate sources
            
        Returns:
            merged_sources: List of merged source locations
        """
        if not candidates:
            return []
        
        merged = []
        used = set()
        
        for i, (y1, x1, intensity1) in enumerate(candidates):
            if i in used:
                continue
                
            # Find all nearby candidates
            nearby_group = [(y1, x1, intensity1)]
            used.add(i)
            
            for j, (y2, x2, intensity2) in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                    
                distance = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                
                if distance <= self.params.merge_distance:
                    nearby_group.append((y2, x2, intensity2))
                    used.add(j)
            
            # Merge nearby candidates using weighted average
            if len(nearby_group) == 1:
                merged.append(nearby_group[0])
            else:
                # Calculate weighted center based on intensity
                total_weight = sum(intensity for _, _, intensity in nearby_group)
                
                weighted_y = sum(y * intensity for y, x, intensity in nearby_group) / total_weight
                weighted_x = sum(x * intensity for y, x, intensity in nearby_group) / total_weight
                max_intensity = max(intensity for _, _, intensity in nearby_group)
                
                # Round to integer coordinates
                merged_y = int(round(weighted_y))
                merged_x = int(round(weighted_x))
                
                merged.append((merged_y, merged_x, max_intensity))
                
                print(f"📍 Merged {len(nearby_group)} nearby candidates into source at ({merged_y}, {merged_x})")
        
        return merged
    
    def validate_sources(self, sources: List[Tuple[int, int, float]], 
                        radiation_field: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Validate detected sources and filter out false positives.
        
        Args:
            sources: List of candidate source locations
            radiation_field: Original radiation field for validation
            
        Returns:
            validated_sources: List of validated source locations
        """
        validated = []
        
        for y, x, intensity in sources:
            # Ensure coordinates are within bounds
            H, W = radiation_field.shape
            if not (0 <= y < H and 0 <= x < W):
                continue
            
            # Check if intensity is still above threshold at actual location
            actual_intensity = radiation_field[y, x]
            if actual_intensity < self.params.detection_threshold:
                continue
            
            # Calculate source confidence based on local intensity distribution
            # Check 3x3 neighborhood around source
            y_start, y_end = max(0, y-1), min(H, y+2)
            x_start, x_end = max(0, x-1), min(W, x+2)
            
            neighborhood = radiation_field[y_start:y_end, x_start:x_end]
            
            # Source should be the maximum in its neighborhood
            if radiation_field[y, x] == neighborhood.max():
                confidence = actual_intensity  # Simple confidence based on intensity
                
                if confidence >= self.params.confidence_threshold:
                    validated.append((y, x, actual_intensity))
        
        print(f"✅ Validated {len(validated)} sources from {len(sources)} candidates")
        return validated
    
    def detect_sources(self, radiation_field: np.ndarray) -> Dict[str, Any]:
        """
        Main source detection pipeline with enhanced circular pattern analysis.
        
        Args:
            radiation_field: (H, W) predicted radiation field
            
        Returns:
            detection_result: Dict containing detected sources and metadata
        """
        # Preprocess field
        processed_field = self.preprocess_field(radiation_field)
        
        # Find local maxima candidates
        candidates = self.find_local_maxima(processed_field)
        
        # Limit number of candidates to process
        if len(candidates) > self.params.max_sources * 3:  # Process 3x max for better selection
            candidates = candidates[:self.params.max_sources * 3]
        
        # Enhanced scoring with circular pattern analysis
        if self.params.use_circular_analysis and candidates:
            print(f"🔄 Applying enhanced circular pattern analysis...")
            scored_candidates = self.score_source_candidates(candidates, processed_field)
            
            # Convert back to (y, x, intensity) format for existing pipeline
            candidates = [(y, x, intensity) for y, x, intensity, score in scored_candidates]
            print(f"🎯 Re-ranked candidates using circular analysis and peak scoring")
        
        # Merge nearby sources
        merged_sources = self.merge_nearby_sources(candidates)
        
        # Validate sources
        validated_sources = self.validate_sources(merged_sources, processed_field)
        
        # Limit to maximum number of sources
        if len(validated_sources) > self.params.max_sources:
            validated_sources = validated_sources[:self.params.max_sources]
        
        # Extract coordinates for compatibility
        source_locations = [(y, x) for y, x, _ in validated_sources]
        source_intensities = [intensity for _, _, intensity in validated_sources]
        
        # Prepare metadata
        metadata = {
            'parameters': {
                'detection_threshold': self.params.detection_threshold,
                'peak_distance': self.params.peak_distance,
                'merge_distance': self.params.merge_distance,
                'max_sources': self.params.max_sources
            },
            'detection_stats': {
                'candidates_found': len(candidates),
                'after_merging': len(merged_sources),
                'final_validated': len(validated_sources),
                'detection_success': len(validated_sources) > 0
            },
            'source_details': {
                'locations': source_locations,
                'intensities': source_intensities,
                'mean_intensity': float(np.mean(source_intensities)) if source_intensities else 0.0,
                'max_intensity': float(np.max(source_intensities)) if source_intensities else 0.0
            }
        }
        
        return {
            'source_locations': source_locations,
            'source_intensities': source_intensities,
            'processed_field': processed_field,
            'metadata': metadata
        }


# Convenience function for standalone use
def detect_radiation_sources(radiation_field: np.ndarray,
                           parameters: Optional[SourceDetectionParameters] = None) -> Dict[str, Any]:
    """
    Standalone function to detect radiation sources.
    
    Args:
        radiation_field: (H, W) predicted radiation field
        parameters: optional detection parameters
        
    Returns:
        detection_result: Dict containing detected sources and metadata
    """
    detector = SourceDetector(parameters)
    return detector.detect_sources(radiation_field)


if __name__ == "__main__":
    # Test the module
    print("Testing Source Detection Module...")
    
    # Create test field with known sources
    field = np.zeros((256, 256), dtype=np.float32)
    
    # Add well-separated sources
    true_sources = [(80, 80), (180, 120), (120, 200)]
    
    Y, X = np.ogrid[:256, :256]
    
    for y, x in true_sources:
        # Create Gaussian source
        source_field = 0.8 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 15**2))
        field += source_field
    
    # Add some noise
    field += np.random.normal(0, 0.02, field.shape)
    field = np.clip(field, 0, 1)
    
    print(f"📊 Test field range: [{field.min():.3f}, {field.max():.3f}]")
    print(f"🎯 True source locations: {true_sources}")
    
    # Test source detection
    result = detect_radiation_sources(field)
    
    detected_sources = result['source_locations']
    print(f"\n🔍 Detection Results:")
    print(f"  Detected sources: {detected_sources}")
    print(f"  Detection accuracy: {len(detected_sources)}/{len(true_sources)}")
    print(f"  Source intensities: {[f'{i:.3f}' for i in result['source_intensities']]}")
    
    # Calculate detection accuracy
    if detected_sources and true_sources:
        distances = []
        for true_y, true_x in true_sources:
            min_dist = min(np.sqrt((true_y - det_y)**2 + (true_x - det_x)**2) 
                          for det_y, det_x in detected_sources)
            distances.append(min_dist)
        
        print(f"  Average detection error: {np.mean(distances):.1f} pixels")
        print(f"  Max detection error: {np.max(distances):.1f} pixels")
    
    print(f"\n📈 Detection Statistics:")
    stats = result['metadata']['detection_stats']
    print(f"  Candidates found: {stats['candidates_found']}")
    print(f"  After merging: {stats['after_merging']}")
    print(f"  Final validated: {stats['final_validated']}")
    
    print("Source detection test complete!")