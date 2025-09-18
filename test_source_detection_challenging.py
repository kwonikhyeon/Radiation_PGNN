#!/usr/bin/env python3
"""
Test source detection with challenging scenarios
"""

import numpy as np
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from source_detection import detect_radiation_sources, SourceDetectionParameters
from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def test_challenging_source_detection():
    """Test source detection with challenging scenarios"""
    print("🧪 Testing Source Detection with Challenging Scenarios...")
    
    # Test Case 1: Very close sources (potential merging issue)
    print("\n📍 Test Case 1: Very Close Sources")
    field1 = np.zeros((256, 256), dtype=np.float32)
    
    # Two sources very close to each other (should be merged)
    close_sources = [(100, 100), (105, 105)]  # Only 7 pixels apart
    far_source = [(200, 200)]
    
    Y, X = np.ogrid[:256, :256]
    
    for y, x in close_sources + far_source:
        source_field = 0.7 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 12**2))
        field1 += source_field
    
    field1 = np.clip(field1, 0, 1)
    
    # Configure source detection for merging nearby sources
    detection_params = SourceDetectionParameters(
        detection_threshold=0.3,
        merge_distance=15,  # Should merge sources within 15 pixels
        peak_distance=10
    )
    
    result1 = detect_radiation_sources(field1, detection_params)
    detected1 = result1['source_locations']
    
    print(f"  Expected 2 sources (after merging): {close_sources + far_source}")
    print(f"  Detected {len(detected1)} sources: {detected1}")
    print(f"  Merging successful: {len(detected1) == 2}")
    
    # Test Case 2: Overlapping sources with different intensities
    print("\n📍 Test Case 2: Overlapping Sources")
    field2 = np.zeros((256, 256), dtype=np.float32)
    
    # Overlapping sources with different intensities
    sources2 = [(120, 120, 0.8, 20), (140, 120, 0.6, 15)]  # Overlapping
    
    for y, x, intensity, sigma in sources2:
        source_field = intensity * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * sigma**2))
        field2 += source_field
    
    field2 = np.clip(field2, 0, 1)
    
    # Test with different detection parameters
    strict_params = SourceDetectionParameters(
        detection_threshold=0.4,  # Higher threshold
        peak_prominence=0.15,
        merge_distance=25
    )
    
    result2 = detect_radiation_sources(field2, strict_params)
    detected2 = result2['source_locations']
    
    print(f"  Expected complex overlapping pattern")
    print(f"  Detected {len(detected2)} sources: {detected2}")
    print(f"  Detection stats: {result2['metadata']['detection_stats']}")
    
    # Test Case 3: Low intensity sources with noise
    print("\n📍 Test Case 3: Low Intensity + Noise")
    field3 = np.zeros((256, 256), dtype=np.float32)
    
    # Add weak sources
    weak_sources = [(80, 80), (180, 180)]
    for y, x in weak_sources:
        source_field = 0.4 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 15**2))
        field3 += source_field
    
    # Add significant noise
    field3 += np.random.normal(0, 0.05, field3.shape)
    field3 = np.clip(field3, 0, 1)
    
    # Use noise-robust parameters
    robust_params = SourceDetectionParameters(
        detection_threshold=0.25,  # Lower threshold for weak sources
        gaussian_sigma=1.5,        # More smoothing for noise
        peak_prominence=0.08,
        min_source_area=5
    )
    
    result3 = detect_radiation_sources(field3, robust_params)
    detected3 = result3['source_locations']
    
    print(f"  Expected 2 weak sources: {weak_sources}")
    print(f"  Detected {len(detected3)} sources: {detected3}")
    print(f"  Noise robustness test: {len(detected3) >= 1}")
    
    # Calculate accuracy for each test case
    print(f"\n📊 Overall Results:")
    print(f"  Test 1 (Close sources): {len(detected1)} detected")
    print(f"  Test 2 (Overlapping): {len(detected2)} detected") 
    print(f"  Test 3 (Noisy): {len(detected3)} detected")
    
    # Test traversability integration with challenging case
    print(f"\n🚀 Testing Traversability Integration:")
    
    trav_params = TraversabilityParameters(
        source_detection_params=robust_params,  # Use robust detection
        proximity_weight=0.7
    )
    
    trav_result = calculate_traversability_layer(
        field3, 
        parameters=trav_params
    )
    
    print(f"  Traversability sources detected: {len(trav_result['source_locations'])}")
    print(f"  Integration successful: {len(trav_result['source_locations']) > 0}")
    print(f"  Detection metadata available: {'source_detection' in trav_result['metadata']}")
    
    return result1, result2, result3, trav_result

if __name__ == "__main__":
    results = test_challenging_source_detection()
    print(f"\n🏁 Challenging Source Detection Tests Complete!")