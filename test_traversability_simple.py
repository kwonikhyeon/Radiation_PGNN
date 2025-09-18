#!/usr/bin/env python3
"""
Simple test for the new traversability layer functionality
"""

import numpy as np
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer

def test_traversability_with_sources():
    """Test traversability layer with well-defined sources"""
    print("🧪 Testing Traversability Layer with Multiple Sources...")
    
    # Create radiation field with clear source patterns
    field = np.zeros((256, 256), dtype=np.float32)
    
    # Add multiple sources with different intensities
    sources = [
        (80, 80, 0.9, 12),    # High intensity, small source
        (180, 120, 0.7, 18),  # Medium intensity, larger source  
        (120, 200, 0.8, 15),  # High intensity, medium source
    ]
    
    Y, X = np.ogrid[:256, :256]
    
    for y, x, intensity, sigma in sources:
        source_field = intensity * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * sigma**2))
        field += source_field
    
    # Clip to [0, 1] range
    field = np.clip(field, 0, 1)
    
    # Create measurement mask
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    measurement_points = [(50, 50), (100, 100), (150, 150), (200, 200)]
    
    for y, x in measurement_points:
        measurement_mask[y, x] = 1
    
    print(f"📊 Radiation field range: [{field.min():.3f}, {field.max():.3f}]")
    print(f"📍 Measurement points: {measurement_mask.sum()}")
    print(f"🎯 Expected sources: {len(sources)}")
    
    # Calculate traversability layer
    result = calculate_traversability_layer(field, measurement_mask)
    
    print(f"\n✅ Traversability Results:")
    print(f"  Total traversability range: [{result['total_traversability'].min():.4f}, {result['total_traversability'].max():.4f}]")
    print(f"  Mean traversability: {result['metadata']['statistics']['mean_traversability']:.4f}")
    print(f"  Sources detected: {result['metadata']['statistics']['num_sources_detected']}")
    print(f"  High traversability area: {result['metadata']['statistics']['high_traversability_area']*100:.1f}%")
    print(f"  Low traversability area: {result['metadata']['statistics']['low_traversability_area']*100:.1f}%")
    
    # Analyze source detection accuracy
    detected_sources = result['source_locations']
    print(f"\n🔍 Source Detection Analysis:")
    print(f"  Expected sources: {[(y, x) for y, x, _, _ in sources]}")
    print(f"  Detected sources: {detected_sources}")
    
    # Check proximity effect around sources
    if detected_sources:
        print(f"\n📏 Proximity Analysis:")
        for i, (det_y, det_x) in enumerate(detected_sources):
            # Check traversability at source location vs nearby
            trav_at_source = result['total_traversability'][det_y, det_x]
            
            # Check nearby location (20 pixels away)
            nearby_y = min(det_y + 20, 255)
            nearby_x = min(det_x + 20, 255)
            trav_nearby = result['total_traversability'][nearby_y, nearby_x]
            
            # Check far location (50 pixels away)
            far_y = min(det_y + 50, 255) 
            far_x = min(det_x + 50, 255)
            trav_far = result['total_traversability'][far_y, far_x]
            
            print(f"    Source {i+1} at ({det_y}, {det_x}):")
            print(f"      At source: {trav_at_source:.4f}")
            print(f"      Nearby (+20px): {trav_nearby:.4f}")
            print(f"      Far (+50px): {trav_far:.4f}")
            print(f"      Proximity effect: {trav_at_source > trav_far}")
    
    # Component analysis
    print(f"\n📋 Component Analysis:")
    print(f"  Source proximity range: [{result['source_proximity'].min():.4f}, {result['source_proximity'].max():.4f}]")
    print(f"  Low radiation bonus range: [{result['low_radiation_bonus'].min():.4f}, {result['low_radiation_bonus'].max():.4f}]")
    print(f"  Proximity weight: {result['metadata']['parameters']['proximity_weight']}")
    print(f"  Proximity decay rate: {result['metadata']['parameters']['proximity_decay']}")
    
    return result

if __name__ == "__main__":
    result = test_traversability_with_sources()
    print(f"\n🏁 Traversability Layer Test Complete!")