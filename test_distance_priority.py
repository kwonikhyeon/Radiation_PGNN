#!/usr/bin/env python3
"""
Test distance uncertainty prioritization over gradient
"""

import numpy as np
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from information_gain import calculate_information_gain_layer

def test_distance_vs_gradient_priority():
    """Test that distance uncertainty has priority over gradient"""
    print("🧪 Testing Distance vs Gradient Priority...")
    
    # Create a field with strong gradient at far distances
    x = np.linspace(-4, 4, 256)
    y = np.linspace(-4, 4, 256)
    X, Y = np.meshgrid(x, y)
    
    # Create a field with sharp edges far from center
    prediction = np.zeros((256, 256), dtype=np.float32)
    
    # Add a sharp step function far from center (high gradient, far from measurements)
    prediction[180:200, 100:150] = 0.8  # High value region
    prediction[160:180, 100:150] = 0.1  # Low value region (creates sharp edge)
    
    # Smooth slightly to make it more realistic
    from scipy.ndimage import gaussian_filter
    prediction = gaussian_filter(prediction, sigma=0.5)
    
    # Create measurement mask at center (far from the high gradient area)
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    measurement_mask[128, 128] = 1  # Center measurement
    measurement_mask[120, 120] = 1  # Nearby measurement
    
    print(f"📊 Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"📍 Measurement points: {measurement_mask.sum()}")
    print(f"📏 Distance from measurements to high gradient: ~{np.sqrt((180-128)**2 + (125-128)**2):.1f} pixels")
    
    # Calculate information gain layer
    result = calculate_information_gain_layer(
        single_prediction=prediction,
        measurement_mask=measurement_mask
    )
    
    # Analyze results at specific locations
    center_y, center_x = 128, 128  # Measurement point (low distance, low gradient)
    edge_y, edge_x = 170, 125      # High gradient area (high distance, high gradient)
    far_y, far_x = 200, 200       # Far area (high distance, low gradient)
    
    print(f"\n📍 Analysis at Key Locations:")
    
    # At measurement center
    center_uncertainty = result['uncertainty_component'][center_y, center_x]
    center_gradient = result['gradient_component'][center_y, center_x]
    center_total = result['information_gain'][center_y, center_x]
    
    print(f"  📌 Center (measurement point):")
    print(f"    Uncertainty: {center_uncertainty:.4f}")
    print(f"    Gradient: {center_gradient:.4f}")
    print(f"    Total Gain: {center_total:.4f}")
    
    # At high gradient edge (far from measurements)
    edge_uncertainty = result['uncertainty_component'][edge_y, edge_x]
    edge_gradient = result['gradient_component'][edge_y, edge_x]
    edge_total = result['information_gain'][edge_y, edge_x]
    
    print(f"  🔥 High Gradient Edge (far from measurements):")
    print(f"    Uncertainty: {edge_uncertainty:.4f}")
    print(f"    Gradient: {edge_gradient:.4f}")
    print(f"    Total Gain: {edge_total:.4f}")
    
    # At far low gradient area
    far_uncertainty = result['uncertainty_component'][far_y, far_x]
    far_gradient = result['gradient_component'][far_y, far_x]
    far_total = result['information_gain'][far_y, far_x]
    
    print(f"  🌀 Far Low Gradient Area:")
    print(f"    Uncertainty: {far_uncertainty:.4f}")
    print(f"    Gradient: {far_gradient:.4f}")
    print(f"    Total Gain: {far_total:.4f}")
    
    # Check prioritization
    print(f"\n🎯 Priority Analysis:")
    print(f"  Distance uncertainty dominance: {edge_uncertainty > edge_gradient}")
    print(f"  High distance beats high gradient: {edge_total > far_total}")
    print(f"  Uncertainty contribution at edge: {(0.7 * edge_uncertainty) / edge_total * 100:.1f}%")
    print(f"  Gradient contribution at edge: {(0.3 * edge_gradient) / edge_total * 100:.1f}%")
    
    # Parameters summary
    print(f"\n⚙️  Current Parameters:")
    print(f"  Distance weight in uncertainty: {result['metadata']['parameters']['distance_weight']}")
    print(f"  Distance sigma: {result['metadata']['parameters']['distance_sigma']}")
    print(f"  Uncertainty weight in final: {result['metadata']['gradient_info']['uncertainty_weight']}")
    print(f"  Gradient weight in final: {result['metadata']['gradient_info']['gradient_weight']}")
    
    return result

if __name__ == "__main__":
    result = test_distance_vs_gradient_priority()
    print(f"\n🏁 Distance Priority Test Complete!")