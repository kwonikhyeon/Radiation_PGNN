#!/usr/bin/env python3
"""
Test fastest path navigation algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def test_fastest_path_navigation():
    """Test fastest path navigation with clear visualization"""
    print("🚀 Testing Fastest Path Navigation Algorithm...")
    
    # Create simple radiation field with one clear source
    field = np.zeros((100, 100), dtype=np.float32)
    
    # Add single source at (75, 75) 
    source_location = (75, 75)
    Y, X = np.ogrid[:100, :100]
    source_field = 0.9 * np.exp(-((Y - 75)**2 + (X - 75)**2) / (2 * 10**2))
    field += source_field
    field = np.clip(field, 0, 1)
    
    # Create measurement mask with a few points
    measurement_mask = np.zeros((100, 100), dtype=np.uint8)
    measurement_points = [(20, 20), (40, 40), (60, 60)]
    for y, x in measurement_points:
        measurement_mask[y, x] = 1
    
    # Test Case: Robot at bottom-left, target at top-right
    robot_position = (25, 25)
    robot_heading = np.pi/4  # 45 degrees toward target
    
    print(f"📊 Test scenario:")
    print(f"  Source at: {source_location}")
    print(f"  Robot at: {robot_position}")
    print(f"  Robot heading: {robot_heading:.2f} rad ({np.degrees(robot_heading):.0f}°)")
    
    # Calculate traversability with fastest path
    result = calculate_traversability_layer(
        field, measurement_mask, robot_position, robot_heading
    )
    
    print(f"\n🎯 Results:")
    print(f"  Sources detected: {len(result['source_locations'])}")
    if result['source_locations']:
        detected_source = result['source_locations'][0]
        print(f"  Detected source at: {detected_source}")
        
        # Calculate distances
        robot_y, robot_x = robot_position
        source_y, source_x = detected_source
        direct_distance = np.sqrt((robot_y - source_y)**2 + (robot_x - source_x)**2)
        print(f"  Direct distance to source: {direct_distance:.1f} pixels")
    
    if 'directional_navigation' in result:
        dir_nav = result['directional_navigation']
        print(f"  Directional navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
        print(f"  Mean directional value: {dir_nav.mean():.3f}")
        
        # Check highest traversability points (should form path to source)
        total_trav = result['total_traversability']
        high_trav_mask = total_trav > 0.8
        high_points = np.where(high_trav_mask)
        print(f"  High traversability points (>0.8): {len(high_points[0])}")
        
        # Check if path leads toward source
        if len(high_points[0]) > 0:
            mean_high_y = np.mean(high_points[0])
            mean_high_x = np.mean(high_points[1])
            print(f"  Mean of high traversability points: ({mean_high_y:.1f}, {mean_high_x:.1f})")
            
            # Should be between robot and source
            expected_y = (robot_y + source_y) / 2
            expected_x = (robot_x + source_x) / 2
            print(f"  Expected path center: ({expected_y:.1f}, {expected_x:.1f})")
    
    # Visualize the result
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original field with robot and source
    im1 = axes[0, 0].imshow(field, cmap='hot', origin='lower')
    axes[0, 0].scatter(robot_position[1], robot_position[0], c='blue', s=100, marker='o', label='Robot')
    axes[0, 0].scatter(source_location[1], source_location[0], c='white', s=100, marker='x', label='True Source')
    if result['source_locations']:
        det_source = result['source_locations'][0]
        axes[0, 0].scatter(det_source[1], det_source[0], c='red', s=80, marker='+', label='Detected')
    axes[0, 0].set_title('Radiation Field')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Directional navigation map
    if 'directional_navigation' in result:
        im2 = axes[0, 1].imshow(result['directional_navigation'], cmap='viridis', origin='lower')
        axes[0, 1].scatter(robot_position[1], robot_position[0], c='blue', s=100, marker='o')
        if result['source_locations']:
            det_source = result['source_locations'][0]
            axes[0, 1].scatter(det_source[1], det_source[0], c='red', s=80, marker='+')
        axes[0, 1].set_title('Directional Navigation Map')
        plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Total traversability
    im3 = axes[1, 0].imshow(result['total_traversability'], cmap='plasma', origin='lower')
    axes[1, 0].scatter(robot_position[1], robot_position[0], c='blue', s=100, marker='o')
    if result['source_locations']:
        det_source = result['source_locations'][0]
        axes[1, 0].scatter(det_source[1], det_source[0], c='red', s=80, marker='+')
    axes[1, 0].set_title('Total Traversability')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot 4: Source proximity
    im4 = axes[1, 1].imshow(result['source_proximity'], cmap='Greens', origin='lower')
    axes[1, 1].scatter(robot_position[1], robot_position[0], c='blue', s=100, marker='o')
    if result['source_locations']:
        det_source = result['source_locations'][0]
        axes[1, 1].scatter(det_source[1], det_source[0], c='red', s=80, marker='+')
    axes[1, 1].set_title('Source Proximity')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('fastest_path_test.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to 'fastest_path_test.png'")
    
    return result

if __name__ == "__main__":
    result = test_fastest_path_navigation()
    print(f"\n🏁 Fastest Path Navigation Test Complete!")