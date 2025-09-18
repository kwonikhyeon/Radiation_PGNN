#!/usr/bin/env python3
"""
Test multi-source path navigation with various scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def test_multi_source_scenarios():
    """Test different scenarios for multi-source path navigation"""
    print("🗺️ Testing Multi-Source Path Navigation...")
    
    # Scenario 1: Multiple sources at different distances
    print("\n📍 Scenario 1: Multiple sources at different distances")
    field1 = np.zeros((150, 150), dtype=np.float32)
    
    # Add sources at various distances from robot
    sources1 = [(40, 40), (80, 50), (120, 100)]  # Close, medium, far
    robot_pos1 = (25, 25)
    
    Y, X = np.ogrid[:150, :150]
    for y, x in sources1:
        source_field = 0.8 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 12**2))
        field1 += source_field
    field1 = np.clip(field1, 0, 1)
    
    # Calculate distances from robot to each source
    for i, (sy, sx) in enumerate(sources1):
        distance = np.sqrt((robot_pos1[0] - sy)**2 + (robot_pos1[1] - sx)**2)
        print(f"  Source {i+1} at ({sy}, {sx}): distance {distance:.1f}")
    
    result1 = calculate_traversability_layer(
        field1, None, robot_pos1, 0.0
    )
    
    print(f"  Result: {len(result1['source_locations'])} sources detected")
    if 'directional_navigation' in result1:
        dir_nav = result1['directional_navigation']
        print(f"  Directional navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
    
    # Scenario 2: No sources detected (exploration mode)
    print("\n🔍 Scenario 2: No sources detected (exploration mode)")
    field2 = np.random.normal(0.1, 0.02, (100, 100))  # Just noise
    field2 = np.clip(field2, 0, 1)
    
    robot_pos2 = (50, 50)
    
    result2 = calculate_traversability_layer(
        field2, None, robot_pos2, np.pi/2
    )
    
    print(f"  Result: {len(result2['source_locations'])} sources detected")
    if 'directional_navigation' in result2:
        dir_nav = result2['directional_navigation']
        print(f"  Exploration navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
        print(f"  Exploration navigation mean: {dir_nav.mean():.3f}")
    
    # Scenario 3: Sources at edge of detection range
    print("\n🎯 Scenario 3: Sources near max detection distance")
    field3 = np.zeros((200, 200), dtype=np.float32)
    
    # Add sources at various distances, some beyond max range
    sources3 = [(50, 50), (150, 150), (180, 180)]  # Within, edge, beyond
    robot_pos3 = (25, 25)
    
    Y3, X3 = np.ogrid[:200, :200]
    for y, x in sources3:
        source_field = 0.9 * np.exp(-((Y3 - y)**2 + (X3 - x)**2) / (2 * 15**2))
        field3 += source_field
    field3 = np.clip(field3, 0, 1)
    
    # Test with default max_source_distance (100.0)
    result3 = calculate_traversability_layer(
        field3, None, robot_pos3, 0.0
    )
    
    for i, (sy, sx) in enumerate(sources3):
        distance = np.sqrt((robot_pos3[0] - sy)**2 + (robot_pos3[1] - sx)**2)
        within_range = distance <= 100.0  # default max_source_distance
        print(f"  Source {i+1} at ({sy}, {sx}): distance {distance:.1f}, within range: {within_range}")
    
    print(f"  Result: {len(result3['source_locations'])} sources detected")
    
    # Scenario 4: Custom parameters test
    print("\n⚙️ Scenario 4: Custom parameters test")
    
    # Test with tighter path spread
    tight_params = TraversabilityParameters(
        path_spread_sigma=4.0,      # Tighter paths
        distance_decay_rate=0.05,   # Faster decay for distant sources
        max_source_distance=80.0    # Shorter max range
    )
    
    result4 = calculate_traversability_layer(
        field1, None, robot_pos1, 0.0, tight_params
    )
    
    print(f"  Custom params result: {len(result4.get('source_locations', []))} sources processed")
    if 'directional_navigation' in result4:
        dir_nav = result4['directional_navigation']
        print(f"  Custom navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
    
    # Visualization for Scenario 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original field
    im1 = axes[0, 0].imshow(field1, cmap='hot', origin='lower')
    axes[0, 0].scatter(robot_pos1[1], robot_pos1[0], c='blue', s=100, marker='o', label='Robot')
    for i, (sy, sx) in enumerate(sources1):
        axes[0, 0].scatter(sx, sy, c='white', s=80, marker='x', label=f'Source {i+1}')
    axes[0, 0].set_title('Multi-Source Field')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Directional navigation
    if 'directional_navigation' in result1:
        im2 = axes[0, 1].imshow(result1['directional_navigation'], cmap='viridis', origin='lower')
        axes[0, 1].scatter(robot_pos1[1], robot_pos1[0], c='blue', s=100, marker='o')
        for i, (sy, sx) in enumerate(sources1):
            axes[0, 1].scatter(sx, sy, c='red', s=60, marker='+')
        axes[0, 1].set_title('Multi-Source Paths')
        plt.colorbar(im2, ax=axes[0, 1])
    
    # Total traversability
    im3 = axes[1, 0].imshow(result1['total_traversability'], cmap='plasma', origin='lower')
    axes[1, 0].scatter(robot_pos1[1], robot_pos1[0], c='blue', s=100, marker='o')
    axes[1, 0].set_title('Total Traversability')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Exploration mode (Scenario 2)
    if 'directional_navigation' in result2:
        im4 = axes[1, 1].imshow(result2['directional_navigation'], cmap='cividis', origin='lower')
        axes[1, 1].scatter(robot_pos2[1], robot_pos2[0], c='blue', s=100, marker='o')
        axes[1, 1].set_title('Exploration Mode')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('multi_source_paths_test.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to 'multi_source_paths_test.png'")
    
    return result1, result2, result3, result4

if __name__ == "__main__":
    results = test_multi_source_scenarios()
    print(f"\n🏁 Multi-Source Path Navigation Tests Complete!")