#!/usr/bin/env python3
"""
Test enhanced traversability layer with robot navigation
"""

import numpy as np
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def test_enhanced_traversability():
    """Test enhanced traversability layer with robot position and heading"""
    print("🧪 Testing Enhanced Traversability Layer with Robot Navigation...")
    
    # Create radiation field with clear sources
    field = np.zeros((256, 256), dtype=np.float32)
    
    # Add sources at known locations
    sources = [(80, 80), (180, 120), (120, 200)]
    Y, X = np.ogrid[:256, :256]
    
    for y, x in sources:
        source_field = 0.8 * np.exp(-((Y - y)**2 + (X - x)**2) / (2 * 15**2))
        field += source_field
    
    field = np.clip(field, 0, 1)
    
    # Create measurement mask
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    measurement_points = [(50, 50), (100, 100), (150, 150), (200, 200)]
    
    for y, x in measurement_points:
        measurement_mask[y, x] = 1
    
    print(f"📊 Created field with {len(sources)} sources")
    print(f"📍 Robot test scenarios:")
    
    # Test Case 1: Robot close to first source
    robot_pos1 = (90, 90)  # Close to source at (80, 80)
    robot_heading1 = np.pi/4  # 45 degrees toward first source
    
    result1 = calculate_traversability_layer(
        field, measurement_mask, robot_pos1, robot_heading1
    )
    
    print(f"\n🤖 Test 1 - Robot at {robot_pos1}, heading {robot_heading1:.2f}rad:")
    print(f"  Sources detected: {len(result1['source_locations'])}")
    print(f"  Directional navigation enabled: {result1['metadata']['robot_info']['directional_navigation_enabled']}")
    print(f"  Traversability range: [{result1['total_traversability'].min():.3f}, {result1['total_traversability'].max():.3f}]")
    
    # Compare traversability at robot position vs. far away
    robot_y, robot_x = robot_pos1
    trav_at_robot = result1['total_traversability'][robot_y, robot_x]
    trav_far = result1['total_traversability'][200, 200]
    
    print(f"  Traversability at robot: {trav_at_robot:.3f}")
    print(f"  Traversability far away: {trav_far:.3f}")
    
    # Test Case 2: Robot far from all sources
    robot_pos2 = (30, 30)  # Far from all sources
    robot_heading2 = 3*np.pi/4  # 135 degrees
    
    result2 = calculate_traversability_layer(
        field, measurement_mask, robot_pos2, robot_heading2
    )
    
    print(f"\n🤖 Test 2 - Robot at {robot_pos2}, heading {robot_heading2:.2f}rad:")
    print(f"  Sources detected: {len(result2['source_locations'])}")
    print(f"  Traversability range: [{result2['total_traversability'].min():.3f}, {result2['total_traversability'].max():.3f}]")
    
    # Test Case 3: No robot info (baseline)
    result3 = calculate_traversability_layer(field, measurement_mask)
    
    print(f"\n🤖 Test 3 - No robot info (baseline):")
    print(f"  Sources detected: {len(result3['source_locations'])}")
    print(f"  Directional navigation enabled: {result3['metadata']['robot_info']['directional_navigation_enabled']}")
    print(f"  Traversability range: [{result3['total_traversability'].min():.3f}, {result3['total_traversability'].max():.3f}]")
    
    # Compare component contributions
    print(f"\n📊 Component Analysis (Test 1 with robot):")
    if 'directional_navigation' in result1:
        dir_nav = result1['directional_navigation']
        print(f"  Directional navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
        print(f"  Directional navigation mean: {dir_nav.mean():.3f}")
    
    source_prox = result1['source_proximity']
    low_rad = result1['low_radiation_bonus']
    print(f"  Source proximity range: [{source_prox.min():.3f}, {source_prox.max():.3f}]")
    print(f"  Low radiation bonus range: [{low_rad.min():.3f}, {low_rad.max():.3f}]")
    
    # Test parameter impact
    print(f"\n⚙️ Parameter Impact Test:")
    
    # High directional weight
    high_dir_params = TraversabilityParameters(robot_directional_weight=0.8)
    result_high_dir = calculate_traversability_layer(
        field, measurement_mask, robot_pos1, robot_heading1, high_dir_params
    )
    
    print(f"  High directional weight (0.8): max traversability = {result_high_dir['total_traversability'].max():.3f}")
    
    # Low directional weight  
    low_dir_params = TraversabilityParameters(robot_directional_weight=0.2)
    result_low_dir = calculate_traversability_layer(
        field, measurement_mask, robot_pos1, robot_heading1, low_dir_params
    )
    
    print(f"  Low directional weight (0.2): max traversability = {result_low_dir['total_traversability'].max():.3f}")
    
    return result1, result2, result3

if __name__ == "__main__":
    results = test_enhanced_traversability()
    print(f"\n🏁 Enhanced Traversability Tests Complete!")