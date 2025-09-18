#!/usr/bin/env python3
"""
Investigate why traversability layer sometimes saturates to values near 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def investigate_saturation_causes():
    """Investigate different scenarios that might cause saturation"""
    print("🔍 Investigating Traversability Layer Saturation...")
    
    scenarios = []
    
    # Scenario 1: No sources detected (exploration mode)
    print("\n📍 Scenario 1: No sources detected")
    field1 = np.random.normal(0.05, 0.01, (100, 100))  # Very low noise field
    field1 = np.clip(field1, 0, 1)
    robot_pos1 = (50, 50)
    
    result1 = calculate_traversability_layer(field1, None, robot_pos1, 0.0)
    scenarios.append(("No Sources", result1, field1))
    
    print(f"  Total traversability range: [{result1['total_traversability'].min():.3f}, {result1['total_traversability'].max():.3f}]")
    print(f"  Mean: {result1['total_traversability'].mean():.3f}")
    
    # Scenario 2: Very close sources (robot almost at source)
    print("\n📍 Scenario 2: Robot very close to source")
    field2 = np.zeros((100, 100), dtype=np.float32)
    source_pos = (52, 48)  # Very close to robot at (50, 50)
    robot_pos2 = (50, 50)
    
    Y, X = np.ogrid[:100, :100]
    source_field = 0.9 * np.exp(-((Y - source_pos[0])**2 + (X - source_pos[1])**2) / (2 * 10**2))
    field2 += source_field
    field2 = np.clip(field2, 0, 1)
    
    result2 = calculate_traversability_layer(field2, None, robot_pos2, 0.0)
    scenarios.append(("Very Close Source", result2, field2))
    
    print(f"  Distance to source: {np.sqrt((robot_pos2[0] - source_pos[0])**2 + (robot_pos2[1] - source_pos[1])**2):.1f}")
    print(f"  Total traversability range: [{result2['total_traversability'].min():.3f}, {result2['total_traversability'].max():.3f}]")
    print(f"  Mean: {result2['total_traversability'].mean():.3f}")
    
    # Scenario 3: Multiple sources all close to robot
    print("\n📍 Scenario 3: Multiple sources close to robot")
    field3 = np.zeros((100, 100), dtype=np.float32)
    close_sources = [(45, 45), (55, 45), (45, 55), (55, 55)]  # All around robot
    robot_pos3 = (50, 50)
    
    for sy, sx in close_sources:
        source_field = 0.7 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 8**2))
        field3 += source_field
    field3 = np.clip(field3, 0, 1)
    
    result3 = calculate_traversability_layer(field3, None, robot_pos3, 0.0)
    scenarios.append(("Multiple Close Sources", result3, field3))
    
    print(f"  Number of close sources: {len(close_sources)}")
    print(f"  Total traversability range: [{result3['total_traversability'].min():.3f}, {result3['total_traversability'].max():.3f}]")
    print(f"  Mean: {result3['total_traversability'].mean():.3f}")
    
    # Scenario 4: Very low radiation field (high low_radiation_bonus)
    print("\n📍 Scenario 4: Very low radiation everywhere")
    field4 = np.full((100, 100), 0.02, dtype=np.float32)  # Very low uniform radiation
    robot_pos4 = (50, 50)
    
    # Add tiny source that might not be detected
    field4[80, 20] = 0.15  # Small peak that might be below detection threshold
    
    result4 = calculate_traversability_layer(field4, None, robot_pos4, 0.0)
    scenarios.append(("Very Low Radiation", result4, field4))
    
    print(f"  Max radiation in field: {field4.max():.3f}")
    print(f"  Total traversability range: [{result4['total_traversability'].min():.3f}, {result4['total_traversability'].max():.3f}]")
    print(f"  Mean: {result4['total_traversability'].mean():.3f}")
    
    # Scenario 5: Parameter sensitivity test
    print("\n📍 Scenario 5: Parameter sensitivity")
    field5 = np.zeros((100, 100), dtype=np.float32)
    source_pos5 = (80, 20)
    robot_pos5 = (20, 80)
    
    source_field = 0.8 * np.exp(-((Y - source_pos5[0])**2 + (X - source_pos5[1])**2) / (2 * 10**2))
    field5 += source_field
    field5 = np.clip(field5, 0, 1)
    
    # Test with high directional weight
    high_weight_params = TraversabilityParameters(
        robot_directional_weight=0.95,  # Very high
        path_spread_sigma=20.0,         # Very wide
        low_radiation_bonus=0.8         # High bonus for low radiation
    )
    
    result5 = calculate_traversability_layer(field5, None, robot_pos5, 0.0, high_weight_params)
    scenarios.append(("High Parameter Values", result5, field5))
    
    print(f"  Robot directional weight: {high_weight_params.robot_directional_weight}")
    print(f"  Total traversability range: [{result5['total_traversability'].min():.3f}, {result5['total_traversability'].max():.3f}]")
    print(f"  Mean: {result5['total_traversability'].mean():.3f}")
    
    # Analyze component contributions for each scenario
    print(f"\n📊 Component Analysis:")
    
    for name, result, field in scenarios:
        print(f"\n  {name}:")
        
        # Check if directional navigation exists
        if 'directional_navigation' in result:
            dir_nav = result['directional_navigation']
            print(f"    Directional nav range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}], mean: {dir_nav.mean():.3f}")
        else:
            print(f"    Directional nav: Not available")
        
        # Check other components
        source_prox = result['source_proximity']
        low_rad = result['low_radiation_bonus']
        total = result['total_traversability']
        
        print(f"    Source proximity range: [{source_prox.min():.3f}, {source_prox.max():.3f}], mean: {source_prox.mean():.3f}")
        print(f"    Low radiation bonus range: [{low_rad.min():.3f}, {low_rad.max():.3f}], mean: {low_rad.mean():.3f}")
        print(f"    Total traversability range: [{total.min():.3f}, {total.max():.3f}], mean: {total.mean():.3f}")
        
        # Check for saturation (mean > 0.8)
        if total.mean() > 0.8:
            print(f"    ⚠️  SATURATION DETECTED! Mean: {total.mean():.3f}")
            
            # Identify dominant component
            if 'directional_navigation' in result:
                if dir_nav.mean() > 0.7:
                    print(f"    🔴 Directional navigation is dominant (mean: {dir_nav.mean():.3f})")
            if low_rad.mean() > 0.7:
                print(f"    🟡 Low radiation bonus is dominant (mean: {low_rad.mean():.3f})")
            if source_prox.mean() > 0.7:
                print(f"    🟢 Source proximity is dominant (mean: {source_prox.mean():.3f})")
    
    # Create visualization for the most problematic cases
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select scenarios with highest saturation
    saturated_scenarios = [(name, result, field) for name, result, field in scenarios 
                          if result['total_traversability'].mean() > 0.7]
    
    for i, (name, result, field) in enumerate(saturated_scenarios[:3]):
        if i < 3:
            # Original field
            im1 = axes[0, i].imshow(field, cmap='hot', origin='lower')
            axes[0, i].set_title(f'{name}\nOriginal Field')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
            
            # Total traversability
            im2 = axes[1, i].imshow(result['total_traversability'], cmap='plasma', origin='lower')
            axes[1, i].set_title(f'{name}\nTraversability (mean: {result["total_traversability"].mean():.3f})')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('saturation_investigation.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved investigation to 'saturation_investigation.png'")
    
    return scenarios

if __name__ == "__main__":
    scenarios = investigate_saturation_causes()
    print(f"\n🏁 Saturation Investigation Complete!")