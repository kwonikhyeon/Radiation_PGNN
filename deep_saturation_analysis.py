#!/usr/bin/env python3
"""
Deep analysis of persistent traversability layer saturation issues
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters, TraversabilityCalculator

def deep_saturation_analysis():
    """Perform deep analysis of saturation causes"""
    print("🔬 Deep Analysis of Traversability Layer Saturation...")
    
    # Let's simulate the exact conditions from the app
    print("\n📱 Simulating Real App Conditions...")
    
    # Test Case 1: Realistic radiation field from model prediction
    print("\n🧪 Test Case 1: Realistic Model Prediction Simulation")
    
    # Create field that mimics model prediction output
    field_realistic = np.random.beta(2, 5, (256, 256)).astype(np.float32)  # Realistic distribution
    field_realistic = np.clip(field_realistic, 0, 1)
    
    # Add some weak sources that might be borderline for detection
    weak_sources = [(80, 80), (180, 120)]
    Y, X = np.ogrid[:256, :256]
    for sy, sx in weak_sources:
        weak_source = 0.35 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 20**2))
        field_realistic += weak_source
    field_realistic = np.clip(field_realistic, 0, 1)
    
    # Random robot position (like in app)
    robot_pos = (np.random.randint(50, 206), np.random.randint(50, 206))
    robot_heading = np.random.uniform(0, 2*np.pi)
    
    print(f"  Robot position: {robot_pos}")
    print(f"  Robot heading: {robot_heading:.2f} rad")
    print(f"  Field range: [{field_realistic.min():.3f}, {field_realistic.max():.3f}]")
    print(f"  Field mean: {field_realistic.mean():.3f}")
    
    result1 = calculate_traversability_layer(field_realistic, None, robot_pos, robot_heading)
    
    print(f"  Sources detected: {len(result1['source_locations'])}")
    print(f"  Total traversability range: [{result1['total_traversability'].min():.3f}, {result1['total_traversability'].max():.3f}]")
    print(f"  Total traversability mean: {result1['total_traversability'].mean():.3f}")
    
    if result1['total_traversability'].mean() > 0.7:
        print("  🚨 SATURATION DETECTED in realistic scenario!")
    
    # Test Case 2: Step-by-step component analysis
    print("\n🔍 Test Case 2: Component-by-Component Analysis")
    
    calculator = TraversabilityCalculator()
    
    # Manually call each component
    source_locations, detection_metadata = calculator.detect_radiation_sources(field_realistic)
    source_proximity = calculator.calculate_source_proximity_map(field_realistic, source_locations)
    low_radiation_bonus = calculator.calculate_low_radiation_bonus(field_realistic)
    
    print(f"  Source locations found: {source_locations}")
    print(f"  Source proximity range: [{source_proximity.min():.3f}, {source_proximity.max():.3f}], mean: {source_proximity.mean():.3f}")
    print(f"  Low radiation bonus range: [{low_radiation_bonus.min():.3f}, {low_radiation_bonus.max():.3f}], mean: {low_radiation_bonus.mean():.3f}")
    
    if source_locations:
        directional_navigation = calculator.calculate_directional_navigation_map(
            field_realistic, source_locations, robot_pos, robot_heading
        )
        print(f"  Directional navigation range: [{directional_navigation.min():.3f}, {directional_navigation.max():.3f}], mean: {directional_navigation.mean():.3f}")
        
        # Manual combination calculation
        base_weight = 1 - calculator.params.robot_directional_weight
        manual_total = (
            base_weight * calculator.params.proximity_weight * source_proximity + 
            base_weight * (1 - calculator.params.proximity_weight) * low_radiation_bonus +
            calculator.params.robot_directional_weight * directional_navigation
        )
        
        print(f"  Manual combination range: [{manual_total.min():.3f}, {manual_total.max():.3f}], mean: {manual_total.mean():.3f}")
        
        # Check individual component contributions
        prox_contrib = base_weight * calculator.params.proximity_weight * source_proximity
        low_rad_contrib = base_weight * (1 - calculator.params.proximity_weight) * low_radiation_bonus
        dir_contrib = calculator.params.robot_directional_weight * directional_navigation
        
        print(f"  Proximity contribution: mean {prox_contrib.mean():.3f}, max {prox_contrib.max():.3f}")
        print(f"  Low radiation contribution: mean {low_rad_contrib.mean():.3f}, max {low_rad_contrib.max():.3f}")
        print(f"  Directional contribution: mean {dir_contrib.mean():.3f}, max {dir_contrib.max():.3f}")
        
        # Check which component is dominant
        if prox_contrib.mean() > 0.4:
            print("  🟢 Proximity component is dominant!")
        if low_rad_contrib.mean() > 0.4:
            print("  🟡 Low radiation component is dominant!")
        if dir_contrib.mean() > 0.4:
            print("  🔵 Directional component is dominant!")
    else:
        # No sources - exploration mode
        directional_navigation = calculator.calculate_directional_navigation_map(
            field_realistic, source_locations, robot_pos, robot_heading
        )
        print(f"  Exploration mode directional range: [{directional_navigation.min():.3f}, {directional_navigation.max():.3f}], mean: {directional_navigation.mean():.3f}")
        
        # In exploration mode, only directional navigation and low radiation bonus
        manual_total = (
            (1 - calculator.params.robot_directional_weight) * low_radiation_bonus +
            calculator.params.robot_directional_weight * directional_navigation
        )
        
        print(f"  Exploration manual combination: [{manual_total.min():.3f}, {manual_total.max():.3f}], mean: {manual_total.mean():.3f}")
        
        low_rad_contrib = (1 - calculator.params.robot_directional_weight) * low_radiation_bonus
        dir_contrib = calculator.params.robot_directional_weight * directional_navigation
        
        print(f"  Low radiation contribution: mean {low_rad_contrib.mean():.3f}")
        print(f"  Exploration directional contribution: mean {dir_contrib.mean():.3f}")
        
        if low_rad_contrib.mean() > 0.4:
            print("  🟡 Low radiation bonus is causing saturation in exploration mode!")
        if dir_contrib.mean() > 0.4:
            print("  🔵 Exploration directional is causing saturation!")
    
    # Test Case 3: Multiple random scenarios like the app generates
    print("\n🎲 Test Case 3: Multiple Random App-like Scenarios")
    
    saturation_count = 0
    total_tests = 20
    saturation_details = []
    
    for i in range(total_tests):
        # Generate random field like the app
        test_field = np.random.beta(2, 5, (256, 256)).astype(np.float32)
        
        # Sometimes add sources, sometimes don't
        if np.random.random() > 0.3:  # 70% chance of sources
            n_sources = np.random.randint(1, 4)
            for _ in range(n_sources):
                sy = np.random.randint(30, 226)
                sx = np.random.randint(30, 226)
                intensity = np.random.uniform(0.3, 0.8)
                sigma = np.random.uniform(10, 25)
                source_field = intensity * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * sigma**2))
                test_field += source_field
        
        test_field = np.clip(test_field, 0, 1)
        
        test_robot_pos = (np.random.randint(50, 206), np.random.randint(50, 206))
        test_robot_heading = np.random.uniform(0, 2*np.pi)
        
        test_result = calculate_traversability_layer(test_field, None, test_robot_pos, test_robot_heading)
        
        mean_trav = test_result['total_traversability'].mean()
        
        if mean_trav > 0.7:
            saturation_count += 1
            saturation_details.append({
                'test_id': i,
                'mean': mean_trav,
                'field_mean': test_field.mean(),
                'field_max': test_field.max(),
                'sources_detected': len(test_result['source_locations']),
                'robot_pos': test_robot_pos
            })
    
    print(f"  Saturation rate: {saturation_count}/{total_tests} ({saturation_count/total_tests*100:.1f}%)")
    
    if saturation_details:
        print(f"  Saturated cases details:")
        for detail in saturation_details[:5]:  # Show first 5
            print(f"    Test {detail['test_id']}: mean={detail['mean']:.3f}, "
                  f"field_mean={detail['field_mean']:.3f}, sources={detail['sources_detected']}")
    
    # Test Case 4: Examine the combination formula itself
    print("\n🧮 Test Case 4: Formula Analysis")
    
    params = TraversabilityParameters()
    print(f"  Current parameters:")
    print(f"    robot_directional_weight: {params.robot_directional_weight}")
    print(f"    proximity_weight: {params.proximity_weight}")
    print(f"    low_radiation_bonus: {params.low_radiation_bonus}")
    
    # Theoretical maximum values
    base_weight = 1 - params.robot_directional_weight
    max_proximity_contrib = base_weight * params.proximity_weight * 1.0  # max source proximity
    max_low_rad_contrib = base_weight * (1 - params.proximity_weight) * params.low_radiation_bonus  # max low rad bonus
    max_dir_contrib = params.robot_directional_weight * 1.0  # max directional nav
    
    theoretical_max = max_proximity_contrib + max_low_rad_contrib + max_dir_contrib
    
    print(f"  Theoretical maximum contributions:")
    print(f"    Max proximity: {max_proximity_contrib:.3f}")
    print(f"    Max low radiation: {max_low_rad_contrib:.3f}")
    print(f"    Max directional: {max_dir_contrib:.3f}")
    print(f"    Theoretical total max: {theoretical_max:.3f}")
    
    if theoretical_max > 0.9:
        print("  🚨 FORMULA ISSUE: Theoretical maximum exceeds 0.9!")
        print("     This means the combination can easily saturate!")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test case 1 visualization
    im1 = axes[0, 0].imshow(field_realistic, cmap='hot', origin='lower')
    axes[0, 0].scatter(robot_pos[1], robot_pos[0], c='blue', s=50, marker='o')
    axes[0, 0].set_title(f'Realistic Field\n(mean: {field_realistic.mean():.3f})')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    if 'directional_navigation' in result1:
        im2 = axes[0, 1].imshow(result1['directional_navigation'], cmap='viridis', origin='lower')
        axes[0, 1].set_title(f'Directional Nav\n(mean: {result1["directional_navigation"].mean():.3f})')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(result1['total_traversability'], cmap='plasma', origin='lower')
    axes[0, 2].set_title(f'Total Traversability\n(mean: {result1["total_traversability"].mean():.3f})')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Component contributions
    if source_locations:
        im4 = axes[1, 0].imshow(prox_contrib, cmap='Greens', origin='lower')
        axes[1, 0].set_title(f'Proximity Contrib\n(mean: {prox_contrib.mean():.3f})')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
        
        im5 = axes[1, 1].imshow(low_rad_contrib, cmap='Blues', origin='lower')
        axes[1, 1].set_title(f'Low Rad Contrib\n(mean: {low_rad_contrib.mean():.3f})')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
        
        im6 = axes[1, 2].imshow(dir_contrib, cmap='Purples', origin='lower')
        axes[1, 2].set_title(f'Directional Contrib\n(mean: {dir_contrib.mean():.3f})')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('deep_saturation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved deep analysis to 'deep_saturation_analysis.png'")
    
    return saturation_count / total_tests

if __name__ == "__main__":
    saturation_rate = deep_saturation_analysis()
    print(f"\n🏁 Deep Analysis Complete! Saturation rate: {saturation_rate*100:.1f}%")