#!/usr/bin/env python3
"""
Debug shortest path weight application
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def debug_path_weights():
    """Debug the shortest path weight calculation"""
    print("🔍 Debugging Shortest Path Weight Application...")
    
    # Create simple test case with clear visualization
    field = np.zeros((100, 100), dtype=np.float32)
    
    # Add single source at (80, 20) - easy to see path
    source_pos = (80, 20)
    robot_pos = (20, 80)
    
    Y, X = np.ogrid[:100, :100]
    source_field = 0.9 * np.exp(-((Y - source_pos[0])**2 + (X - source_pos[1])**2) / (2 * 10**2))
    field += source_field
    field = np.clip(field, 0, 1)
    
    print(f"📊 Test setup:")
    print(f"  Robot at: {robot_pos}")
    print(f"  Source at: {source_pos}")
    
    # Calculate direct distance
    direct_distance = np.sqrt((robot_pos[0] - source_pos[0])**2 + (robot_pos[1] - source_pos[1])**2)
    print(f"  Direct distance: {direct_distance:.1f} pixels")
    
    # Test with verbose parameters to see what's happening
    debug_params = TraversabilityParameters(
        path_spread_sigma=5.0,      # Narrow path for clear visualization
        distance_decay_rate=0.01,   # Weak decay to see effect
        max_source_distance=200.0,  # Large range
        robot_directional_weight=1.0  # Pure directional navigation
    )
    
    result = calculate_traversability_layer(
        field, None, robot_pos, 0.0, debug_params
    )
    
    print(f"\n🎯 Results:")
    print(f"  Sources detected: {len(result['source_locations'])}")
    
    if 'directional_navigation' in result:
        dir_nav = result['directional_navigation']
        print(f"  Directional navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
        print(f"  Directional navigation mean: {dir_nav.mean():.3f}")
        
        # Check values along the expected path
        print(f"\n📏 Path analysis:")
        
        # Sample points along the direct path
        n_samples = 10
        for i in range(n_samples):
            t = i / (n_samples - 1)
            path_y = int(robot_pos[0] + t * (source_pos[0] - robot_pos[0]))
            path_x = int(robot_pos[1] + t * (source_pos[1] - robot_pos[1]))
            
            if 0 <= path_y < 100 and 0 <= path_x < 100:
                path_value = dir_nav[path_y, path_x]
                print(f"    Point {i}: ({path_y}, {path_x}) = {path_value:.3f}")
        
        # Check off-path points for comparison
        print(f"\n📍 Off-path comparison:")
        # Points perpendicular to path
        mid_y = (robot_pos[0] + source_pos[0]) // 2
        mid_x = (robot_pos[1] + source_pos[1]) // 2
        
        offsets = [5, 10, 15]
        for offset in offsets:
            off_y = min(99, max(0, mid_y + offset))
            off_x = min(99, max(0, mid_x))
            off_value = dir_nav[off_y, off_x]
            print(f"    Off-path +{offset}: ({off_y}, {off_x}) = {off_value:.3f}")
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original field
    im1 = axes[0, 0].imshow(field, cmap='hot', origin='lower')
    axes[0, 0].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o', label='Robot')
    axes[0, 0].scatter(source_pos[1], source_pos[0], c='white', s=100, marker='x', label='Source')
    # Draw expected path line
    axes[0, 0].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                   'cyan', linewidth=2, alpha=0.7, label='Expected path')
    axes[0, 0].set_title('Original Field')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Directional navigation
    if 'directional_navigation' in result:
        im2 = axes[0, 1].imshow(result['directional_navigation'], cmap='viridis', origin='lower')
        axes[0, 1].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o')
        axes[0, 1].scatter(source_pos[1], source_pos[0], c='red', s=100, marker='+')
        axes[0, 1].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                       'cyan', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Directional Navigation')
        plt.colorbar(im2, ax=axes[0, 1])
    
    # Total traversability
    im3 = axes[0, 2].imshow(result['total_traversability'], cmap='plasma', origin='lower')
    axes[0, 2].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o')
    axes[0, 2].scatter(source_pos[1], source_pos[0], c='red', s=100, marker='+')
    axes[0, 2].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                   'cyan', linewidth=2, alpha=0.7)
    axes[0, 2].set_title('Total Traversability')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Cross-section along the path
    if 'directional_navigation' in result:
        # Create line samples
        n_line_samples = int(direct_distance)
        line_values = []
        line_positions = []
        
        for i in range(n_line_samples):
            t = i / (n_line_samples - 1)
            ly = int(robot_pos[0] + t * (source_pos[0] - robot_pos[0]))
            lx = int(robot_pos[1] + t * (source_pos[1] - robot_pos[1]))
            
            if 0 <= ly < 100 and 0 <= lx < 100:
                line_values.append(result['directional_navigation'][ly, lx])
                line_positions.append(i)
        
        axes[1, 0].plot(line_positions, line_values, 'b-', linewidth=2, label='Path values')
        axes[1, 0].set_title('Values Along Path')
        axes[1, 0].set_xlabel('Distance from robot')
        axes[1, 0].set_ylabel('Directional weight')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # Cross-section perpendicular to path
    if 'directional_navigation' in result:
        mid_y = (robot_pos[0] + source_pos[0]) // 2
        mid_x = (robot_pos[1] + source_pos[1]) // 2
        
        perp_values = []
        perp_positions = []
        
        for offset in range(-30, 31):
            py = mid_y + offset
            px = mid_x
            
            if 0 <= py < 100 and 0 <= px < 100:
                perp_values.append(result['directional_navigation'][py, px])
                perp_positions.append(offset)
        
        axes[1, 1].plot(perp_positions, perp_values, 'r-', linewidth=2, label='Perpendicular values')
        axes[1, 1].axvline(x=0, color='cyan', linestyle='--', alpha=0.7, label='Path center')
        axes[1, 1].set_title('Values Perpendicular to Path')
        axes[1, 1].set_xlabel('Distance from path center')
        axes[1, 1].set_ylabel('Directional weight')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
    # Statistics
    axes[1, 2].text(0.1, 0.9, f"Robot: {robot_pos}", transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.8, f"Source: {source_pos}", transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.7, f"Distance: {direct_distance:.1f}", transform=axes[1, 2].transAxes, fontsize=12)
    
    if 'directional_navigation' in result:
        dir_nav = result['directional_navigation']
        axes[1, 2].text(0.1, 0.5, f"Dir nav range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]", 
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.4, f"Dir nav mean: {dir_nav.mean():.3f}", 
                       transform=axes[1, 2].transAxes, fontsize=10)
        
        # Check if path has higher values
        path_center_y = (robot_pos[0] + source_pos[0]) // 2
        path_center_x = (robot_pos[1] + source_pos[1]) // 2
        path_value = dir_nav[path_center_y, path_center_x]
        
        axes[1, 2].text(0.1, 0.3, f"Path center value: {path_value:.3f}", 
                       transform=axes[1, 2].transAxes, fontsize=10)
    
    axes[1, 2].set_title('Debug Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_path_weights.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved debug visualization to 'debug_path_weights.png'")
    
    return result

if __name__ == "__main__":
    result = debug_path_weights()
    print(f"\n🏁 Debug Analysis Complete!")