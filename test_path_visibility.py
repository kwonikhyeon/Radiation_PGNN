#!/usr/bin/env python3
"""
Test path visibility with different parameter settings
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def test_path_visibility():
    """Test different parameter settings for path visibility"""
    print("👁️ Testing Path Visibility with Different Parameters...")
    
    # Create simple test case
    field = np.zeros((100, 100), dtype=np.float32)
    
    # Add two sources for clear comparison
    sources = [(80, 20), (20, 80)]
    robot_pos = (50, 50)  # Center position
    
    Y, X = np.ogrid[:100, :100]
    for sy, sx in sources:
        source_field = 0.8 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 10**2))
        field += source_field
    field = np.clip(field, 0, 1)
    
    print(f"📊 Test setup:")
    print(f"  Robot at: {robot_pos}")
    print(f"  Sources at: {sources}")
    
    # Test different parameter combinations
    test_configs = [
        {
            'name': 'Default',
            'params': TraversabilityParameters()
        },
        {
            'name': 'High Directional Weight',
            'params': TraversabilityParameters(
                robot_directional_weight=0.8,  # Much higher
                path_spread_sigma=5.0           # Narrow paths
            )
        },
        {
            'name': 'Pure Directional',
            'params': TraversabilityParameters(
                robot_directional_weight=1.0,  # Only directional
                path_spread_sigma=4.0           # Very narrow paths
            )
        },
        {
            'name': 'Wide Paths',
            'params': TraversabilityParameters(
                robot_directional_weight=0.7,
                path_spread_sigma=12.0          # Wide paths
            )
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 Testing {config['name']}...")
        
        result = calculate_traversability_layer(
            field, None, robot_pos, 0.0, config['params']
        )
        
        results.append((config['name'], result))
        
        if 'directional_navigation' in result:
            dir_nav = result['directional_navigation']
            total_trav = result['total_traversability']
            
            print(f"  Directional range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
            print(f"  Total range: [{total_trav.min():.3f}, {total_trav.max():.3f}]")
            print(f"  Directional mean: {dir_nav.mean():.3f}")
            print(f"  Total mean: {total_trav.mean():.3f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original field
    im_orig = axes[0, 0].imshow(field, cmap='hot', origin='lower')
    axes[0, 0].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o', label='Robot')
    for i, (sy, sx) in enumerate(sources):
        axes[0, 0].scatter(sx, sy, c='white', s=80, marker='x', label=f'Source {i+1}')
        # Draw expected paths
        axes[0, 0].plot([robot_pos[1], sx], [robot_pos[0], sy], 
                       'cyan', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Original Field + Expected Paths')
    axes[0, 0].legend()
    plt.colorbar(im_orig, ax=axes[0, 0], fraction=0.046)
    
    # Plot results for each configuration
    for i, (name, result) in enumerate(results):
        col = i + 1
        
        # Directional navigation
        if 'directional_navigation' in result:
            im_dir = axes[0, col].imshow(result['directional_navigation'], cmap='viridis', origin='lower')
            axes[0, col].scatter(robot_pos[1], robot_pos[0], c='blue', s=80, marker='o')
            for sy, sx in sources:
                axes[0, col].scatter(sx, sy, c='red', s=60, marker='+')
            axes[0, col].set_title(f'{name}\nDirectional Navigation')
            plt.colorbar(im_dir, ax=axes[0, col], fraction=0.046)
        
        # Total traversability
        im_total = axes[1, col].imshow(result['total_traversability'], cmap='plasma', origin='lower')
        axes[1, col].scatter(robot_pos[1], robot_pos[0], c='blue', s=80, marker='o')
        for sy, sx in sources:
            axes[1, col].scatter(sx, sy, c='red', s=60, marker='+')
        axes[1, col].set_title(f'{name}\nTotal Traversability')
        plt.colorbar(im_total, ax=axes[1, col], fraction=0.046)
        
        # Difference from baseline (first result)
        if i > 0:
            baseline_total = results[0][1]['total_traversability']
            diff = result['total_traversability'] - baseline_total
            im_diff = axes[2, col].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-0.5, vmax=0.5)
            axes[2, col].scatter(robot_pos[1], robot_pos[0], c='blue', s=80, marker='o')
            axes[2, col].set_title(f'{name}\nDifference from Default')
            plt.colorbar(im_diff, ax=axes[2, col], fraction=0.046)
        else:
            # Statistics for baseline
            if 'directional_navigation' in result:
                dir_nav = result['directional_navigation']
                axes[2, col].text(0.1, 0.8, f"Dir weight: {test_configs[i]['params'].robot_directional_weight}", 
                                 transform=axes[2, col].transAxes)
                axes[2, col].text(0.1, 0.6, f"Path sigma: {test_configs[i]['params'].path_spread_sigma}", 
                                 transform=axes[2, col].transAxes)
                axes[2, col].text(0.1, 0.4, f"Dir range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]", 
                                 transform=axes[2, col].transAxes)
                axes[2, col].text(0.1, 0.2, f"Dir mean: {dir_nav.mean():.3f}", 
                                 transform=axes[2, col].transAxes)
            axes[2, col].set_title('Baseline Stats')
            axes[2, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('path_visibility_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved comparison to 'path_visibility_comparison.png'")
    
    return results

if __name__ == "__main__":
    results = test_path_visibility()
    print(f"\n🏁 Path Visibility Tests Complete!")