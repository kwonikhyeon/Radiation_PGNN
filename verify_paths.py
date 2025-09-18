#!/usr/bin/env python3
"""
Quick verification of path visibility improvement
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer

def verify_path_improvement():
    """Quick verification that paths are now more visible"""
    print("✅ Verifying Path Visibility Improvement...")
    
    # Simple test case
    field = np.zeros((80, 80), dtype=np.float32)
    
    # Single source for clear path
    source_pos = (70, 10)
    robot_pos = (10, 70)
    
    Y, X = np.ogrid[:80, :80]
    source_field = 0.9 * np.exp(-((Y - source_pos[0])**2 + (X - source_pos[1])**2) / (2 * 8**2))
    field += source_field
    field = np.clip(field, 0, 1)
    
    result = calculate_traversability_layer(field, None, robot_pos, 0.0)
    
    if 'directional_navigation' in result:
        dir_nav = result['directional_navigation']
        total_trav = result['total_traversability']
        
        print(f"📊 Results:")
        print(f"  Directional navigation range: [{dir_nav.min():.3f}, {dir_nav.max():.3f}]")
        print(f"  Directional navigation mean: {dir_nav.mean():.3f}")
        print(f"  Total traversability range: [{total_trav.min():.3f}, {total_trav.max():.3f}]")
        
        # Check path contrast
        path_y = (robot_pos[0] + source_pos[0]) // 2
        path_x = (robot_pos[1] + source_pos[1]) // 2
        path_value = total_trav[path_y, path_x]
        
        # Check off-path value
        off_path_y = path_y + 15
        off_path_x = path_x
        if off_path_y < 80:
            off_path_value = total_trav[off_path_y, off_path_x]
            contrast = path_value - off_path_value
            print(f"  Path contrast: {contrast:.3f} (path: {path_value:.3f}, off-path: {off_path_value:.3f})")
        
        # Create simple visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original field
        im1 = axes[0].imshow(field, cmap='hot', origin='lower')
        axes[0].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o', label='Robot')
        axes[0].scatter(source_pos[1], source_pos[0], c='white', s=100, marker='x', label='Source')
        axes[0].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                    'cyan', linewidth=3, alpha=0.8, label='Expected path')
        axes[0].set_title('Original Field')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0])
        
        # Directional navigation
        im2 = axes[1].imshow(dir_nav, cmap='viridis', origin='lower')
        axes[1].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o')
        axes[1].scatter(source_pos[1], source_pos[0], c='red', s=100, marker='+')
        axes[1].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                    'cyan', linewidth=3, alpha=0.8)
        axes[1].set_title('Directional Navigation\n(Should show clear path)')
        plt.colorbar(im2, ax=axes[1])
        
        # Total traversability
        im3 = axes[2].imshow(total_trav, cmap='plasma', origin='lower')
        axes[2].scatter(robot_pos[1], robot_pos[0], c='blue', s=100, marker='o')
        axes[2].scatter(source_pos[1], source_pos[0], c='red', s=100, marker='+')
        axes[2].plot([robot_pos[1], source_pos[1]], [robot_pos[0], source_pos[0]], 
                    'cyan', linewidth=3, alpha=0.8)
        axes[2].set_title('Total Traversability\n(Final result)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('path_verification.png', dpi=150, bbox_inches='tight')
        print(f"💾 Saved verification to 'path_verification.png'")
        
        return dir_nav.mean() > 0.25  # Should be much higher now
    
    return False

if __name__ == "__main__":
    success = verify_path_improvement()
    if success:
        print("\n✅ Path visibility improvement VERIFIED!")
    else:
        print("\n❌ Path visibility still needs work")
    print("🏁 Verification Complete!")