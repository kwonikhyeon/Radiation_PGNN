#!/usr/bin/env python3
"""
Analyze saved traversability debug files to identify high value issues
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

def analyze_debug_file(filepath):
    """Analyze a single traversability debug file"""
    
    print(f"🔍 Analyzing debug file: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            debug_data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Extract data
    stats = debug_data['stats']
    traversability = debug_data['traversability_layer']['total_traversability']
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Timestamp: {debug_data['timestamp']}")
    print(f"  Seed: {debug_data.get('seed', 'Unknown')}")
    print(f"  Mean traversability: {stats['total_traversability_mean']:.4f}")
    print(f"  Max traversability: {stats['total_traversability_max']:.4f}")
    print(f"  Min traversability: {stats['total_traversability_min']:.4f}")
    print(f"  Std traversability: {stats['total_traversability_std']:.4f}")
    print(f"  Sources detected: {stats['sources_detected']}")
    print(f"  Measurement points: {stats['n_measurements']}")
    
    # Check for high values
    is_high_value = stats['total_traversability_mean'] > 0.7
    if is_high_value:
        print(f"  🚨 HIGH MEAN VALUE CASE: {stats['total_traversability_mean']:.4f}")
    
    # Component analysis
    print(f"\n🧩 Component Analysis:")
    print(f"  Source proximity mean: {stats['source_proximity_mean']:.4f}")
    print(f"  Low radiation bonus mean: {stats['low_radiation_bonus_mean']:.4f}")
    print(f"  Directional navigation mean: {stats['directional_navigation_mean']:.4f}")
    
    # Identify dominant component
    components = {
        'proximity': stats['source_proximity_mean'],
        'low_radiation': stats['low_radiation_bonus_mean'],
        'directional': stats['directional_navigation_mean']
    }
    
    dominant_component = max(components, key=components.get)
    dominant_value = components[dominant_component]
    
    print(f"  🎯 Dominant component: {dominant_component} ({dominant_value:.4f})")
    
    if dominant_value > 0.4:
        print(f"  ⚠️  Component {dominant_component} may be causing saturation!")
    
    # Field analysis
    print(f"\n🌍 Field Analysis:")
    print(f"  GT field mean: {stats['field_mean']:.4f}")
    print(f"  GT field max: {stats['field_max']:.4f}")
    
    # Check for formula issues
    theoretical_max = (stats['source_proximity_mean'] + 
                      stats['low_radiation_bonus_mean'] + 
                      stats['directional_navigation_mean'])
    
    print(f"\n🧮 Formula Analysis:")
    print(f"  Theoretical component sum: {theoretical_max:.4f}")
    print(f"  Actual mean: {stats['total_traversability_mean']:.4f}")
    
    if theoretical_max > 1.1:
        print(f"  🚨 FORMULA ISSUE: Components sum to {theoretical_max:.4f} > 1.0!")
        print(f"     This indicates weighted sum instead of weighted average!")
    
    # Value distribution analysis
    print(f"\n📈 Value Distribution:")
    high_value_pixels = np.sum(traversability > 0.8)
    total_pixels = traversability.size
    high_value_percentage = (high_value_pixels / total_pixels) * 100
    
    print(f"  Pixels > 0.8: {high_value_pixels}/{total_pixels} ({high_value_percentage:.1f}%)")
    
    very_high_pixels = np.sum(traversability > 0.9)
    very_high_percentage = (very_high_pixels / total_pixels) * 100
    print(f"  Pixels > 0.9: {very_high_pixels}/{total_pixels} ({very_high_percentage:.1f}%)")
    
    # Potential causes analysis
    print(f"\n🕵️ Potential Causes Analysis:")
    
    if is_high_value:
        causes = []
        
        if stats['sources_detected'] == 0:
            causes.append("No sources detected - exploration mode may be saturating")
        
        if stats['low_radiation_bonus_mean'] > 0.5:
            causes.append("Low radiation bonus component is very high")
        
        if stats['directional_navigation_mean'] > 0.5:
            causes.append("Directional navigation component is very high")
        
        if stats['field_mean'] < 0.1:
            causes.append("Very low field values may trigger high low-radiation bonus")
        
        if theoretical_max > 1.0:
            causes.append("Formula using weighted sum instead of weighted average")
        
        if causes:
            print("  Likely causes:")
            for cause in causes:
                print(f"    - {cause}")
        else:
            print("  No obvious causes identified - needs deeper investigation")
    
    return debug_data, is_high_value

def visualize_debug_data(debug_data, save_path=None):
    """Create visualization of the debug data"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get data
    gt_field = debug_data.get('gt_field')
    prediction = debug_data.get('prediction')
    traversability_data = debug_data['traversability_layer']
    measurement_points = debug_data.get('measurement_points', [])
    
    # Row 1: Input data
    if gt_field is not None:
        im1 = axes[0, 0].imshow(gt_field, cmap='hot', origin='lower')
        axes[0, 0].set_title(f'GT Field\n(mean: {gt_field.mean():.3f})')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Add measurement points
        if len(measurement_points) > 0:
            mp_array = np.array(measurement_points)
            axes[0, 0].scatter(mp_array[:, 1], mp_array[:, 0], 
                             c='white', s=20, marker='o', alpha=0.8)
    
    if prediction is not None:
        im2 = axes[0, 1].imshow(prediction, cmap='hot', origin='lower')
        axes[0, 1].set_title(f'Prediction\n(mean: {prediction.mean():.3f})')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Prediction\nAvailable', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Prediction\n(Not Available)')
    
    # Total traversability
    total_trav = traversability_data['total_traversability']
    im3 = axes[0, 2].imshow(total_trav, cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Total Traversability\n(mean: {total_trav.mean():.3f})')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Components
    source_prox = traversability_data.get('source_proximity', np.zeros_like(total_trav))
    im4 = axes[1, 0].imshow(source_prox, cmap='Greens', origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Source Proximity\n(mean: {source_prox.mean():.3f})')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    low_rad = traversability_data.get('low_radiation_bonus', np.zeros_like(total_trav))
    im5 = axes[1, 1].imshow(low_rad, cmap='Blues', origin='lower', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Low Radiation Bonus\n(mean: {low_rad.mean():.3f})')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    directional = traversability_data.get('directional_navigation', np.zeros_like(total_trav))
    im6 = axes[1, 2].imshow(directional, cmap='Purples', origin='lower', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Directional Navigation\n(mean: {directional.mean():.3f})')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Add source locations if available
    source_locations = traversability_data.get('source_locations', [])
    if source_locations:
        source_array = np.array(source_locations)
        for ax in axes.flat:
            ax.scatter(source_array[:, 1], source_array[:, 0], 
                      c='red', s=60, marker='x', linewidth=2, alpha=0.9)
    
    # Add overall info
    stats = debug_data['stats']
    info_text = (f"Timestamp: {debug_data['timestamp']}\n"
                f"Mean: {stats['total_traversability_mean']:.3f}, "
                f"Sources: {stats['sources_detected']}, "
                f"Measurements: {stats['n_measurements']}")
    fig.suptitle(info_text, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze traversability debug files")
    parser.add_argument("filepath", type=str, help="Path to debug .pkl file")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--save-vis", type=str, help="Save visualization to file")
    
    args = parser.parse_args()
    
    # Analyze the file
    debug_data, is_high_value = analyze_debug_file(args.filepath)
    
    if debug_data is None:
        return
    
    # Create visualization if requested
    if args.visualize or args.save_vis:
        visualize_debug_data(debug_data, args.save_vis)
    
    if is_high_value:
        print(f"\n🚨 This is a HIGH VALUE case - needs investigation!")
    else:
        print(f"\n✅ This appears to be a normal case")

if __name__ == "__main__":
    main()