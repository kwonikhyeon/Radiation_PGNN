#!/usr/bin/env python3
"""
Analyze traversability layer high value issues by running multiple test cases
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from traversability_layer import calculate_traversability_layer, TraversabilityParameters

def generate_test_field(field_type="random", size=256):
    """Generate different types of test radiation fields"""
    
    if field_type == "random":
        # Random field with some sources
        field = np.random.beta(2, 5, (size, size)).astype(np.float32)
        field = np.clip(field, 0, 1)
        
        # Add random sources
        n_sources = np.random.randint(1, 4)
        Y, X = np.ogrid[:size, :size]
        
        for _ in range(n_sources):
            sy = np.random.randint(30, size-30)
            sx = np.random.randint(30, size-30)
            intensity = np.random.uniform(0.3, 0.8)
            sigma = np.random.uniform(10, 25)
            source = intensity * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * sigma**2))
            field += source
            
        field = np.clip(field, 0, 1)
        
    elif field_type == "low_background":
        # Very low background with weak sources
        field = np.full((size, size), 0.05, dtype=np.float32)
        Y, X = np.ogrid[:size, :size]
        
        # Add weak sources
        weak_sources = [(80, 80), (180, 120), (120, 200)]
        for sy, sx in weak_sources:
            if sy < size and sx < size:
                weak_source = 0.25 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 15**2))
                field += weak_source
                
        field = np.clip(field, 0, 1)
        
    elif field_type == "high_background":
        # High background radiation
        field = np.random.normal(0.4, 0.1, (size, size)).astype(np.float32)
        field = np.clip(field, 0, 1)
        
        # Add sources on top
        Y, X = np.ogrid[:size, :size]
        sources = [(100, 100), (150, 180)]
        for sy, sx in sources:
            source = 0.4 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 20**2))
            field += source
            
        field = np.clip(field, 0, 1)
        
    elif field_type == "single_large_source":
        # Single large source
        field = np.random.normal(0.02, 0.01, (size, size)).astype(np.float32)
        field = np.clip(field, 0, 1)
        
        Y, X = np.ogrid[:size, :size]
        sy, sx = size//2, size//2
        large_source = 0.9 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 40**2))
        field += large_source
        field = np.clip(field, 0, 1)
        
    elif field_type == "multiple_close_sources":
        # Multiple sources close together
        field = np.random.normal(0.03, 0.01, (size, size)).astype(np.float32)
        field = np.clip(field, 0, 1)
        
        Y, X = np.ogrid[:size, :size]
        close_sources = [(120, 120), (130, 125), (125, 135), (115, 130)]
        for sy, sx in close_sources:
            source = 0.6 * np.exp(-((Y - sy)**2 + (X - sx)**2) / (2 * 12**2))
            field += source
            
        field = np.clip(field, 0, 1)
        
    return field

def analyze_traversability_batch():
    """Run multiple traversability calculations and analyze patterns"""
    
    print("🔬 Analyzing Traversability Layer High Value Issues...")
    
    test_cases = [
        "random",
        "low_background", 
        "high_background",
        "single_large_source",
        "multiple_close_sources"
    ]
    
    high_value_cases = []
    total_cases = 0
    
    # Test each case type multiple times
    for case_type in test_cases:
        print(f"\n📊 Testing {case_type} scenarios...")
        
        case_high_values = []
        
        for i in range(10):  # 10 tests per case type
            field = generate_test_field(case_type)
            
            # Random robot position
            robot_pos = (np.random.randint(50, 206), np.random.randint(50, 206))
            robot_heading = np.random.uniform(0, 2*np.pi)
            
            try:
                result = calculate_traversability_layer(field, None, robot_pos, robot_heading)
                
                total_trav = result['total_traversability']
                mean_val = total_trav.mean()
                max_val = total_trav.max()
                min_val = total_trav.min()
                
                total_cases += 1
                
                # Check for high values
                if mean_val > 0.7:
                    case_info = {
                        'case_type': case_type,
                        'test_id': i,
                        'mean': mean_val,
                        'max': max_val,
                        'min': min_val,
                        'robot_pos': robot_pos,
                        'robot_heading': robot_heading,
                        'field_stats': {
                            'field_mean': field.mean(),
                            'field_max': field.max(),
                            'field_min': field.min()
                        },
                        'sources_detected': len(result.get('source_locations', [])),
                        'components': {
                            'proximity_mean': result.get('source_proximity', np.zeros_like(field)).mean(),
                            'low_rad_mean': result.get('low_radiation_bonus', np.zeros_like(field)).mean(),
                            'directional_mean': result.get('directional_navigation', np.zeros_like(field)).mean() if 'directional_navigation' in result else 0.0
                        },
                        'field': field,
                        'result': result
                    }
                    high_value_cases.append(case_info)
                    case_high_values.append(mean_val)
                    
                    print(f"  🚨 HIGH VALUES: Test {i} - mean={mean_val:.3f}, max={max_val:.3f}")
                else:
                    print(f"  ✅ Normal: Test {i} - mean={mean_val:.3f}, max={max_val:.3f}")
                    
            except Exception as e:
                print(f"  ❌ Error in test {i}: {e}")
        
        if case_high_values:
            avg_high = np.mean(case_high_values)
            print(f"  📈 {case_type}: {len(case_high_values)}/10 cases with high values (avg: {avg_high:.3f})")
        else:
            print(f"  ✅ {case_type}: No high value cases")
    
    # Analysis summary
    high_value_rate = len(high_value_cases) / total_cases if total_cases > 0 else 0
    print(f"\n📊 Overall Analysis:")
    print(f"  Total cases tested: {total_cases}")
    print(f"  High value cases: {len(high_value_cases)}")
    print(f"  High value rate: {high_value_rate:.1%}")
    
    if high_value_cases:
        print(f"\n🔍 High Value Case Analysis:")
        
        # Group by case type
        by_type = {}
        for case in high_value_cases:
            case_type = case['case_type']
            if case_type not in by_type:
                by_type[case_type] = []
            by_type[case_type].append(case)
        
        for case_type, cases in by_type.items():
            print(f"\n  {case_type}: {len(cases)} cases")
            
            # Component analysis
            prox_means = [c['components']['proximity_mean'] for c in cases]
            low_rad_means = [c['components']['low_rad_mean'] for c in cases]
            dir_means = [c['components']['directional_mean'] for c in cases]
            
            print(f"    Avg proximity component: {np.mean(prox_means):.3f}")
            print(f"    Avg low radiation component: {np.mean(low_rad_means):.3f}")
            print(f"    Avg directional component: {np.mean(dir_means):.3f}")
            
            # Check which component is dominating
            if np.mean(prox_means) > 0.4:
                print(f"    🟢 Proximity component likely causing saturation")
            if np.mean(low_rad_means) > 0.4:
                print(f"    🟡 Low radiation component likely causing saturation")
            if np.mean(dir_means) > 0.4:
                print(f"    🔵 Directional component likely causing saturation")
        
        # Visualize the worst cases
        print(f"\n🖼️  Creating visualization of worst cases...")
        visualize_worst_cases(high_value_cases[:6])  # Show top 6 worst cases
    
    return high_value_cases, high_value_rate

def visualize_worst_cases(cases):
    """Visualize the worst high-value cases"""
    
    n_cases = min(len(cases), 6)
    if n_cases == 0:
        return
        
    fig, axes = plt.subplots(n_cases, 4, figsize=(16, 4*n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    for i, case in enumerate(cases[:n_cases]):
        field = case['field']
        result = case['result']
        robot_pos = case['robot_pos']
        
        # Original field
        im1 = axes[i, 0].imshow(field, cmap='hot', origin='lower')
        axes[i, 0].scatter(robot_pos[1], robot_pos[0], c='blue', s=50, marker='o')
        axes[i, 0].set_title(f'{case["case_type"]}\nField (mean: {case["field_stats"]["field_mean"]:.3f})')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # Total traversability
        im2 = axes[i, 1].imshow(result['total_traversability'], cmap='plasma', origin='lower')
        axes[i, 1].scatter(robot_pos[1], robot_pos[0], c='blue', s=50, marker='o')
        axes[i, 1].set_title(f'Total Traversability\n(mean: {case["mean"]:.3f})')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        # Component analysis
        if 'directional_navigation' in result:
            im3 = axes[i, 2].imshow(result['directional_navigation'], cmap='viridis', origin='lower')
            axes[i, 2].set_title(f'Directional Nav\n(mean: {case["components"]["directional_mean"]:.3f})')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
        else:
            axes[i, 2].text(0.5, 0.5, 'No Directional\nNavigation', ha='center', va='center', transform=axes[i, 2].transAxes)
            axes[i, 2].set_title('Directional Nav\n(Not Available)')
        
        # Low radiation bonus
        im4 = axes[i, 3].imshow(result['low_radiation_bonus'], cmap='Blues', origin='lower')
        axes[i, 3].set_title(f'Low Rad Bonus\n(mean: {case["components"]["low_rad_mean"]:.3f})')
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.046)
        
        # Add case info
        info_text = f"Type: {case['case_type']}, Sources: {case['sources_detected']}, Robot: {robot_pos}"
        fig.text(0.02, 0.98 - i/n_cases, info_text, fontsize=8, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('traversability_high_value_analysis.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved analysis to 'traversability_high_value_analysis.png'")

if __name__ == "__main__":
    high_value_cases, rate = analyze_traversability_batch()
    print(f"\n🏁 Analysis Complete! High value rate: {rate:.1%}")
    
    if rate > 0.1:  # If more than 10% of cases have high values
        print(f"🚨 Significant high value issue detected - needs fixing!")
    else:
        print(f"✅ High value rate is acceptable")