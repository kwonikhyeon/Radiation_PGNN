#!/usr/bin/env python3
"""
Test the fixed data generation pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

import dataset.generate_truth as gt
import dataset.trajectory_sampler as ts
from dataset.dataset_generator import _make_single_sample

def test_trajectory_sampler():
    """Test the fixed trajectory sampler"""
    print("Testing fixed trajectory_sampler...")
    
    # Generate test field
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n_src = rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1)
    c, a, s = gt.sample_sources(gt.GRID, n_src, rng=rng)
    field = gt.gaussian_field(gt.GRID, c, a, s)
    
    # Normalize field
    field = np.clip(field, 0, None)
    field_99th = np.percentile(field[field > 0], 99) if np.any(field > 0) else 1.0
    normalization_factor = max(field_99th, 1.0)
    field = np.clip(field / normalization_factor, 0, 1.0)
    
    print(f"  Generated field: range=[{field.min():.6f}, {field.max():.6f}]")
    
    # Generate waypoints and measurements
    waypoints = ts.generate_waypoints(rng=rng)
    meas, mask = ts.sparse_from_waypoints(field, waypoints, rng=rng)
    
    print(f"  Generated {len(waypoints)} waypoints")
    print(f"  Measurements: range=[{meas.min():.6f}, {meas.max():.6f}]")
    print(f"  Mask: {mask.sum()} measured points")
    
    # Check consistency
    measured_positions = np.where(mask > 0)
    consistency_errors = 0
    max_error = 0.0
    
    for i in range(len(measured_positions[0])):
        y, x = measured_positions[0][i], measured_positions[1][i]
        field_val = field[y, x]
        meas_val = meas[y, x]
        error = abs(field_val - meas_val)
        max_error = max(max_error, error)
        
        if error > 1e-6:
            consistency_errors += 1
    
    print(f"  Consistency check: {consistency_errors}/{len(measured_positions[0])} errors")
    print(f"  Max error: {max_error:.6f}")
    
    if consistency_errors == 0:
        print("  ✅ All measured values match field values!")
    else:
        print("  ❌ Data inconsistency detected!")
    
    return field, waypoints, meas, mask

def test_dataset_generator():
    """Test the fixed dataset generator"""
    print("\nTesting fixed dataset_generator...")
    
    rng = np.random.default_rng(42)
    
    # Generate multiple samples
    consistent_samples = 0
    total_samples = 10
    
    for i in range(total_samples):
        field, inp, mask = _make_single_sample(rng)
        
        # Extract channels
        meas = inp[0]  # measured values
        mask_ch = inp[1]  # mask
        
        # Check consistency
        measured_positions = np.where(mask_ch > 0)
        sample_errors = 0
        
        for j in range(len(measured_positions[0])):
            y, x = measured_positions[0][j], measured_positions[1][j]
            field_val = field[y, x]
            meas_val = meas[y, x]
            
            if abs(field_val - meas_val) > 1e-6:
                sample_errors += 1
        
        if sample_errors == 0:
            consistent_samples += 1
        
        if i < 3:  # Print details for first 3 samples
            print(f"  Sample {i+1}:")
            print(f"    Field range: [{field.min():.6f}, {field.max():.6f}]")
            print(f"    Measured range: [{meas.min():.6f}, {meas.max():.6f}]")
            print(f"    Measured points: {len(measured_positions[0])}")
            print(f"    Consistency errors: {sample_errors}")
    
    print(f"\nConsistency summary: {consistent_samples}/{total_samples} samples are consistent")
    
    if consistent_samples == total_samples:
        print("✅ All samples have consistent data!")
        return True
    else:
        print("❌ Some samples still have inconsistencies!")
        return False

def visualize_fixed_sample():
    """Create visualization of a fixed sample"""
    print("\nCreating visualization of fixed sample...")
    
    rng = np.random.default_rng(42)
    field, inp, mask = _make_single_sample(rng)
    
    # Extract channels
    meas = inp[0]  # measured values
    mask_ch = inp[1]  # mask
    log_meas = inp[2]  # log measured
    coord_y = inp[3]  # Y coordinates
    coord_x = inp[4]  # X coordinates
    distance_map = inp[5]  # distance map
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Fixed Data Generation Pipeline - Sample Visualization', fontsize=16)
    
    # Ground truth field
    im1 = axes[0, 0].imshow(field, cmap='hot', origin='upper')
    axes[0, 0].set_title(f'Ground Truth Field\nRange: [{field.min():.3f}, {field.max():.3f}]')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Measured values
    im2 = axes[0, 1].imshow(meas, cmap='hot', origin='upper')
    axes[0, 1].set_title(f'Measured Values\nRange: [{meas.min():.3f}, {meas.max():.3f}]')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Mask
    axes[0, 2].imshow(mask_ch, cmap='gray', origin='upper')
    axes[0, 2].set_title(f'Measurement Mask\n{mask_ch.sum():.0f} points')
    
    # Overlay: GT with measured points
    axes[0, 3].imshow(field, cmap='hot', origin='upper', alpha=0.7)
    measured_y, measured_x = np.where(mask_ch > 0)
    axes[0, 3].scatter(measured_x, measured_y, c='blue', s=20, alpha=0.8)
    axes[0, 3].set_title('GT + Measured Points')
    
    # Log measured
    im3 = axes[1, 0].imshow(log_meas, cmap='hot', origin='upper')
    axes[1, 0].set_title(f'Log Measured\nRange: [{log_meas.min():.3f}, {log_meas.max():.3f}]')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Coordinates
    im4 = axes[1, 1].imshow(coord_y, cmap='viridis', origin='upper')
    axes[1, 1].set_title('Y Coordinates')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    im5 = axes[1, 2].imshow(coord_x, cmap='viridis', origin='upper')
    axes[1, 2].set_title('X Coordinates')
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)
    
    # Distance map
    im6 = axes[1, 3].imshow(distance_map, cmap='viridis', origin='upper')
    axes[1, 3].set_title('Distance Map')
    plt.colorbar(im6, ax=axes[1, 3], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('fixed_data_pipeline_test.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'fixed_data_pipeline_test.png'")
    
    # Check data consistency for this sample
    print("\nData consistency check for visualized sample:")
    measured_positions = np.where(mask_ch > 0)
    for i in range(min(5, len(measured_positions[0]))):
        y, x = measured_positions[0][i], measured_positions[1][i]
        field_val = field[y, x]
        meas_val = meas[y, x]
        error = abs(field_val - meas_val)
        print(f"  Point {i+1} at ({y:3d},{x:3d}): field={field_val:.6f}, meas={meas_val:.6f}, error={error:.6f}")

def compare_old_vs_new():
    """Compare the distribution characteristics of old vs new data generation"""
    print("\nComparing data generation characteristics...")
    
    rng = np.random.default_rng(42)
    
    # Generate multiple samples with new method
    all_measured_values = []
    all_gt_values_at_measured = []
    inconsistency_count = 0
    total_points = 0
    
    for i in range(20):  # Test 20 samples
        field, inp, mask = _make_single_sample(rng)
        meas = inp[0]
        mask_ch = inp[1]
        
        measured_positions = np.where(mask_ch > 0)
        total_points += len(measured_positions[0])
        
        for j in range(len(measured_positions[0])):
            y, x = measured_positions[0][j], measured_positions[1][j]
            field_val = field[y, x]
            meas_val = meas[y, x]
            
            all_measured_values.append(meas_val)
            all_gt_values_at_measured.append(field_val)
            
            if abs(field_val - meas_val) > 1e-6:
                inconsistency_count += 1
    
    all_measured_values = np.array(all_measured_values)
    all_gt_values_at_measured = np.array(all_gt_values_at_measured)
    
    print(f"New data generation statistics:")
    print(f"  Total measurement points: {total_points}")
    print(f"  Inconsistent points: {inconsistency_count} ({100*inconsistency_count/total_points:.1f}%)")
    print(f"  Measured values range: [{all_measured_values.min():.6f}, {all_measured_values.max():.6f}]")
    print(f"  GT values at measured pos range: [{all_gt_values_at_measured.min():.6f}, {all_gt_values_at_measured.max():.6f}]")
    print(f"  Non-zero measured values: {(all_measured_values > 0).sum()}/{len(all_measured_values)} ({100*(all_measured_values > 0).sum()/len(all_measured_values):.1f}%)")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution comparison
    axes[0].hist(all_measured_values, bins=50, alpha=0.7, label='Measured Values', density=True)
    axes[0].hist(all_gt_values_at_measured, bins=50, alpha=0.7, label='GT at Measured Pos', density=True)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Value Distributions (Fixed Pipeline)')
    axes[0].legend()
    
    # Scatter plot
    axes[1].scatter(all_gt_values_at_measured, all_measured_values, alpha=0.5, s=1)
    max_val = max(all_gt_values_at_measured.max(), all_measured_values.max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Match')
    axes[1].set_xlabel('GT Value')
    axes[1].set_ylabel('Measured Value')
    axes[1].set_title('Measured vs GT Values')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('fixed_pipeline_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved as 'fixed_pipeline_comparison.png'")

def main():
    """Main test function"""
    print("Testing Fixed Data Generation Pipeline")
    print("=" * 60)
    
    # Test individual components
    test_trajectory_sampler()
    
    # Test complete pipeline
    pipeline_ok = test_dataset_generator()
    
    # Create visualizations
    visualize_fixed_sample()
    compare_old_vs_new()
    
    print("\n" + "=" * 60)
    if pipeline_ok:
        print("✅ Data generation pipeline is FIXED and ready for dataset regeneration!")
        print("\nTo regenerate the dataset with fixed data, run:")
        print("python3 -m src.dataset.dataset_generator --n_train 1000 --n_val 100 --n_test 50")
    else:
        print("❌ Pipeline still has issues that need to be resolved.")
    
    return pipeline_ok

if __name__ == "__main__":
    main()