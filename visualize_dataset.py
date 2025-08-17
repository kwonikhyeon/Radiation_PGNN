#!/usr/bin/env python3
"""
Comprehensive dataset visualization script
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

def load_dataset(split='test'):
    """Load dataset split"""
    data_path = ROOT / "data" / f"{split}.npz"
    if not data_path.exists():
        print(f"Dataset {split}.npz not found at {data_path}")
        return None
    
    data = np.load(data_path)
    print(f"Loaded {split} dataset:")
    print(f"  Keys: {list(data.keys())}")
    
    if "inp" in data and "gt" in data:
        inp_data = data["inp"]  # (N, 6, H, W)
        gt_data = data["gt"]    # (N, 1, H, W)
        print(f"  Input shape: {inp_data.shape}")
        print(f"  GT shape: {gt_data.shape}")
        return inp_data, gt_data
    else:
        print(f"  Expected keys 'inp' and 'gt' not found")
        return None

def analyze_dataset_statistics(inp_data, gt_data, split_name):
    """Analyze dataset statistics"""
    print(f"\n=== {split_name.upper()} Dataset Statistics ===")
    
    N, C, H, W = inp_data.shape
    print(f"Number of samples: {N}")
    print(f"Input channels: {C}")
    print(f"Image size: {H}x{W}")
    
    # Channel analysis
    channel_names = ["measured_values", "mask", "log_measured", "coord_Y", "coord_X", "distance_map"]
    print(f"\nChannel analysis:")
    
    for c in range(C):
        ch_data = inp_data[:, c]
        print(f"  {c}: {channel_names[c] if c < len(channel_names) else f'Channel_{c}'}")
        print(f"      Range: {ch_data.min():.6f} to {ch_data.max():.6f}")
        print(f"      Mean: {ch_data.mean():.6f}, Std: {ch_data.std():.6f}")
        
        # Special analysis for mask channel
        if c == 1:  # mask channel
            total_pixels = N * H * W
            masked_pixels = (ch_data > 0).sum()
            mask_ratio = masked_pixels / total_pixels
            print(f"      Mask ratio: {mask_ratio:.6f} ({masked_pixels}/{total_pixels})")
            
            # Per-sample mask analysis
            masks_per_sample = []
            for i in range(N):
                mask_count = (inp_data[i, 1] > 0).sum()
                masks_per_sample.append(mask_count)
            
            print(f"      Measurements per sample: min={min(masks_per_sample)}, max={max(masks_per_sample)}, avg={np.mean(masks_per_sample):.1f}")
        
        # Special analysis for measured values
        if c == 0:  # measured_values channel
            non_zero_values = ch_data[ch_data > 0]
            if len(non_zero_values) > 0:
                print(f"      Non-zero values: {len(non_zero_values)}")
                print(f"      Non-zero range: {non_zero_values.min():.6f} to {non_zero_values.max():.6f}")
            else:
                print(f"      ⚠️  NO NON-ZERO MEASURED VALUES FOUND!")
    
    # GT analysis
    print(f"\nGround Truth analysis:")
    print(f"  Range: {gt_data.min():.6f} to {gt_data.max():.6f}")
    print(f"  Mean: {gt_data.mean():.6f}, Std: {gt_data.std():.6f}")
    
    # Per-sample GT max analysis
    gt_max_per_sample = []
    for i in range(N):
        gt_max = gt_data[i].max()
        gt_max_per_sample.append(gt_max)
    
    print(f"  Max per sample: min={min(gt_max_per_sample):.3f}, max={max(gt_max_per_sample):.3f}, avg={np.mean(gt_max_per_sample):.3f}")

def visualize_sample_detailed(inp_data, gt_data, sample_idx, split_name):
    """Create detailed visualization of a single sample"""
    if sample_idx >= len(inp_data):
        sample_idx = 0
    
    inp_sample = inp_data[sample_idx]  # (6, H, W)
    gt_sample = gt_data[sample_idx, 0]  # (H, W)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{split_name.title()} Dataset - Sample {sample_idx}', fontsize=16)
    
    # Channel names
    channel_names = ["Measured Values", "Mask", "Log Measured", "Coord Y", "Coord X", "Distance Map"]
    
    # Plot input channels
    for i in range(6):
        row = i // 4
        col = i % 4
        
        if i < 4:
            ax = axes[0, col]
        else:
            ax = axes[1, col-4]
        
        data = inp_sample[i]
        
        if i == 1:  # mask channel
            im = ax.imshow(data, cmap='gray', origin='upper')
            # Mark measured positions
            measured_y, measured_x = np.where(data > 0)
            ax.scatter(measured_x, measured_y, c='red', s=10, alpha=0.7)
            ax.set_title(f'{channel_names[i]} ({len(measured_y)} points)')
        else:
            im = ax.imshow(data, cmap='viridis' if i in [3, 4, 5] else 'hot', origin='upper')
            ax.set_title(f'{channel_names[i]}\nRange: [{data.min():.3f}, {data.max():.3f}]')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot ground truth
    ax_gt = axes[1, 2]
    im_gt = ax_gt.imshow(gt_sample, cmap='hot', origin='upper')
    ax_gt.set_title(f'Ground Truth\nRange: [{gt_sample.min():.3f}, {gt_sample.max():.3f}]')
    plt.colorbar(im_gt, ax=ax_gt, shrink=0.8)
    
    # Plot measurement vs GT comparison
    ax_comp = axes[1, 3]
    measured_values = inp_sample[0]
    mask = inp_sample[1]
    
    # Create comparison plot
    comparison = np.zeros_like(gt_sample)
    comparison = gt_sample.copy()
    
    # Highlight measured positions
    measured_y, measured_x = np.where(mask > 0)
    if len(measured_y) > 0:
        # Create overlay showing measured vs GT values
        overlay = ax_comp.imshow(comparison, cmap='hot', origin='upper', alpha=0.7)
        
        # Mark measured positions with their values
        for i in range(len(measured_y)):
            y, x = measured_y[i], measured_x[i]
            measured_val = measured_values[y, x]
            gt_val = gt_sample[y, x]
            
            # Color code: green if match, red if mismatch
            color = 'green' if abs(measured_val - gt_val) < 1e-6 else 'red'
            ax_comp.scatter(x, y, c=color, s=30, alpha=0.8)
            
            # Add text showing values
            if i < 10:  # Only show first 10 to avoid clutter
                ax_comp.text(x+2, y+2, f'M:{measured_val:.2f}\nG:{gt_val:.2f}', 
                           fontsize=6, color='white', weight='bold')
    
    ax_comp.set_title('Measured vs GT Comparison\n(Green=Match, Red=Mismatch)')
    plt.colorbar(overlay, ax=ax_comp, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'dataset_sample_{sample_idx}_{split_name}.png', dpi=150, bbox_inches='tight')
    print(f"Detailed sample visualization saved as 'dataset_sample_{sample_idx}_{split_name}.png'")

def visualize_dataset_overview(splits=['train', 'val', 'test']):
    """Create overview visualization of all dataset splits"""
    fig, axes = plt.subplots(len(splits), 6, figsize=(24, 4*len(splits)))
    if len(splits) == 1:
        axes = axes.reshape(1, -1)
    
    for split_idx, split in enumerate(splits):
        data_result = load_dataset(split)
        if data_result is None:
            continue
            
        inp_data, gt_data = data_result
        
        # Select a random sample for visualization
        sample_idx = np.random.randint(0, len(inp_data))
        inp_sample = inp_data[sample_idx]
        gt_sample = gt_data[sample_idx, 0]
        
        channel_names = ["Measured", "Mask", "Log Meas", "Coord Y", "Coord X", "Distance"]
        
        for ch in range(6):
            ax = axes[split_idx, ch] if len(splits) > 1 else axes[ch]
            
            if ch < inp_sample.shape[0]:
                data = inp_sample[ch]
                
                if ch == 1:  # mask
                    im = ax.imshow(data, cmap='gray', origin='upper')
                    measured_y, measured_x = np.where(data > 0)
                    ax.scatter(measured_x, measured_y, c='red', s=2, alpha=0.5)
                    ax.set_title(f'{split.upper()}: {channel_names[ch]}\n{len(measured_y)} points')
                else:
                    cmap = 'hot' if ch in [0, 2] else 'viridis'
                    im = ax.imshow(data, cmap=cmap, origin='upper')
                    ax.set_title(f'{split.upper()}: {channel_names[ch]}\n[{data.min():.2f}, {data.max():.2f}]')
            else:
                # Plot GT in the last available position
                im = ax.imshow(gt_sample, cmap='hot', origin='upper')
                ax.set_title(f'{split.upper()}: GT\n[{gt_sample.min():.2f}, {gt_sample.max():.2f}]')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=150, bbox_inches='tight')
    print("Dataset overview saved as 'dataset_overview.png'")

def analyze_measurement_distribution():
    """Analyze the distribution of measurements across all samples"""
    print(f"\n=== Measurement Distribution Analysis ===")
    
    all_measured_values = []
    all_gt_values_at_measured = []
    all_mask_counts = []
    
    for split in ['train', 'val', 'test']:
        data_result = load_dataset(split)
        if data_result is None:
            continue
            
        inp_data, gt_data = data_result
        print(f"\nAnalyzing {split} split...")
        
        problem_samples = 0
        
        for i in range(len(inp_data)):
            measured_values = inp_data[i, 0]  # measured values channel
            mask = inp_data[i, 1]            # mask channel
            gt_field = gt_data[i, 0]         # GT field
            
            # Find measured positions
            measured_y, measured_x = np.where(mask > 0)
            all_mask_counts.append(len(measured_y))
            
            if len(measured_y) > 0:
                # Extract measured values and corresponding GT values
                measured_vals = measured_values[measured_y, measured_x]
                gt_vals_at_measured = gt_field[measured_y, measured_x]
                
                all_measured_values.extend(measured_vals)
                all_gt_values_at_measured.extend(gt_vals_at_measured)
                
                # Check for mismatches
                mismatches = np.abs(measured_vals - gt_vals_at_measured) > 1e-6
                if np.any(mismatches):
                    problem_samples += 1
        
        print(f"  Problem samples (measured ≠ GT): {problem_samples}/{len(inp_data)} ({100*problem_samples/len(inp_data):.1f}%)")
    
    # Convert to numpy arrays
    all_measured_values = np.array(all_measured_values)
    all_gt_values_at_measured = np.array(all_gt_values_at_measured)
    all_mask_counts = np.array(all_mask_counts)
    
    print(f"\nOverall statistics:")
    print(f"  Total measurement points: {len(all_measured_values)}")
    print(f"  Measured values range: {all_measured_values.min():.6f} to {all_measured_values.max():.6f}")
    print(f"  GT values at measured positions: {all_gt_values_at_measured.min():.6f} to {all_gt_values_at_measured.max():.6f}")
    print(f"  Measurements per sample: {all_mask_counts.min()} to {all_mask_counts.max()} (avg: {all_mask_counts.mean():.1f})")
    
    # Check if all measured values are zero
    non_zero_measured = all_measured_values[all_measured_values > 0]
    print(f"  Non-zero measured values: {len(non_zero_measured)}/{len(all_measured_values)} ({100*len(non_zero_measured)/len(all_measured_values):.1f}%)")
    
    if len(non_zero_measured) == 0:
        print(f"  ⚠️  ALL MEASURED VALUES ARE ZERO! This is the root cause of the problem.")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Measured values distribution
    axes[0, 0].hist(all_measured_values, bins=50, alpha=0.7, label='Measured Values')
    axes[0, 0].set_title('Distribution of Measured Values')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # GT values at measured positions
    axes[0, 1].hist(all_gt_values_at_measured, bins=50, alpha=0.7, label='GT at Measured Pos', color='orange')
    axes[0, 1].set_title('Distribution of GT Values at Measured Positions')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Measurements per sample
    axes[1, 0].hist(all_mask_counts, bins=30, alpha=0.7, label='Measurements per Sample', color='green')
    axes[1, 0].set_title('Distribution of Measurements per Sample')
    axes[1, 0].set_xlabel('Number of Measurements')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Measured vs GT scatter plot
    if len(all_measured_values) > 0:
        axes[1, 1].scatter(all_gt_values_at_measured, all_measured_values, alpha=0.5, s=1)
        axes[1, 1].plot([0, max(all_gt_values_at_measured.max(), all_measured_values.max())], 
                       [0, max(all_gt_values_at_measured.max(), all_measured_values.max())], 
                       'r--', label='Perfect Match')
        axes[1, 1].set_xlabel('GT Value')
        axes[1, 1].set_ylabel('Measured Value')
        axes[1, 1].set_title('Measured vs GT Values')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('measurement_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Measurement distribution analysis saved as 'measurement_distribution_analysis.png'")

def main():
    """Main visualization function"""
    print("Starting comprehensive dataset visualization...")
    print("=" * 60)
    
    # Check available datasets
    data_dir = ROOT / "data"
    available_splits = []
    for split in ['train', 'val', 'test']:
        if (data_dir / f"{split}.npz").exists():
            available_splits.append(split)
    
    print(f"Available dataset splits: {available_splits}")
    
    if not available_splits:
        print("No dataset files found! Please generate datasets first.")
        return
    
    # Analyze each split
    for split in available_splits:
        data_result = load_dataset(split)
        if data_result is not None:
            inp_data, gt_data = data_result
            analyze_dataset_statistics(inp_data, gt_data, split)
            
            # Create detailed visualization for a few samples
            for sample_idx in [0, 1, len(inp_data)//2]:
                if sample_idx < len(inp_data):
                    visualize_sample_detailed(inp_data, gt_data, sample_idx, split)
    
    # Create overview visualization
    visualize_dataset_overview(available_splits)
    
    # Analyze measurement distribution
    analyze_measurement_distribution()
    
    print("\n" + "=" * 60)
    print("Dataset visualization completed!")
    print("Generated files:")
    print("  - dataset_overview.png")
    print("  - dataset_sample_*_*.png")
    print("  - measurement_distribution_analysis.png")

if __name__ == "__main__":
    main()