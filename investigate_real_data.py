#!/usr/bin/env python3
"""
Investigate the high value issue with real evaluation data
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from model.conv_next_pgnn import ConvNeXtUNetWithMaskAttention as ConvNeXtUNet

def load_real_test_data():
    """Load actual test data to investigate the issue"""
    try:
        # Load test data
        data_path = ROOT / "data" / "test.npz"
        if not data_path.exists():
            print(f"Test data not found at {data_path}")
            return None
            
        data = np.load(data_path)
        print(f"Loaded test data with keys: {list(data.keys())}")
        
        if "inp" in data:
            inp_data = data["inp"]  # Shape: (N, 6, H, W)
            gt_data = data["gt"]    # Shape: (N, 1, H, W)
        else:
            print("Expected 'inp' key not found in test data")
            return None
            
        print(f"Input data shape: {inp_data.shape}")
        print(f"GT data shape: {gt_data.shape}")
        
        return inp_data, gt_data
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def investigate_sample(sample_idx=0):
    """Investigate a specific sample from real test data"""
    data_result = load_real_test_data()
    if data_result is None:
        print("Cannot load real test data, using synthetic data instead")
        return investigate_synthetic_issue()
        
    inp_data, gt_data = data_result
    
    # Select a sample
    if sample_idx >= len(inp_data):
        sample_idx = 0
        
    inp_sample = torch.from_numpy(inp_data[sample_idx]).unsqueeze(0).float()
    gt_sample = torch.from_numpy(gt_data[sample_idx]).unsqueeze(0).float()
    
    print(f"\n=== Investigating Real Sample {sample_idx} ===")
    
    # Extract channels
    measured_values = inp_sample[0, 0]  # [H, W]
    mask = inp_sample[0, 1]            # [H, W]
    gt_field = gt_sample[0, 0]         # [H, W]
    
    print(f"Input measured values range: {measured_values.min():.3f} to {measured_values.max():.3f}")
    print(f"Mask sum: {mask.sum()}")
    print(f"GT field range: {gt_field.min():.3f} to {gt_field.max():.3f}")
    
    # Find measured positions
    measured_positions = torch.where(mask > 0)
    if len(measured_positions[0]) > 0:
        print(f"Number of measured points: {len(measured_positions[0])}")
        
        # Show first few measured points
        for i in range(min(5, len(measured_positions[0]))):
            y, x = measured_positions[0][i], measured_positions[1][i]
            input_val = measured_values[y, x].item()
            gt_val = gt_field[y, x].item()
            print(f"  Point {i+1} at ({y:3d},{x:3d}): input={input_val:.3f}, gt={gt_val:.3f}")
            
            # Check if input and GT match at measured positions
            if abs(input_val - gt_val) > 1e-6:
                print(f"    ⚠️  INPUT-GT MISMATCH! This could be the issue!")
            else:
                print(f"    ✅ Input matches GT")
    
    # Test model prediction
    model = ConvNeXtUNet(in_channels=6)
    model.eval()
    
    with torch.no_grad():
        pred = model(inp_sample)
    
    pred_field = pred[0, 0]
    print(f"Prediction range: {pred_field.min():.3f} to {pred_field.max():.3f}")
    
    # Check predicted values at measured positions
    print(f"\nPredicted values at measured positions:")
    for i in range(min(5, len(measured_positions[0]))):
        y, x = measured_positions[0][i], measured_positions[1][i]
        input_val = measured_values[y, x].item()
        pred_val = pred_field[y, x].item()
        gt_val = gt_field[y, x].item()
        
        print(f"  Point {i+1}: input={input_val:.3f}, pred={pred_val:.3f}, gt={gt_val:.3f}")
        
        if abs(pred_val - input_val) > 1e-6:
            print(f"    ❌ PREDICTION DOESN'T MATCH INPUT! Pred={pred_val:.3f}, Input={input_val:.3f}")
            print(f"    This suggests the preservation mechanism failed!")
        else:
            print(f"    ✅ Prediction matches input")
    
    # Visualize the sample
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Input measurements
    im1 = axes[0].imshow(measured_values.numpy(), cmap='hot', origin='upper')
    axes[0].set_title(f'Input Measurements (Sample {sample_idx})')
    axes[0].scatter(measured_positions[1][:10], measured_positions[0][:10], c='blue', s=20, alpha=0.7)
    plt.colorbar(im1, ax=axes[0])
    
    # Ground truth
    im2 = axes[1].imshow(gt_field.numpy(), cmap='hot', origin='upper')
    axes[1].set_title('Ground Truth')
    axes[1].scatter(measured_positions[1][:10], measured_positions[0][:10], c='blue', s=20, alpha=0.7)
    plt.colorbar(im2, ax=axes[1])
    
    # Prediction
    im3 = axes[2].imshow(pred_field.numpy(), cmap='hot', origin='upper')
    axes[2].set_title('Model Prediction')
    axes[2].scatter(measured_positions[1][:10], measured_positions[0][:10], c='blue', s=20, alpha=0.7)
    plt.colorbar(im3, ax=axes[2])
    
    # Difference (Prediction - GT)
    diff = (pred_field - gt_field).numpy()
    im4 = axes[3].imshow(diff, cmap='RdBu_r', origin='upper', 
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[3].set_title('Prediction - GT')
    axes[3].scatter(measured_positions[1][:10], measured_positions[0][:10], c='green', s=20, alpha=0.7)
    plt.colorbar(im4, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(f'real_sample_{sample_idx}_investigation.png', dpi=150, bbox_inches='tight')
    print(f"\nReal sample investigation saved as 'real_sample_{sample_idx}_investigation.png'")
    
    return inp_sample, gt_sample, pred

def investigate_data_preprocessing():
    """Investigate if the issue is in data preprocessing"""
    print(f"\n=== Investigating Data Preprocessing ===")
    
    # Check dataset_generator.py processing
    data_result = load_real_test_data()
    if data_result is None:
        return
        
    inp_data, gt_data = data_result
    
    # Check a few samples for consistency
    for i in range(min(3, len(inp_data))):
        inp_sample = inp_data[i]  # Shape: (6, H, W)
        gt_sample = gt_data[i]    # Shape: (1, H, W)
        
        measured_values = inp_sample[0]  # [H, W]
        mask = inp_sample[1]            # [H, W]
        gt_field = gt_sample[0]         # [H, W]
        
        # Find measured positions
        measured_y, measured_x = np.where(mask > 0)
        
        print(f"\nSample {i}:")
        print(f"  Measured points: {len(measured_y)}")
        
        if len(measured_y) > 0:
            # Check if measured values match GT at those positions
            for j in range(min(3, len(measured_y))):
                y, x = measured_y[j], measured_x[j]
                input_val = measured_values[y, x]
                gt_val = gt_field[y, x]
                
                print(f"    Point {j+1} at ({y:3d},{x:3d}): input={input_val:.3f}, gt={gt_val:.3f}")
                
                if abs(input_val - gt_val) > 1e-6:
                    print(f"      ❌ DATA INCONSISTENCY! Input and GT don't match at measured position!")
                    print(f"      This is likely the root cause of the high value issue!")
                else:
                    print(f"      ✅ Input matches GT at measured position")

def investigate_synthetic_issue():
    """Investigate with synthetic data that might show the issue"""
    print(f"\n=== Creating Synthetic Test Case ===")
    
    # Create a case where input might have higher values than intended
    H, W = 128, 128
    
    # Simulate potential data corruption
    measured_values = np.zeros((H, W))
    mask = np.zeros((H, W))
    
    # Add some normal measurements
    normal_points = [(30, 30, 0.3), (70, 70, 0.6)]
    for y, x, val in normal_points:
        measured_values[y, x] = val
        mask[y, x] = 1.0
    
    # Add some potentially problematic high measurements
    high_points = [(50, 50, 2.5), (90, 90, 3.0)]  # Very high values
    for y, x, val in high_points:
        measured_values[y, x] = val
        mask[y, x] = 1.0
    
    print(f"Created synthetic data with high measured values:")
    all_points = normal_points + high_points
    for i, (y, x, val) in enumerate(all_points):
        print(f"  Point {i+1} at ({y:3d},{x:3d}): {val:.3f}")
    
    # Create other channels
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys_norm = (ys / (H - 1) * 2 - 1).astype(np.float32)
    xs_norm = (xs / (W - 1) * 2 - 1).astype(np.float32)
    log_measured = np.log1p(measured_values)
    
    from scipy.ndimage import distance_transform_edt
    raw_dist = distance_transform_edt(1 - mask).astype(np.float32)
    raw_dist[mask == 1] = 0.0
    distance_map = raw_dist / (H + 1e-6)
    
    # Stack channels
    inp = np.stack([measured_values, mask, log_measured, ys_norm, xs_norm, distance_map], axis=0)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
    
    # Test model
    model = ConvNeXtUNet(in_channels=6)
    model.eval()
    
    with torch.no_grad():
        pred = model(inp_tensor)
    
    # Check results
    print(f"\nModel prediction results:")
    print(f"  Prediction range: {pred.min():.3f} to {pred.max():.3f}")
    
    for i, (y, x, expected_val) in enumerate(all_points):
        pred_val = pred[0, 0, y, x].item()
        print(f"  Point {i+1}: expected={expected_val:.3f}, predicted={pred_val:.3f}")
        
        if abs(pred_val - expected_val) > 1e-6:
            print(f"    ❌ Value not preserved! This could be the issue.")
        else:
            print(f"    ✅ Value correctly preserved")

if __name__ == "__main__":
    print("Investigating high values at measured positions...")
    print("=" * 60)
    
    investigate_sample(0)
    investigate_data_preprocessing()
    investigate_synthetic_issue()
    
    print("\n" + "=" * 60)
    print("Investigation completed!")