#!/usr/bin/env python3
"""
Simple test for the new information gain layer functionality
"""

import numpy as np
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "test" / "modules"))

from information_gain import calculate_information_gain_layer

def test_simple_case():
    """Test with simple synthetic data"""
    print("🧪 Testing Information Gain Layer with simple case...")
    
    # Create a simple Gaussian-like field
    x = np.linspace(-3, 3, 256)
    y = np.linspace(-3, 3, 256)
    X, Y = np.meshgrid(x, y)
    
    # Single prediction: Gaussian with some noise
    prediction = np.exp(-(X**2 + Y**2)/2) * 0.8
    prediction = np.clip(prediction, 0, 1).astype(np.float32)
    
    # Create measurement mask at strategic locations
    measurement_mask = np.zeros((256, 256), dtype=np.uint8)
    measurement_mask[128, 128] = 1  # Center
    measurement_mask[100, 100] = 1  # Off-center
    measurement_mask[150, 150] = 1  # Off-center
    measurement_mask[180, 128] = 1  # Edge area (high gradient expected)
    
    print(f"📊 Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"📍 Measurement points: {measurement_mask.sum()}")
    
    # Calculate information gain layer
    result = calculate_information_gain_layer(
        single_prediction=prediction,
        measurement_mask=measurement_mask
    )
    
    # Print results
    print("\n✅ Results:")
    print(f"  Information Gain range: [{result['information_gain'].min():.4f}, {result['information_gain'].max():.4f}]")
    print(f"  Uncertainty component range: [{result['uncertainty_component'].min():.4f}, {result['uncertainty_component'].max():.4f}]")
    print(f"  Gradient component range: [{result['gradient_component'].min():.4f}, {result['gradient_component'].max():.4f}]")
    print(f"  Excluded points: {result['metadata']['exclusion_info']['excluded_points']}")
    
    # Check gradient makes sense - should be high near Gaussian edge
    gradient_center = result['gradient_component'][128, 128]  # Center (low gradient expected)
    gradient_edge = result['gradient_component'][180, 128]    # Edge area (high gradient expected)
    
    print(f"  Gradient at center: {gradient_center:.4f}")
    print(f"  Gradient at edge (before exclusion): {gradient_edge:.4f}")
    print(f"  Edge/Center gradient ratio: {gradient_edge/max(gradient_center, 0.001):.2f}")
    
    # Check exclusion effectiveness
    measurement_points = np.where(measurement_mask > 0)
    gain_at_measurements = result['information_gain'][measurement_points]
    gain_before_exclusion = (result['uncertainty_component'] + result['gradient_component'])[measurement_points]
    
    print(f"\n🚫 Exclusion Effectiveness:")
    print(f"  Info gain at measurements (before exclusion): [{gain_before_exclusion.min():.4f}, {gain_before_exclusion.max():.4f}]")
    print(f"  Info gain at measurements (after exclusion): [{gain_at_measurements.min():.4f}, {gain_at_measurements.max():.4f}]")
    print(f"  Average reduction: {(1 - gain_at_measurements.mean()/max(gain_before_exclusion.mean(), 0.001))*100:.1f}%")
    
    print("\n🎯 Information Gain Layer Test Complete!")
    return result

def test_multiple_predictions():
    """Test with multiple predictions for uncertainty calculation"""
    print("\n🧪 Testing with Multiple Predictions...")
    
    # Create base Gaussian field  
    x = np.linspace(-3, 3, 256)
    y = np.linspace(-3, 3, 256)
    X, Y = np.meshgrid(x, y)
    base_field = np.exp(-(X**2 + Y**2)/2) * 0.8
    
    # Create multiple predictions with small variations
    n_predictions = 10
    predictions = np.stack([
        np.clip(base_field + np.random.normal(0, 0.05, base_field.shape), 0, 1)
        for _ in range(n_predictions)
    ]).astype(np.float32)
    
    # Create measurement mask
    measurement_mask = np.zeros((256, 256), dtype=np.uint8) 
    for i in range(8):
        y = np.random.randint(50, 206)
        x = np.random.randint(50, 206)
        measurement_mask[y, x] = 1
    
    print(f"📊 Multiple predictions shape: {predictions.shape}")
    print(f"📍 Measurement points: {measurement_mask.sum()}")
    
    # Calculate information gain layer
    result = calculate_information_gain_layer(
        predictions=predictions,
        measurement_mask=measurement_mask
    )
    
    print("\n✅ Results with Multiple Predictions:")
    print(f"  Information Gain range: [{result['information_gain'].min():.4f}, {result['information_gain'].max():.4f}]")
    print(f"  Model uncertainty available: {result['model_uncertainty'].max() > 0}")
    print(f"  Gradient component range: [{result['gradient_component'].min():.4f}, {result['gradient_component'].max():.4f}]")
    
    print("\n🎯 Multiple Predictions Test Complete!")
    return result

if __name__ == "__main__":
    # Test both cases
    result1 = test_simple_case()
    result2 = test_multiple_predictions()
    
    print(f"\n🏁 All Information Gain Tests Completed Successfully!")
    
    # Optionally save results for inspection
    # np.savez('info_gain_test_results.npz', 
    #          simple_case=result1['information_gain'],
    #          multiple_predictions=result2['information_gain'])