#!/usr/bin/env python3
"""
Sample Visualization Generator for Comparative Analysis
=======================================================

This script creates sample plots based on the analysis data to demonstrate
the capabilities of the comparative analysis tool.

Usage:
    python3 view_analysis_sample.py

This will generate sample plots showing:
1. Key metrics comparison
2. Performance radar chart
3. Distribution analysis

Author: Claude Code Analysis System
Date: 2025-09-04
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Sample data based on actual analysis results
SAMPLE_DATA = {
    'GT-Physics': {
        'ssim': 0.532,
        'psnr': 19.22,
        'field_mae': 0.0449,
        'intensity_ratio': 4.233,
        'intensity_std': 2.625,
        'peak_distance': 106.6,
        'good_quality': 0.0,
        'color': '#E74C3C'
    },
    'Simplified': {
        'ssim': 0.628,
        'psnr': 17.98,
        'field_mae': 0.0505,
        'intensity_ratio': 0.996,
        'intensity_std': 0.029,
        'peak_distance': 112.9,
        'good_quality': 2.0,
        'color': '#3498DB'
    }
}

def create_sample_metric_comparison():
    """Create sample metric comparison chart."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Radiation Field Prediction Models: Key Metrics Comparison', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('SSIM Score', 'ssim', 'Higher is Better'),
        ('PSNR (dB)', 'psnr', 'Higher is Better'),  
        ('Field MAE', 'field_mae', 'Lower is Better'),
        ('Intensity Ratio', 'intensity_ratio', 'Closer to 1.0 is Better'),
        ('Peak Distance (px)', 'peak_distance', 'Lower is Better'),
        ('Good Quality (%)', 'good_quality', 'Higher is Better')
    ]
    
    for idx, (title, key, description) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        
        models = ['GT-Physics', 'Simplified']
        values = [SAMPLE_DATA[model][key] for model in models]
        colors = [SAMPLE_DATA[model]['color'] for model in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight better performance
        if 'Higher is Better' in description:
            better_idx = np.argmax(values)
        elif 'Lower is Better' in description:
            better_idx = np.argmin(values)
        else:  # Closer to 1.0 is Better
            better_idx = np.argmin([abs(v - 1.0) for v in values])
        
        bars[better_idx].set_edgecolor('gold')
        bars[better_idx].set_linewidth(4)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, f'{v:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title(f'{title}\n({description})', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Special formatting for intensity ratio
        if key == 'intensity_ratio':
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal (1.0)')
            ax.legend()
    
    plt.tight_layout()
    return fig

def create_sample_radar_chart():
    """Create sample radar chart for performance comparison."""
    # Define normalized metrics for radar chart
    metrics = ['SSIM', 'PSNR\n(Normalized)', 'Field Accuracy\n(1-MAE)', 
               'Intensity Control', 'Peak Accuracy', 'Consistency']
    
    # Normalize values to 0-1 scale
    gt_values = [
        SAMPLE_DATA['GT-Physics']['ssim'],
        SAMPLE_DATA['GT-Physics']['psnr'] / 25,  # Normalize PSNR
        1 - SAMPLE_DATA['GT-Physics']['field_mae'] * 10,  # Invert MAE
        1 / (1 + abs(SAMPLE_DATA['GT-Physics']['intensity_ratio'] - 1.0)),  # Intensity control
        1 / (1 + SAMPLE_DATA['GT-Physics']['peak_distance'] / 100),  # Peak accuracy
        1 / (1 + SAMPLE_DATA['GT-Physics']['intensity_std'])  # Consistency
    ]
    
    simp_values = [
        SAMPLE_DATA['Simplified']['ssim'],
        SAMPLE_DATA['Simplified']['psnr'] / 25,
        1 - SAMPLE_DATA['Simplified']['field_mae'] * 10,
        1 / (1 + abs(SAMPLE_DATA['Simplified']['intensity_ratio'] - 1.0)),
        1 / (1 + SAMPLE_DATA['Simplified']['peak_distance'] / 100),
        1 / (1 + SAMPLE_DATA['Simplified']['intensity_std'])
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    gt_values = gt_values + [gt_values[0]]  # Complete the circle
    simp_values = simp_values + [simp_values[0]]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot the models
    ax.plot(angles, gt_values, 'o-', linewidth=3, label='GT-Physics Model', 
            color=SAMPLE_DATA['GT-Physics']['color'])
    ax.fill(angles, gt_values, alpha=0.25, color=SAMPLE_DATA['GT-Physics']['color'])
    
    ax.plot(angles, simp_values, 'o-', linewidth=3, label='Simplified Model', 
            color=SAMPLE_DATA['Simplified']['color'])
    ax.fill(angles, simp_values, alpha=0.25, color=SAMPLE_DATA['Simplified']['color'])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.title('Model Performance Comparison\n(Normalized Metrics)', 
              size=16, fontweight='bold', y=1.12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    
    return fig

def create_sample_distribution_chart():
    """Create sample distribution comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Sample Metric Distributions Comparison', fontsize=16, fontweight='bold')
    
    # Generate sample distributions based on reported statistics
    np.random.seed(42)  # For reproducible results
    
    # Field MAE distributions
    gt_mae = np.random.normal(SAMPLE_DATA['GT-Physics']['field_mae'], 0.017, 100)
    simp_mae = np.random.normal(SAMPLE_DATA['Simplified']['field_mae'], 0.022, 100)
    
    axes[0].hist(gt_mae, bins=20, alpha=0.6, label='GT-Physics', 
                color=SAMPLE_DATA['GT-Physics']['color'], density=True)
    axes[0].hist(simp_mae, bins=20, alpha=0.6, label='Simplified', 
                color=SAMPLE_DATA['Simplified']['color'], density=True)
    axes[0].axvline(np.mean(gt_mae), color=SAMPLE_DATA['GT-Physics']['color'], 
                   linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(simp_mae), color=SAMPLE_DATA['Simplified']['color'], 
                   linestyle='--', linewidth=2)
    axes[0].set_title('Field MAE Distribution', fontweight='bold')
    axes[0].set_xlabel('Mean Absolute Error')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Intensity ratio distributions
    gt_ratio = np.random.normal(SAMPLE_DATA['GT-Physics']['intensity_ratio'], 2.6, 100)
    simp_ratio = np.random.normal(SAMPLE_DATA['Simplified']['intensity_ratio'], 0.03, 100)
    
    axes[1].hist(gt_ratio, bins=20, alpha=0.6, label='GT-Physics', 
                color=SAMPLE_DATA['GT-Physics']['color'], density=True)
    axes[1].hist(simp_ratio, bins=20, alpha=0.6, label='Simplified', 
                color=SAMPLE_DATA['Simplified']['color'], density=True)
    axes[1].axvline(1.0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Ideal (1.0)')
    axes[1].axvline(np.mean(gt_ratio), color=SAMPLE_DATA['GT-Physics']['color'], 
                   linestyle='--', linewidth=2)
    axes[1].axvline(np.mean(simp_ratio), color=SAMPLE_DATA['Simplified']['color'], 
                   linestyle='--', linewidth=2)
    axes[1].set_title('Intensity Ratio Distribution', fontweight='bold')
    axes[1].set_xlabel('Predicted/Actual Intensity Ratio')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Peak distance distributions
    gt_peak = np.random.normal(SAMPLE_DATA['GT-Physics']['peak_distance'], 55, 100)
    simp_peak = np.random.normal(SAMPLE_DATA['Simplified']['peak_distance'], 60, 100)
    
    axes[2].hist(gt_peak, bins=20, alpha=0.6, label='GT-Physics', 
                color=SAMPLE_DATA['GT-Physics']['color'], density=True)
    axes[2].hist(simp_peak, bins=20, alpha=0.6, label='Simplified', 
                color=SAMPLE_DATA['Simplified']['color'], density=True)
    axes[2].axvline(np.mean(gt_peak), color=SAMPLE_DATA['GT-Physics']['color'], 
                   linestyle='--', linewidth=2)
    axes[2].axvline(np.mean(simp_peak), color=SAMPLE_DATA['Simplified']['color'], 
                   linestyle='--', linewidth=2)
    axes[2].set_title('Peak Distance Error Distribution', fontweight='bold')
    axes[2].set_xlabel('Distance (pixels)')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Generate and display sample analysis plots."""
    print("Generating Sample Comparative Analysis Plots...")
    print("=" * 50)
    
    # Set output directory
    output_dir = Path("sample_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate charts
    print("1. Creating metric comparison chart...")
    fig1 = create_sample_metric_comparison()
    fig1.savefig(output_dir / "sample_metric_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("2. Creating radar performance chart...")
    fig2 = create_sample_radar_chart()
    fig2.savefig(output_dir / "sample_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("3. Creating distribution comparison chart...")
    fig3 = create_sample_distribution_chart()
    fig3.savefig(output_dir / "sample_distributions.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\n✓ Sample plots saved to: {output_dir}")
    print("✓ Key findings:")
    print("  - Simplified Model shows superior intensity control (ratio ≈ 1.0 vs 4.2)")
    print("  - Simplified Model has better SSIM (structural similarity)")
    print("  - Both models struggle with peak localization accuracy")
    print("  - GT-Physics has marginally better PSNR (signal fidelity)")
    print("\n  Recommendation: Use Simplified Model for better intensity control and stability")

if __name__ == "__main__":
    main()