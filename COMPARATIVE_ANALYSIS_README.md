# Radiation Field Prediction Model Comparative Analysis

This document provides instructions for using the comprehensive comparative analysis tools created to compare radiation field prediction models.

## Overview

The analysis tools compare two ConvNeXt-based PGNN models:
1. **GT-Physics Model (convnext_gt_physics_exp3)** - Complex physics-guided model
2. **Simplified Model (convnext_simple_gt_exp8)** - Simplified architecture with better control

## Files Created

### Main Analysis Script
- **`comparative_analysis.py`** - Complete comparative analysis tool
- **`view_analysis_sample.py`** - Sample visualization generator

### Generated Results (in `comparative_analysis_results/`)
- **`comparison_summary.txt`** - Detailed textual comparison
- **`recommendations.txt`** - Strategic recommendations and findings
- **`statistical_analysis.json`** - Statistical significance test results
- **`summary_table.csv`** - Key metrics in table format
- **`metric_comparison.png`** - Side-by-side metric comparison chart
- **`distribution_plots.png`** - Metric distribution visualizations
- **`radar_comparison.png`** - Radar chart performance comparison
- **`error_analysis.png`** - Error analysis and quality distribution

### Sample Plots (in `sample_plots/`)
- **`sample_metric_comparison.png`** - Sample metrics visualization
- **`sample_radar_chart.png`** - Sample radar performance chart
- **`sample_distributions.png`** - Sample distribution comparisons

## Usage Instructions

### Running the Complete Analysis

```bash
# Run the comprehensive comparative analysis
python3 comparative_analysis.py
```

This will:
1. Load evaluation data from both experiments
2. Generate all visualizations and analysis files
3. Perform statistical significance testing
4. Create detailed recommendations
5. Save all results to `comparative_analysis_results/`

### Viewing Sample Plots

```bash
# Generate sample visualizations
python3 view_analysis_sample.py
```

This creates sample plots in the `sample_plots/` directory.

### Dependencies

Required Python packages:
- `matplotlib` - For plotting and visualization
- `numpy` - For numerical computations
- `scipy` - For statistical analysis
- `pandas` (optional) - For CSV output

Install with:
```bash
pip3 install matplotlib numpy scipy pandas --user
```

## Analysis Features

### 1. Comprehensive Metrics Comparison
- **SSIM Score** - Structural similarity index
- **PSNR** - Peak signal-to-noise ratio
- **Field MAE** - Mean absolute error in field reconstruction
- **Intensity Ratio** - Predicted vs actual intensity accuracy
- **Peak Distance** - Spatial accuracy of peak localization
- **Quality Distribution** - Overall prediction quality categories

### 2. Statistical Analysis
- **T-tests** - Parametric significance testing
- **Mann-Whitney U tests** - Non-parametric significance testing
- **Effect size analysis** - Cohen's d for practical significance
- **Distribution analysis** - Mean, standard deviation, min/max values

### 3. Visualizations
- **Bar charts** - Direct metric comparisons with highlighting
- **Distribution plots** - Histograms showing metric distributions
- **Radar charts** - Multi-dimensional performance comparison
- **Error analysis** - Quality distribution and issue frequency

### 4. Strategic Recommendations
- **Short-term improvements** - Immediate actionable steps
- **Mid-term research directions** - Future development paths
- **Long-term development** - Strategic research initiatives

## Key Findings Summary

### Winner: Simplified Model (convnext_simple_gt_exp8)

**Superior Performance:**
- **Intensity Control**: Perfect ratio (0.996 vs 4.233) with minimal variation
- **Structural Similarity**: Better SSIM (0.628 vs 0.532)
- **Stability**: Dramatically lower intensity standard deviation (0.029 vs 2.625)
- **Quality Distribution**: 2% good quality cases vs 0% for GT-Physics

**Areas for Improvement (Both Models):**
- **Peak Localization**: Both models show >98% peak position errors >10px
- **Spatial Accuracy**: Mean peak distances >100px indicate poor localization

**Recommendations:**
1. **Adopt Simplified Architecture** for production use
2. **Focus on Spatial Localization** improvements
3. **Investigate Attention Mechanisms** for better peak detection
4. **Validate on Real-World Data** before deployment

## Data Sources

The analysis uses evaluation data from:
- `/home/ikhyeon/research_ws/Radiation_PGNN/eval/convnext_gt_physics_exp3/`
- `/home/ikhyeon/research_ws/Radiation_PGNN/eval/convnext_simple_gt_exp8/`

Each contains:
- Individual test reports (100 samples each)
- Aggregate summary statistics
- Visualization files
- Structured evaluation data

## Customization

### Adding New Models
To compare additional models, modify `comparative_analysis.py`:

1. Add model information to the `models` dictionary in `__init__`
2. Ensure evaluation data follows the same JSON structure
3. Update visualization colors and labels as needed

### Custom Metrics
To add new comparison metrics:

1. Update the `extract_metrics_arrays()` method
2. Add metric extraction logic for individual reports
3. Include in statistical testing and visualization functions

### Visualization Styles
Modify visualization appearance by:
- Changing colors in the `models` dictionary
- Updating matplotlib rcParams at the top of the script
- Customizing plot styles in individual visualization functions

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Ensure evaluation directories exist and contain required JSON files
   - Check file paths in the script match your directory structure

2. **Import Errors**
   - Install required dependencies: `pip3 install matplotlib numpy scipy pandas`
   - For matplotlib 3D warning, this is non-critical and can be ignored

3. **Memory Issues**
   - Large datasets may require more memory
   - Consider reducing the number of individual reports processed

4. **Permission Errors**
   - Ensure write permissions in the output directory
   - Use `chmod` to fix permission issues if needed

### Performance Notes

- Analysis of 100 samples per model takes ~30 seconds
- Image generation adds ~10-15 seconds
- Statistical testing is computationally efficient
- Memory usage peaks around 200-300MB for typical datasets

## Future Enhancements

Potential improvements to the analysis tools:

1. **Interactive Dashboards** - Web-based visualization interface
2. **Real-time Monitoring** - Live performance tracking during training
3. **Multi-model Comparison** - Support for comparing >2 models simultaneously
4. **Advanced Statistics** - Bayesian analysis, confidence intervals
5. **Export Formats** - LaTeX tables, PowerPoint-ready charts
6. **Automated Reporting** - PDF report generation with findings

## Contact and Support

This analysis tool was created as part of the Radiation PGNN project for comparing physics-guided neural network models in radiation field prediction.

For questions about the analysis methodology or extending the tools, refer to the code documentation and comments within the scripts.