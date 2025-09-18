# ──────────────────────────────────────────────────────────────
# test/modules/visualizer.py - Visualization Module for 3-Channel App
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Dict, Any, Tuple

__all__ = [
    "RadiationVisualizer", "plot_ground_truth", "plot_measurements", 
    "plot_three_channels", "plot_comparison"
]


class RadiationVisualizer:
    """Comprehensive visualization module for radiation field analysis."""
    
    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.default_cmap = "hot"
        self.path_color = "lime"
        self.measurement_color = "white"
        
    def plot_ground_truth(self, field: np.ndarray, sources_info: Optional[Dict] = None, 
                         ax: Optional[Axes] = None) -> Axes:
        """Plot ground truth radiation field with measurement points overlay."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        
        im = ax.imshow(field, cmap=self.default_cmap, origin="lower", 
                      vmin=0, vmax=1)
        
        # Add measurement points overlay if available
        if sources_info and 'measurement_points' in sources_info:
            measurement_points = sources_info['measurement_points']
            n_measurements = sources_info.get('n_measurements', len(measurement_points))
            ax.scatter(measurement_points[:, 1], measurement_points[:, 0], 
                      c='white', s=20, marker='o', edgecolor='black', 
                      linewidth=0.5, alpha=0.8, label=f'{n_measurements} measurements')
            ax.legend(frameon=True, fancybox=True, shadow=True, 
                     loc='upper right', fontsize=10)
        
        ax.set_title("Ground Truth with Measurement Points")
        ax.axis("off")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax
    
    def plot_measurements(self, measurements: np.ndarray, mask: np.ndarray, 
                         waypoints: Optional[np.ndarray] = None, 
                         ax: Optional[Axes] = None) -> Axes:
        """Plot sparse measurements with trajectory."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        
        im = ax.imshow(measurements, cmap=self.default_cmap, origin="lower",
                      vmin=0, vmax=1)
        
        # Add trajectory path if available
        if waypoints is not None:
            ax.plot(waypoints[:, 1], waypoints[:, 0], "-o", 
                   c=self.measurement_color, lw=2, ms=4, alpha=0.7, 
                   label=f"Path ({len(waypoints)} points)")
            ax.legend(frameon=False)
        
        ax.set_title(f"Sparse Measurements ({mask.sum()} points)")
        ax.axis("off")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax
    
    def plot_input_channel(self, channel_data: np.ndarray, channel_name: str, 
                          ax: Optional[Axes] = None, cmap: Optional[str] = None) -> Axes:
        """Plot a single input channel."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        
        if cmap is None:
            # Choose appropriate colormap based on channel name
            if "coord" in channel_name.lower() or "distance" in channel_name.lower():
                cmap = "coolwarm"
            elif "log" in channel_name.lower():
                cmap = "viridis"
            elif "mask" in channel_name.lower():
                cmap = "binary"
            else:
                cmap = self.default_cmap
        
        im = ax.imshow(channel_data, cmap=cmap, origin="lower")
        ax.set_title(f"Channel: {channel_name}")
        ax.axis("off")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax
    
    def plot_three_channels(self, input_channels: np.ndarray, 
                           channel_indices: tuple[int, int, int] = (0, 1, 2),
                           channel_names: Optional[tuple[str, str, str]] = None) -> Figure:
        """Plot three specific input channels side by side."""
        if channel_names is None:
            channel_names = (
                f"Channel {channel_indices[0]}", 
                f"Channel {channel_indices[1]}", 
                f"Channel {channel_indices[2]}"
            )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)
        
        for i, (ch_idx, ch_name) in enumerate(zip(channel_indices, channel_names)):
            self.plot_input_channel(input_channels[ch_idx], ch_name, axes[i])
        
        plt.tight_layout()
        return fig
    
    def plot_all_channels(self, input_channels: np.ndarray) -> Figure:
        """Plot all 6 input channels in a 2x3 grid."""
        channel_names = [
            "Measurements", "Mask", "Log Measurements",
            "Y Coordinates", "X Coordinates", "Distance Map"
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        for i in range(6):
            self.plot_input_channel(input_channels[i], channel_names[i], axes[i])
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, ground_truth: np.ndarray, prediction: Optional[np.ndarray] = None, 
                       measurements: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, 
                       waypoints: Optional[np.ndarray] = None, sources_info: Optional[Dict] = None) -> Figure:
        """Plot GT vs prediction comparison."""
        if prediction is not None:
            # GT vs Prediction comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            
            # Ground truth with measurement points
            self.plot_ground_truth(ground_truth, sources_info, axes[0])
            
            # Prediction
            im = axes[1].imshow(prediction, cmap=self.default_cmap, origin="lower",
                               vmin=0, vmax=1)
            axes[1].set_title("Model Prediction")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
        else:
            # Fallback to GT vs measurements
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            
            # Ground truth
            self.plot_ground_truth(ground_truth, sources_info, axes[0])
            
            # Measurements
            if measurements is not None and mask is not None:
                self.plot_measurements(measurements, mask, waypoints, axes[1])
            else:
                axes[1].text(0.5, 0.5, 'No prediction or measurements available', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive_analysis(self, field: np.ndarray, input_channels: np.ndarray,
                                   mask: np.ndarray, waypoints: Optional[np.ndarray] = None,
                                   sources_info: Optional[Dict] = None) -> Figure:
        """Create a comprehensive visualization with GT, measurements, and key channels."""
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        
        # Create grid layout: 3 rows, 4 columns
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Ground truth and measurements
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_ground_truth(field, sources_info, ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        measurements = input_channels[0]  # First channel is measurements
        self.plot_measurements(measurements, mask, waypoints, ax2)
        
        # Row 1: Key statistics
        ax_stats = fig.add_subplot(gs[0, 2:])
        ax_stats.axis('off')
        stats_text = self._generate_stats_text(field, mask, sources_info)
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     transform=ax_stats.transAxes)
        
        # Row 2-3: All input channels
        channel_names = [
            "Measurements", "Mask", "Log Measurements",
            "Y Coordinates", "X Coordinates", "Distance Map"
        ]
        
        for i in range(6):
            row = 1 + i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            self.plot_input_channel(input_channels[i], channel_names[i], ax)
        
        # Overall title
        fig.suptitle("Comprehensive Radiation Field Analysis", fontsize=16, fontweight='bold')
        
        return fig
    
    def _generate_stats_text(self, field: np.ndarray, mask: np.ndarray, 
                            sources_info: Optional[Dict]) -> str:
        """Generate statistics text for display."""
        stats = [
            "Dataset Statistics:",
            f"• Field size: {field.shape[0]}×{field.shape[1]}",
            f"• Field range: [{field.min():.4f}, {field.max():.4f}]",
            f"• Field mean: {field.mean():.4f}",
            f"• Measurements: {mask.sum()} points",
            f"• Coverage: {mask.sum()/(field.shape[0]*field.shape[1])*100:.2f}%",
        ]
        
        if sources_info:
            stats.extend([
                "",
                "Source Information:",
                f"• Number of sources: {sources_info.get('n_sources', 'N/A')}",
                f"• Max intensity: {sources_info.get('max_intensity', 'N/A'):.4f}",
                f"• Total intensity: {sources_info.get('total_intensity', 'N/A'):.2f}",
            ])
        
        return "\n".join(stats)


# Convenience functions for standalone use
def plot_ground_truth(field: np.ndarray, sources_info: Optional[Dict] = None) -> Figure:
    """Standalone function to plot ground truth."""
    visualizer = RadiationVisualizer()
    fig, ax = plt.subplots(figsize=(8, 8))
    visualizer.plot_ground_truth(field, sources_info, ax)
    return fig


def plot_measurements(measurements: np.ndarray, mask: np.ndarray, 
                     waypoints: Optional[np.ndarray] = None) -> Figure:
    """Standalone function to plot measurements."""
    visualizer = RadiationVisualizer()
    fig, ax = plt.subplots(figsize=(8, 8))
    visualizer.plot_measurements(measurements, mask, waypoints, ax)
    return fig


def plot_three_channels(input_channels: np.ndarray, 
                       channel_indices: tuple[int, int, int] = (0, 1, 2),
                       channel_names: Optional[tuple[str, str, str]] = None) -> Figure:
    """Standalone function to plot three channels."""
    visualizer = RadiationVisualizer()
    return visualizer.plot_three_channels(input_channels, channel_indices, channel_names)


def plot_comparison(ground_truth: np.ndarray, measurements: np.ndarray, 
                   mask: np.ndarray, waypoints: Optional[np.ndarray] = None,
                   sources_info: Optional[Dict] = None) -> Figure:
    """Standalone function to plot GT vs measurements."""
    visualizer = RadiationVisualizer()
    return visualizer.plot_comparison(ground_truth, measurements, mask, waypoints, sources_info)


if __name__ == "__main__":
    # Test the module
    print("Testing RadiationVisualizer...")
    
    # Create dummy data for testing
    field = np.random.exponential(0.3, (256, 256)).astype(np.float32)
    measurements = field * np.random.binomial(1, 0.05, field.shape)
    mask = (measurements > 0).astype(np.uint8)
    
    visualizer = RadiationVisualizer()
    fig = visualizer.plot_comparison(field, measurements, mask)
    plt.show()
    
    print("RadiationVisualizer test complete!")