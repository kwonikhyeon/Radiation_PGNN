#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
# test/3ch_visualize_app.py - 3-Channel Radiation Field Visualization App
# ──────────────────────────────────────────────────────────────
"""
Ground Truth Radiation Field Visualization App

This application generates ground truth radiation fields using the existing
src/dataset/ modules and provides visualization of the ground truth field
with radiation sources marked.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import dataset.generate_truth as gt

# Add modules to path
sys.path.append(str(Path(__file__).parent))
from modules.visualizer import RadiationVisualizer
from modules.model_inference import ModelInference
from modules.safety_layer import SafetyCalculator, SafetyParameters
from modules.information_gain import calculate_information_gain_layer, InformationGainParameters
from modules.traversability_layer import calculate_traversability_layer, TraversabilityParameters


class RadiationFieldApp:
    """Main application class for ground truth radiation field visualization."""
    
    def __init__(self, seed: Optional[int] = None, checkpoint_path: Optional[str] = None):
        self.seed = seed
        self.visualizer = RadiationVisualizer()
        self.current_field = None
        self.current_metadata = None
        self.current_prediction = None
        self.current_safety_layer = None
        self.current_information_gain = None
        self.current_traversability_layer = None
        
        # Initialize model if checkpoint provided
        self.model_inference = None
        if checkpoint_path:
            try:
                self.model_inference = ModelInference(checkpoint_path)
                print(f"✅ Model loaded from {checkpoint_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.model_inference = None
        
        # Initialize safety calculator
        self.safety_calculator = SafetyCalculator()
        
    def generate_sample(self, n_sources: Optional[int] = None) -> Dict[str, Any]:
        """Generate a ground truth radiation field sample using src/dataset functions directly.
        
        Returns:
            dict with keys: 'field', 'metadata'
        """
        print(f"Generating radiation field sample...")
        
        # Create new RNG for each sample (ensures randomness when seed=None)
        rng = np.random.default_rng(self.seed)
        
        # Use src/dataset functions exactly as they are
        if n_sources is None:
            n_sources = rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1)
        
        coords, amps, sigmas = gt.sample_sources(gt.GRID, n_sources, rng=rng)
        field = gt.gaussian_field(gt.GRID, coords, amps, sigmas)
        
        # Apply same normalization as interactive_uncertainty_app_patched.py
        field -= field.min()
        if field.max() > 0:
            field /= field.max()
        
        # Generate random measurement points (5-60 points)
        n_measurements = rng.integers(15, 61)  # 5 to 60 points
        measurement_points = []
        
        for _ in range(n_measurements):
            y = rng.integers(0, gt.GRID)
            x = rng.integers(0, gt.GRID)
            measurement_points.append((y, x))
        
        measurement_points = np.array(measurement_points)
        
        metadata = {
            'n_sources': n_sources,
            'coords': coords,
            'amplitudes': amps,
            'sigmas': sigmas,
            'max_intensity': field.max(),
            'total_intensity': field.sum(),
            'measurement_points': measurement_points,
            'n_measurements': n_measurements
        }
        
        # Store current state
        self.current_field = field
        self.current_metadata = metadata
        
        # Generate model prediction if model is available
        self.current_prediction = None
        if self.model_inference:
            try:
                # Create measurements and mask from measurement points
                measurements = np.zeros((gt.GRID, gt.GRID), dtype=np.float32)
                mask = np.zeros((gt.GRID, gt.GRID), dtype=np.uint8)
                
                for y, x in measurement_points:
                    measurements[y, x] = field[y, x]  # Sample GT at measurement points
                    mask[y, x] = 1
                
                # Get model prediction
                prediction = self.model_inference.predict(measurements, mask)
                self.current_prediction = prediction
                
                # Calculate safety layer from prediction
                safety_result = self.safety_calculator.calculate_safety_layer(
                    prediction, mask
                )
                self.current_safety_layer = safety_result
                
                # Calculate information gain layer from prediction
                information_gain_result = calculate_information_gain_layer(
                    single_prediction=prediction, 
                    measurement_mask=mask
                )
                self.current_information_gain = information_gain_result
                
                # Generate robot position and heading for traversability calculation
                # Select random measurement point as robot position
                robot_position = None
                robot_heading = None
                if len(measurement_points) > 0:
                    robot_idx = rng.integers(0, len(measurement_points))
                    robot_position = tuple(measurement_points[robot_idx])  # (y, x)
                    robot_heading = rng.uniform(0, 2 * np.pi)  # Random heading in radians
                
                # Calculate traversability layer from prediction with robot navigation
                traversability_result = calculate_traversability_layer(
                    radiation_field=prediction,
                    measurement_mask=mask,
                    robot_position=robot_position,
                    robot_heading=robot_heading
                )
                self.current_traversability_layer = traversability_result
                
                print(f"✅ Model prediction, safety layer, information gain, and traversability completed")
                
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                self.current_prediction = None
                self.current_safety_layer = None
                self.current_information_gain = None
                self.current_traversability_layer = None
        
        print(f"Generated sample with {metadata['n_sources']} sources, {metadata['n_measurements']} measurement points")
        
        return {
            'field': field,
            'metadata': metadata,
            'prediction': self.current_prediction,
            'safety_layer': self.current_safety_layer,
            'information_gain': self.current_information_gain,
            'traversability_layer': self.current_traversability_layer
        }
    
    def visualize_ground_truth(self, save_path: Optional[str] = None):
        """Visualize ground truth radiation field or GT vs prediction comparison."""
        if self.current_field is None:
            print("No sample generated. Call generate_sample() first.")
            return
        
        if self.current_prediction is not None:
            print("Creating GT vs Prediction comparison...")
            self.visualizer.plot_comparison(
                self.current_field, 
                prediction=self.current_prediction,
                sources_info=self.current_metadata
            )
        else:
            print("Creating ground truth visualization...")
            self.visualizer.plot_ground_truth(self.current_field, self.current_metadata)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def regenerate_gt(self, n_sources: Optional[int] = None) -> Dict[str, Any]:
        """Generate new GT and measurements, update model prediction."""
        return self.generate_sample(n_sources)
    
    def regenerate_measurements(self) -> Dict[str, Any]:
        """Generate new measurement points with existing GT, update model prediction."""
        if self.current_field is None:
            print("No GT field available. Generate sample first.")
            return {}
        
        print("Regenerating measurement points...")
        
        # Create new RNG for each sample
        rng = np.random.default_rng(self.seed)
        field = self.current_field
        
        # Generate random measurement points (15-60 points)
        n_measurements = rng.integers(15, 61)
        measurement_points = []
        
        for _ in range(n_measurements):
            y = rng.integers(0, gt.GRID)
            x = rng.integers(0, gt.GRID)
            measurement_points.append((y, x))
        
        measurement_points = np.array(measurement_points)
        
        # Update metadata
        self.current_metadata.update({
            'measurement_points': measurement_points,
            'n_measurements': n_measurements
        })
        
        # Generate model prediction if model is available
        self.current_prediction = None
        if self.model_inference:
            try:
                # Create measurements and mask from measurement points
                measurements = np.zeros((gt.GRID, gt.GRID), dtype=np.float32)
                mask = np.zeros((gt.GRID, gt.GRID), dtype=np.uint8)
                
                for y, x in measurement_points:
                    measurements[y, x] = field[y, x]  # Sample GT at measurement points
                    mask[y, x] = 1
                
                # Get model prediction
                prediction = self.model_inference.predict(measurements, mask)
                self.current_prediction = prediction
                
                # Calculate safety layer from prediction
                safety_result = self.safety_calculator.calculate_safety_layer(
                    prediction, mask
                )
                self.current_safety_layer = safety_result
                
                # Calculate information gain layer from prediction
                information_gain_result = calculate_information_gain_layer(
                    single_prediction=prediction, 
                    measurement_mask=mask
                )
                self.current_information_gain = information_gain_result
                
                # Generate robot position and heading for traversability calculation
                # Select random measurement point as robot position
                robot_position = None
                robot_heading = None
                if len(measurement_points) > 0:
                    robot_idx = rng.integers(0, len(measurement_points))
                    robot_position = tuple(measurement_points[robot_idx])  # (y, x)
                    robot_heading = rng.uniform(0, 2 * np.pi)  # Random heading in radians
                
                # Calculate traversability layer from prediction with robot navigation
                traversability_result = calculate_traversability_layer(
                    radiation_field=prediction,
                    measurement_mask=mask,
                    robot_position=robot_position,
                    robot_heading=robot_heading
                )
                self.current_traversability_layer = traversability_result
                
                print(f"✅ Model prediction, safety layer, information gain, and traversability updated")
                
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                self.current_prediction = None
                self.current_safety_layer = None
                self.current_information_gain = None
                self.current_traversability_layer = None
        
        print(f"Regenerated {n_measurements} measurement points")
        
        return {
            'field': field,
            'metadata': self.current_metadata,
            'prediction': self.current_prediction,
            'safety_layer': self.current_safety_layer,
            'information_gain': self.current_information_gain,
            'traversability_layer': self.current_traversability_layer
        }


class RadiationFieldGUI:
    """Interactive GUI for radiation field visualization with buttons."""
    
    def __init__(self, app: RadiationFieldApp):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Radiation Field Visualization")
        self.root.geometry("1800x1200")
        
        # Create matplotlib figure with 2 rows: 3 plots in row 1, 3 plots in row 2
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle("Radiation Field Analysis")
        
        # Setup GUI
        self.setup_gui()
        
        # Initial visualization
        self.update_visualization()
    
    def setup_gui(self):
        """Setup the GUI layout with buttons and canvas."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        ttk.Button(button_frame, text="Generate New GT", 
                  command=self.generate_new_gt).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="New Measurements", 
                  command=self.generate_new_measurements).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save Plot", 
                  command=self.save_plot).pack(side=tk.LEFT, padx=(0, 10))
        
        # Info label
        self.info_label = ttk.Label(button_frame, text="")
        self.info_label.pack(side=tk.RIGHT)
        
        # Canvas for matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_new_gt(self):
        """Generate new GT and update visualization."""
        self.info_label.config(text="Generating new GT...")
        self.root.update_idletasks()
        
        self.app.regenerate_gt()
        self.update_visualization()
        
        meta = self.app.current_metadata
        self.info_label.config(text=f"New GT: {meta['n_sources']} sources, {meta['n_measurements']} measurements")
    
    def generate_new_measurements(self):
        """Generate new measurements and update visualization."""
        if self.app.current_field is None:
            self.info_label.config(text="No GT available. Generate GT first.")
            return
        
        self.info_label.config(text="Generating new measurements...")
        self.root.update_idletasks()
        
        self.app.regenerate_measurements()
        self.update_visualization()
        
        meta = self.app.current_metadata
        self.info_label.config(text=f"New measurements: {meta['n_measurements']} points")
    
    def update_visualization(self):
        """Update the matplotlib visualization."""
        if self.app.current_field is None:
            return
        
        # Clear figure completely and recreate subplots
        self.fig.clear()
        self.axes = self.fig.subplots(2, 3)
        self.fig.suptitle("Radiation Field Analysis")
        
        if self.app.current_prediction is not None:
            # Row 1, Col 1: GT with measurement points
            self.app.visualizer.plot_ground_truth(self.app.current_field, self.app.current_metadata, self.axes[0, 0])
            
            # Row 1, Col 2: Prediction with measurement points
            im1 = self.axes[0, 1].imshow(self.app.current_prediction, cmap="hot", origin="lower", vmin=0, vmax=1)
            
            # Add measurement points overlay on prediction
            if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                measurement_points = self.app.current_metadata['measurement_points']
                n_measurements = self.app.current_metadata.get('n_measurements', len(measurement_points))
                self.axes[0, 1].scatter(measurement_points[:, 1], measurement_points[:, 0], 
                                       c='white', s=20, marker='o', edgecolor='black', 
                                       linewidth=0.5, alpha=0.8, label=f'{n_measurements} measurements')
                self.axes[0, 1].legend(frameon=True, fancybox=True, shadow=True, 
                                      loc='upper right', fontsize=10)
            
            self.axes[0, 1].set_title("Model Prediction")
            self.axes[0, 1].axis("off")
            plt.colorbar(im1, ax=self.axes[0, 1], fraction=0.046, pad=0.04)
            
            # Row 1, Col 3: GT with Source Overlay (new plot)
            im_overlay = self.axes[0, 2].imshow(self.app.current_field, cmap="hot", origin="lower", vmin=0, vmax=1)
            
            # Add measurement points
            if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                self.axes[0, 2].scatter(measurement_points[:, 1], measurement_points[:, 0], 
                                       c='white', s=15, marker='o', edgecolor='black', 
                                       linewidth=0.5, alpha=0.7)
            
            # Add true source locations from metadata
            if self.app.current_metadata and 'coords' in self.app.current_metadata:
                true_sources = self.app.current_metadata['coords']
                if len(true_sources) > 0:
                    true_array = np.array(true_sources)
                    self.axes[0, 2].scatter(true_array[:, 1], true_array[:, 0], 
                                           c='cyan', s=80, marker='x', linewidth=3, 
                                           alpha=0.9, label=f'{len(true_sources)} true sources')
            
            # Add detected source locations if available
            if (self.app.current_traversability_layer and 
                'source_locations' in self.app.current_traversability_layer):
                detected_sources = self.app.current_traversability_layer['source_locations']
                if detected_sources:
                    detected_array = np.array(detected_sources)
                    self.axes[0, 2].scatter(detected_array[:, 1], detected_array[:, 0], 
                                           c='r', s=60, marker='x', linewidth=2, 
                                           alpha=0.9, label=f'{len(detected_sources)} detected')
            
            self.axes[0, 2].legend(frameon=True, fancybox=True, shadow=True, 
                                  loc='upper right', fontsize=8)
            self.axes[0, 2].set_title("GT with Source Detection")
            self.axes[0, 2].axis("off")
            plt.colorbar(im_overlay, ax=self.axes[0, 2], fraction=0.046, pad=0.04)
            
            # Row 2, Col 1: Safety layer
            if self.app.current_safety_layer is not None:
                total_safety = self.app.current_safety_layer['total_safety']
                im2 = self.axes[1, 0].imshow(total_safety, cmap="Greens", origin="lower", vmin=0, vmax=1)
                
                # Add measurement points overlay on safety layer
                if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                    self.axes[1, 0].scatter(measurement_points[:, 1], measurement_points[:, 0], 
                                           c='white', s=15, marker='o', edgecolor='black', 
                                           linewidth=0.5, alpha=0.7)
                
                self.axes[1, 0].set_title("Safety Layer")
                self.axes[1, 0].axis("off")
                plt.colorbar(im2, ax=self.axes[1, 0], fraction=0.046, pad=0.04)
            else:
                self.axes[1, 0].text(0.5, 0.5, 'Safety layer\ncalculation failed', 
                                    ha='center', va='center', transform=self.axes[1, 0].transAxes, fontsize=12)
                self.axes[1, 0].set_title("Safety Layer")
                self.axes[1, 0].axis('off')
            
            # Row 2, Col 2: Information gain layer
            if self.app.current_information_gain is not None:
                information_gain = self.app.current_information_gain['information_gain']
                im3 = self.axes[1, 1].imshow(information_gain, cmap="plasma", origin="lower", vmin=0, vmax=1)
                
                # Add measurement points overlay on information gain layer
                if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                    self.axes[1, 1].scatter(measurement_points[:, 1], measurement_points[:, 0], 
                                           c='white', s=15, marker='o', edgecolor='black', 
                                           linewidth=0.5, alpha=0.7)
                
                self.axes[1, 1].set_title("Information Gain Layer")
                self.axes[1, 1].axis("off")
                plt.colorbar(im3, ax=self.axes[1, 1], fraction=0.046, pad=0.04)
            else:
                self.axes[1, 1].text(0.5, 0.5, 'Information gain\ncalculation failed', 
                                    ha='center', va='center', transform=self.axes[1, 1].transAxes, fontsize=12)
                self.axes[1, 1].set_title("Information Gain Layer")
                self.axes[1, 1].axis('off')
            
            # Row 2, Col 3: Traversability layer
            if self.app.current_traversability_layer is not None:
                total_traversability = self.app.current_traversability_layer['total_traversability']
                im4 = self.axes[1, 2].imshow(total_traversability, cmap="viridis", origin="lower", vmin=0, vmax=1)
                
                # Add measurement points overlay on traversability layer
                if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                    self.axes[1, 2].scatter(measurement_points[:, 1], measurement_points[:, 0], 
                                           c='white', s=15, marker='o', edgecolor='black', 
                                           linewidth=0.5, alpha=0.7)
                
                # Add detected source locations as red X marks
                if 'source_locations' in self.app.current_traversability_layer:
                    source_locs = self.app.current_traversability_layer['source_locations']
                    if source_locs:
                        source_array = np.array(source_locs)
                        self.axes[1, 2].scatter(source_array[:, 1], source_array[:, 0], 
                                               c='red', s=50, marker='x', linewidth=2, 
                                               alpha=0.9, label=f'{len(source_locs)} sources')
                
                # Add robot position and heading if available
                if (self.app.current_traversability_layer and 
                    'metadata' in self.app.current_traversability_layer and
                    'robot_info' in self.app.current_traversability_layer['metadata']):
                    robot_info = self.app.current_traversability_layer['metadata']['robot_info']
                    if robot_info['position'] is not None and robot_info['heading_rad'] is not None:
                        robot_y, robot_x = robot_info['position']
                        robot_heading = robot_info['heading_rad']
                        
                        # Plot robot position as blue circle
                        self.axes[1, 2].scatter(robot_x, robot_y, c='blue', s=80, marker='o', 
                                               edgecolor='white', linewidth=2, alpha=0.9, 
                                               label='Robot')
                        
                        # Plot robot heading as arrow
                        arrow_length = 10
                        dx = arrow_length * np.cos(robot_heading)
                        dy = arrow_length * np.sin(robot_heading)
                        self.axes[1, 2].arrow(robot_x, robot_y, dx, dy, 
                                             head_width=3, head_length=2, fc='blue', ec='blue', 
                                             alpha=0.8, linewidth=2)
                
                self.axes[1, 2].legend(frameon=True, fancybox=True, shadow=True, 
                                      loc='upper right', fontsize=8)
                
                self.axes[1, 2].set_title("Traversability Layer")
                self.axes[1, 2].axis("off")
                plt.colorbar(im4, ax=self.axes[1, 2], fraction=0.046, pad=0.04)
            else:
                self.axes[1, 2].text(0.5, 0.5, 'Traversability\ncalculation failed', 
                                    ha='center', va='center', transform=self.axes[1, 2].transAxes, fontsize=12)
                self.axes[1, 2].set_title("Traversability Layer")
                self.axes[1, 2].axis('off')
        else:
            # Only GT available - Row 1
            self.app.visualizer.plot_ground_truth(self.app.current_field, self.app.current_metadata, self.axes[0, 0])
            
            self.axes[0, 1].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                ha='center', va='center', transform=self.axes[0, 1].transAxes, fontsize=12)
            self.axes[0, 1].set_title("Model Prediction")
            self.axes[0, 1].axis('off')
            
            self.axes[0, 2].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                ha='center', va='center', transform=self.axes[0, 2].transAxes, fontsize=12)
            self.axes[0, 2].set_title("GT with Source Detection")
            self.axes[0, 2].axis('off')
            
            # Row 2 - Analysis layers
            self.axes[1, 0].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                ha='center', va='center', transform=self.axes[1, 0].transAxes, fontsize=12)
            self.axes[1, 0].set_title("Safety Layer")
            self.axes[1, 0].axis('off')
            
            self.axes[1, 1].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                ha='center', va='center', transform=self.axes[1, 1].transAxes, fontsize=12)
            self.axes[1, 1].set_title("Information Gain Layer")
            self.axes[1, 1].axis('off')
            
            self.axes[1, 2].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                ha='center', va='center', transform=self.axes[1, 2].transAxes, fontsize=12)
            self.axes[1, 2].set_title("Traversability Layer")
            self.axes[1, 2].axis('off')
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_plot(self):
        """Save current plot to file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            self.info_label.config(text=f"Saved to {filename}")
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Ground Truth Radiation Field Visualization App",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Generation options
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for reproducible results (default: random)")
    parser.add_argument("--sources", type=int, 
                       help="Number of radiation sources (default: random 1-4)")
    
    # Model options
    parser.add_argument("--checkpoint", type=str, default="checkpoints/convnext_simple_gt_exp8/ckpt_best.pth",
                       help="Path to model checkpoint for prediction")
    
    # Visualization options
    parser.add_argument("--show-gt", action="store_true", default=False,
                       help="Show ground truth visualization (non-interactive)")
    parser.add_argument("--gui", action="store_true", default=True,
                       help="Show interactive GUI (default)")
    
    # Output options
    parser.add_argument("--save-dir", type=str,
                       help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # Create app instance
    app = RadiationFieldApp(seed=args.seed, checkpoint_path=args.checkpoint)
    
    # Generate sample
    app.generate_sample(n_sources=args.sources)
    
    # Prepare save paths if needed
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose visualization mode
    if args.show_gt:
        # Non-interactive mode
        save_path = None
        if save_dir:
            save_path = save_dir / "ground_truth.png"
        app.visualize_ground_truth(save_path)
    elif args.gui:
        # Interactive GUI mode (default)
        gui = RadiationFieldGUI(app)
        gui.run()
    else:
        print("Use --show-gt for non-interactive mode or --gui for interactive mode.")


if __name__ == "__main__":
    main()