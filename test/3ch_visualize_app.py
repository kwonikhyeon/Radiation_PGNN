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
from typing import Optional, Dict, Any, List, Tuple

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
from modules.rrt_path_planner import RRTPathPlanner, RRTParameters, plan_rrt_path


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
        self.root.geometry("1800x1600")  # Increased height for 3rd row
        
        # Create matplotlib figure with 3 rows: 3 plots per row
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 16))
        self.fig.suptitle("Radiation Field Analysis")
        
        # Default layer weights (must sum to 1.0)
        self.layer_weights = {
            'safety': 0.4,
            'information_gain': 0.4, 
            'traversability': 0.2
        }
        
        # Simulation mode variables
        self.simulation_mode = False
        self.simulation_measurements = []  # List of (y, x) measurement points
        self.simulation_values = []        # List of measured values

        # Auto mode variables
        self.auto_mode = False
        self.auto_job_id = None  # For scheduled auto measurements
        self.auto_iteration = 0
        self.auto_termination_criteria = {
            'max_iterations': 50,
            'convergence_threshold': 0.01,
            'min_measurements': 20,
            'max_measurements': 100
        }

        # Path planning method selection
        self.path_planning_method = "iterative"  # "iterative" or "rrt"
        self.rrt_planner = None
        self.last_rrt_result = None  # Store last RRT planning result for visualization
        
        # Coordinate conversion: 256x256 pixels = 10m x 10m
        self.field_size_pixels = 256
        self.field_size_meters = 10.0
        self.pixels_per_meter = self.field_size_pixels / self.field_size_meters  # ~25.6 pixels/meter
        
        # Starting position: 1m, 1m from origin (0,0) in meters -> convert to pixels
        start_x_meters = 1.0
        start_y_meters = 1.0
        self.simulation_start_pos = (int(start_y_meters * self.pixels_per_meter), 
                                   int(start_x_meters * self.pixels_per_meter))  # (~26, ~26) pixels
        self.simulation_current_pos = None
        
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
        ttk.Button(button_frame, text="Save T-Layer Debug", 
                  command=self.save_traversability_debug).pack(side=tk.LEFT, padx=(0, 10))
        
        # Path planning method selection
        path_method_frame = ttk.LabelFrame(main_frame, text="Path Planning Method")
        path_method_frame.pack(fill=tk.X, pady=(0, 10))

        self.path_method_var = tk.StringVar(value="iterative")

        ttk.Radiobutton(path_method_frame, text="Iterative (Current)",
                       variable=self.path_method_var, value="iterative",
                       command=self.on_path_method_change).pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Radiobutton(path_method_frame, text="RRT-based",
                       variable=self.path_method_var, value="rrt",
                       command=self.on_path_method_change).pack(side=tk.LEFT, padx=10, pady=5)

        # RRT parameters (initially hidden)
        self.rrt_params_frame = ttk.Frame(path_method_frame)

        ttk.Label(self.rrt_params_frame, text="Max Iter:").pack(side=tk.LEFT, padx=(20, 5))
        self.rrt_max_iter_var = tk.IntVar(value=500)
        ttk.Spinbox(self.rrt_params_frame, from_=100, to=2000, textvariable=self.rrt_max_iter_var,
                   width=6).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(self.rrt_params_frame, text="Step Size(m):").pack(side=tk.LEFT, padx=(10, 5))
        self.rrt_step_size_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(self.rrt_params_frame, from_=0.5, to=2.0, increment=0.1,
                   textvariable=self.rrt_step_size_var, width=6).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(self.rrt_params_frame, text="Min Step(m):").pack(side=tk.LEFT, padx=(10, 5))
        self.rrt_min_step_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(self.rrt_params_frame, from_=0.3, to=1.0, increment=0.1,
                   textvariable=self.rrt_min_step_var, width=6).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(self.rrt_params_frame, text="Exploration Radius(m):").pack(side=tk.LEFT, padx=(10, 5))
        self.rrt_radius_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(self.rrt_params_frame, from_=2.0, to=5.0, increment=0.5,
                   textvariable=self.rrt_radius_var, width=6).pack(side=tk.LEFT, padx=(0, 10))

        # Simulation mode controls
        sim_frame = ttk.LabelFrame(main_frame, text="Simulation Mode")
        sim_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side: mode toggle and measurement count
        sim_left_frame = ttk.Frame(sim_frame)
        sim_left_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.sim_mode_button = ttk.Button(sim_left_frame, text="Enter Simulation Mode", 
                                         command=self.toggle_simulation_mode)
        self.sim_mode_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Measurement count control
        ttk.Label(sim_left_frame, text="Points per measure:").pack(side=tk.LEFT, padx=(10, 5))
        self.n_measurements_var = tk.IntVar(value=5)
        self.n_measurements_spinbox = ttk.Spinbox(sim_left_frame, from_=1, to=20, 
                                                 textvariable=self.n_measurements_var, 
                                                 width=5)
        self.n_measurements_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Measure button (initially disabled)
        self.measure_button = ttk.Button(sim_left_frame, text="Measure",
                                        command=self.perform_measurement,
                                        state="disabled")
        self.measure_button.pack(side=tk.LEFT, padx=(10, 5))

        # Auto button (initially disabled)
        self.auto_button = ttk.Button(sim_left_frame, text="Auto",
                                     command=self.toggle_auto_mode,
                                     state="disabled")
        self.auto_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Right side: simulation status
        sim_right_frame = ttk.Frame(sim_frame)
        sim_right_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.sim_status_label = ttk.Label(sim_right_frame, text="Simulation: Inactive")
        self.sim_status_label.pack()
        
        # Weight control frame
        weight_frame = ttk.LabelFrame(main_frame, text="Layer Weights (must sum to 1.0)")
        weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create weight sliders
        self.weight_vars = {}
        self.weight_sliders = {}
        
        # Safety weight
        safety_frame = ttk.Frame(weight_frame)
        safety_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(safety_frame, text="Safety").pack()
        self.weight_vars['safety'] = tk.DoubleVar(value=self.layer_weights['safety'])
        self.weight_sliders['safety'] = ttk.Scale(safety_frame, from_=0.0, to=1.0, 
                                                 variable=self.weight_vars['safety'], 
                                                 orient=tk.VERTICAL, length=80,
                                                 command=self.on_weight_change)
        self.weight_sliders['safety'].pack()
        self.safety_label = ttk.Label(safety_frame, text=f"{self.layer_weights['safety']:.2f}")
        self.safety_label.pack()
        
        # Information Gain weight
        info_frame = ttk.Frame(weight_frame)
        info_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(info_frame, text="Info Gain").pack()
        self.weight_vars['information_gain'] = tk.DoubleVar(value=self.layer_weights['information_gain'])
        self.weight_sliders['information_gain'] = ttk.Scale(info_frame, from_=0.0, to=1.0,
                                                           variable=self.weight_vars['information_gain'],
                                                           orient=tk.VERTICAL, length=80,
                                                           command=self.on_weight_change)
        self.weight_sliders['information_gain'].pack()
        self.info_label_weight = ttk.Label(info_frame, text=f"{self.layer_weights['information_gain']:.2f}")
        self.info_label_weight.pack()
        
        # Traversability weight
        trav_frame = ttk.Frame(weight_frame)
        trav_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(trav_frame, text="Traversability").pack()
        self.weight_vars['traversability'] = tk.DoubleVar(value=self.layer_weights['traversability'])
        self.weight_sliders['traversability'] = ttk.Scale(trav_frame, from_=0.0, to=1.0,
                                                          variable=self.weight_vars['traversability'],
                                                          orient=tk.VERTICAL, length=80,
                                                          command=self.on_weight_change)
        self.weight_sliders['traversability'].pack()
        self.trav_label = ttk.Label(trav_frame, text=f"{self.layer_weights['traversability']:.2f}")
        self.trav_label.pack()
        
        # Reset button
        ttk.Button(weight_frame, text="Reset to Default", 
                  command=self.reset_weights).pack(side=tk.LEFT, padx=20, pady=5)
        
        # Sum display
        self.sum_label = ttk.Label(weight_frame, text="Sum: 1.00", foreground="green")
        self.sum_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
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

        # Reset RRT planner and path visualization when generating new GT
        if self.path_planning_method == "rrt":
            self.rrt_planner = None
            self.last_rrt_result = None  # Clear previous RRT result
            print("🔄 Reset RRT planner and path visualization for new GT")

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
    
    def on_weight_change(self, value=None):
        """Handle weight slider changes and update visualization.
        Automatically adjust other weights to maintain sum = 1.0."""
        
        # Prevent recursive calls during adjustment
        if hasattr(self, '_adjusting_weights') and self._adjusting_weights:
            return
        
        self._adjusting_weights = True
        
        try:
            # Get current values from sliders
            current_weights = {
                'safety': self.weight_vars['safety'].get(),
                'information_gain': self.weight_vars['information_gain'].get(), 
                'traversability': self.weight_vars['traversability'].get()
            }
            
            # Find which slider was changed by comparing with stored values
            changed_key = None
            for key, current_val in current_weights.items():
                if abs(current_val - self.layer_weights[key]) > 0.001:  # Small tolerance for float comparison
                    changed_key = key
                    break
            
            if changed_key is not None:
                # Calculate remaining weight to distribute
                new_value = current_weights[changed_key]
                remaining_weight = 1.0 - new_value
                
                # Get the other two weights that need to be adjusted
                other_keys = [k for k in current_weights.keys() if k != changed_key]
                
                # Get current sum of the other two weights
                other_sum = sum(self.layer_weights[k] for k in other_keys)
                
                if other_sum > 0 and remaining_weight >= 0:
                    # Redistribute remaining weight proportionally
                    for key in other_keys:
                        proportion = self.layer_weights[key] / other_sum
                        new_weight = remaining_weight * proportion
                        # Clamp to valid range
                        new_weight = max(0.0, min(1.0, new_weight))
                        self.layer_weights[key] = new_weight
                        self.weight_vars[key].set(new_weight)
                elif remaining_weight >= 0:
                    # If other_sum is 0, distribute equally
                    equal_weight = remaining_weight / len(other_keys)
                    for key in other_keys:
                        self.layer_weights[key] = equal_weight
                        self.weight_vars[key].set(equal_weight)
                else:
                    # If remaining weight is negative, clamp the changed value
                    new_value = min(new_value, 1.0)
                    current_weights[changed_key] = new_value
                    self.weight_vars[changed_key].set(new_value)
                    remaining_weight = 1.0 - new_value
                    
                    # Set other weights to zero if no remaining weight
                    for key in other_keys:
                        self.layer_weights[key] = 0.0
                        self.weight_vars[key].set(0.0)
                
                # Update the changed weight
                self.layer_weights[changed_key] = new_value
            
            # Update all labels
            self.safety_label.config(text=f"{self.layer_weights['safety']:.2f}")
            self.info_label_weight.config(text=f"{self.layer_weights['information_gain']:.2f}")
            self.trav_label.config(text=f"{self.layer_weights['traversability']:.2f}")
            
            # Display sum (should always be 1.0 now)
            total_sum = sum(self.layer_weights.values())
            self.sum_label.config(text=f"Sum: {total_sum:.2f}", foreground="green")
            
            # Update visualization
            self.update_visualization()
            
        finally:
            self._adjusting_weights = False
    
    def reset_weights(self):
        """Reset weights to default values."""
        # Prevent weight change callbacks during reset
        self._adjusting_weights = True
        
        try:
            default_weights = {'safety': 0.4, 'information_gain': 0.4, 'traversability': 0.2}
            
            for layer, weight in default_weights.items():
                self.layer_weights[layer] = weight
                self.weight_vars[layer].set(weight)
            
            # Update labels
            self.safety_label.config(text=f"{self.layer_weights['safety']:.2f}")
            self.info_label_weight.config(text=f"{self.layer_weights['information_gain']:.2f}")
            self.trav_label.config(text=f"{self.layer_weights['traversability']:.2f}")
            
            # Update sum display
            self.sum_label.config(text="Sum: 1.00", foreground="green")
            
            # Update visualization
            self.update_visualization()
            
        finally:
            self._adjusting_weights = False

    def on_path_method_change(self):
        """Handle path planning method change."""
        method = self.path_method_var.get()
        self.path_planning_method = method

        if method == "rrt":
            # Show RRT parameters
            self.rrt_params_frame.pack(side=tk.LEFT, padx=10, pady=5)
            # Initialize RRT planner
            self._initialize_rrt_planner()
            self.info_label.config(text="Switched to RRT-based path planning")
        else:
            # Hide RRT parameters
            self.rrt_params_frame.pack_forget()
            self.rrt_planner = None
            self.info_label.config(text="Switched to iterative path planning")

    def _initialize_rrt_planner(self):
        """Initialize RRT planner with current parameters."""
        params = RRTParameters(
            max_iterations=self.rrt_max_iter_var.get(),
            step_size_meters=self.rrt_step_size_var.get(),
            min_step_size_meters=self.rrt_min_step_var.get(),
            exploration_radius_meters=self.rrt_radius_var.get(),
            n_steps=self.n_measurements_var.get(),
            use_rrt_star=True,
            goal_bias=0.2,
            source_avoidance=False,  # Disable planning-time avoidance
            post_process_source_avoidance=True,  # Enable post-processing avoidance
            dynamic_avoidance=True  # Use dynamic radius based on intensity
        )
        self.rrt_planner = RRTPathPlanner(params)

    def _update_rrt_parameters(self):
        """Update RRT planner parameters from UI values."""
        if self.rrt_planner is not None:
            # Update all parameters from current UI values
            self.rrt_planner.params.max_iterations = self.rrt_max_iter_var.get()
            self.rrt_planner.params.step_size_meters = self.rrt_step_size_var.get()
            self.rrt_planner.params.min_step_size_meters = self.rrt_min_step_var.get()
            self.rrt_planner.params.exploration_radius_meters = self.rrt_radius_var.get()

            # Ensure post-processing parameters are set
            self.rrt_planner.params.source_avoidance = False
            self.rrt_planner.params.post_process_source_avoidance = True
            self.rrt_planner.params.dynamic_avoidance = True

            print(f"🔄 Updated RRT parameters: max_iter={self.rrt_planner.params.max_iterations}, "
                  f"step={self.rrt_planner.params.step_size_meters:.1f}m, "
                  f"min_step={self.rrt_planner.params.min_step_size_meters:.1f}m, "
                  f"radius={self.rrt_planner.params.exploration_radius_meters:.1f}m, "
                  f"post_process_avoidance={self.rrt_planner.params.post_process_source_avoidance}")

            # Reset pixels_per_meter to ensure proper conversion
            if hasattr(self.rrt_planner, 'pixels_per_meter'):
                self.rrt_planner.pixels_per_meter = 26.0  # 256 pixels / 10 meters

    def toggle_simulation_mode(self):
        """Toggle between normal mode and simulation mode."""
        if not self.simulation_mode:
            # Enter simulation mode
            if self.app.current_field is None:
                self.info_label.config(text="No GT available. Generate GT first.")
                return
            
            self.simulation_mode = True
            self.simulation_measurements = []
            self.simulation_values = []
            self.simulation_current_pos = self.simulation_start_pos
            
            # Reset app to clean state (no measurements)
            self.app.current_prediction = None
            self.app.current_safety_layer = None
            self.app.current_information_gain = None
            self.app.current_traversability_layer = None
            
            # Update metadata to reflect no measurements
            self.app.current_metadata = {
                'measurement_points': np.empty((0, 2)),  # Ensure correct 2D shape
                'n_measurements': 0,
                'coords': []
            }
            
            # Update UI
            self.sim_mode_button.config(text="Exit Simulation Mode")
            self.measure_button.config(state="normal")
            self.auto_button.config(state="normal")
            # Display position in meters for user-friendly format
            start_meter_y = self.simulation_current_pos[0] / self.pixels_per_meter
            start_meter_x = self.simulation_current_pos[1] / self.pixels_per_meter
            self.sim_status_label.config(text=f"Simulation: Active | Position: ({start_meter_x:.1f}m, {start_meter_y:.1f}m)")
            
            # Disable other buttons in simulation mode
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for button in widget.winfo_children():
                            if isinstance(button, ttk.Button) and button != self.sim_mode_button and button != self.measure_button:
                                button.config(state="disabled")
            
            self.info_label.config(text="Simulation mode activated. Ready to measure.")
            
        else:
            # Exit simulation mode
            self.simulation_mode = False
            self.simulation_measurements = []
            self.simulation_values = []
            self.simulation_current_pos = None
            
            # Update UI
            self.sim_mode_button.config(text="Enter Simulation Mode")
            self.measure_button.config(state="disabled")
            self.auto_button.config(state="disabled")
            self.sim_status_label.config(text="Simulation: Inactive")

            # Stop auto mode if active
            if self.auto_mode:
                self._stop_auto_mode()
            
            # Re-enable other buttons
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for button in widget.winfo_children():
                            if isinstance(button, ttk.Button):
                                button.config(state="normal")
            
            self.info_label.config(text="Simulation mode deactivated.")
        
        # Update visualization
        self.update_visualization()
    
    def perform_measurement(self):
        """Perform n measurement points starting from current position."""
        try:
            if not self.simulation_mode:
                print("❌ Not in simulation mode")
                return
            
            if self.app.current_field is None:
                print("❌ No ground truth field available")
                return
            
            n = self.n_measurements_var.get()
            
            print(f"🎯 Simulation mode: Performing measurement with n={n}")
            print(f"📍 Current measurements: {len(self.simulation_measurements)}")
            print(f"🔧 Debug: simulation_current_pos = {self.simulation_current_pos}")
            
            # For initial measurements, start from the predefined start position
            if not self.simulation_measurements:
                current_y, current_x = self.simulation_current_pos
                # Convert to meters for display
                start_meter_y = current_y / self.pixels_per_meter
                start_meter_x = current_x / self.pixels_per_meter
                print(f"🚀 Starting measurements from initial position: ({start_meter_y:.1f}m, {start_meter_x:.1f}m) = ({current_y}, {current_x}) pixels")
            else:
                # Continue from last measurement position
                current_y, current_x = self.simulation_current_pos
                # Convert to meters for display
                current_meter_y = current_y / self.pixels_per_meter
                current_meter_x = current_x / self.pixels_per_meter
                print(f"🔄 Continuing measurements from position: ({current_meter_y:.1f}m, {current_meter_x:.1f}m) = ({current_y}, {current_x}) pixels")
                
        except Exception as e:
            print(f"❌ Error in perform_measurement setup: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Determine measurement strategy
        new_measurements = []
        new_values = []
        
        if not self.simulation_measurements:
            # First measurement cycle: use initial downward path
            print("📏 Using initial downward measurement pattern")
            
            for i in range(n):
                # Calculate measurement position (going downward by 1 meter steps)
                meter_step = int(1.0 * self.pixels_per_meter)  # ~26 pixels per meter
                measure_y = current_y + (i * meter_step)
                measure_x = current_x
                
                # Check bounds
                H, W = self.app.current_field.shape
                if measure_y >= H or measure_x >= W or measure_y < 0 or measure_x < 0:
                    meter_y = measure_y / self.pixels_per_meter
                    meter_x = measure_x / self.pixels_per_meter
                    print(f"Measurement point ({meter_y:.1f}m, {meter_x:.1f}m) = ({measure_y}, {measure_x}) pixels is out of bounds")
                    break
                
                # Get measurement value from GT field
                measurement_value = self.app.current_field[measure_y, measure_x]
                
                # Convert coordinates back to meters for display
                meter_y = measure_y / self.pixels_per_meter
                meter_x = measure_x / self.pixels_per_meter
                print(f"  📍 Measurement {len(self.simulation_measurements)+1}: ({meter_y:.1f}m, {meter_x:.1f}m) = ({measure_y}, {measure_x}) pixels, value={measurement_value:.4f}")
                
                new_measurements.append((measure_y, measure_x))
                new_values.append(measurement_value)
                
                # Add to simulation data
                self.simulation_measurements.append((measure_y, measure_x))
                self.simulation_values.append(measurement_value)
                
        else:
            # Subsequent measurements: choose method based on selected planning approach
            # Start from current position
            current_pos = (current_y, current_x)

            if self.path_planning_method == "rrt":
                print("🎯 Using RRT-based path planning approach")
                new_measurements, new_values = self._perform_rrt_measurement(current_pos, n)
            else:
                # Default iterative 2m radius approach
                print("🎯 Using 2m radius iterative measurement approach")
                new_measurements, new_values = self._perform_iterative_measurement(current_pos, n)

            # Add new measurements to simulation data
            for pos, val in zip(new_measurements, new_values):
                self.simulation_measurements.append(pos)
                self.simulation_values.append(val)
        
        # Update current position to last measurement
        if new_measurements:
            self.simulation_current_pos = new_measurements[-1]
        
        # Create measurements array and mask for model prediction
        H, W = self.app.current_field.shape
        
        # Create measurements array in the correct format (N, 2) for visualization
        if self.simulation_measurements:
            measurements_array = np.array(self.simulation_measurements)
        else:
            measurements_array = np.empty((0, 2))
        
        # Create measurement matrix and mask
        measurements = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.uint8)
        
        for (y, x), value in zip(self.simulation_measurements, self.simulation_values):
            measurements[y, x] = value
            mask[y, x] = 1
        
        # Update app metadata
        self.app.current_metadata = {
            'measurement_points': measurements_array,
            'n_measurements': len(self.simulation_measurements),
            'coords': self.app.current_metadata.get('coords', []) if self.app.current_metadata else []
        }
        
        # Get model prediction if available
        if self.app.model_inference:
            try:
                prediction = self.app.model_inference.predict(measurements, mask)
                self.app.current_prediction = prediction
                
                # Calculate layers
                # Safety layer - using the correct class method
                safety_result = self.app.safety_calculator.calculate_safety_layer(prediction, mask)
                self.app.current_safety_layer = safety_result
                
                # Information gain layer - using correct function signature
                from modules.information_gain import calculate_information_gain_layer
                info_result = calculate_information_gain_layer(
                    single_prediction=prediction, 
                    measurement_mask=mask
                )
                self.app.current_information_gain = info_result
                
                # Traversability layer - using correct function signature
                from modules.traversability_layer import calculate_traversability_layer
                if len(self.simulation_measurements) > 0:
                    robot_idx = np.random.randint(len(self.simulation_measurements))
                    robot_pos = self.simulation_measurements[robot_idx]
                    robot_heading = np.random.uniform(0, 2*np.pi)
                    trav_result = calculate_traversability_layer(
                        radiation_field=prediction,
                        measurement_mask=mask,
                        robot_position=robot_pos,
                        robot_heading=robot_heading
                    )
                    self.app.current_traversability_layer = trav_result
                
                print(f"✅ Model prediction, safety layer, information gain, and traversability completed")
                
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                self.app.current_prediction = None
        
        # Update status and visualization
        total_points = len(self.simulation_measurements)
        
        try:
            # Update status with current position in meters
            current_meter_y = self.simulation_current_pos[0] / self.pixels_per_meter
            current_meter_x = self.simulation_current_pos[1] / self.pixels_per_meter
            self.sim_status_label.config(text=f"Simulation: Active | Position: ({current_meter_x:.1f}m, {current_meter_y:.1f}m) | Points: {total_points}")
            self.info_label.config(text=f"Measured {n} points. Total: {total_points} points.")
            
            self.update_visualization()
            print(f"✅ Successfully completed measurement cycle. Total points: {total_points}")
            
        except Exception as e:
            print(f"❌ Error in measurement finalization: {e}")
            import traceback
            traceback.print_exc()

    def toggle_auto_mode(self):
        """Toggle auto mode on/off."""
        if not self.auto_mode:
            self._start_auto_mode()
        else:
            self._stop_auto_mode()

    def _start_auto_mode(self):
        """Start auto mode."""
        if not self.simulation_mode:
            self.info_label.config(text="Auto mode requires simulation mode")
            return

        self.auto_mode = True
        self.auto_iteration = 0

        # Update UI
        self.auto_button.config(text="Stop Auto")
        self.measure_button.config(state="disabled")  # Disable manual measure during auto

        # Start auto measurement
        self._schedule_auto_measurement()
        self.info_label.config(text="Auto mode started...")

    def _stop_auto_mode(self):
        """Stop auto mode."""
        self.auto_mode = False

        # Cancel any scheduled auto measurement
        if self.auto_job_id is not None:
            self.root.after_cancel(self.auto_job_id)
            self.auto_job_id = None

        # Update UI
        self.auto_button.config(text="Auto")
        if self.simulation_mode:
            self.measure_button.config(state="normal")  # Re-enable manual measure

        self.info_label.config(text=f"Auto mode stopped after {self.auto_iteration} iterations")

    def _schedule_auto_measurement(self):
        """Schedule next auto measurement with 1 second delay."""
        if self.auto_mode:
            self.auto_job_id = self.root.after(1000, self._perform_auto_measurement)

    def _perform_auto_measurement(self):
        """Perform auto measurement and check termination conditions."""
        if not self.auto_mode:
            return

        self.auto_iteration += 1
        print(f"🤖 Auto iteration {self.auto_iteration}")

        # Perform measurement
        self.perform_measurement()

        # Check termination criteria
        if self._check_auto_termination():
            self._stop_auto_mode()
        else:
            # Schedule next measurement
            self._schedule_auto_measurement()

    def _check_auto_termination(self) -> bool:
        """Check if auto mode should terminate based on criteria."""
        criteria = self.auto_termination_criteria
        n_measurements = len(self.simulation_measurements)

        # Max iterations check
        if self.auto_iteration >= criteria['max_iterations']:
            self.info_label.config(text=f"Auto terminated: max iterations ({criteria['max_iterations']})")
            return True

        # Max measurements check
        if n_measurements >= criteria['max_measurements']:
            self.info_label.config(text=f"Auto terminated: max measurements ({criteria['max_measurements']})")
            return True

        # Min measurements not yet reached
        if n_measurements < criteria['min_measurements']:
            return False

        # Convergence check (if we have enough measurements and a prediction)
        if (n_measurements >= criteria['min_measurements'] and
            self.app.current_prediction is not None and
            len(self.simulation_measurements) >= 10):  # Need some history for convergence

            # Simple convergence: check if recent measurements are in similar radiation areas
            recent_measurements = self.simulation_measurements[-5:]  # Last 5 measurements
            recent_values = self.simulation_values[-5:]

            if len(recent_values) >= 5:
                value_std = np.std(recent_values)
                if value_std < criteria['convergence_threshold']:
                    self.info_label.config(text=f"Auto terminated: convergence (std={value_std:.3f})")
                    return True

        return False

    def calculate_weighted_map(self):
        """Calculate weighted combination of all layers."""
        if (self.app.current_safety_layer is None or 
            self.app.current_information_gain is None or 
            self.app.current_traversability_layer is None):
            return None, "Missing layer data"
        
        # Get layer data
        safety = self.app.current_safety_layer['total_safety']
        info_gain = self.app.current_information_gain['information_gain']
        traversability = self.app.current_traversability_layer['total_traversability']
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.layer_weights.values())
        if total_weight == 0:
            return None, "All weights are zero"
        
        normalized_weights = {k: v/total_weight for k, v in self.layer_weights.items()}
        
        # Calculate weighted combination
        weighted_map = (normalized_weights['safety'] * safety + 
                       normalized_weights['information_gain'] * info_gain + 
                       normalized_weights['traversability'] * traversability)
        
        # Create info string
        info_str = (f"Weights: S={normalized_weights['safety']:.2f}, "
                   f"I={normalized_weights['information_gain']:.2f}, "
                   f"T={normalized_weights['traversability']:.2f}")
        
        return weighted_map, info_str
    
    def generate_candidate_positions(self, n_candidates=100):
        """Generate random candidate positions for measurement selection."""
        H, W = self.app.current_field.shape
        
        # Generate random positions
        candidates = []
        for _ in range(n_candidates):
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
            candidates.append((y, x))
        
        return candidates

    def generate_candidates_in_radius(self, center_pos, radius_meters=2.0, n_candidates=50):
        """Generate random candidate positions within 1-2m range from center position."""
        import random
        import math

        H, W = self.app.current_field.shape
        center_y, center_x = center_pos

        # Convert radius from meters to pixels
        min_radius_pixels = 1.0 * self.pixels_per_meter  # 1m minimum
        max_radius_pixels = radius_meters * self.pixels_per_meter  # 2m maximum

        candidates = []
        attempts = 0
        max_attempts = n_candidates * 10  # Prevent infinite loop

        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1

            # Generate random point in annulus (ring between 1m and 2m)
            # Use polar coordinates for uniform distribution
            r_min_sq = min_radius_pixels ** 2
            r_max_sq = max_radius_pixels ** 2
            r = math.sqrt(r_min_sq + (r_max_sq - r_min_sq) * random.random())
            theta = 2 * math.pi * random.random()

            # Convert to cartesian coordinates
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)

            # Calculate candidate position
            cand_x = int(center_x + dx)
            cand_y = int(center_y + dy)

            # Check bounds
            if 0 <= cand_y < H and 0 <= cand_x < W:
                candidates.append((cand_y, cand_x))

        return candidates

    def _perform_iterative_measurement(self, current_pos: Tuple[int, int], n: int) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Perform iterative 2m radius measurement approach."""
        new_measurements = []
        new_values = []

        # Perform n iterations of measurement
        for step in range(n):
            print(f"🔄 Step {step + 1}/{n}")

            # Generate candidate points within 2m radius
            candidates = self.generate_candidates_in_radius(current_pos, radius_meters=2.0)

            if not candidates:
                print(f"  ❌ No valid candidates in 2m radius from {current_pos}")
                break

            # Calculate weighted map with current measurements
            weighted_map, _ = self.calculate_weighted_map()

            if weighted_map is None:
                print(f"  ❌ Could not calculate weighted map for step {step + 1}")
                break

            # Select candidate with highest combined weight (traversability + direction continuity)
            best_pos = None
            best_combined_weight = -1.0

            # Calculate previous direction if we have measurement history
            prev_direction = None
            if len(self.simulation_measurements) >= 2:
                prev_y1, prev_x1 = self.simulation_measurements[-2]
                prev_y2, prev_x2 = self.simulation_measurements[-1]
                prev_direction = (prev_y2 - prev_y1, prev_x2 - prev_x1)
                # Normalize
                prev_len = (prev_direction[0]**2 + prev_direction[1]**2)**0.5
                if prev_len > 0:
                    prev_direction = (prev_direction[0]/prev_len, prev_direction[1]/prev_len)

            for candidate in candidates:
                cand_y, cand_x = candidate
                cur_y, cur_x = current_pos

                # Get base traversability weight
                base_weight = weighted_map[cand_y, cand_x]

                # Calculate direction continuity bonus
                direction_bonus = 0.0
                if prev_direction is not None:
                    # Calculate current direction
                    cur_direction = (cand_y - cur_y, cand_x - cur_x)
                    cur_len = (cur_direction[0]**2 + cur_direction[1]**2)**0.5
                    if cur_len > 0:
                        cur_direction = (cur_direction[0]/cur_len, cur_direction[1]/cur_len)

                        # Calculate dot product (cosine of angle between directions)
                        dot_product = (prev_direction[0] * cur_direction[0] +
                                     prev_direction[1] * cur_direction[1])

                        # Bonus for continuing in similar direction (avoid sharp turns)
                        if dot_product > 0.5:  # Within 60 degrees
                            direction_bonus = 0.2 * dot_product  # Up to 0.2 bonus

                # Combine weights
                combined_weight = base_weight + direction_bonus

                if combined_weight > best_combined_weight:
                    best_combined_weight = combined_weight
                    best_pos = candidate

            if best_pos is None:
                print(f"  ❌ No valid best position found for step {step + 1}")
                break

            # Move to best position and take measurement
            measure_y, measure_x = best_pos
            current_pos = best_pos

            # Get measurement value from GT field
            measurement_value = self.app.current_field[measure_y, measure_x]

            # Convert coordinates to meters for display
            meter_y = measure_y / self.pixels_per_meter
            meter_x = measure_x / self.pixels_per_meter
            print(f"  🎯 Step {step + 1}: Move to ({meter_x:.1f}m, {meter_y:.1f}m) = ({measure_x}, {measure_y}) pixels, value={measurement_value:.4f}, combined_weight={best_combined_weight:.4f}")

            # Add to accumulated measurements
            new_measurements.append((measure_y, measure_x))
            new_values.append(measurement_value)

        return new_measurements, new_values

    def _perform_rrt_measurement(self, current_pos: Tuple[int, int], n: int) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Perform RRT-based measurement approach."""
        new_measurements = []
        new_values = []

        try:
            # Initialize or update RRT planner
            if self.rrt_planner is None:
                self._initialize_rrt_planner()
            else:
                # Update all parameters from UI
                self._update_rrt_parameters()

            # Update n_steps parameter
            self.rrt_planner.params.n_steps = n

            # Set environment for RRT planner
            weighted_map, _ = self.calculate_weighted_map()
            if weighted_map is None:
                print("❌ Could not calculate weighted map for RRT planning")
                return self._perform_iterative_measurement(current_pos, n)  # Fallback

            # Get predicted sources from current field for source avoidance
            predicted_sources = self._get_predicted_sources()

            self.rrt_planner.set_environment(
                radiation_field=self.app.current_field,
                weighted_map=weighted_map,
                predicted_sources=predicted_sources
            )

            # Plan path using RRT
            print(f"🌳 Planning RRT path with {n} steps...")
            result = self.rrt_planner.plan_path(
                start_position=current_pos,
                measurement_positions=self.simulation_measurements
            )

            if result.success and len(result.path) > 1:
                # Use planned path (skip start position as it's current position)
                planned_positions = result.path[1:]  # Skip start position

                print(f"✅ RRT planning successful! Planned {len(planned_positions)} positions")
                print(f"   Total reward: {result.total_reward:.3f}, Cost: {result.total_cost:.3f}")
                print(f"   Planning time: {result.planning_time:.3f}s, Tree size: {len(result.tree_nodes)}")

                # Analyze step distances for minimum distance validation
                pixels_per_meter = 26.0  # 256 pixels / 10 meters
                step_distances = []
                prev_pos_temp = current_pos
                for step_pos in planned_positions:
                    distance_pixels = np.sqrt((step_pos[0] - prev_pos_temp[0])**2 + (step_pos[1] - prev_pos_temp[1])**2)
                    distance_meters = distance_pixels / pixels_per_meter
                    step_distances.append(distance_meters)
                    prev_pos_temp = step_pos

                if step_distances:
                    min_distance = min(step_distances)
                    max_distance = max(step_distances)
                    avg_distance = sum(step_distances) / len(step_distances)
                    total_distance = sum(step_distances)
                    min_threshold = 0.5  # 0.5m minimum

                    print(f"📏 Step distance analysis:")
                    for i, dist in enumerate(step_distances, 1):
                        print(f"   Step {i}: {dist:.3f}m")

                    print(f"📊 Distance summary:")
                    print(f"   Min: {min_distance:.3f}m (threshold: {min_threshold:.3f}m)")
                    print(f"   Max: {max_distance:.3f}m")
                    print(f"   Avg: {avg_distance:.3f}m")
                    print(f"   Total: {total_distance:.3f}m")

                    violations = [i+1 for i, dist in enumerate(step_distances) if dist < min_threshold]
                    if violations:
                        print(f"⚠️  Minimum distance violations at steps: {violations}")
                    else:
                        print(f"✅ All steps satisfy minimum distance constraint")

                # Take measurements at planned positions
                prev_pos = current_pos
                for i, (measure_y, measure_x) in enumerate(planned_positions):
                    # Calculate actual movement distance
                    distance_pixels = np.sqrt((measure_y - prev_pos[0])**2 + (measure_x - prev_pos[1])**2)
                    distance_meters = distance_pixels / self.pixels_per_meter

                    # Get measurement value from GT field
                    measurement_value = self.app.current_field[measure_y, measure_x]

                    # Convert coordinates to meters for display
                    meter_y = measure_y / self.pixels_per_meter
                    meter_x = measure_x / self.pixels_per_meter
                    print(f"  🎯 RRT Step {i + 1}: Move {distance_meters:.2f}m to ({meter_x:.1f}m, {meter_y:.1f}m), value={measurement_value:.4f}")

                    new_measurements.append((measure_y, measure_x))
                    new_values.append(measurement_value)
                    prev_pos = (measure_y, measure_x)

                # Store RRT result for visualization
                self.last_rrt_result = result

            else:
                print(f"❌ RRT planning failed or no path found, falling back to iterative approach")
                return self._perform_iterative_measurement(current_pos, n)

        except Exception as e:
            print(f"❌ RRT measurement error: {e}")
            print("   Falling back to iterative approach")
            return self._perform_iterative_measurement(current_pos, n)

        return new_measurements, new_values

    def _get_predicted_sources(self) -> Optional[List[Tuple[int, int, float]]]:
        """Get predicted source locations with intensity from current field."""
        if self.app.current_field is None:
            return None

        try:
            # Use source detection module to find sources in current field
            from modules.source_detection import detect_radiation_sources, SourceDetectionParameters

            # Create detection parameters
            params = SourceDetectionParameters(
                detection_threshold=0.4,  # Lower threshold to catch more potential sources
                max_sources=10
            )

            # Detect sources using the current field
            result = detect_radiation_sources(self.app.current_field, params)
            sources = result.get('source_locations', [])
            source_intensities = result.get('source_intensities', [])

            if sources and len(sources) > 0:
                # Create source data with intensity: (y, x, intensity)
                source_data = []
                for i, (y, x) in enumerate(sources):
                    # Use detected intensity if available, otherwise sample from field
                    if i < len(source_intensities):
                        intensity = source_intensities[i]
                    else:
                        intensity = self.app.current_field[int(y), int(x)]

                    source_data.append((int(y), int(x), float(intensity)))

                print(f"🔍 Detected {len(source_data)} sources for dynamic avoidance")
                for i, (y, x, intensity) in enumerate(source_data):
                    print(f"   Source {i+1}: ({y}, {x}) intensity={intensity:.3f}")
                return source_data
            else:
                print("🔍 No sources detected for avoidance")
                return []

        except Exception as e:
            print(f"❌ Source detection error: {e}")
            return None

    def draw_measurement_path(self, ax, measurement_points):
        """Draw lines connecting measurement points to show the path (simulation mode only)."""
        if not self.simulation_mode or len(measurement_points) <= 1:
            return

        # Draw RRT tree if available and using RRT method
        if (self.path_planning_method == "rrt" and
            hasattr(self, 'last_rrt_result') and
            self.last_rrt_result is not None):
            self._draw_rrt_tree(ax, self.last_rrt_result)

        # Draw lines connecting consecutive measurement points
        for i in range(len(measurement_points) - 1):
            x1, y1 = measurement_points[i][1], measurement_points[i][0]  # (x, y) format for plot
            x2, y2 = measurement_points[i + 1][1], measurement_points[i + 1][0]
            ax.plot([x1, x2], [y1, y2], 'cyan', linewidth=2, alpha=0.7)

        # Add arrow to show direction at the end
        if len(measurement_points) >= 2:
            x1, y1 = measurement_points[-2][1], measurement_points[-2][0]
            x2, y2 = measurement_points[-1][1], measurement_points[-1][0]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='cyan', lw=2, alpha=0.8))

    def _draw_rrt_tree(self, ax, rrt_result):
        """Draw RRT tree structure on the plot."""
        if not rrt_result.tree_nodes:
            return

        # Draw tree edges in light gray
        for node in rrt_result.tree_nodes:
            if node.parent is not None:
                x1, y1 = node.parent.position[1], node.parent.position[0]
                x2, y2 = node.position[1], node.position[0]
                ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)

        # Draw tree nodes as small dots
        positions = [node.position for node in rrt_result.tree_nodes]
        if positions:
            positions_array = np.array(positions)
            ax.scatter(positions_array[:, 1], positions_array[:, 0],
                      c='lightblue', s=5, alpha=0.5, marker='.')

        # Highlight planned path in red
        if rrt_result.path and len(rrt_result.path) > 1:
            path_array = np.array(rrt_result.path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'red', linewidth=2, alpha=0.8, linestyle='--')
            ax.scatter(path_array[:, 1], path_array[:, 0],
                      c='red', s=20, marker='s', edgecolor='black', linewidth=0.5, alpha=0.7)

    def calculate_path_weight(self, target_positions, weighted_map):
        """Calculate the total weight of a path from start to target positions."""
        total_weight = 0.0
        
        for target_pos in target_positions:
            # Add weight of the target position
            total_weight += weighted_map[target_pos[0], target_pos[1]]
        
        return total_weight
    
    def find_optimal_measurement_path(self, start_pos, n_measurements, weighted_map):
        """Find optimal path for next n measurements using weighted map."""
        print(f"🎯 Finding optimal path for {n_measurements} measurements from {start_pos}")
        
        # Generate many candidate positions
        candidates = self.generate_candidate_positions(n_candidates=500)
        
        # Filter out already measured positions
        measured_positions = set(self.simulation_measurements)
        candidates = [pos for pos in candidates if pos not in measured_positions]
        
        if len(candidates) < n_measurements:
            print(f"⚠️  Not enough candidates ({len(candidates)}) for {n_measurements} measurements")
            # If not enough candidates, generate more in valid areas
            candidates = self.generate_candidate_positions(n_candidates=1000)
            candidates = [pos for pos in candidates if pos not in measured_positions]
        
        print(f"📊 Evaluating {len(candidates)} candidate positions")
        
        # Try different combinations of n measurements
        best_combination = None
        best_weight = -1.0
        max_combinations_to_try = 1000  # Limit for performance

        # Use iterative approach to find best combination
        if len(candidates) >= n_measurements:
            from itertools import combinations
            import math
            import random

            # Calculate expected number of combinations
            expected_combinations = math.comb(len(candidates), n_measurements)
            print(f"📈 Expected combinations: {expected_combinations:,}")

            # If too many combinations, use sampling approach
            if expected_combinations > max_combinations_to_try:
                print(f"⚡ Too many combinations ({expected_combinations:,}), using random sampling")
                combinations_to_try = []
                for _ in range(max_combinations_to_try):
                    combination = tuple(random.sample(candidates, n_measurements))
                    combinations_to_try.append(combination)
            else:
                all_combinations = list(combinations(candidates, n_measurements))
                combinations_to_try = all_combinations
                
            print(f"🔄 Testing {len(combinations_to_try)} combinations...")
            
            for i, combination in enumerate(combinations_to_try):
                if i % 100 == 0:  # Progress update
                    print(f"  Progress: {i}/{len(combinations_to_try)}")
                
                total_weight = self.calculate_path_weight(combination, weighted_map)
                
                if total_weight > best_weight:
                    best_weight = total_weight
                    best_combination = combination
            
            print(f"🎯 Best combination found with weight: {best_weight:.4f}")
            return list(best_combination)
        
        else:
            # Fallback: greedy selection
            print("📈 Using greedy selection fallback")
            selected_positions = []
            remaining_candidates = candidates.copy()
            
            for _ in range(min(n_measurements, len(remaining_candidates))):
                # Find candidate with highest weight
                best_pos = None
                best_weight = -1.0
                
                for pos in remaining_candidates:
                    weight = weighted_map[pos[0], pos[1]]
                    if weight > best_weight:
                        best_weight = weight
                        best_pos = pos
                
                if best_pos is not None:
                    selected_positions.append(best_pos)
                    remaining_candidates.remove(best_pos)
            
            return selected_positions
    
    def update_visualization(self):
        """Update the matplotlib visualization."""
        if self.app.current_field is None:
            return
        
        # Clear figure completely and recreate subplots
        self.fig.clear()
        self.axes = self.fig.subplots(3, 3)  # Changed to 3x3
        self.fig.suptitle("Radiation Field Analysis")
        
        if self.app.current_prediction is not None:
            # Row 1, Col 1: GT with measurement points
            self.app.visualizer.plot_ground_truth(self.app.current_field, self.app.current_metadata, self.axes[0, 0])

            # Add measurement path for GT plot (simulation mode only)
            if (self.simulation_mode and self.app.current_metadata and
                'measurement_points' in self.app.current_metadata):
                measurement_points = self.app.current_metadata['measurement_points']
                self.draw_measurement_path(self.axes[0, 0], measurement_points)
            
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
            has_legend_items = False
            if self.app.current_metadata:
                print(f"🔍 Debug: current_metadata keys: {list(self.app.current_metadata.keys())}")
                if 'coords' in self.app.current_metadata:
                    true_sources = self.app.current_metadata['coords']
                    print(f"🎯 Debug: Found {len(true_sources)} true sources: {true_sources}")
                    if len(true_sources) > 0:
                        true_array = np.array(true_sources)
                        self.axes[0, 2].scatter(true_array[:, 1], true_array[:, 0],
                                               c='cyan', s=80, marker='x', linewidth=3,
                                               alpha=0.9, label=f'{len(true_sources)} true sources')
                        has_legend_items = True
                else:
                    print("❌ Debug: 'coords' key not found in metadata")
            else:
                print("❌ Debug: current_metadata is None")

            # Add detected source locations if available
            if (self.app.current_traversability_layer and
                'source_locations' in self.app.current_traversability_layer):
                detected_sources = self.app.current_traversability_layer['source_locations']
                if detected_sources:
                    detected_array = np.array(detected_sources)
                    self.axes[0, 2].scatter(detected_array[:, 1], detected_array[:, 0],
                                           c='r', s=60, marker='x', linewidth=2,
                                           alpha=0.9, label=f'{len(detected_sources)} detected')
                    has_legend_items = True

            # Only show legend if there are labeled items
            if has_legend_items:
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
                    # Add measurement path (simulation mode only)
                    if self.simulation_mode:
                        self.draw_measurement_path(self.axes[1, 0], measurement_points)
                
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
                    # Add measurement path (simulation mode only)
                    if self.simulation_mode:
                        self.draw_measurement_path(self.axes[1, 1], measurement_points)
                
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
                    # Add measurement path (simulation mode only)
                    if self.simulation_mode:
                        self.draw_measurement_path(self.axes[1, 2], measurement_points)
                
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
        
        # Row 3: Weighted combinations and analysis
        if self.app.current_prediction is not None:
            self.add_third_row_visualizations()
        else:
            # No model loaded - fill Row 3 with placeholders
            for col in range(3):
                self.axes[2, col].text(0.5, 0.5, 'No model loaded\nUse --checkpoint to enable predictions', 
                                      ha='center', va='center', transform=self.axes[2, col].transAxes, fontsize=12)
                self.axes[2, col].axis('off')
            
            self.axes[2, 0].set_title("Final Weighted Map")
            self.axes[2, 1].set_title("Weight Comparison")  
            self.axes[2, 2].set_title("Layer Statistics")
        
        # Adjust layout to prevent overlapping
        self.fig.tight_layout()
        self.canvas.draw()
    
    def add_third_row_visualizations(self):
        """Add visualizations for the third row (weighted combinations)."""
        # Row 3, Col 1: Final weighted map
        weighted_map, weight_info = self.calculate_weighted_map()
        
        if weighted_map is not None:
            im_weighted = self.axes[2, 0].imshow(weighted_map, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
            self.axes[2, 0].set_title(f"Final Weighted Map\n{weight_info}")
            plt.colorbar(im_weighted, ax=self.axes[2, 0], fraction=0.046, pad=0.04)
            
            # Add measurement points overlay
            if self.app.current_metadata and 'measurement_points' in self.app.current_metadata:
                measurement_points = self.app.current_metadata['measurement_points']
                self.axes[2, 0].scatter(measurement_points[:, 1], measurement_points[:, 0],
                                       c='white', s=15, marker='o', edgecolor='black',
                                       linewidth=0.5, alpha=0.7)
                # Add measurement path (simulation mode only)
                if self.simulation_mode:
                    self.draw_measurement_path(self.axes[2, 0], measurement_points)
        else:
            self.axes[2, 0].text(0.5, 0.5, f'Weighted map\ncalculation failed:\n{weight_info}', 
                                ha='center', va='center', transform=self.axes[2, 0].transAxes, fontsize=10)
            self.axes[2, 0].set_title("Final Weighted Map")
        
        self.axes[2, 0].axis('off')
        
        # Row 3, Col 2: Weight comparison bar chart
        self.axes[2, 1].clear()
        weights = list(self.layer_weights.values())
        labels = ['Safety', 'Info Gain', 'Traversability']
        colors = ['green', 'orange', 'purple']
        
        bars = self.axes[2, 1].bar(labels, weights, color=colors, alpha=0.7)
        self.axes[2, 1].set_ylim(0, 1)
        self.axes[2, 1].set_ylabel('Weight')
        self.axes[2, 1].set_title('Layer Weights')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            self.axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{weight:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Row 3, Col 3: Layer statistics
        self.axes[2, 2].clear()
        self.axes[2, 2].axis('off')
        
        # Calculate statistics for each layer
        stats_text = "Layer Statistics:\n\n"
        
        if self.app.current_safety_layer is not None:
            safety_mean = self.app.current_safety_layer['total_safety'].mean()
            stats_text += f"Safety:\n  Mean: {safety_mean:.3f}\n\n"
        
        if self.app.current_information_gain is not None:
            info_mean = self.app.current_information_gain['information_gain'].mean()
            stats_text += f"Info Gain:\n  Mean: {info_mean:.3f}\n\n"
        
        if self.app.current_traversability_layer is not None:
            trav_mean = self.app.current_traversability_layer['total_traversability'].mean()
            stats_text += f"Traversability:\n  Mean: {trav_mean:.3f}\n\n"
        
        if weighted_map is not None:
            weighted_mean = weighted_map.mean()
            stats_text += f"Final Weighted:\n  Mean: {weighted_mean:.3f}"
        
        self.axes[2, 2].text(0.05, 0.95, stats_text, transform=self.axes[2, 2].transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.axes[2, 2].set_title("Layer Statistics")
    
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
    
    def save_traversability_debug(self):
        """Save current traversability layer data for debugging high values."""
        import pickle
        from datetime import datetime
        
        if self.app.current_traversability_layer is None:
            self.info_label.config(text="No traversability data to save!")
            return
        
        # Create debug data package
        debug_data = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.app.seed,
            
            # Core field data
            'gt_field': self.app.current_field.copy() if self.app.current_field is not None else None,
            'prediction': self.app.current_prediction.copy() if self.app.current_prediction is not None else None,
            'measurement_points': self.app.current_metadata.get('measurement_points', []).copy() if self.app.current_metadata else [],
            
            # Traversability layer data
            'traversability_layer': {
                key: value.copy() if isinstance(value, np.ndarray) else value 
                for key, value in self.app.current_traversability_layer.items()
            },
            
            # Statistics for quick analysis
            'stats': {
                'total_traversability_mean': self.app.current_traversability_layer['total_traversability'].mean(),
                'total_traversability_max': self.app.current_traversability_layer['total_traversability'].max(),
                'total_traversability_min': self.app.current_traversability_layer['total_traversability'].min(),
                'total_traversability_std': self.app.current_traversability_layer['total_traversability'].std(),
                
                'source_proximity_mean': self.app.current_traversability_layer.get('source_proximity', np.array([0])).mean(),
                'low_radiation_bonus_mean': self.app.current_traversability_layer.get('low_radiation_bonus', np.array([0])).mean(),
                'directional_navigation_mean': self.app.current_traversability_layer.get('directional_navigation', np.array([0])).mean(),
                
                'sources_detected': len(self.app.current_traversability_layer.get('source_locations', [])),
                'n_measurements': len(self.app.current_metadata.get('measurement_points', [])) if self.app.current_metadata else 0,
                
                'field_mean': self.app.current_field.mean() if self.app.current_field is not None else 0,
                'field_max': self.app.current_field.max() if self.app.current_field is not None else 0,
            }
        }
        
        # Add metadata if available
        if self.app.current_metadata:
            debug_data['metadata'] = self.app.current_metadata.copy()
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        mean_val = debug_data['stats']['total_traversability_mean']
        filename = f"t-layer samples/t_layer_debug_{timestamp_str}_mean{mean_val:.3f}.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(debug_data, f)
            
            info_text = (f"T-Layer debug saved to {filename}\n"
                        f"Mean: {mean_val:.3f}, Max: {debug_data['stats']['total_traversability_max']:.3f}, "
                        f"Sources: {debug_data['stats']['sources_detected']}")
            self.info_label.config(text=info_text)
            
            print(f"🔍 T-Layer Debug Data Saved:")
            print(f"  File: {filename}")
            print(f"  Mean traversability: {mean_val:.3f}")
            print(f"  Max traversability: {debug_data['stats']['total_traversability_max']:.3f}")
            print(f"  Sources detected: {debug_data['stats']['sources_detected']}")
            print(f"  Measurement points: {debug_data['stats']['n_measurements']}")
            
            # Alert if high values detected
            if mean_val > 0.7:
                print(f"  🚨 HIGH MEAN VALUE DETECTED: {mean_val:.3f}")
            if debug_data['stats']['total_traversability_max'] > 0.95:
                print(f"  🚨 HIGH MAX VALUE DETECTED: {debug_data['stats']['total_traversability_max']:.3f}")
                
        except Exception as e:
            error_msg = f"Error saving debug data: {str(e)}"
            self.info_label.config(text=error_msg)
            print(f"❌ {error_msg}")
    
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