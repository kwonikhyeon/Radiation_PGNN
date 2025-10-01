#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────
# test/rrt_test_app.py - Standalone RRT Path Planning Test Application
# ──────────────────────────────────────────────────────────────
"""
Standalone application for testing RRT path planning with various parameters.
Allows independent testing of RRT algorithm without the complexity of the full system.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add src to path for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import dataset.generate_truth as gt

# Import RRT module
from modules.rrt_path_planner import RRTPathPlanner, RRTParameters


class RRTTestApp:
    """Standalone RRT testing application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RRT Path Planning Test App")
        self.root.geometry("1400x800")

        # Current environment
        self.radiation_field = None
        self.weighted_map = None
        self.measurement_positions = []  # List of (y, x) measurement points
        self.current_position = None
        self.rrt_planner = None
        self.last_result = None

        # Field properties
        self.field_size_pixels = 256
        self.field_size_meters = 10.0
        self.pixels_per_meter = self.field_size_pixels / self.field_size_meters

        self.setup_ui()
        self.generate_test_environment()

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Environment controls
        env_frame = ttk.LabelFrame(control_frame, text="Environment")
        env_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(env_frame, text="Generate New Field",
                  command=self.generate_test_environment).pack(fill=tk.X, pady=2)

        ttk.Button(env_frame, text="Clear Measurements",
                  command=self.clear_measurements).pack(fill=tk.X, pady=2)

        ttk.Button(env_frame, text="Add Random Measurements",
                  command=self.add_random_measurements).pack(fill=tk.X, pady=2)

        # RRT Parameters
        rrt_frame = ttk.LabelFrame(control_frame, text="RRT Parameters")
        rrt_frame.pack(fill=tk.X, pady=(0, 10))

        # Max iterations
        ttk.Label(rrt_frame, text="Max Iterations:").pack(anchor=tk.W)
        self.max_iter_var = tk.IntVar(value=500)
        ttk.Spinbox(rrt_frame, from_=100, to=2000, textvariable=self.max_iter_var,
                   width=10).pack(fill=tk.X, pady=2)

        # Step size
        ttk.Label(rrt_frame, text="Step Size (m):").pack(anchor=tk.W)
        self.step_size_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rrt_frame, from_=0.5, to=3.0, increment=0.1,
                   textvariable=self.step_size_var, width=10).pack(fill=tk.X, pady=2)

        # Min step size
        ttk.Label(rrt_frame, text="Min Step Size (m):").pack(anchor=tk.W)
        self.min_step_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(rrt_frame, from_=0.1, to=1.0, increment=0.1,
                   textvariable=self.min_step_var, width=10).pack(fill=tk.X, pady=2)

        # Exploration radius
        ttk.Label(rrt_frame, text="Exploration Radius (m):").pack(anchor=tk.W)
        self.radius_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(rrt_frame, from_=1.0, to=6.0, increment=0.5,
                   textvariable=self.radius_var, width=10).pack(fill=tk.X, pady=2)

        # Number of steps
        ttk.Label(rrt_frame, text="Number of Steps:").pack(anchor=tk.W)
        self.n_steps_var = tk.IntVar(value=5)
        ttk.Spinbox(rrt_frame, from_=1, to=10, textvariable=self.n_steps_var,
                   width=10).pack(fill=tk.X, pady=2)

        # Goal bias
        ttk.Label(rrt_frame, text="Goal Bias:").pack(anchor=tk.W)
        self.goal_bias_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(rrt_frame, from_=0.0, to=1.0, increment=0.1,
                   textvariable=self.goal_bias_var, width=10).pack(fill=tk.X, pady=2)

        # Safety threshold
        ttk.Label(rrt_frame, text="Safety Threshold:").pack(anchor=tk.W)
        self.safety_threshold_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(rrt_frame, from_=0.1, to=1.0, increment=0.1,
                   textvariable=self.safety_threshold_var, width=10).pack(fill=tk.X, pady=2)

        # Source avoidance
        self.source_avoidance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rrt_frame, text="Avoid Sources",
                       variable=self.source_avoidance_var).pack(anchor=tk.W, pady=2)

        ttk.Label(rrt_frame, text="Source Avoidance Radius (m):").pack(anchor=tk.W)
        self.source_radius_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rrt_frame, from_=0.5, to=2.0, increment=0.1,
                   textvariable=self.source_radius_var, width=10).pack(fill=tk.X, pady=2)

        # Planning controls
        plan_frame = ttk.LabelFrame(control_frame, text="Path Planning")
        plan_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(plan_frame, text="Set Start Position (Click on Map)",
                  command=self.set_click_mode_start).pack(fill=tk.X, pady=2)

        ttk.Button(plan_frame, text="Plan RRT Path",
                  command=self.plan_rrt_path).pack(fill=tk.X, pady=2)

        ttk.Button(plan_frame, text="Test Multiple Runs",
                  command=self.test_multiple_runs).pack(fill=tk.X, pady=2)

        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Results")
        results_frame.pack(fill=tk.X, pady=(0, 10))

        self.results_text = tk.Text(results_frame, height=8, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Right panel for visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle("RRT Path Planning Test")

        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind click events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.click_mode = None

    def generate_test_environment(self):
        """Generate a test radiation field environment."""
        self.log("Generating new test environment...")

        # Generate radiation field using simple gaussian sources
        n_sources = np.random.randint(2, 6)  # 2-5 sources
        sources = gt.sample_sources(self.field_size_pixels, n_sources)
        coords, amps, sigmas = sources
        self.radiation_field = gt.gaussian_field(self.field_size_pixels, coords, amps, sigmas)

        # Normalize to [0, 1] range
        if self.radiation_field.max() > 0:
            self.radiation_field = self.radiation_field / self.radiation_field.max()

        # Create simple weighted map (inverse of radiation for safety)
        self.weighted_map = 1.0 - np.clip(self.radiation_field, 0, 1)

        # Add some information gain based on gradients
        grad_y, grad_x = np.gradient(self.radiation_field)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        gradient_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

        # Combine safety and information gain
        self.weighted_map = 0.6 * self.weighted_map + 0.4 * gradient_norm

        # Clear previous data
        self.measurement_positions = []
        self.current_position = None
        self.last_result = None

        self.log(f"Generated field with {n_sources} sources")
        self.update_visualization()

    def clear_measurements(self):
        """Clear all measurement positions."""
        self.measurement_positions = []
        self.current_position = None
        self.last_result = None
        self.log("Cleared all measurements")
        self.update_visualization()

    def add_random_measurements(self):
        """Add some random measurement positions."""
        n_measurements = np.random.randint(5, 15)
        self.measurement_positions = []

        for _ in range(n_measurements):
            y = np.random.randint(20, self.field_size_pixels - 20)
            x = np.random.randint(20, self.field_size_pixels - 20)
            self.measurement_positions.append((y, x))

        self.log(f"Added {n_measurements} random measurements")
        self.update_visualization()

    def set_click_mode_start(self):
        """Set mode to select start position by clicking."""
        self.click_mode = "start"
        self.log("Click on the map to set start position")

    def on_click(self, event):
        """Handle click events on the plot."""
        if event.inaxes == self.ax1 and self.click_mode == "start":
            # Convert click coordinates to pixel coordinates
            x_click, y_click = event.xdata, event.ydata
            if x_click is not None and y_click is not None:
                # Convert to integer pixel coordinates
                x_pixel = int(np.clip(x_click, 0, self.field_size_pixels - 1))
                y_pixel = int(np.clip(y_click, 0, self.field_size_pixels - 1))

                self.current_position = (y_pixel, x_pixel)
                self.click_mode = None

                # Convert to meters for display
                x_meters = x_pixel / self.pixels_per_meter
                y_meters = y_pixel / self.pixels_per_meter
                self.log(f"Start position set to: ({x_meters:.1f}m, {y_meters:.1f}m) = pixel ({x_pixel}, {y_pixel})")

                self.update_visualization()

    def create_rrt_planner(self) -> RRTPathPlanner:
        """Create RRT planner with current parameters."""
        params = RRTParameters(
            max_iterations=self.max_iter_var.get(),
            step_size_meters=self.step_size_var.get(),
            min_step_size_meters=self.min_step_var.get(),
            exploration_radius_meters=self.radius_var.get(),
            n_steps=self.n_steps_var.get(),
            goal_bias=self.goal_bias_var.get(),
            min_radiation_threshold=self.safety_threshold_var.get(),
            source_avoidance=False,  # Disable planning-time avoidance
            post_process_source_avoidance=True,  # Enable post-processing avoidance
            source_avoidance_radius_meters=self.source_radius_var.get(),
            use_rrt_star=True
        )

        # Get predicted sources from radiation field
        predicted_sources = self._detect_sources()

        planner = RRTPathPlanner(params)
        planner.set_environment(self.radiation_field, self.weighted_map, predicted_sources=predicted_sources)
        return planner

    def _detect_sources(self) -> Optional[List[Tuple[int, int]]]:
        """Detect sources in the current radiation field."""
        if self.radiation_field is None:
            return None

        try:
            # Import source detection module
            sys.path.append(str(Path(__file__).resolve().parent))
            from modules.source_detection import detect_radiation_sources, SourceDetectionParameters

            # Create detection parameters
            params = SourceDetectionParameters(
                detection_threshold=0.3,
                max_sources=10
            )

            # Detect sources using the current field
            result = detect_radiation_sources(self.radiation_field, params)
            sources = result.get('source_locations', [])

            if sources and len(sources) > 0:
                # Get source intensities from the detection result
                source_intensities = result.get('source_intensities', [])

                # Create source data with intensity information: (y, x, intensity)
                source_data = []
                for i, (y, x) in enumerate(sources):
                    # Use detected intensity if available, otherwise sample from field
                    if i < len(source_intensities):
                        intensity = source_intensities[i]
                    else:
                        intensity = self.radiation_field[int(y), int(x)]

                    source_data.append((int(y), int(x), float(intensity)))

                    # Log for debugging
                    y_m = y / self.pixels_per_meter
                    x_m = x / self.pixels_per_meter
                    avoidance_radius = self._calculate_dynamic_avoidance_radius(intensity)
                    self.log(f"   Source {i+1}: ({x_m:.1f}m, {y_m:.1f}m) intensity={intensity:.3f} → radius={avoidance_radius:.1f}m")

                self.log(f"🔍 Detected {len(source_data)} sources for dynamic avoidance")
                return source_data
            else:
                self.log("🔍 No sources detected")
                return []

        except Exception as e:
            self.log(f"❌ Source detection error: {e}")
            return None

    def _calculate_dynamic_avoidance_radius(self, intensity: float) -> float:
        """Calculate dynamic avoidance radius based on source intensity.

        Args:
            intensity: Source intensity (0.0 to 1.0)

        Returns:
            avoidance_radius_meters: Dynamic avoidance radius in meters
        """
        # Same logic as in RRT path planner
        base_radius = 1.0  # Base avoidance radius
        intensity_scale = 2.0  # Scale factor
        min_radius = 0.5  # Minimum radius
        max_radius = 3.0  # Maximum radius

        # Calculate radius: base + intensity * scale
        dynamic_radius = base_radius + intensity * intensity_scale

        # Clamp to bounds
        return max(min_radius, min(dynamic_radius, max_radius))

    def plan_rrt_path(self):
        """Plan a single RRT path."""
        if self.current_position is None:
            self.log("❌ No start position set. Click 'Set Start Position' and click on map.")
            return

        if self.radiation_field is None:
            self.log("❌ No radiation field available.")
            return

        self.log("🌳 Planning RRT path...")

        try:
            # Create fresh RRT planner
            planner = self.create_rrt_planner()

            # Plan path
            result = planner.plan_path(
                start_position=self.current_position,
                measurement_positions=self.measurement_positions
            )

            self.last_result = result

            if result.success and len(result.path) > 1:
                self.log(f"✅ RRT planning successful!")
                self.log(f"   Path length: {len(result.path)} points")
                self.log(f"   Total reward: {result.total_reward:.3f}")
                self.log(f"   Total cost: {result.total_cost:.3f}")
                self.log(f"   Planning time: {result.planning_time:.3f}s")
                self.log(f"   Tree size: {len(result.tree_nodes)}")

                # Analyze step distances
                self.analyze_step_distances(result.path)

            else:
                self.log("❌ RRT planning failed")

        except Exception as e:
            self.log(f"❌ RRT planning error: {e}")

        self.update_visualization()

    def analyze_step_distances(self, path: List[Tuple[int, int]]):
        """Analyze and log step distances in the path."""
        if len(path) < 2:
            return

        self.log("📏 Step distance analysis:")
        min_distance = float('inf')
        max_distance = 0
        total_distance = 0

        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            dist_meters = dist_pixels / self.pixels_per_meter

            total_distance += dist_meters
            min_distance = min(min_distance, dist_meters)
            max_distance = max(max_distance, dist_meters)

            self.log(f"   Step {i}: {dist_meters:.3f}m")

        avg_distance = total_distance / (len(path) - 1)
        min_threshold = self.min_step_var.get()

        self.log(f"📊 Distance summary:")
        self.log(f"   Min: {min_distance:.3f}m (threshold: {min_threshold:.3f}m)")
        self.log(f"   Max: {max_distance:.3f}m")
        self.log(f"   Avg: {avg_distance:.3f}m")
        self.log(f"   Total: {total_distance:.3f}m")

        # Check if minimum distance constraint is satisfied
        violations = [i for i in range(1, len(path))
                     if np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2) / self.pixels_per_meter < min_threshold * 0.9]

        if violations:
            self.log(f"⚠️  Minimum distance violations at steps: {violations}")
        else:
            self.log(f"✅ All steps satisfy minimum distance constraint")

    def test_multiple_runs(self):
        """Test multiple RRT runs and analyze statistics."""
        if self.current_position is None:
            self.log("❌ No start position set.")
            return

        n_runs = 10
        self.log(f"🔄 Running {n_runs} RRT tests...")

        results = []
        success_count = 0
        min_distances = []

        for run in range(n_runs):
            try:
                planner = self.create_rrt_planner()
                result = planner.plan_path(
                    start_position=self.current_position,
                    measurement_positions=self.measurement_positions
                )

                if result.success and len(result.path) > 1:
                    success_count += 1
                    results.append(result)

                    # Calculate minimum step distance
                    min_dist = float('inf')
                    for i in range(1, len(result.path)):
                        p1, p2 = result.path[i-1], result.path[i]
                        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) / self.pixels_per_meter
                        min_dist = min(min_dist, dist)
                    min_distances.append(min_dist)

            except Exception as e:
                self.log(f"Run {run+1} failed: {e}")

        # Analyze results
        self.log(f"📊 Multiple run analysis ({n_runs} runs):")
        self.log(f"   Success rate: {success_count}/{n_runs} ({success_count/n_runs*100:.1f}%)")

        if results:
            rewards = [r.total_reward for r in results]
            costs = [r.total_cost for r in results]
            times = [r.planning_time for r in results]

            self.log(f"   Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
            self.log(f"   Cost: {np.mean(costs):.3f} ± {np.std(costs):.3f}")
            self.log(f"   Time: {np.mean(times):.3f} ± {np.std(times):.3f}s")

            if min_distances:
                min_threshold = self.min_step_var.get()
                violations = [d for d in min_distances if d < min_threshold * 0.9]
                self.log(f"   Min distances: {np.mean(min_distances):.3f} ± {np.std(min_distances):.3f}m")
                self.log(f"   Distance violations: {len(violations)}/{len(min_distances)} runs")

                if violations:
                    self.log(f"⚠️  Violation details: {violations}")
                else:
                    self.log(f"✅ All runs satisfied minimum distance constraint")

        # Use last successful result for visualization
        if results:
            self.last_result = results[-1]
            self.update_visualization()

    def update_visualization(self):
        """Update the visualization plots."""
        self.ax1.clear()
        self.ax2.clear()

        if self.radiation_field is not None:
            # Plot 1: Radiation field with RRT tree and path
            im1 = self.ax1.imshow(self.radiation_field, cmap='hot', origin='lower',
                                 extent=[0, self.field_size_pixels, 0, self.field_size_pixels])
            self.ax1.set_title('Radiation Field + RRT Path')
            self.ax1.set_xlabel('X (pixels)')
            self.ax1.set_ylabel('Y (pixels)')

            # Plot 2: Weighted map
            im2 = self.ax2.imshow(self.weighted_map, cmap='RdYlGn', origin='lower',
                                 extent=[0, self.field_size_pixels, 0, self.field_size_pixels])
            self.ax2.set_title('Weighted Map + RRT Tree')
            self.ax2.set_xlabel('X (pixels)')
            self.ax2.set_ylabel('Y (pixels)')

            # Draw measurement positions
            if self.measurement_positions:
                meas_array = np.array(self.measurement_positions)
                self.ax1.scatter(meas_array[:, 1], meas_array[:, 0],
                               c='blue', s=30, marker='x', label='Measurements')
                self.ax2.scatter(meas_array[:, 1], meas_array[:, 0],
                               c='blue', s=30, marker='x', label='Measurements')

            # Draw current position
            if self.current_position is not None:
                y, x = self.current_position
                self.ax1.scatter(x, y, c='green', s=100, marker='s',
                               edgecolor='black', linewidth=2, label='Start')
                self.ax2.scatter(x, y, c='green', s=100, marker='s',
                               edgecolor='black', linewidth=2, label='Start')

            # Draw detected sources for avoidance
            predicted_sources = self._detect_sources()
            if predicted_sources and self.source_avoidance_var.get():
                sources_array = np.array(predicted_sources)
                self.ax1.scatter(sources_array[:, 1], sources_array[:, 0],
                               c='red', s=80, marker='^', edgecolor='darkred',
                               linewidth=2, label='Detected Sources')
                self.ax2.scatter(sources_array[:, 1], sources_array[:, 0],
                               c='red', s=80, marker='^', edgecolor='darkred',
                               linewidth=2, label='Detected Sources')

                # Draw avoidance circles with dynamic radius
                if self.source_avoidance_var.get():
                    for source_data in predicted_sources:
                        # Handle both old format (y, x) and new format (y, x, intensity)
                        if len(source_data) == 2:
                            y, x = source_data
                            intensity = self.radiation_field[int(y), int(x)]  # Sample from field
                        else:
                            y, x, intensity = source_data

                        # Calculate dynamic avoidance radius
                        dynamic_radius_meters = self._calculate_dynamic_avoidance_radius(intensity)
                        radius_pixels = dynamic_radius_meters * self.pixels_per_meter

                        # Draw circles with different colors based on intensity
                        if intensity > 0.8:
                            circle_color = 'darkred'
                            alpha = 0.8
                        elif intensity > 0.5:
                            circle_color = 'red'
                            alpha = 0.6
                        else:
                            circle_color = 'orange'
                            alpha = 0.4

                        circle1 = plt.Circle((x, y), radius_pixels, fill=False,
                                           color=circle_color, linestyle='--', alpha=alpha, linewidth=2)
                        circle2 = plt.Circle((x, y), radius_pixels, fill=False,
                                           color=circle_color, linestyle='--', alpha=alpha, linewidth=2)
                        self.ax1.add_patch(circle1)
                        self.ax2.add_patch(circle2)

            # Draw RRT results
            if self.last_result is not None:
                self.draw_rrt_results(self.ax1, self.ax2)

            self.ax1.legend()
            self.ax2.legend()

        self.canvas.draw()

    def draw_rrt_results(self, ax1, ax2):
        """Draw RRT tree and path on both axes."""
        if not self.last_result.tree_nodes:
            return

        # Draw tree edges on ax2 (weighted map)
        for node in self.last_result.tree_nodes:
            if node.parent is not None:
                x1, y1 = node.parent.position[1], node.parent.position[0]
                x2, y2 = node.position[1], node.position[0]
                ax2.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.4, linewidth=0.5)

        # Draw tree nodes
        positions = [node.position for node in self.last_result.tree_nodes]
        if positions:
            positions_array = np.array(positions)
            ax2.scatter(positions_array[:, 1], positions_array[:, 0],
                       c='lightblue', s=8, alpha=0.6, marker='.', label='RRT Nodes')

        # Draw planned path on both plots
        if self.last_result.path and len(self.last_result.path) > 1:
            path_array = np.array(self.last_result.path)

            # Path on radiation field
            ax1.plot(path_array[:, 1], path_array[:, 0], 'cyan', linewidth=3, alpha=0.8, label='RRT Path')
            ax1.scatter(path_array[:, 1], path_array[:, 0],
                       c='cyan', s=40, marker='o', edgecolor='black', linewidth=1)

            # Path on weighted map
            ax2.plot(path_array[:, 1], path_array[:, 0], 'red', linewidth=2, alpha=0.8, label='RRT Path')
            ax2.scatter(path_array[:, 1], path_array[:, 0],
                       c='red', s=30, marker='s', edgecolor='black', linewidth=0.5)

            # Add step numbers
            for i, (y, x) in enumerate(self.last_result.path):
                ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='white', weight='bold')

    def log(self, message: str):
        """Add message to results log."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        print(message)  # Also print to console

    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    print("Starting RRT Test Application...")
    app = RRTTestApp()
    app.run()