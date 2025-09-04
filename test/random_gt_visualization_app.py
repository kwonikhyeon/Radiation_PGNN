#!/usr/bin/env python3
"""
Random Ground Truth Visualization App
Eval 코드의 모델 적용 방식을 그대로 사용하여 랜덤 GT와 측정값을 생성하고
결과를 plot으로만 시각화하는 앱
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import tkinter.messagebox

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from src.model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
    from src.dataset.generate_truth import sample_sources, gaussian_field, N_SOURCES_RANGE
    from src.dataset.trajectory_sampler import generate_waypoints, sparse_from_waypoints
    print(f"✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

class RandomGTVisualizationApp:
    def __init__(self, checkpoint_path="checkpoints/convnext_simple_gt_exp7/ckpt_best.pth"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.grid_size = 256
        
        # Data storage
        self.current_gt = None
        self.current_input = None
        self.current_prediction = None
        self.current_sparse = None
        self.current_mask = None
        
        # Model
        self.model = None
        
        # GUI elements
        self.root = None
        self.fig = None
        self.canvas = None
        self.axes = None
        
        print(f"🎯 Device: {self.device}")
        print(f"📏 Grid size: {self.grid_size}x{self.grid_size}")
    
    def load_model(self):
        """Load model exactly like eval code"""
        try:
            print(f"🔍 Loading model from {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            
            # Extract pred_scale from config like eval code
            pred_scale = 1.0
            if "config" in ckpt:
                pred_scale = ckpt["config"].get("pred_scale", 1.0)
                print(f"📊 Using pred_scale from checkpoint: {pred_scale}")
            
            # Create model exactly like eval code
            self.model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=pred_scale)
            
            # Load state dict exactly like eval code
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
            
            self.model.to(self.device).eval()
            
            # Print model info like eval code
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_random_gt(self):
        """Generate random ground truth exactly like dataset generator"""
        rng = np.random.default_rng(None)  # Random seed each time
        
        # Sample number of sources
        n_sources = rng.integers(N_SOURCES_RANGE[0], N_SOURCES_RANGE[1] + 1)
        
        # Sample source parameters
        coords, amps, sigmas = sample_sources(self.grid_size, n_sources, rng=rng)
        
        # Generate Gaussian field
        gt = gaussian_field(self.grid_size, coords, amps, sigmas)
        
        # Normalize to [0, 1] like dataset
        if gt.max() > 0:
            gt = gt / gt.max()
        
        gt = gt.astype(np.float32)
        
        print(f"✅ Generated GT with {n_sources} sources")
        print(f"   Source positions: {coords.tolist()}")
        print(f"   GT range: [{gt.min():.4f}, {gt.max():.4f}], mean: {gt.mean():.4f}")
        
        return gt, coords, amps, sigmas
    
    def generate_random_measurements(self, gt, n_measurements=None):
        """Generate random measurements using trajectory sampler"""
        if n_measurements is None:
            n_measurements = np.random.randint(8, 15)  # 8-15 measurements for better response
        
        try:
            # Mix of trajectory-based and high-value sampling for better results
            n_trajectory = max(3, n_measurements // 2)  # Half from trajectory
            n_random_high = n_measurements - n_trajectory  # Half from high values
            
            measurements = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            trajectory = []
            
            # 1. Get some points from trajectory sampler
            if n_trajectory > 0:
                waypoints = generate_waypoints(
                    grid=self.grid_size,
                    min_pts=min(3, n_trajectory),
                    max_pts=n_trajectory
                )
                
                for y, x in waypoints:
                    # Ensure coordinates are within bounds
                    y = max(0, min(y, self.grid_size - 1))
                    x = max(0, min(x, self.grid_size - 1))
                    
                    gt_value = gt[y, x]
                    measurements[y, x] += gt_value
                    mask[y, x] = 1.0
                    trajectory.append((y, x))
            
            # 2. Add some high-value measurements for better model response
            if n_random_high > 0:
                gt_flat = gt.flatten()
                high_indices = np.argsort(gt_flat)[-n_random_high*3:]  # Top candidates
                selected_indices = np.random.choice(high_indices, n_random_high, replace=False)
                
                for idx in selected_indices:
                    y, x = np.unravel_index(idx, gt.shape)
                    if mask[y, x] == 0:  # Don't duplicate
                        gt_value = gt[y, x]
                        measurements[y, x] += gt_value
                        mask[y, x] = 1.0
                        trajectory.append((y, x))
            
            # Create sparse input
            sparse = measurements * mask
            
            print(f"✅ Generated {n_measurements} measurements")
            print(f"   Measurement range: [{measurements.min():.4f}, {measurements.max():.4f}]")
            print(f"   Non-zero measurements: {np.count_nonzero(measurements)}")
            
            return measurements, mask, sparse, trajectory
            
        except Exception as e:
            print(f"❌ Error generating measurements: {e}")
            # Fallback: random positions
            measurements = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
            trajectory = []
            for _ in range(n_measurements):
                y = np.random.randint(0, self.grid_size)
                x = np.random.randint(0, self.grid_size)
                gt_value = gt[y, x]
                measurements[y, x] += gt_value
                mask[y, x] = 1.0
                trajectory.append((y, x))
            
            sparse = measurements * mask
            return measurements, mask, sparse, trajectory
    
    def create_model_input(self, measurements, mask):
        """Create 6-channel model input exactly like dataset"""
        # Channel 0: measurements
        # Channel 1: measurement mask  
        # Channel 2: log measurements
        log_measurements = np.log(measurements + 1e-8)
        
        # Channel 3-4: normalized coordinates
        y, x = np.mgrid[0:self.grid_size, 0:self.grid_size]
        coord_x = x / (self.grid_size - 1)  # [0, 1]
        coord_y = y / (self.grid_size - 1)  # [0, 1]
        
        # Channel 5: distance map (simplified version)
        if np.any(mask > 0):
            from scipy.ndimage import distance_transform_edt
            distance_map = distance_transform_edt(mask == 0)
            if distance_map.max() > 0:
                distance_map = distance_map / distance_map.max()
        else:
            distance_map = np.ones((self.grid_size, self.grid_size))
        
        # Stack channels
        input_tensor = np.stack([
            measurements,      # Channel 0
            mask,             # Channel 1 
            log_measurements, # Channel 2
            coord_x,          # Channel 3
            coord_y,          # Channel 4
            distance_map      # Channel 5
        ])
        
        # Convert to torch tensor like eval code
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
        
        return input_tensor
    
    def run_model_inference(self, input_tensor):
        """Run model inference exactly like eval code"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_tensor)
        
        # Convert back to numpy like eval code
        pred_np = pred.cpu().numpy()
        
        return pred_np
    
    def generate_and_predict(self):
        """Generate random data and run prediction"""
        print(f"\n🔄 Generating new random data...")
        
        # Generate random GT
        gt, coords, amps, sigmas = self.generate_random_gt()
        
        # Generate random measurements
        measurements, mask, sparse, trajectory = self.generate_random_measurements(gt)
        
        # Create model input
        input_tensor = self.create_model_input(measurements, mask)
        
        # Run model inference exactly like eval code
        pred = self.run_model_inference(input_tensor)
        
        # Store results
        self.current_gt = gt
        self.current_input = input_tensor
        self.current_prediction = pred
        self.current_sparse = sparse
        self.current_mask = mask
        
        # Extract the actual prediction array for proper logging
        pred_display = pred[0, 0]  # [H, W]
        
        print(f"✅ Model inference completed")
        print(f"   Prediction range: [{pred_display.min():.4f}, {pred_display.max():.4f}], mean: {pred_display.mean():.4f}")
        print(f"   GT vs Prediction max values: GT={gt.max():.4f}, Pred={pred_display.max():.4f}")
        
        # Check if prediction is meaningful
        if pred_display.max() > 0.1:
            print(f"   ✅ Strong prediction response detected")
        elif pred_display.max() > 0.01:
            print(f"   ⚠️  Weak prediction response")  
        else:
            print(f"   🚨 Very weak prediction - might be issue")
        
        return gt, sparse, pred, mask, trajectory
    
    def create_visualization(self, gt, sparse, pred, mask, trajectory):
        """Create visualization exactly like eval code"""
        # Clear previous plot
        for ax in self.axes.flat:
            ax.clear()
        
        # Convert pred to same format as eval code
        pred_display = pred[0, 0]  # [H, W]
        
        # Top row: Input, Prediction, Ground Truth (like eval code)
        images = [sparse, pred_display, gt]
        titles = ["Input: Sparse Measurements", "Prediction (ConvNeXt PGNN)", "Ground Truth"]
        
        for ax, img, title in zip(self.axes[0], images, titles):
            im = ax.imshow(img, cmap="hot", origin="lower", vmin=0, vmax=1)
            ax.set_title(title, fontsize=12, weight='bold')
            ax.axis("off")
            
            # Add colorbar
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add measurement points like eval code
            if np.any(mask > 0):
                y_coords, x_coords = np.where(mask > 0)
                ax.scatter(x_coords, y_coords, c="cyan", s=20, alpha=0.8, marker='+', linewidth=2)
        
        # Bottom row: Analysis plots
        # Error map
        error = np.abs(pred_display - gt)
        im_error = self.axes[1, 0].imshow(error, cmap="plasma", origin="lower")
        self.axes[1, 0].set_title("Absolute Error", fontsize=12, weight='bold')
        self.axes[1, 0].axis("off")
        self.fig.colorbar(im_error, ax=self.axes[1, 0], fraction=0.046, pad=0.04)
        
        # Histogram comparison
        gt_flat = gt.flatten()
        pred_flat = pred_display.flatten()
        
        self.axes[1, 1].hist(gt_flat, bins=50, alpha=0.7, label='Ground Truth', color='red', density=True)
        self.axes[1, 1].hist(pred_flat, bins=50, alpha=0.7, label='Prediction', color='blue', density=True)
        self.axes[1, 1].set_title("Intensity Distribution", fontsize=12, weight='bold')
        self.axes[1, 1].set_xlabel("Intensity")
        self.axes[1, 1].set_ylabel("Density")
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Prediction vs Ground Truth
        self.axes[1, 2].scatter(gt_flat, pred_flat, alpha=0.5, s=1)
        self.axes[1, 2].plot([0, 1], [0, 1], 'r--', linewidth=2)
        self.axes[1, 2].set_title("Prediction vs Ground Truth", fontsize=12, weight='bold')
        self.axes[1, 2].set_xlabel("Ground Truth")
        self.axes[1, 2].set_ylabel("Prediction")
        self.axes[1, 2].grid(True, alpha=0.3)
        
        # Calculate and display metrics
        rmse = np.sqrt(np.mean((pred_display - gt)**2))
        mae = np.mean(np.abs(pred_display - gt))
        
        # Add metrics text
        metrics_text = f"RMSE: {rmse:.4f}\\nMAE: {mae:.4f}\\nMeasurements: {np.count_nonzero(mask)}"
        self.axes[1, 2].text(0.05, 0.95, metrics_text, transform=self.axes[1, 2].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.fig.suptitle(f'Random GT Visualization - RMSE: {rmse:.4f}', fontsize=16, weight='bold')
        self.canvas.draw()
    
    def on_generate_click(self):
        """Handle generate button click"""
        def run_generation():
            try:
                self.generate_button.config(state='disabled', text='Generating...')
                self.root.update()
                
                # Generate and predict
                gt, sparse, pred, mask, trajectory = self.generate_and_predict()
                
                # Update visualization on main thread
                self.root.after(0, lambda: self.create_visualization(gt, sparse, pred, mask, trajectory))
                self.root.after(0, lambda: self.generate_button.config(state='normal', text='Generate New'))
                
            except Exception as e:
                print(f"❌ Error in generation: {e}")
                self.root.after(0, lambda: self.generate_button.config(state='normal', text='Generate New'))
                self.root.after(0, lambda: tk.messagebox.showerror("Error", f"Generation failed: {e}"))
        
        # Run in background thread
        threading.Thread(target=run_generation, daemon=True).start()
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Random GT Visualization App")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.axes = self.fig.subplots(2, 3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Generate button
        self.generate_button = tk.Button(control_frame, text="Generate New", 
                                       command=self.on_generate_click,
                                       bg="#4CAF50", fg="white", font=('Arial', 12, 'bold'))
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_label = tk.Label(control_frame, 
                            text="Click 'Generate New' to create random ground truth and measurements",
                            font=('Arial', 10))
        info_label.pack(side=tk.LEFT, padx=20)
        
        # Status label  
        self.status_label = tk.Label(control_frame, text="Ready", font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        print("✅ GUI setup complete")
    
    def run(self):
        """Run the application"""
        if not self.load_model():
            print("❌ Failed to load model, exiting")
            return
            
        self.setup_gui()
        
        # Generate initial data
        self.on_generate_click()
        
        print("🚀 Starting Random GT Visualization App...")
        self.root.mainloop()

def main():
    """Main function"""
    print("🧪 Random Ground Truth Visualization App")
    print("="*50)
    print("Uses eval code's exact model application method")
    print("Generates random GT and measurements for visualization")
    
    app = RandomGTVisualizationApp()
    app.run()

if __name__ == "__main__":
    main()