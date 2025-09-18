#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-Channel Layer Generator for Radiation Field Path Planning
위험도(Risk) / 정보이득(Information Gain) / 주행가능성(Traversability) 레이어 생성

Based on 3ch_layer_calc.md mathematical formulations
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import matplotlib.pyplot as plt

# Handle OpenCV import with numpy compatibility
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCV not available ({e}). Using alternative gradient computation.")
    CV2_AVAILABLE = False


# ----------------- Robust model import -----------------
def import_model():
    """Import SimplifiedConvNeXtPGNN from src/model directory."""
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    
    try:
        from src.model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
        print("📁 Imported SimplifiedConvNeXtPGNN from src.model")
        return SimplifiedConvNeXtPGNN
    except ImportError:
        try:
            from model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
            print("📁 Imported SimplifiedConvNeXtPGNN from model")
            return SimplifiedConvNeXtPGNN
        except ImportError as e:
            print("❌ Could not import SimplifiedConvNeXtPGNN:", e)
            raise


def import_dataset_modules():
    """Import dataset generation modules from src/dataset directory."""
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    
    try:
        # Import ground truth generation
        import src.dataset.generate_truth as gt
        import src.dataset.trajectory_sampler as ts
        print("📁 Imported dataset modules from src.dataset")
        return gt, ts
    except ImportError:
        try:
            # Fallback import path
            import dataset.generate_truth as gt
            import dataset.trajectory_sampler as ts
            print("📁 Imported dataset modules from dataset")
            return gt, ts
        except ImportError as e:
            print("❌ Could not import dataset modules:", e)
            raise


class LayerGenerator:
    """
    3채널 레이어 생성기:
    - Risk Layer (위험도)
    - Information Gain Layer (정보 이득)  
    - Traversability Layer (주행 가능성)
    """
    
    def __init__(self, checkpoint_path="checkpoints/convnext_simple_gt_exp8/ckpt_best.pth", 
                 model_size=256, n_uncertainty_samples=32):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.n_uncertainty_samples = n_uncertainty_samples
        
        # Model
        self.model = None
        
        # Layer parameters (from 3ch_layer_calc.md)
        self.params = {
            # Information Gain parameters
            'gamma': 10.0,          # sigmoid sharpness for gating
            'tau_r': 0.3,           # radiation threshold for gating
            'alpha_g': 1.0,         # gradient weight
            'alpha_u': 1.0,         # uncertainty weight
            
            # Traversability parameters  
            'lambda_d': 0.4,        # distance cost
            'lambda_theta': 0.6,    # rotation cost
            'lambda_o': 0.5,        # obstacle proximity cost
            'lambda_v': 0.3,        # velocity limitation cost
            'd_max': 100.0,         # maximum distance for normalization
            
            # Risk parameters
            'tau_E': 0.8,           # exposure limit threshold
            'kappa': 5.0,           # soft cutoff steepness
            
            # Layer combination weights
            'lambda_r': 0.4,        # risk weight
            'lambda_i': 0.4,        # information gain weight  
            'lambda_t': 0.2,        # traversability weight
        }
        
        print(f"🔧 Device: {self.device} | Model size: {self.model_size}")
    
    def load_model(self):
        """Load the SimplifiedConvNeXtPGNN model (same as interactive app)."""
        Model = import_model()
        print(f"🔍 Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        pred_scale = 1.0
        if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
            pred_scale = ckpt["config"].get("pred_scale", 1.0)
            print(f"📊 pred_scale from ckpt: {pred_scale}")

        self.model = Model(in_channels=6, pred_scale=pred_scale)

        # Resolve state_dict and log integrity
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        ik = self.model.load_state_dict(sd, strict=False)
        print(f"🔧 missing={len(ik.missing_keys)} unexpected={len(ik.unexpected_keys)}")

        self.model.to(self.device).eval()
        tot = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model ready: {tot/1e6:.1f}M params")
        return True

    def create_model_input(self, measurements, measurement_mask):
        """Create 6-channel input tensor like interactive app."""
        H = W = self.model_size
        
        # Channel 0-1: measurements, mask
        meas = measurements.astype(np.float32)
        mask = measurement_mask.astype(np.float32)
        
        # Channel 2: masked log1p
        logm = np.zeros_like(meas, dtype=np.float32)
        nz = mask > 0
        logm[nz] = np.log1p(meas[nz])
        
        # Channel 3-4: normalized coordinates
        yy, xx = np.mgrid[0:H, 0:W]
        coord_x = (xx / (W-1)).astype(np.float32)
        coord_y = (yy / (H-1)).astype(np.float32)
        
        # Channel 5: distance map
        if np.any(mask > 0):
            dist = distance_transform_edt(mask == 0).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        else:
            dist = np.ones((H, W), np.float32)
        
        inp = np.stack([meas, mask, logm, coord_x, coord_y, dist], axis=0)
        inp_t = torch.from_numpy(inp[None]).float().to(self.device)
        return inp_t

    @torch.no_grad()
    def predict_radiation_field(self, measurements, measurement_mask):
        """Predict radiation field using the model."""
        inp_t = self.create_model_input(measurements, measurement_mask)
        self.model.eval()
        pred = self.model(inp_t)
        return pred[0, 0].detach().cpu().numpy()

    @torch.no_grad()  
    def predict_uncertainty(self, measurements, measurement_mask):
        """Predict uncertainty using TTA with measurement noise."""
        inp_t = self.create_model_input(measurements, measurement_mask)
        self.model.eval()
        
        preds = []
        for _ in range(self.n_uncertainty_samples):
            noisy = inp_t.clone()
            # Add noise to measurement channel only
            noisy[:, 0:1] += torch.randn_like(noisy[:, 0:1]) * 0.01
            preds.append(self.model(noisy).detach().cpu().numpy())
        
        preds = np.array(preds)  # [K, B, 1, H, W]
        preds = preds[:, 0, 0]   # [K, H, W]
        
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        
        # Distance-based uncertainty weighting
        mask = inp_t[0, 1].detach().cpu().numpy()
        if np.any(mask > 0):
            dist = distance_transform_edt(mask == 0).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        else:
            dist = np.ones_like(std_pred)
        
        # Combine model uncertainty with distance
        distance_weight = 0.6
        distance_sigma = 40.0
        w = np.exp(dist * distance_sigma / 100.0)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        
        std_norm = (std_pred - std_pred.min()) / (std_pred.max() - std_pred.min() + 1e-8)
        uncertainty = (1 - distance_weight) * std_norm + distance_weight * w
        
        return mean_pred, uncertainty

    def normalize_field(self, field):
        """Normalize field to [0,1] range with epsilon for stability."""
        eps = 1e-8
        return (field - field.min()) / (field.max() - field.min() + eps)

    def calculate_risk_layer(self, radiation_field):
        """
        Calculate risk layer L_r(x) = 1 - norm(r_hat)(x)
        Lower radiation = higher preference (safer)
        """
        r_norm = self.normalize_field(radiation_field)
        L_r = 1.0 - r_norm
        
        # Optional: soft exposure limit (equation from md file)
        tau_E = self.params['tau_E']
        kappa = self.params['kappa']
        soft_cutoff = 1.0 / (1.0 + np.exp(-kappa * (tau_E - radiation_field)))
        L_r_clamped = L_r * soft_cutoff
        
        return L_r_clamped

    def calculate_information_gain_layer(self, radiation_field, uncertainty):
        """
        Calculate information gain layer combining gradient and uncertainty.
        L_i(x) = norm(alpha_g ||r||_2) * g(x) + norm(alpha_u sigma_r) * (1-g(x))
        """
        # Compute gradient magnitude
        if CV2_AVAILABLE:
            # Use OpenCV Sobel filter
            grad_x = cv2.Sobel(radiation_field, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(radiation_field, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # Use scipy sobel filter as fallback
            grad_x = ndimage.sobel(radiation_field, axis=1)
            grad_y = ndimage.sobel(radiation_field, axis=0)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth with Gaussian to reduce noise
        grad_mag = ndimage.gaussian_filter(grad_mag, sigma=1.0)
        
        # Gating function g(x) = sigma(gamma(r_hat - tau_r))
        gamma = self.params['gamma']
        tau_r = self.params['tau_r'] 
        g = 1.0 / (1.0 + np.exp(-gamma * (radiation_field - tau_r)))
        
        # Weighted combination
        alpha_g = self.params['alpha_g']
        alpha_u = self.params['alpha_u']
        
        grad_term = self.normalize_field(alpha_g * grad_mag) * g
        uncertainty_term = self.normalize_field(alpha_u * uncertainty) * (1 - g)
        
        L_i = grad_term + uncertainty_term
        
        return L_i

    def calculate_traversability_layer(self, robot_pos, robot_theta, 
                                     obstacle_distance_map=None, velocity_map=None):
        """
        Calculate traversability layer L_t(x) considering distance, rotation, obstacles.
        
        Args:
            robot_pos: (x, y) current robot position
            robot_theta: current robot heading angle (radians)
            obstacle_distance_map: distance to nearest obstacles (optional)
            velocity_map: local velocity limits (optional)
        """
        H, W = self.model_size, self.model_size
        yy, xx = np.mgrid[0:H, 0:W]
        
        x0, y0 = robot_pos
        theta0 = robot_theta
        
        # Distance from robot position
        distances = np.sqrt((xx - x0)**2 + (yy - y0)**2)
        
        # Angle difference from robot heading
        angles_to_points = np.arctan2(yy - y0, xx - x0)
        angle_diffs = np.abs(self.wrap_angle(angles_to_points - theta0))
        
        # Traversability components
        lambda_d = self.params['lambda_d']
        lambda_theta = self.params['lambda_theta'] 
        lambda_o = self.params['lambda_o']
        lambda_v = self.params['lambda_v']
        d_max = self.params['d_max']
        
        # Distance cost (normalized)
        dist_cost = lambda_d * (distances / d_max)
        
        # Rotation cost (normalized by pi)
        rot_cost = lambda_theta * (angle_diffs / np.pi)
        
        # Obstacle cost (if provided)
        obs_cost = 0.0
        if obstacle_distance_map is not None:
            obs_inverse = 1.0 / (obstacle_distance_map + 1e-6)
            obs_cost = lambda_o * self.normalize_field(obs_inverse)
        
        # Velocity cost (if provided)
        vel_cost = 0.0
        if velocity_map is not None:
            vel_inverse = 1.0 / (velocity_map + 1e-6)  
            vel_cost = lambda_v * self.normalize_field(vel_inverse)
        
        # Total traversability (exponential form from md file)
        total_cost = dist_cost + rot_cost + obs_cost + vel_cost
        L_t = np.exp(-total_cost)
        
        return L_t

    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] range."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def combine_layers(self, risk_layer, info_layer, traversability_layer):
        """
        Combine three layers with weights: W(x) = lambda_r L_r + lambda_i L_i + lambda_t L_t
        Returns combined weight map for path planning.
        """
        lambda_r = self.params['lambda_r']
        lambda_i = self.params['lambda_i'] 
        lambda_t = self.params['lambda_t']
        
        # Ensure weights sum to 1
        total_weight = lambda_r + lambda_i + lambda_t
        lambda_r /= total_weight
        lambda_i /= total_weight
        lambda_t /= total_weight
        
        combined_weight = (lambda_r * risk_layer + 
                          lambda_i * info_layer + 
                          lambda_t * traversability_layer)
        
        return combined_weight

    def get_layers(self, measurements, measurement_mask, robot_pos, robot_theta,
                   obstacle_distance_map=None, velocity_map=None):
        """
        Main function to generate all 3 layers.
        
        Args:
            measurements: [H, W] radiation measurements
            measurement_mask: [H, W] binary mask of measured locations
            robot_pos: (x, y) robot position
            robot_theta: robot heading angle in radians
            obstacle_distance_map: [H, W] distance to obstacles (optional)
            velocity_map: [H, W] velocity limits (optional)
            
        Returns:
            layers: [3, H, W] numpy array containing:
                    - Channel 0: Risk layer
                    - Channel 1: Information gain layer  
                    - Channel 2: Traversability layer
            radiation_field: [H, W] predicted radiation field
            uncertainty: [H, W] uncertainty map
            combined_weight: [H, W] combined weight map for path planning
        """
        print("⚡ Predicting radiation field and uncertainty...")
        
        # Get predictions from model
        radiation_field, uncertainty = self.predict_uncertainty(measurements, measurement_mask)
        
        print("🧮 Calculating layers...")
        
        # Calculate individual layers
        risk_layer = self.calculate_risk_layer(radiation_field)
        info_layer = self.calculate_information_gain_layer(radiation_field, uncertainty)
        traversability_layer = self.calculate_traversability_layer(
            robot_pos, robot_theta, obstacle_distance_map, velocity_map)
        
        # Stack layers
        layers = np.stack([risk_layer, info_layer, traversability_layer], axis=0)
        
        # Calculate combined weight map
        combined_weight = self.combine_layers(risk_layer, info_layer, traversability_layer)
        
        print("✅ Layer generation complete")
        
        return layers, radiation_field, uncertainty, combined_weight


def visualize_layers(layers, radiation_field, uncertainty, combined_weight, 
                    measurements, measurement_mask, waypoints, robot_pos, robot_theta, ground_truth=None):
    """
    Visualize all generated layers and related data.
    
    Args:
        layers: [3, H, W] risk, info gain, traversability layers
        radiation_field: [H, W] predicted radiation field
        uncertainty: [H, W] uncertainty map
        combined_weight: [H, W] combined weight map
        measurements: [H, W] sparse measurements
        measurement_mask: [H, W] measurement mask
        waypoints: [N, 2] trajectory waypoints
        robot_pos: (x, y) robot position
        robot_theta: robot heading angle
        ground_truth: [H, W] ground truth radiation field (optional)
    """
    if ground_truth is not None:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('3-Channel Layer Generator Results with Ground Truth', fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('3-Channel Layer Generator Results', fontsize=16, fontweight='bold')
    
    # Extract individual layers
    risk_layer = layers[0]
    info_layer = layers[1]
    traversability_layer = layers[2]
    
    # Draw robot heading arrow (used in multiple plots)
    arrow_length = 20
    dx = arrow_length * np.cos(robot_theta)
    dy = arrow_length * np.sin(robot_theta)
    
    if ground_truth is not None:
        # Row 0: Ground Truth and Comparison
        # (0,0) Ground truth field (normalized)
        gt_normalized = ground_truth / ground_truth.max() if ground_truth.max() > 0 else ground_truth
        im_gt = axes[0,0].imshow(gt_normalized, cmap='hot', origin='lower', vmax=1.0)
        if len(waypoints) > 0:
            axes[0,0].plot(waypoints[:, 1], waypoints[:, 0], '-o', c='lime', lw=1.5, ms=2, alpha=0.8)
        axes[0,0].scatter(robot_pos[0], robot_pos[1], c='cyan', s=100, marker='*', 
                         edgecolor='black', linewidth=1)
        axes[0,0].arrow(robot_pos[0], robot_pos[1], dx, dy, 
                       head_width=8, head_length=6, fc='cyan', ec='black')
        axes[0,0].set_title('Ground Truth (Normalized)')
        axes[0,0].axis('off')
        plt.colorbar(im_gt, ax=axes[0,0], fraction=0.046, pad=0.04)
        
        # (0,1) Predicted field (normalized)
        field_normalized = radiation_field / radiation_field.max() if radiation_field.max() > 0 else radiation_field
        im_pred = axes[0,1].imshow(field_normalized, cmap='hot', origin='lower', vmax=1.0)
        axes[0,1].set_title('Predicted Field (Normalized)')
        axes[0,1].axis('off')
        plt.colorbar(im_pred, ax=axes[0,1], fraction=0.046, pad=0.04)
        
        # (0,2) Prediction error (absolute difference of normalized values)
        if ground_truth.shape == radiation_field.shape:
            pred_error = np.abs(gt_normalized - field_normalized)
            im_err = axes[0,2].imshow(pred_error, cmap='Reds', origin='lower', vmax=1.0)
            axes[0,2].set_title('Prediction Error (Normalized)')
            axes[0,2].axis('off')
            plt.colorbar(im_err, ax=axes[0,2], fraction=0.046, pad=0.04)
        else:
            axes[0,2].text(0.5, 0.5, 'Size Mismatch\nGT vs Pred', ha='center', va='center', 
                          transform=axes[0,2].transAxes, fontsize=12)
            axes[0,2].set_title('Prediction Error')
            axes[0,2].axis('off')
        
        # (0,3) Sparse measurements
        im_meas = axes[0,3].imshow(measurements, cmap='hot', origin='lower')
        if len(waypoints) > 0:
            axes[0,3].plot(waypoints[:, 1], waypoints[:, 0], '-o', c='lime', lw=1.5, ms=2, alpha=0.8)
        axes[0,3].scatter(robot_pos[0], robot_pos[1], c='cyan', s=100, marker='*', 
                         edgecolor='black', linewidth=1)
        axes[0,3].arrow(robot_pos[0], robot_pos[1], dx, dy, 
                       head_width=8, head_length=6, fc='cyan', ec='black')
        axes[0,3].set_title('Sparse Measurements')
        axes[0,3].axis('off')
        plt.colorbar(im_meas, ax=axes[0,3], fraction=0.046, pad=0.04)
        
        # Row 1: Predictions and analysis
        pred_row = 1
    else:
        # Row 0: Input data and predictions (original layout without GT)
        # (0,0) Sparse measurements
        im1 = axes[0,0].imshow(measurements, cmap='hot', origin='lower')
        if len(waypoints) > 0:
            axes[0,0].plot(waypoints[:, 1], waypoints[:, 0], '-o', c='lime', lw=1.5, ms=2, alpha=0.8)
        axes[0,0].scatter(robot_pos[0], robot_pos[1], c='cyan', s=100, marker='*', 
                         edgecolor='black', linewidth=1)
        axes[0,0].arrow(robot_pos[0], robot_pos[1], dx, dy, 
                       head_width=8, head_length=6, fc='cyan', ec='black')
        axes[0,0].set_title('Sparse Measurements & Trajectory')
        axes[0,0].axis('off')
        plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
        
        pred_row = 0
    
    # Prediction and analysis row (pred_row)
    if ground_truth is None or pred_row > 0:  # Only plot these if not already done in GT section
        # Predicted radiation field (with max=1 scaling)
        field_normalized = radiation_field / radiation_field.max() if radiation_field.max() > 0 else radiation_field
        im2 = axes[pred_row,1].imshow(field_normalized, cmap='hot', origin='lower', vmax=1.0)
        axes[pred_row,1].set_title('Predicted Field (Normalized)')
        axes[pred_row,1].axis('off')
        plt.colorbar(im2, ax=axes[pred_row,1], fraction=0.046, pad=0.04)
        
        # Uncertainty map
        im3 = axes[pred_row,2].imshow(uncertainty, cmap='viridis', origin='lower')
        axes[pred_row,2].set_title('Uncertainty Map')
        axes[pred_row,2].axis('off')
        plt.colorbar(im3, ax=axes[pred_row,2], fraction=0.046, pad=0.04)
        
        # Combined weight map
        im4 = axes[pred_row,3].imshow(combined_weight, cmap='plasma', origin='lower')
        axes[pred_row,3].scatter(robot_pos[0], robot_pos[1], c='white', s=100, marker='*', 
                         edgecolor='black', linewidth=2)
        axes[pred_row,3].set_title('Combined Weight Map')
        axes[pred_row,3].axis('off')
        plt.colorbar(im4, ax=axes[pred_row,3], fraction=0.046, pad=0.04)
    
    # Row for Individual layers
    if ground_truth is not None:
        layer_row = 2  # GT uses rows 0-1, layers start at row 2
    else:
        layer_row = 1  # No GT, layers start at row 1
    
    # Risk layer
    im5 = axes[layer_row,0].imshow(risk_layer, cmap='RdYlGn', origin='lower')
    axes[layer_row,0].set_title('Risk Layer (Safety)')
    axes[layer_row,0].axis('off')
    plt.colorbar(im5, ax=axes[layer_row,0], fraction=0.046, pad=0.04)
    
    # Information gain layer
    im6 = axes[layer_row,1].imshow(info_layer, cmap='Blues', origin='lower')
    axes[layer_row,1].set_title('Information Gain Layer')
    axes[layer_row,1].axis('off')
    plt.colorbar(im6, ax=axes[layer_row,1], fraction=0.046, pad=0.04)
    
    # Traversability layer
    im7 = axes[layer_row,2].imshow(traversability_layer, cmap='Oranges', origin='lower')
    # Show robot position and orientation
    axes[layer_row,2].scatter(robot_pos[0], robot_pos[1], c='red', s=100, marker='*', 
                     edgecolor='black', linewidth=2)
    axes[layer_row,2].arrow(robot_pos[0], robot_pos[1], dx, dy, 
                   head_width=8, head_length=6, fc='red', ec='black', alpha=0.8)
    axes[layer_row,2].set_title('Traversability Layer')
    axes[layer_row,2].axis('off')
    plt.colorbar(im7, ax=axes[layer_row,2], fraction=0.046, pad=0.04)
    
    # Statistics panel
    axes[layer_row,3].axis('off')
    
    # Add GT statistics if available
    if ground_truth is not None:
        stats_text = f"""Statistics:

Ground Truth:
  Min: {ground_truth.min():.3f}
  Max: {ground_truth.max():.3f}
  Mean: {ground_truth.mean():.3f}

Prediction:
  Min: {radiation_field.min():.3f}
  Max: {radiation_field.max():.3f}
  Mean: {radiation_field.mean():.3f}

Risk Layer:
  Min: {risk_layer.min():.3f}
  Max: {risk_layer.max():.3f}
  Mean: {risk_layer.mean():.3f}

Info Gain Layer:
  Min: {info_layer.min():.3f}
  Max: {info_layer.max():.3f}
  Mean: {info_layer.mean():.3f}

Traversability:
  Min: {traversability_layer.min():.3f}
  Max: {traversability_layer.max():.3f}
  Mean: {traversability_layer.mean():.3f}

Robot: ({robot_pos[0]}, {robot_pos[1]})
Heading: {np.degrees(robot_theta):.1f}°"""
    else:
        stats_text = f"""Layer Statistics:

Risk Layer:
  Min: {risk_layer.min():.3f}
  Max: {risk_layer.max():.3f}
  Mean: {risk_layer.mean():.3f}

Info Gain Layer:
  Min: {info_layer.min():.3f}
  Max: {info_layer.max():.3f}
  Mean: {info_layer.mean():.3f}

Traversability Layer:
  Min: {traversability_layer.min():.3f}
  Max: {traversability_layer.max():.3f}
  Mean: {traversability_layer.mean():.3f}

Combined Weight:
  Min: {combined_weight.min():.3f}
  Max: {combined_weight.max():.3f}
  Mean: {combined_weight.mean():.3f}

Robot State:
  Position: ({robot_pos[0]}, {robot_pos[1]})
  Heading: {np.degrees(robot_theta):.1f}°
"""
    
    axes[layer_row,3].text(0.05, 0.95, stats_text, transform=axes[layer_row,3].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[layer_row,3].set_title('Statistics')
    
    plt.tight_layout()
    plt.show()


def main():
    """Example usage of LayerGenerator."""
    parser = argparse.ArgumentParser(description="3-Channel Layer Generator")
    parser.add_argument("--checkpoint", default="checkpoints/convnext_simple_gt_exp8/ckpt_best.pth")
    parser.add_argument("--model_size", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=32)
    args = parser.parse_args()
    
    # Initialize generator
    generator = LayerGenerator(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        n_uncertainty_samples=args.n_samples
    )
    
    if not generator.load_model():
        return
    
    # Import dataset modules
    gt, ts = import_dataset_modules()
    
    # Create example scenario using dataset generation code
    print("🔧 Creating example scenario with dataset modules...")
    H, W = args.model_size, args.model_size
    
    # Generate ground truth using dataset modules
    rng = np.random.default_rng()
    n_sources = rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1)
    coords, amps, sigmas = gt.sample_sources(gt.GRID, n_sources, rng=rng)
    ground_truth = gt.gaussian_field(gt.GRID, coords, amps, sigmas)
    
    # Resize to model size if needed
    if gt.GRID != args.model_size:
        if CV2_AVAILABLE:
            ground_truth = cv2.resize(ground_truth, (args.model_size, args.model_size), interpolation=cv2.INTER_LINEAR)
        else:
            # Use scipy zoom as fallback
            from scipy.ndimage import zoom
            scale_factor = args.model_size / gt.GRID
            ground_truth = zoom(ground_truth, scale_factor, order=1)
    
    # Generate trajectory and sparse measurements using trajectory sampler
    waypoints = ts.generate_waypoints(args.model_size, min_pts=15, max_pts=40, rng=rng)
    measurements, measurement_mask = ts.sparse_from_waypoints(ground_truth, waypoints, rng=rng)
    
    print(f"📊 Generated {n_sources} radiation sources")
    print(f"🛤️  Generated trajectory with {len(waypoints)} waypoints")
    
    # Robot state
    robot_pos = (W//4, H//4)  # bottom-left quadrant
    robot_theta = np.pi/4     # 45 degrees
    
    n_measurements = np.sum(measurement_mask > 0)
    print(f"🤖 Robot position: {robot_pos}, heading: {robot_theta:.2f} rad ({np.degrees(robot_theta):.1f}°)")
    print(f"📍 Measurements: {n_measurements} points")
    
    # Generate layers
    layers, radiation_field, uncertainty, combined_weight = generator.get_layers(
        measurements, measurement_mask, robot_pos, robot_theta
    )
    
    # Print results
    print(f"\n📋 Results:")
    print(f"  Layers shape: {layers.shape}")  
    print(f"  Radiation field range: [{radiation_field.min():.4f}, {radiation_field.max():.4f}]")
    print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print(f"  Combined weight range: [{combined_weight.min():.4f}, {combined_weight.max():.4f}]")
    print(f"  Risk layer range: [{layers[0].min():.4f}, {layers[0].max():.4f}]")
    print(f"  Info layer range: [{layers[1].min():.4f}, {layers[1].max():.4f}]")
    print(f"  Traversability range: [{layers[2].min():.4f}, {layers[2].max():.4f}]")
    
    print("\n🎉 Layer generation completed successfully!")
    
    # Visualize the results
    print("\n📊 Displaying visualization...")
    visualize_layers(layers, radiation_field, uncertainty, combined_weight,
                    measurements, measurement_mask, waypoints, robot_pos, robot_theta, ground_truth)


if __name__ == "__main__":
    main()