# ──────────────────────────────────────────────────────────────
# test/modules/model_inference.py - Model Inference Module
# ──────────────────────────────────────────────────────────────
"""
Model inference module for radiation field prediction.
Based on interactive_uncertainty_app_patched.py implementation.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from typing import Optional, Tuple

__all__ = [
    "ModelInference", "import_model", "create_model_input"
]


def import_model():
    """
    Import SimplifiedConvNeXtPGNN from project structure.
    Based on interactive_uncertainty_app_patched.py logic.
    """
    # Get project root (parent of test directory)
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    
    try:
        # Try src/model first (current project structure)
        from src.model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
        print("✅ Imported SimplifiedConvNeXtPGNN from src.model")
        return SimplifiedConvNeXtPGNN
    except ImportError:
        try:
            # Fallback to model directory
            from model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
            print("✅ Imported SimplifiedConvNeXtPGNN from model")
            return SimplifiedConvNeXtPGNN
        except ImportError as e:
            print("❌ Could not import SimplifiedConvNeXtPGNN from either location:", e)
            print(f"   Project root: {ROOT}")
            print(f"   Available paths: {list(ROOT.glob('**/simplified_conv_next_pgnn.py'))}")
            raise


class ModelInference:
    """Model inference class for radiation field prediction."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu", n_samples: int = 10):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.n_samples = n_samples  # for uncertainty estimation
        self.model_size = 256
        self.model = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load model from checkpoint. Based on interactive_uncertainty_app_patched.py."""
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
        print(f"[ckpt] missing={len(ik.missing_keys)} unexpected={len(ik.unexpected_keys)}")
        if ik.missing_keys:
            print("   → missing keys sample:", ik.missing_keys[:5])
        if ik.unexpected_keys:
            print("   → unexpected keys sample:", ik.unexpected_keys[:5])

        self.model.to(self.device).eval()
    
    def create_model_input(self, measurements: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """
        Create 6-channel input tensor from measurements and mask.
        Based on interactive_uncertainty_app_patched.py implementation.
        """
        H = W = self.model_size
        
        # channels 0..1: measurements, mask
        meas = measurements.astype(np.float32)
        mask = mask.astype(np.float32)

        # channel 2: masked log1p(meas); zero where mask=0
        logm = np.zeros_like(meas, dtype=np.float32)
        nz = mask > 0
        logm[nz] = np.log1p(meas[nz])

        # channels 3..4: normalized coordinates in [0,1]
        yy, xx = np.mgrid[0:H, 0:W]
        coord_x = (xx / (W-1)).astype(np.float32)
        coord_y = (yy / (H-1)).astype(np.float32)

        # channel 5: normalized distance to nearest measurement (1 far, 0 near)
        if np.any(mask > 0):
            dist = distance_transform_edt(mask == 0).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        else:
            dist = np.ones((H, W), np.float32)

        # Stack all channels
        inp = np.stack([meas, mask, logm, coord_x, coord_y, dist], axis=0)  # [6,H,W]
        inp_t = torch.from_numpy(inp[None]).float().to(self.device)         # [1,6,H,W]
        return inp_t
    
    @torch.no_grad()
    def predict(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Make prediction from measurements and mask.
        Based on interactive_uncertainty_app_patched.py.
        """
        inp_t = self.create_model_input(measurements, mask)
        self.model.eval()
        out = self.model(inp_t)
        return out[0, 0].detach().cpu().numpy()
    
    @torch.no_grad()
    def predict_with_uncertainty(self, measurements: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction with uncertainty estimation using TTA.
        Based on interactive_uncertainty_app_patched.py.
        """
        inp_t = self.create_model_input(measurements, mask)
        self.model.eval()
        
        preds = []
        for _ in range(self.n_samples):
            noisy = inp_t.clone()
            # add small Gaussian noise on measurement channel only
            noisy[:, 0:1] += torch.randn_like(noisy[:, 0:1]) * 0.01
            preds.append(self.model(noisy).detach().cpu().numpy())
        
        preds = np.asarray(preds)            # [K,B,1,H,W]
        preds = preds[:, 0, 0]               # [K,H,W]

        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # distance-based weighting
        mask_t = inp_t[0, 1].detach().cpu().numpy()
        if np.any(mask_t > 0):
            dist = distance_transform_edt(mask_t == 0).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        else:
            dist = np.ones_like(mask_t)
        
        # Weight uncertainty by distance
        weighted_std = std_pred * (1.0 + dist * 2.0)  # Higher uncertainty farther from measurements
        
        return mean_pred, weighted_std


# Convenience functions for standalone use
def create_model_input(measurements: np.ndarray, mask: np.ndarray, model_size: int = 256) -> torch.Tensor:
    """
    Standalone function to create model input tensor.
    """
    H = W = model_size
    
    # channels 0..1: measurements, mask
    meas = measurements.astype(np.float32)
    mask = mask.astype(np.float32)

    # channel 2: masked log1p(meas); zero where mask=0
    logm = np.zeros_like(meas, dtype=np.float32)
    nz = mask > 0
    logm[nz] = np.log1p(meas[nz])

    # channels 3..4: normalized coordinates in [0,1]
    yy, xx = np.mgrid[0:H, 0:W]
    coord_x = (xx / (W-1)).astype(np.float32)
    coord_y = (yy / (H-1)).astype(np.float32)

    # channel 5: normalized distance to nearest measurement (1 far, 0 near)
    if np.any(mask > 0):
        dist = distance_transform_edt(mask == 0).astype(np.float32)
        if dist.max() > 0:
            dist /= dist.max()
    else:
        dist = np.ones((H, W), np.float32)

    # Stack all channels
    inp = np.stack([meas, mask, logm, coord_x, coord_y, dist], axis=0)  # [6,H,W]
    inp_t = torch.from_numpy(inp[None]).float()                        # [1,6,H,W]
    return inp_t


if __name__ == "__main__":
    # Test the module
    print("Testing ModelInference module...")
    
    # Create dummy measurements and mask
    measurements = np.zeros((256, 256), dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some dummy measurement points
    for i in range(10):
        y, x = np.random.randint(0, 256, 2)
        measurements[y, x] = np.random.uniform(0.1, 1.0)
        mask[y, x] = 1
    
    # Test input creation
    inp_t = create_model_input(measurements, mask)
    print(f"Input tensor shape: {inp_t.shape}")
    print(f"Input channels range: {inp_t.min():.4f} to {inp_t.max():.4f}")
    
    print("ModelInference module test complete!")