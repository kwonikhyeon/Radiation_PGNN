#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Uncertainty Mapping Application (Patched)
- Robust model import & checkpoint loading with integrity logs
- Eval-only inference (no BN corruption via train())
- Masked log1p preprocessing for log channel
- Optional auto-process: clicks immediately trigger measurement + inference
- Uncertainty via eval-only TTA (noise on measurement channel) + distance weighting
"""

import os, sys, argparse, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import distance_transform_edt
import tkinter as tk

# ----------------- Robust model import (align with eval script) -----------------
def import_model():
    """
    Try to import SimplifiedConvNeXtPGNN from:
      1) project_root/src/model/simplified_conv_next_pgnn.py
      2) project_root/model/simplified_conv_next_pgnn.py
    """
    # Get project root (parent of test directory)
    ROOT = Path(__file__).resolve().parents[1]
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

# ----------------- App -----------------
class InteractiveUncertaintyApp:
    def __init__(self, checkpoint_path="checkpoints/convnext_simple_gt_exp7/ckpt_best.pth", n_samples=32):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GUI / canvas
        self.model_size = 256
        self.canvas_size = 768
        self.scale_factor = self.canvas_size // self.model_size

        # Uncertainty params
        self.n_samples = n_samples
        self.distance_weight = 0.6
        self.distance_sigma = 40.0  # larger -> faster growth with distance

        # Inputs
        self.ground_truth = None
        self.measurements = np.zeros((self.model_size, self.model_size), np.float32)
        self.measurement_mask = np.zeros((self.model_size, self.model_size), np.float32)

        # Click storage
        self.click_positions = []   # list of (canvas_x, canvas_y)
        self.model_positions = []   # list of (x, y) in model grid
        self.pending_clicks = []    # list of (canvas_x, canvas_y, model_x, model_y)

        # Model
        self.model = None

        # Auto process clicks
        self.auto_process = True

        # GUI holders
        self.root = None
        self.fig = None
        self.ax_main = None
        self.ax_unc = None
        self.ax_gt = None
        self.canvas = None
        self.status_label = None

        # Current outputs
        self.current_prediction = None
        self.current_uncertainty = None

        print(f"🎯 Device: {self.device} | Model grid: {self.model_size} | Canvas: {self.canvas_size}")

    # ----------------- Model -----------------
    def load_model(self):
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

        # param count
        tot = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model ready: {tot/1e6:.1f}M params")
        return True

    # ----------------- Data generation -----------------
    def generate_random_ground_truth(self, seed=None):
        """Generate a smooth field with a few Gaussian sources (for demo)."""
        rng = np.random.default_rng(seed)
        H = W = self.model_size
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        field = np.zeros((H, W), np.float32)

        n_src = rng.integers(1, 4+1)
        for _ in range(n_src):
            cx = rng.uniform(0, W-1)
            cy = rng.uniform(0, H-1)
            amp = rng.uniform(0.3, 1.0)
            sig = rng.uniform(8.0, 25.0)
            field += amp * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sig**2))

        # normalize to [0,1]
        field -= field.min()
        if field.max() > 0:
            field /= field.max()
        self.ground_truth = field
        print(f"🎲 New ground truth with {n_src} sources.")

    # ----------------- Preprocess -----------------
    def create_model_input(self):
        """Create 6-channel input tensor with masked log1p and distance map."""
        H = W = self.model_size
        # channels 0..1: measurements, mask
        meas = self.measurements.astype(np.float32)
        mask = self.measurement_mask.astype(np.float32)

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

        # Debug stats
        print("🔎 Input stats: "
              f"meas[min={meas.min():.4f}, max={meas.max():.4f}, nz={int((meas>0).sum())}] | "
              f"mask[nz={int(mask.sum())}] | "
              f"logm[min={logm.min():.4f}, max={logm.max():.4f}] | "
              f"dist[min={dist.min():.4f}, max={dist.max():.4f}]")

        inp = np.stack([meas, mask, logm, coord_x, coord_y, dist], axis=0)  # [6,H,W]
        inp_t = torch.from_numpy(inp[None]).float().to(self.device)         # [1,6,H,W]
        return inp_t

    # ----------------- Inference -----------------
    @torch.no_grad()
    def predict(self, inp_t: torch.Tensor):
        self.model.eval()
        out = self.model(inp_t)
        return out[0, 0].detach().cpu().numpy()

    @torch.no_grad()
    def predict_uncertainty(self, inp_t: torch.Tensor):
        """Eval-only TTA: noise on measurement channel; + distance weighting."""
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
        mask = inp_t[0, 1].detach().cpu().numpy()
        if np.any(mask > 0):
            dist = distance_transform_edt(mask == 0).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        else:
            dist = np.ones_like(std_pred, np.float32)

        # exponential weight, then normalize
        w = np.exp(dist * self.distance_sigma / 100.0)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)

        stdn = (std_pred - std_pred.min()) / (std_pred.max() - std_pred.min() + 1e-8)
        unc = (1 - self.distance_weight) * stdn + self.distance_weight * w
        return mean_pred, unc

    # ----------------- Click handling -----------------
    def on_click(self, event):
        if event.inaxes != self.ax_main or event.xdata is None or event.ydata is None:
            return
        cx, cy = int(event.xdata), int(event.ydata)
        mx, my = cx // self.scale_factor, cy // self.scale_factor
        mx = max(0, min(mx, self.model_size-1))
        my = max(0, min(my, self.model_size-1))

        if self.auto_process:
            self.add_measurement(cx, cy, mx, my)
        else:
            self.pending_clicks.append((cx, cy, mx, my))
            self.update_visualization()
            self.status_label.config(text=f"Pending clicks: {len(self.pending_clicks)} — press 'Process Measurements'")

    def add_measurement(self, cx, cy, mx, my):
        gt_v = self.ground_truth[my, mx]
        self.measurements[my, mx] += float(gt_v)
        self.measurement_mask[my, mx] = 1.0
        self.click_positions.append((cx, cy))
        self.model_positions.append((mx, my))

        self.status_label.config(text=f"Processing measurement #{len(self.click_positions)}...")
        self.root.update_idletasks()
        threading.Thread(target=self.run_inference_and_update, daemon=True).start()

    def process_pending_clicks(self):
        while self.pending_clicks:
            cx, cy, mx, my = self.pending_clicks.pop(0)
            self.add_measurement(cx, cy, mx, my)

    def reset_measurements(self):
        self.measurements.fill(0.0)
        self.measurement_mask.fill(0.0)
        self.click_positions.clear()
        self.model_positions.clear()
        self.pending_clicks.clear()
        self.current_prediction = None
        self.current_uncertainty = None
        self.update_visualization()
        self.status_label.config(text="Reset complete — click to add measurements")

    def generate_new_gt(self):
        self.generate_random_ground_truth()
        self.reset_measurements()
        self.status_label.config(text="New ground truth generated")

    # ----------------- Visualization -----------------
    def upscale(self, arr: np.ndarray):
        return np.repeat(np.repeat(arr, self.scale_factor, 0), self.scale_factor, 1)

    def run_inference_and_update(self):
        try:
            inp_t = self.create_model_input()
            pred = self.predict(inp_t)
            _, unc = self.predict_uncertainty(inp_t)
            self.current_prediction = pred
            self.current_uncertainty = unc
            self.root.after(0, self.update_visualization)
        except Exception as e:
            print("❌ Inference error:", e)
            self.root.after(0, lambda: self.status_label.config(text=f"Inference error: {e}"))

    def update_visualization(self):
        if self.fig is None:
            return
        self.ax_main.clear()
        self.ax_unc.clear()
        self.ax_gt.clear()

        self.ax_main.set_title("Prediction (click to add measurements)")
        self.ax_main.set_xlim(0, self.canvas_size); self.ax_main.set_ylim(self.canvas_size, 0); self.ax_main.set_aspect("equal")

        self.ax_unc.set_title("Uncertainty")
        self.ax_unc.set_xlim(0, self.canvas_size); self.ax_unc.set_ylim(self.canvas_size, 0); self.ax_unc.set_aspect("equal")

        self.ax_gt.set_title(f"Ground Truth + Points ({len(self.click_positions)})")
        self.ax_gt.set_xlim(0, self.canvas_size); self.ax_gt.set_ylim(self.canvas_size, 0); self.ax_gt.set_aspect("equal")

        # main panel
        if self.current_prediction is not None:
            self.ax_main.imshow(self.upscale(self.current_prediction), cmap="hot", vmin=0, vmax=1,
                                extent=[0, self.canvas_size, self.canvas_size, 0])
        else:
            self.ax_main.imshow(np.zeros((self.canvas_size, self.canvas_size)), cmap="gray", vmin=0, vmax=1,
                                extent=[0, self.canvas_size, self.canvas_size, 0])
        # Show processed click positions as white crosses
        if self.click_positions:
            xs, ys = zip(*self.click_positions)
            self.ax_main.scatter(xs, ys, c="white", s=100, marker="+", linewidths=3, alpha=0.9)
        # Show pending clicks as yellow circles
        if self.pending_clicks:
            px, py = zip(*[(x, y) for x, y, _, _ in self.pending_clicks])
            self.ax_main.scatter(px, py, c="yellow", s=70, marker="o", linewidths=2, alpha=0.8)

        # uncertainty panel
        if self.current_uncertainty is not None:
            self.ax_unc.imshow(self.upscale(self.current_uncertainty), cmap="plasma", vmin=0, vmax=1,
                               extent=[0, self.canvas_size, self.canvas_size, 0])
        else:
            self.ax_unc.imshow(np.zeros((self.canvas_size, self.canvas_size)), cmap="gray", vmin=0, vmax=1,
                               extent=[0, self.canvas_size, self.canvas_size, 0])
            self.ax_unc.text(self.canvas_size//2, self.canvas_size//2, "No uncertainty yet",
                             ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        if self.click_positions:
            xs, ys = zip(*self.click_positions)
            self.ax_unc.scatter(xs, ys, c="white", s=100, marker="+", linewidths=3, alpha=0.8)

        # GT panel
        self.ax_gt.imshow(self.upscale(self.ground_truth), cmap="hot", vmin=0, vmax=1,
                          extent=[0, self.canvas_size, self.canvas_size, 0])
        if self.click_positions:
            xs, ys = zip(*self.click_positions)
            self.ax_gt.scatter(xs, ys, c="white", s=100, marker="+", linewidths=3, alpha=0.9)
        if self.pending_clicks:
            px, py = zip(*[(x, y) for x, y, _, _ in self.pending_clicks])
            self.ax_gt.scatter(px, py, c="yellow", s=70, marker="o", linewidths=2, alpha=0.8)

        self.canvas.draw_idle()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Interactive Uncertainty Mapping")
        self.root.geometry("2000x820")

        self.fig, (self.ax_main, self.ax_unc, self.ax_gt) = plt.subplots(1, 3, figsize=(20, 6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        ctrl = tk.Frame(self.root); ctrl.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(ctrl, text="Reset", command=self.reset_measurements).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Generate New GT", command=self.generate_new_gt).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text="Process Measurements", command=self.process_pending_clicks, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(ctrl, text="Exit", command=self.root.quit, bg="#f44336", fg="white").pack(side=tk.RIGHT, padx=5)

        # Auto-process toggle
        self.auto_var = tk.BooleanVar(value=self.auto_process)
        def toggle_auto():
            self.auto_process = self.auto_var.get()
        tk.Checkbutton(ctrl, text="Auto-process clicks", variable=self.auto_var, command=toggle_auto).pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.root, text="Click on Prediction panel to add measurements")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        # initial draw
        self.update_visualization()

    # ----------------- Run -----------------
    def run(self):
        print("🚀 Starting Interactive Uncertainty Mapping App")
        if not self.load_model():
            return
        self.generate_random_ground_truth()
        self.setup_gui()
        self.root.mainloop()

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/convnext_simple_gt_exp8/ckpt_best.pth")
    ap.add_argument("--n_samples", type=int, default=32)
    args = ap.parse_args()

    app = InteractiveUncertaintyApp(checkpoint_path=args.checkpoint, n_samples=args.n_samples)
    app.run()

if __name__ == "__main__":
    main()
