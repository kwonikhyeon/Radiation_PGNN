#!/usr/bin/env python3
"""
Peak Constraint Methods Comparison - V2 (Fixed Learning Issues)
기본 모델의 스케일 문제를 해결하여 제대로 학습되도록 수정
"""

import argparse
import json
import pathlib
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from model.conv_next_pgnn import (
    ConvNeXtUNetWithMaskAttention as ConvNeXtUNetBase,
    laplacian_loss
)

# Dataset class
class RadFieldDataset(Dataset):
    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path)
        if "inp" in self.data:
            self.inp = self.data["inp"].astype(np.float32)
        else:
            chans = [self.data[k].astype(np.float32)
                      for k in ["M", "mask", "logM", "X", "Y", "distance"]]
            self.inp = np.stack(chans, axis=1)
        self.gt = self.data["gt"].astype(np.float32)
        self.names = [f"{npz_path.stem}_{i}" for i in range(len(self.inp))]

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inp[idx]),
                torch.from_numpy(self.gt[idx]),
                self.names[idx])

# SSIM metric
from kornia.metrics import ssim as kornia_ssim

def ssim(pred, gt):
    try:
        return kornia_ssim(pred, gt, window_size=11, reduction="none").mean()
    except TypeError:
        return kornia_ssim(pred, gt, window_size=11).mean()

# =====================================================================
# Improved Constraint Models
# =====================================================================
class Method1_HardClip(nn.Module):
    """Hard Clipping with proper scaling"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        # First normalize to reasonable range, then clip
        pred = pred / 4.0  # Rough normalization based on observed range
        pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred

class Method2_Sigmoid(nn.Module):
    """Sigmoid activation with proper scaling"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        # Scale to reasonable range before sigmoid
        pred = pred / 6.0  # Adjust scale for sigmoid
        pred = torch.sigmoid(pred)
        return pred

class Method3_Tanh(nn.Module):
    """Tanh + shift activation"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        # Scale to reasonable range before tanh
        pred = pred / 3.0  # Adjust scale for tanh
        pred = (torch.tanh(pred) + 1.0) / 2.0
        return pred

class Method4_LossConstrained(nn.Module):
    """Original model with loss-based constraints"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        # Normalize to [0,1] range approximately
        pred = pred / 4.0  # Basic normalization
        return pred

class Method5_Adaptive(nn.Module):
    """Adaptive normalization"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        
        # Adaptive normalization per batch
        pred = F.softplus(pred)  # Ensure positive
        batch_size = pred.shape[0]
        
        # Normalize each sample in batch individually
        for i in range(batch_size):
            sample = pred[i:i+1]
            sample_max = sample.max()
            if sample_max > 1e-8:
                pred[i:i+1] = sample / sample_max
        
        return pred

# =====================================================================
# Loss Functions
# =====================================================================
def intensity_constraint_loss(pred, target_max=1.0, penalty_weight=100.0):
    """Softer intensity constraint loss"""
    over_penalty = torch.relu(pred - target_max).mean()
    under_penalty = torch.relu(-pred).mean()
    return penalty_weight * (over_penalty + under_penalty)

def train_one_epoch(model, loader, optimizer, epoch, cfg, method_name):
    model.train()
    losses = []
    ssims = []
    pred_ranges = []
    
    pbar = tqdm(loader, desc=f"{method_name} Epoch {epoch:3d}")
    for batch_idx, (inp, gt, names) in enumerate(pbar):
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        
        optimizer.zero_grad()
        
        mask = inp[:, 1:2]
        pred = model(inp)

        # Track prediction range
        pred_ranges.append((pred.min().item(), pred.max().item()))

        # Basic losses
        dist = inp[:, 5:6]
        weight = torch.exp(-2.0 * dist)
        
        unmeasured_mask = (1 - mask)
        loss_unmeasured = F.mse_loss(
            pred * unmeasured_mask * weight, 
            gt * unmeasured_mask * weight
        )
        
        loss_all = F.mse_loss(pred, gt)
        
        # Measurement constraint
        mask_float = mask.float()
        measured_values = inp[:, 0:1]
        measurement_loss = F.mse_loss(pred * mask_float, measured_values * mask_float)
        
        # Spike prevention (lighter)
        spike_penalty = torch.tensor(0.0, device=cfg.device)
        if mask_float.sum() > 0:
            kernel = torch.ones(1, 1, 3, 3, device=cfg.device) / 9.0
            pred_smooth = F.conv2d(pred, kernel, padding=1)
            measured_pred = pred * mask_float
            measured_smooth = pred_smooth * mask_float
            spike_penalty = F.relu(measured_pred - measured_smooth - 0.1).mean()  # Reduced threshold
        
        # Physics loss (lighter)
        if epoch >= 5:
            lambda_pg = min(0.02, (epoch - 5) / 30.0 * 0.02)  # Reduced weight
            lap_loss = laplacian_loss(pred, mask)
        else:
            lambda_pg = 0.0
            lap_loss = torch.tensor(0.0, device=cfg.device)
        
        # Method-specific loss
        if method_name == "method4_loss_constrained":
            intensity_penalty = intensity_constraint_loss(pred, target_max=1.0, penalty_weight=50.0)  # Reduced weight
        else:
            intensity_penalty = torch.tensor(0.0, device=cfg.device)
        
        # Total loss (rebalanced)
        loss = (
            3.0 * loss_unmeasured +     # Increased from 2.0
            1.0 * loss_all +            # Increased from 0.3
            200.0 * measurement_loss +  # Reduced from 500.0
            20.0 * spike_penalty +      # Reduced from 100.0
            lambda_pg * lap_loss +
            intensity_penalty
        )
        
        if torch.isnan(loss) or torch.isinf(loss) or loss > 100.0:
            print(f"Skipping batch {batch_idx} due to abnormal loss: {loss.item()}")
            continue
        
        loss.backward()
        
        # Gentler gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Increased from 1.0
        
        optimizer.step()
        
        losses.append(loss.item())
        
        with torch.no_grad():
            ssim_val = ssim(pred, gt).item()
            ssims.append(ssim_val)
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "SSIM": f"{ssim_val:.4f}",
            "Range": f"[{pred.min().item():.3f}, {pred.max().item():.3f}]"
        })
    
    # Print range statistics
    if pred_ranges:
        min_vals = [r[0] for r in pred_ranges]
        max_vals = [r[1] for r in pred_ranges]
        print(f"Pred range stats - Min: {np.mean(min_vals):.3f}±{np.std(min_vals):.3f}, "
              f"Max: {np.mean(max_vals):.3f}±{np.std(max_vals):.3f}")
    
    return np.mean(losses), np.mean(ssims)

def validate(model, loader, cfg):
    model.eval()
    losses = []
    ssims = []
    pred_ranges = []
    
    with torch.no_grad():
        for inp, gt, names in tqdm(loader, desc="Validation"):
            inp, gt = inp.to(cfg.device), gt.to(cfg.device)
            
            pred = model(inp)
            pred_ranges.append((pred.min().item(), pred.max().item()))
            
            loss = F.mse_loss(pred, gt)
            losses.append(loss.item())
            
            ssim_val = ssim(pred, gt).item()
            ssims.append(ssim_val)
    
    # Range statistics
    if pred_ranges:
        min_vals = [r[0] for r in pred_ranges]
        max_vals = [r[1] for r in pred_ranges]
        avg_min, avg_max = np.mean(min_vals), np.mean(max_vals)
    else:
        avg_min, avg_max = 0.0, 0.0
    
    return np.mean(losses), np.mean(ssims), avg_min, avg_max

# =====================================================================
# Training Configuration
# =====================================================================
class TrainConfig:
    def __init__(self, method_name, **kwargs):
        self.method_name = method_name
        self.epochs = kwargs.get('epochs', 30)
        self.batch_size = kwargs.get('batch_size', 8)
        self.lr = kwargs.get('lr', 1e-4)  # Reduced learning rate
        
        self.data_dir = Path(kwargs.get('data_dir', 'data'))
        self.save_dir = Path(kwargs.get('save_dir', f'checkpoints/peak_comparison_v2/{method_name}'))
        
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# Model Factory
# =====================================================================
def create_model(method_name):
    # Create base model with appropriate pred_scale
    base_model = ConvNeXtUNetBase(in_channels=6, pred_scale=1.0)  # Reduced from 1.5
    
    if method_name == "method1_hard_clip":
        return Method1_HardClip(base_model)
    elif method_name == "method2_sigmoid":
        return Method2_Sigmoid(base_model)
    elif method_name == "method3_tanh":
        return Method3_Tanh(base_model)
    elif method_name == "method4_loss_constrained":
        return Method4_LossConstrained(base_model)
    elif method_name == "method5_adaptive":
        return Method5_Adaptive(base_model)
    else:
        raise ValueError(f"Unknown method: {method_name}")

# =====================================================================
# Training Runner
# =====================================================================
def train_method(method_name, data_dir="data", base_save_dir="checkpoints/peak_comparison_v2", epochs=30):
    print(f"\n🚀 Training {method_name}")
    print("="*60)
    
    cfg = TrainConfig(
        method_name=method_name,
        data_dir=data_dir,
        save_dir=f"{base_save_dir}/{method_name}",
        epochs=epochs
    )
    
    # Create save directory
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        'method_name': method_name,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'lr': cfg.lr,
        'data_dir': str(cfg.data_dir),
        'save_dir': str(cfg.save_dir)
    }
    
    with open(cfg.save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Data loaders
    try:
        train_ds = RadFieldDataset(cfg.data_dir / "train.npz")
        val_ds = RadFieldDataset(cfg.data_dir / "val.npz")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None, None
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    try:
        model = create_model(method_name).to(cfg.device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None
        
    # Optimizer with different settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-5)  # Reduced weight decay
    
    # More aggressive scheduler for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.lr * 2,  # Peak at 2x learning rate
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3  # Reach peak at 30% of training
    )
    
    best_val_ssim = 0.0
    training_history = []
    
    print(f"🎯 Device: {cfg.device}")
    print(f"📈 Starting training for {cfg.epochs} epochs...")
    
    for epoch in range(1, cfg.epochs + 1):
        try:
            # Training
            train_loss, train_ssim = train_one_epoch(model, train_loader, optimizer, epoch, cfg, method_name)
            
            # Validation
            val_loss, val_ssim, val_min, val_max = validate(model, val_loader, cfg)
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_ssim': train_ssim,
                'val_loss': val_loss,
                'val_ssim': val_ssim,
                'val_pred_min': val_min,
                'val_pred_max': val_max,
                'lr': current_lr
            }
            training_history.append(epoch_data)
            
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train: Loss={train_loss:.4f}, SSIM={train_ssim:.4f} | "
                  f"Val: Loss={val_loss:.4f}, SSIM={val_ssim:.4f} | "
                  f"Range: [{val_min:.3f}, {val_max:.3f}] | "
                  f"LR={current_lr:.1e}")
            
            # Save best model
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ssim': val_ssim,
                    'config': config_dict
                }, cfg.save_dir / "ckpt_best.pth")
                print(f"✅ New best SSIM: {val_ssim:.4f}")
            
            # Save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ssim': val_ssim,
                'config': config_dict
            }, cfg.save_dir / "ckpt_last.pth")
            
        except Exception as e:
            print(f"❌ Error in epoch {epoch}: {e}")
            continue
    
    # Save training history
    with open(cfg.save_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print(f"🎉 Training completed!")
    print(f"📈 Best validation SSIM: {best_val_ssim:.4f}")
    print(f"💾 Models saved to: {cfg.save_dir}")
    
    return best_val_ssim, cfg.save_dir

# =====================================================================
# Evaluation and Report (same as before)
# =====================================================================
def run_evaluation(method_name, checkpoint_dir, data_file="data/test.npz"):
    print(f"\n📊 Evaluating {method_name}")
    print("="*60)
    
    eval_script = "src/eval/eval_convnext.py"
    checkpoint_path = f"{checkpoint_dir}/ckpt_best.pth"
    output_dir = f"eval_comparison_v2/{method_name}"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    cmd = [
        "python3", eval_script,
        "--ckpt", checkpoint_path,
        "--data_file", data_file,
        "--out_dir", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Evaluation completed: {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed for {method_name}")
        print(f"Error: {e.stderr}")
        return None

def generate_comparison_report(results):
    print(f"\n📋 Generating Comparison Report")
    print("="*60)
    
    report_dir = Path("eval_comparison_v2/comparison_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        "experiment_info": {
            "description": "Peak Constraint Methods Comparison V2 - Fixed Learning",
            "methods": list(results.keys()),
            "training_epochs": 30,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    # Save comparison data
    with open(report_dir / "comparison_results.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    # Generate summary table
    print("\n📊 COMPARISON SUMMARY")
    print("-" * 80)
    print(f"{'Method':<25} {'Best SSIM':<12} {'Status':<15} {'Eval Dir'}")
    print("-" * 80)
    
    for method, data in results.items():
        status = "✅ Success" if data['eval_dir'] else "❌ Failed"
        ssim_str = f"{data['best_ssim']:.4f}" if data['best_ssim'] else "N/A"
        eval_dir = data['eval_dir'] if data['eval_dir'] else "N/A"
        print(f"{method:<25} {ssim_str:<12} {status:<15} {eval_dir}")
    
    print("-" * 80)
    print(f"\n💾 Detailed results saved to: {report_dir}/comparison_results.json")

# =====================================================================
# Main Execution
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Peak Constraint Methods Comparison V2")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/peak_comparison_v2")
    parser.add_argument("--methods", nargs="+", 
                       default=["method1_hard_clip", "method2_sigmoid", "method3_tanh", 
                               "method4_loss_constrained", "method5_adaptive"],
                       help="Methods to compare")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation, only train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    
    args = parser.parse_args()
    
    print("🎯 Peak Constraint Methods Comparison V2 - Fixed Learning")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💾 Save directory: {args.save_dir}")
    print(f"🔬 Methods: {', '.join(args.methods)}")
    print()
    
    results = {}
    
    for method in args.methods:
        results[method] = {
            'best_ssim': None,
            'checkpoint_dir': None,
            'eval_dir': None
        }
        
        # Training
        if not args.skip_training:
            try:
                best_ssim, checkpoint_dir = train_method(method, args.data_dir, args.save_dir, args.epochs)
                if best_ssim is not None:
                    results[method]['best_ssim'] = best_ssim
                    results[method]['checkpoint_dir'] = str(checkpoint_dir)
            except Exception as e:
                print(f"❌ Training failed for {method}: {e}")
                continue
        else:
            # Use existing checkpoint
            checkpoint_dir = Path(args.save_dir) / method
            if checkpoint_dir.exists():
                results[method]['checkpoint_dir'] = str(checkpoint_dir)
        
        # Evaluation
        if not args.skip_evaluation and results[method]['checkpoint_dir']:
            eval_dir = run_evaluation(method, results[method]['checkpoint_dir'])
            results[method]['eval_dir'] = eval_dir
    
    # Generate comparison report
    generate_comparison_report(results)
    
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    main()