#!/usr/bin/env python3
"""
ConvNeXt PGNN training with anti-blur mechanisms
Enhanced training specifically designed to prevent foggy/blurry predictions
"""
import argparse, json, math, os, random, pathlib, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))            # …/src
from model.conv_next_pgnn import (
    ConvNeXtUNetWithMaskAttention as ConvNeXtUNet,
    laplacian_loss,
    inverse_square_law_loss,
    estimate_sources_from_field,
    extract_measured_data,
    anti_blur_loss,
    sharpness_enhancement_loss,
    total_variation_loss,
    spatial_consistency_loss,
    intensity_preservation_loss
)

class RadFieldDataset(Dataset):
    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path)
        if "inp" in self.data:                       # (N,5,H,W)  형태
            self.inp = self.data["inp"].astype(np.float32)
        else:                                        # 개별 채널 → stack
            chans = [self.data[k].astype(np.float32)
                      for k in ["M", "mask", "logM", "X", "Y", "distance"]]
            self.inp = np.stack(chans, axis=1)       # (N,5,H,W)
        self.gt  = self.data["gt"].astype(np.float32)  # (N,1,H,W)
        self.names = [f"{npz_path.stem}_{i}" for i in range(len(self.inp))]

    def __len__(self):  return len(self.inp)

    def __getitem__(self, idx):
        return ( torch.from_numpy(self.inp[idx]),
                 torch.from_numpy(self.gt[idx]),
                 self.names[idx] )


from kornia.metrics import ssim as kornia_ssim

def ssim(pred, gt):
    """
    kornia >=0.6 : ssim(..., reduction='mean|none')
    kornia <0.6  : reduction 인자 없음 → (B,) tensor 반환
    """
    try:
        return kornia_ssim(pred, gt, window_size=11, reduction="none").mean()
    except TypeError:
        return kornia_ssim(pred, gt, window_size=11).mean()

def compute_anti_blur_loss_batch(pred, gt, inp, epoch, cfg):
    """Enhanced anti-blur loss computation"""
    mask = inp[:, 1:2]
    
    # 1. Core anti-blur loss
    anti_blur = anti_blur_loss(pred, gt, mask)
    
    # 2. Sharpness enhancement for peaks
    sharpness = sharpness_enhancement_loss(pred, gt, mask)
    
    # 3. Spatial consistency
    spatial_consistency = spatial_consistency_loss(pred, gt, mask)
    
    # 4. Intensity preservation for high-value regions
    intensity_preservation = intensity_preservation_loss(pred, gt)
    
    # 5. Total variation regularization (moderate)
    tv_loss = total_variation_loss(pred, mask, alpha=0.02)
    
    # 6. Enhanced gradient preservation
    # Compute gradients
    pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    gt_grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    gt_grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    
    # Weight gradients more near measurements
    mask_x = F.interpolate(mask, size=pred_grad_x.shape[2:], mode='bilinear', align_corners=False)
    mask_y = F.interpolate(mask, size=pred_grad_y.shape[2:], mode='bilinear', align_corners=False)
    
    weight_x = 1.0 + 2.0 * mask_x[:, :, :, :-1]  # Higher weight near measurements
    weight_y = 1.0 + 2.0 * mask_y[:, :, :-1, :]
    
    enhanced_gradient_loss = (
        F.mse_loss(pred_grad_x * weight_x, gt_grad_x * weight_x) +
        F.mse_loss(pred_grad_y * weight_y, gt_grad_y * weight_y)
    ) * 0.5
    
    # 7. Local variance preservation to prevent over-smoothing
    def local_variance(x, kernel_size=3):
        B, C, H, W = x.shape
        padding = kernel_size // 2
        x_unfold = F.unfold(x, kernel_size, padding=padding)
        x_var = x_unfold.var(dim=1, keepdim=True)
        return x_var.view(B, 1, H, W)
    
    pred_var = local_variance(pred)
    gt_var = local_variance(gt)
    
    # Only apply variance preservation in unmeasured regions
    unmeasured_mask = (1 - mask).float()
    variance_loss = F.mse_loss(pred_var * unmeasured_mask, gt_var * unmeasured_mask)
    
    # Progressive weighting based on training epoch
    epoch_factor = min(1.0, epoch / 30.0)  # Ramp up over 30 epochs
    
    # Combine all anti-blur components with progressive weighting
    total_anti_blur_loss = epoch_factor * (
        0.3 * anti_blur +                    # Core anti-blur loss
        0.25 * sharpness +                   # Peak sharpness
        0.2 * spatial_consistency +          # Spatial consistency  
        0.1 * intensity_preservation +       # High-intensity preservation
        0.1 * enhanced_gradient_loss +       # Enhanced gradient preservation
        0.05 * variance_loss                 # Variance preservation
    ) + 0.03 * tv_loss                       # TV regularization (always applied)
    
    return total_anti_blur_loss

def train_one_epoch(model, loader, optimizer, epoch, cfg):
    model.train()
    losses = []
    ssims = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for batch_idx, (inp, gt, names) in enumerate(pbar):
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        
        optimizer.zero_grad()
        
        mask = inp[:, 1:2]                  # measurement mask
        dist = inp[:, 5:6]                  # distance map

        pred = model(inp)                   # [B, 1, H, W] - with measurement preservation

        # Basic reconstruction losses
        weight = torch.exp(-3.0 * dist)
        loss_unmeasured = F.mse_loss(pred * (1 - mask) * weight, gt * (1 - mask) * weight)
        loss_measured   = F.mse_loss(pred * mask, gt * mask)  # Should be minimal due to direct substitution
        loss_all = F.mse_loss(pred, gt)
        
        # Physics-guided smoothness
        lap = laplacian_loss(pred, mask)

        # Enhanced center loss for high-intensity regions
        center_weight = (gt > 0.6).float()
        loss_center = F.mse_loss(pred * center_weight, gt * center_weight)
        
        # Suppression loss for far unmeasured regions
        far_unmeasured = ((inp[:, 5:6] > 0.7) & (mask == 0)).float()
        far_pred = pred * far_unmeasured
        far_gt = gt * far_unmeasured
        suppression_loss = F.relu(far_pred - far_gt).mean()
        
        # Measurement preservation verification
        measured_preservation_loss = F.mse_loss(pred * mask, gt * mask)
        
        # Physics loss (if enabled)
        physics_loss = torch.tensor(0.0, device=cfg.device)
        if epoch > cfg.physics_start_epoch:
            try:
                source_estimates = estimate_sources_from_field(pred, 
                                                             threshold=cfg.source_threshold,
                                                             max_sources=4)
                measured_positions, measured_values = extract_measured_data(inp)
                physics_loss = inverse_square_law_loss(
                    pred, measured_positions, measured_values, source_estimates,
                    background_level=cfg.background_level, 
                    beta=cfg.air_attenuation
                )
                
                if physics_loss > 10.0:
                    physics_loss = torch.tensor(10.0, device=cfg.device)
                    
            except Exception as e:
                physics_loss = torch.tensor(0.0, device=cfg.device)

        # Anti-blur loss computation
        anti_blur_total = compute_anti_blur_loss_batch(pred, gt, inp, epoch, cfg)
        
        # Loss weight scheduling
        lambda_pg = min(1.0, epoch / cfg.pg_warmup) * cfg.pg_weight
        lambda_physics = min(1.0, (epoch - cfg.physics_start_epoch) / 20.0) * cfg.physics_weight if epoch > cfg.physics_start_epoch else 0.0
        
        # Intensity weight - emphasize intensity reconstruction
        intensity_weight = 1.0 + 0.8 * min(1.0, epoch / 25.0)
        
        # Anti-blur weight - progressive increase
        anti_blur_weight = min(0.5, epoch / 40.0)  # Gradually increase to 0.5 over 40 epochs
        
        # Total loss with anti-blur components
        loss = (
            intensity_weight * loss_unmeasured +    # Primary reconstruction loss
            0.1 * loss_measured +                   # Measurement consistency (should be minimal)
            cfg.alpha_unmask * loss_all +           # Overall reconstruction
            lambda_pg * lap +                       # Physics-guided smoothness
            cfg.center_lambda * loss_center +       # High-intensity region focus
            lambda_physics * physics_loss +         # Physics consistency
            0.2 * suppression_loss +                # Far region suppression
            0.3 * measured_preservation_loss +      # Measurement preservation
            anti_blur_weight * anti_blur_total      # Anti-blur comprehensive loss
        )
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at epoch {epoch}, batch {batch_idx}, skipping")
            continue
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # SSIM computation
        with torch.no_grad():
            ssim_val = ssim(pred, gt).item()
            ssims.append(ssim_val)
        
        # Update progress bar
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "SSIM": f"{ssim_val:.4f}",
            "AntiBlur": f"{anti_blur_total.item():.4f}"
        })
    
    return np.mean(losses), np.mean(ssims)

def validate(model, loader, cfg):
    model.eval()
    losses = []
    ssims = []
    
    with torch.no_grad():
        for inp, gt, names in tqdm(loader, desc="Validation"):
            inp, gt = inp.to(cfg.device), gt.to(cfg.device)
            
            pred = model(inp)
            loss = F.mse_loss(pred, gt)
            losses.append(loss.item())
            
            ssim_val = ssim(pred, gt).item()
            ssims.append(ssim_val)
    
    return np.mean(losses), np.mean(ssims)

class TrainConfig:
    def __init__(self, **kwargs):
        # Training parameters
        self.epochs = kwargs.get('epochs', 80)
        self.batch_size = kwargs.get('batch_size', 16)
        self.lr = kwargs.get('lr', 1e-4)
        
        # Anti-blur specific parameters
        self.pred_scale = kwargs.get('pred_scale', 3.5)  # Higher scale for sharper predictions
        
        # Loss weights
        self.pg_weight = kwargs.get('pg_weight', 0.12)
        self.pg_warmup = kwargs.get('pg_warmup', 15)
        self.alpha_unmask = kwargs.get('alpha_unmask', 0.8)
        self.center_lambda = kwargs.get('center_lambda', 1.2)
        
        # Physics parameters
        self.physics_weight = kwargs.get('physics_weight', 0.08)
        self.physics_start_epoch = kwargs.get('physics_start_epoch', 20)
        self.source_threshold = kwargs.get('source_threshold', 0.18)
        self.background_level = kwargs.get('background_level', 0.005)
        self.air_attenuation = kwargs.get('air_attenuation', 0.03)
        
        # Paths
        self.data_dir = Path(kwargs.get('data_dir', 'data'))
        self.save_dir = Path(kwargs.get('save_dir', 'checkpoints/convnext_anti_blur_exp1'))
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description="ConvNeXt PGNN Anti-Blur Training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/convnext_anti_blur_exp1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pred_scale", type=float, default=3.5)
    parser.add_argument("--pg_weight", type=float, default=0.12)
    parser.add_argument("--physics_weight", type=float, default=0.08)
    parser.add_argument("--physics_start_epoch", type=int, default=20)
    parser.add_argument("--source_threshold", type=float, default=0.18)
    parser.add_argument("--background_level", type=float, default=0.005)
    parser.add_argument("--air_attenuation", type=float, default=0.03)
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pred_scale=args.pred_scale,
        pg_weight=args.pg_weight,
        physics_weight=args.physics_weight,
        physics_start_epoch=args.physics_start_epoch,
        source_threshold=args.source_threshold,
        background_level=args.background_level,
        air_attenuation=args.air_attenuation,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    
    print(f"🔥 Anti-Blur ConvNeXt PGNN Training")
    print(f"📁 Data: {cfg.data_dir}")
    print(f"💾 Save: {cfg.save_dir}")
    print(f"🎯 Device: {cfg.device}")
    print(f"⚡ Pred Scale: {cfg.pred_scale} (enhanced for sharpness)")
    
    # Create save directory
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = vars(cfg)
    config_dict['save_dir'] = str(config_dict['save_dir'])
    config_dict['data_dir'] = str(config_dict['data_dir'])
    with open(cfg.save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Data loaders
    train_ds = RadFieldDataset(cfg.data_dir / "train.npz")
    val_ds = RadFieldDataset(cfg.data_dir / "val.npz")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model with enhanced pred_scale for sharper predictions
    model = ConvNeXtUNet(in_channels=6, pred_scale=cfg.pred_scale).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    best_val_ssim = 0.0
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting anti-blur training for {cfg.epochs} epochs...")
    
    for epoch in range(1, cfg.epochs + 1):
        # Training
        train_loss, train_ssim = train_one_epoch(model, train_loader, optimizer, epoch, cfg)
        
        # Validation
        val_loss, val_ssim = validate(model, val_loader, cfg)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train: Loss={train_loss:.4f}, SSIM={train_ssim:.4f} | "
              f"Val: Loss={val_loss:.4f}, SSIM={val_ssim:.4f} | "
              f"LR={current_lr:.2e}")
        
        # Save best model based on validation SSIM
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
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ssim': val_ssim,
            'config': config_dict
        }, cfg.save_dir / "ckpt_last.pth")
    
    print(f"\n🎉 Training completed!")
    print(f"📈 Best validation SSIM: {best_val_ssim:.4f}")
    print(f"💾 Models saved to: {cfg.save_dir}")

if __name__ == "__main__":
    main()