#!/usr/bin/env python3
"""
ConvNeXt PGNN - SIMPLIFIED TRAINING (문제 해결용)
복잡한 손실 함수들을 제거하고 핵심만 남긴 안정화된 훈련 스크립트
"""
import argparse, json, math, os, random, pathlib, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from model.conv_next_pgnn import (
    ConvNeXtUNetWithMaskAttention as ConvNeXtUNet,
    laplacian_loss,
    physics_loss_unified
)

class RadFieldDataset(Dataset):
    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path)
        if "inp" in self.data:
            self.inp = self.data["inp"].astype(np.float32)
        else:
            chans = [self.data[k].astype(np.float32)
                      for k in ["M", "mask", "logM", "X", "Y", "distance"]]
            self.inp = np.stack(chans, axis=1)
        self.gt  = self.data["gt"].astype(np.float32)
        self.names = [f"{npz_path.stem}_{i}" for i in range(len(self.inp))]

    def __len__(self):  return len(self.inp)

    def __getitem__(self, idx):
        return ( torch.from_numpy(self.inp[idx]),
                 torch.from_numpy(self.gt[idx]),
                 self.names[idx] )

from kornia.metrics import ssim as kornia_ssim

def ssim(pred, gt):
    try:
        return kornia_ssim(pred, gt, window_size=11, reduction="none").mean()
    except TypeError:
        return kornia_ssim(pred, gt, window_size=11).mean()

def train_one_epoch_simple(model, loader, optimizer, epoch, cfg):
    """단순화된 훈련 함수 - 핵심 손실만 사용"""
    model.train()
    losses = []
    ssims = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for batch_idx, (inp, gt, names) in enumerate(pbar):
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        
        optimizer.zero_grad()
        
        mask = inp[:, 1:2]  # measurement mask
        pred = model(inp)   # [B, 1, H, W]

        # 1. CORE LOSS: Basic reconstruction with distance weighting
        dist = inp[:, 5:6]
        weight = torch.exp(-2.0 * dist)  # Simple distance weighting
        
        # Unmeasured region loss (핵심)
        unmeasured_mask = (1 - mask)
        loss_unmeasured = F.mse_loss(
            pred * unmeasured_mask * weight, 
            gt * unmeasured_mask * weight
        )
        
        # Overall reconstruction loss
        loss_all = F.mse_loss(pred, gt)
        
        # 2. PHYSICS LOSS: Laplacian smoothness (점진적 적용)
        if epoch >= 5:  # 5 에포크부터 적용
            lambda_pg = min(0.05, (epoch - 5) / 20.0 * 0.05)  # 최대 0.05까지 점진적 증가
            lap_loss = laplacian_loss(pred, mask)
        else:
            lambda_pg = 0.0
            lap_loss = torch.tensor(0.0, device=cfg.device)
        
        # 3. UNIFIED PHYSICS LOSS (선택적 적용)
        if epoch >= 15 and hasattr(cfg, 'use_physics') and cfg.use_physics:
            try:
                physics_loss = physics_loss_unified(pred, gt, inp)
                lambda_physics = min(0.02, (epoch - 15) / 30.0 * 0.02)  # 최대 0.02
            except:
                physics_loss = torch.tensor(0.0, device=cfg.device)
                lambda_physics = 0.0
        else:
            physics_loss = torch.tensor(0.0, device=cfg.device)
            lambda_physics = 0.0
        
        # 4. MEASUREMENT CONSTRAINT LOSS (gradient-preserving)
        mask_float = mask.float()
        measured_values = inp[:, 0:1]
        measurement_loss = F.mse_loss(pred * mask_float, measured_values * mask_float)
        
        # 5. TOTAL LOSS (단순화됨 + 측정 제약)
        loss = (
            2.0 * loss_unmeasured +      # 주요 재구성 손실
            0.3 * loss_all +             # 전체 재구성
            1000.0 * measurement_loss +  # 강한 측정값 제약 (그래디언트 보존)
            lambda_pg * lap_loss +       # 물리 평활성 (점진적)
            lambda_physics * physics_loss # 물리 법칙 (선택적)
        )
        
        # 안전 장치
        if torch.isnan(loss) or torch.isinf(loss) or loss > 50.0:
            print(f"Warning: Abnormal loss {loss:.4f} at epoch {epoch}, batch {batch_idx}")
            continue
        
        loss.backward()
        
        # 그래디언트 클리핑 (안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # SSIM 계산
        with torch.no_grad():
            ssim_val = ssim(pred, gt).item()
            ssims.append(ssim_val)
        
        # 진행률 표시
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "SSIM": f"{ssim_val:.4f}",
            "LapW": f"{lambda_pg:.4f}",
            "PhysW": f"{lambda_physics:.4f}"
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

class SimpleTrainConfig:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs', 60)
        self.batch_size = kwargs.get('batch_size', 16)
        self.lr = kwargs.get('lr', 2e-4)
        
        # 단순화된 모델 설정
        self.pred_scale = kwargs.get('pred_scale', 2.0)  # 보수적 스케일
        
        # 경로
        self.data_dir = Path(kwargs.get('data_dir', 'data'))
        self.save_dir = Path(kwargs.get('save_dir', 'checkpoints/convnext_simple_fix'))
        
        # 물리 손실 사용 여부
        self.use_physics = kwargs.get('use_physics', False)  # 기본적으로 비활성화
        
        # 디바이스
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description="ConvNeXt PGNN - Simplified Training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/convnext_simple_exp1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pred_scale", type=float, default=2.0)
    parser.add_argument("--use_physics", action='store_true', help="Enable physics loss")
    
    args = parser.parse_args()
    
    cfg = SimpleTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pred_scale=args.pred_scale,
        use_physics=args.use_physics,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    
    print(f"🚀 SIMPLIFIED ConvNeXt PGNN Training")
    print(f"📁 Data: {cfg.data_dir}")
    print(f"💾 Save: {cfg.save_dir}")
    print(f"🎯 Device: {cfg.device}")
    print(f"⚡ Pred Scale: {cfg.pred_scale}")
    print(f"🔬 Physics Loss: {cfg.use_physics}")
    
    # 저장 디렉토리 생성
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    config_dict = {
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'lr': cfg.lr,
        'pred_scale': cfg.pred_scale,
        'use_physics': cfg.use_physics,
        'data_dir': str(cfg.data_dir),
        'save_dir': str(cfg.save_dir)
    }
    
    with open(cfg.save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # 데이터 로더
    train_ds = RadFieldDataset(cfg.data_dir / "train.npz")
    val_ds = RadFieldDataset(cfg.data_dir / "val.npz")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # 단순화된 모델
    model = ConvNeXtUNet(in_channels=6, pred_scale=cfg.pred_scale).to(cfg.device)
    
    # 보수적 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # 간단한 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1
    )
    
    best_val_ssim = 0.0
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting simplified training for {cfg.epochs} epochs...")
    
    for epoch in range(1, cfg.epochs + 1):
        # 훈련
        train_loss, train_ssim = train_one_epoch_simple(model, train_loader, optimizer, epoch, cfg)
        
        # 검증
        val_loss, val_ssim = validate(model, val_loader, cfg)
        
        # 학습률 업데이트
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train: Loss={train_loss:.4f}, SSIM={train_ssim:.4f} | "
              f"Val: Loss={val_loss:.4f}, SSIM={val_ssim:.4f} | "
              f"LR={current_lr:.2e}")
        
        # 최고 모델 저장
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
        
        # 최신 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_ssim': val_ssim,
            'config': config_dict
        }, cfg.save_dir / "ckpt_last.pth")
        
        # 조기 종료 (검증 손실이 개선되지 않을 때)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 검증 메트릭이 너무 오래 정체되면 경고
        if epoch > 10 and val_ssim < 0.1:
            print(f"⚠️  Warning: SSIM still very low ({val_ssim:.4f}) at epoch {epoch}")
            if epoch > 20 and val_ssim < 0.05:
                print(f"❌ Training appears to be failing. Consider debugging the model or data.")
    
    print(f"\n🎉 Training completed!")
    print(f"📈 Best validation SSIM: {best_val_ssim:.4f}")
    print(f"💾 Models saved to: {cfg.save_dir}")

if __name__ == "__main__":
    main()