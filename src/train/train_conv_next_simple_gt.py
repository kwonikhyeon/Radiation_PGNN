#!/usr/bin/env python3
"""
Simplified ConvNeXt PGNN Training Script
단순화된 ConvNeXt PGNN (53M 파라미터) 학습 스크립트
- 기존 99M → 53M 파라미터로 복잡성 감소
- Saturation 문제 해결된 안정적인 아키텍처
- GT-Physics 손실 포함
"""
import argparse, json, pathlib, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 경로 설정
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  

# 단순화된 모델 임포트
try:
    from model.simplified_conv_next_pgnn import (
        SimplifiedConvNeXtPGNN,
        simplified_laplacian_loss,
        simplified_physics_loss
    )
    print(f"✅ Successfully imported SimplifiedConvNeXtPGNN from {ROOT}")
except ImportError as e:
    print(f"❌ Error importing simplified model: {e}")
    print(f"Current ROOT path: {ROOT}")
    print(f"Simplified model file exists: {(ROOT / 'simplified_conv_next_pgnn.py').exists()}")
    sys.exit(1)

# GT-Physics 관련 함수들 (기존 모듈에서 가져오기)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from model.conv_next_pgnn import (
    gt_based_inverse_square_law_loss,
    extract_measured_data
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

def train_one_epoch_simple_gt(model, loader, optimizer, epoch, cfg):
    """Simplified ConvNeXt PGNN + GT-Physics 훈련 함수"""
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
        weight = torch.exp(-2.0 * dist)
        
        unmeasured_mask = (1 - mask)
        loss_unmeasured = F.mse_loss(
            pred * unmeasured_mask * weight, 
            gt * unmeasured_mask * weight
        )
        
        loss_all = F.mse_loss(pred, gt)
        
        # 2. MEASUREMENT CONSTRAINT LOSS (간소화됨 - 직접 보존으로 인해)
        mask_float = mask.float()
        measured_values = inp[:, 0:1]
        
        # 🔥 핵심 변경: 측정값 직접 보존으로 인해 측정점 손실 대폭 간소화
        # 측정점에서는 이미 정확한 값이 보존되므로 별도 제약 불필요
        
        # A. 측정점 보존 확인 (디버깅용, 실제로는 항상 0이어야 함)
        measurement_preservation_check = F.mse_loss(pred * mask_float, measured_values * mask_float)
        
        # B. 측정값 스파이크 억제 손실 제거 (직접 보존으로 해결됨)
        spike_penalty = torch.tensor(0.0, device=cfg.device)
        
        # 3. PHYSICS LOSS: Laplacian smoothness (점진적 적용) - 단순화된 버전 사용
        if epoch >= 5:
            lambda_pg = min(0.05, (epoch - 5) / 20.0 * 0.05)
            lap_loss = simplified_laplacian_loss(pred, mask)
        else:
            lambda_pg = 0.0
            lap_loss = torch.tensor(0.0, device=cfg.device)
        
        # 4. GT-PHYSICS LOSS (핵심 추가!) 
        gt_physics_loss = torch.tensor(0.0, device=cfg.device)
        if epoch >= cfg.gt_physics_start_epoch and cfg.use_gt_physics:
            try:
                # 측정 데이터 추출
                measured_positions, measured_values_list = extract_measured_data(inp)
                
                # GT 기반 물리 손실 계산
                gt_physics_loss = gt_based_inverse_square_law_loss(
                    pred, gt, measured_positions, measured_values_list,
                    background_level=cfg.background_level,
                    beta=cfg.air_attenuation
                )
                
                # 안전 장치
                if torch.isnan(gt_physics_loss) or torch.isinf(gt_physics_loss):
                    gt_physics_loss = torch.tensor(0.0, device=cfg.device)
                elif gt_physics_loss > 5.0:
                    gt_physics_loss = torch.clamp(gt_physics_loss, max=5.0)
                    
                # 점진적 가중치 증가
                lambda_gt_physics = min(cfg.gt_physics_weight, 
                                      (epoch - cfg.gt_physics_start_epoch) / cfg.gt_physics_warmup * cfg.gt_physics_weight)
            except Exception as e:
                # print(f"GT-Physics loss calculation failed: {e}")
                gt_physics_loss = torch.tensor(0.0, device=cfg.device)
                lambda_gt_physics = 0.0
        else:
            lambda_gt_physics = 0.0
        
        # 5. 가우시안 형태 제약 (소스 확산 방지)
        gaussian_loss = torch.tensor(0.0, device=cfg.device)
        if epoch >= 10:  # 어느정도 학습 후 적용
            # GT에서 피크 위치 찾기
            B = gt.shape[0]
            for b in range(B):
                gt_b = gt[b, 0]  # [H, W]
                pred_b = pred[b, 0]  # [H, W]
                
                if gt_b.max() > 0.5:  # 유의미한 소스가 있는 경우
                    # GT 피크 위치
                    peak_idx = torch.argmax(gt_b.flatten())
                    peak_y, peak_x = peak_idx // gt_b.shape[1], peak_idx % gt_b.shape[1]
                    
                    # 피크 주변 영역 정의 (15x15)
                    y_min, y_max = max(0, peak_y-7), min(gt_b.shape[0], peak_y+8)
                    x_min, x_max = max(0, peak_x-7), min(gt_b.shape[1], peak_x+8)
                    
                    # 해당 영역에서 가우시안 형태 유도
                    pred_region = pred_b[y_min:y_max, x_min:x_max]
                    gt_region = gt_b[y_min:y_max, x_min:x_max]
                    
                    # 중심에서 거리별 가중치 (가우시안 감쇠 유도)
                    center_y, center_x = pred_region.shape[0]//2, pred_region.shape[1]//2
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(pred_region.shape[0], device=cfg.device),
                        torch.arange(pred_region.shape[1], device=cfg.device),
                        indexing='ij'
                    )
                    distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                    
                    # 거리에 따른 감쇠 비율이 GT와 유사해야 함
                    if gt_region.max() > 0:
                        gt_decay = gt_region / gt_region.max()
                        pred_decay = pred_region / (pred_region.max() + 1e-8)
                        
                        # 먼 거리에서의 감쇠가 GT와 유사해야 함
                        far_mask = (distances > 3.0).float()
                        gaussian_loss += F.mse_loss(pred_decay * far_mask, gt_decay * far_mask) / B
        
        # 6. 강도 제한 손실 (정규화된 범위 준수)
        intensity_limit_loss = torch.tensor(0.0, device=cfg.device)
        if epoch >= 5:
            # GT가 [0,1] 정규화되어 있으므로 예측도 동일 범위로 제한
            intensity_penalty = F.relu(pred - 1.0).mean()  # 1.0 이상 억제 (1.5→1.0)
            
            # 피크 주변이 아닌 곳에서 강한 예측 억제
            for b in range(gt.shape[0]):
                gt_b = gt[b, 0]
                pred_b = pred[b, 0]
                
                if gt_b.max() > 0.3:
                    # GT 피크에서 멀리 떨어진 곳의 강한 예측 억제
                    peak_idx = torch.argmax(gt_b.flatten())
                    peak_y, peak_x = peak_idx // gt_b.shape[1], peak_idx % gt_b.shape[1]
                    
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(gt_b.shape[0], device=cfg.device),
                        torch.arange(gt_b.shape[1], device=cfg.device),
                        indexing='ij'
                    )
                    distances = torch.sqrt((y_coords - peak_y)**2 + (x_coords - peak_x)**2)
                    
                    # 피크에서 멀리 떨어진 곳 (거리 > 20)에서 강한 예측 억제
                    far_region = (distances > 20.0).float()
                    intensity_limit_loss += F.relu(pred_b * far_region - 0.05).mean() / gt.shape[0]  # 0.1→0.05
        
        # 7. TOTAL LOSS (직접 측정값 보존 버전 - 대폭 간소화)
        loss = (
            3.0 * loss_unmeasured +                           # 주요 재구성 (증가: 2.0→3.0)
            0.5 * loss_all +                                  # 전체 재구성 (증가: 0.3→0.5)
            10.0 * measurement_preservation_check +           # 측정값 보존 확인 (대폭 감소: 500→10)
            0.0 * spike_penalty +                             # 스파이크 억제 제거 (100→0)
            lambda_pg * lap_loss +                            # 라플라시안 (점진적)
            lambda_gt_physics * gt_physics_loss +             # GT-Physics (점진적)
            0.3 * gaussian_loss +                             # 가우시안 형태 제약 (증가: 0.2→0.3)
            0.7 * intensity_limit_loss                        # 강도 제한 (증가: 0.5→0.7)
        )
        
        # 안전 장치
        if torch.isnan(loss) or torch.isinf(loss) or loss > 50.0:
            print(f"Warning: Abnormal loss {loss:.4f} at epoch {epoch}, batch {batch_idx}")
            continue
        
        loss.backward()
        
        # 그래디언트 클리핑
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
            "GTPhysW": f"{lambda_gt_physics:.4f}",
            "Gauss": f"{gaussian_loss.item():.4f}",
            "IntLim": f"{intensity_limit_loss.item():.4f}"
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

class SimpleGTTrainConfig:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs', 60)
        self.batch_size = kwargs.get('batch_size', 16)
        self.lr = kwargs.get('lr', 2e-4)
        
        # 모델 설정
        self.pred_scale = kwargs.get('pred_scale', 1.0)  # 1.0에서 1.5로 증가 (보수적)
        
        # GT-Physics 설정
        self.use_gt_physics = kwargs.get('use_gt_physics', True)
        self.gt_physics_start_epoch = kwargs.get('gt_physics_start_epoch', 15)
        self.gt_physics_weight = kwargs.get('gt_physics_weight', 0.05)  # 보수적 시작
        self.gt_physics_warmup = kwargs.get('gt_physics_warmup', 20)
        
        # 물리 파라미터
        self.background_level = kwargs.get('background_level', 0.0)
        self.air_attenuation = kwargs.get('air_attenuation', 0.01)
        
        # 경로
        self.data_dir = Path(kwargs.get('data_dir', 'data'))
        self.save_dir = Path(kwargs.get('save_dir', 'checkpoints/convnext_simple_gt_exp_00'))
        
        # 디바이스
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description="ConvNeXt PGNN - Simple + GT-Physics")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints/convnext_simple_gt_exp7")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pred_scale", type=float, default=1.0)
    
    # GT-Physics 전용 옵션
    parser.add_argument("--use_gt_physics", action='store_true', default=True, 
                       help="Enable GT-based physics loss")
    parser.add_argument("--gt_physics_weight", type=float, default=0.2,
                       help="GT-Physics loss weight")
    parser.add_argument("--gt_physics_start_epoch", type=int, default=20,
                       help="Epoch to start GT-Physics loss")
    
    args = parser.parse_args()
    
    cfg = SimpleGTTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pred_scale=args.pred_scale,
        use_gt_physics=args.use_gt_physics,
        gt_physics_weight=args.gt_physics_weight,
        gt_physics_start_epoch=args.gt_physics_start_epoch,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    
    print(f"🚀 Simplified ConvNeXt PGNN Training (53M params)")
    print(f"📁 Data: {cfg.data_dir}")
    print(f"💾 Save: {cfg.save_dir}")
    print(f"🎯 Device: {cfg.device}")
    print(f"⚡ Pred Scale: {cfg.pred_scale}")
    print(f"🔬 GT-Physics: {cfg.use_gt_physics}")
    if cfg.use_gt_physics:
        print(f"   - Weight: {cfg.gt_physics_weight}")
        print(f"   - Start Epoch: {cfg.gt_physics_start_epoch}")
    print(f"🏗️  Architecture: Simplified (53M vs 99M original)")
    
    # 저장 디렉토리 생성
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    config_dict = {
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'lr': cfg.lr,
        'pred_scale': cfg.pred_scale,
        'use_gt_physics': cfg.use_gt_physics,
        'gt_physics_weight': cfg.gt_physics_weight,
        'gt_physics_start_epoch': cfg.gt_physics_start_epoch,
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
    
    # 모델 - 단순화된 ConvNeXt PGNN 사용
    model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=cfg.pred_scale).to(cfg.device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1
    )
    
    best_val_ssim = 0.0
    
    print(f"\n🚀 Starting Simplified ConvNeXt PGNN training for {cfg.epochs} epochs...")
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    for epoch in range(1, cfg.epochs + 1):
        # 훈련
        train_loss, train_ssim = train_one_epoch_simple_gt(model, train_loader, optimizer, epoch, cfg)
        
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
    
    print(f"\n🎉 Training completed!")
    print(f"📈 Best validation SSIM: {best_val_ssim:.4f}")
    print(f"💾 Models saved to: {cfg.save_dir}")

if __name__ == "__main__":
    main()