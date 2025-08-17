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
    gt_based_inverse_square_law_loss,  # 새로운 GT 기반 물리 손실
    extract_measured_data
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
    

def train_one_epoch(model, loader, opt, epoch, cfg):
    model.train()
    total = 0.0
    for inp, gt, _ in tqdm(loader, desc=f"Train {epoch}"):
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        mask = inp[:, 1:2]          # 측정 위치
        dist = inp[:, 5:6]          # 거리 맵 (normalized)

        pred = model(inp)                          # [B, 1, H, W] - sparse values already preserved in model

        # PGNN loss weighting
        weight = torch.exp(-3.0 * dist)
        loss_unmeasured = F.mse_loss(pred * (1 - mask) * weight, gt * (1 - mask) * weight)
        loss_measured   = F.mse_loss(pred * mask, gt * mask)  # Should be 0 since sparse values are preserved
        loss_all = F.mse_loss(pred, gt)
        
        # 라플라시안 평활성 손실 (마스크 정보 활용)
        lap = laplacian_loss(pred, mask)

        # Center-specific loss
        center_weight = (gt > 0.7).float()
        loss_center = F.mse_loss(pred * center_weight, gt * center_weight)
        
        # Sparse region suppression loss - 원거리 비측정 영역의 과도한 예측 억제
        far_unmeasured = ((inp[:, 5:6] > 0.7) & (mask == 0)).float()  # 원거리 & 비측정
        far_pred = pred * far_unmeasured
        far_gt = gt * far_unmeasured
        # 원거리에서는 예측값이 실제값보다 크면 페널티
        suppression_loss = F.relu(far_pred - far_gt).mean()
        
        # Measured value preservation loss - 측정값 완전 보존 확인
        measured_preservation_loss = F.mse_loss(pred * mask, gt * mask)  # 이미 보장되어야 하지만 추가 확인
        
        # Spatial consistency loss - 공간적 일관성 강화
        # 측정점 주변에서의 그래디언트 일관성
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        gt_grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        gt_grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
        
        gradient_loss = (F.mse_loss(pred_grad_x, gt_grad_x) + F.mse_loss(pred_grad_y, gt_grad_y)) * 0.5

        # GT 기반 역제곱 법칙 물리 손실 (개선된 버전)
        gt_physics_loss = torch.tensor(0.0, device=cfg.device)
        if epoch > cfg.physics_start_epoch:  # 일정 에포크 후부터 물리 손실 적용
            try:
                # 측정 데이터 추출
                measured_positions, measured_values = extract_measured_data(inp)
                
                # GT 기반 역제곱 법칙 손실 계산
                gt_physics_loss = gt_based_inverse_square_law_loss(
                    pred, gt, measured_positions, measured_values,
                    background_level=cfg.background_level, 
                    beta=cfg.air_attenuation
                )
                
                # 물리 손실이 비정상적으로 클 경우 제한 (더 엄격한 제한)
                if gt_physics_loss > 5.0:  # 15.0에서 5.0으로 감소
                    print(f"Warning: Large GT physics loss ({gt_physics_loss:.4f}) at epoch {epoch}, clamping to 5.0")
                    gt_physics_loss = torch.tensor(5.0, device=cfg.device)
                    
            except Exception as e:
                # 물리 손실 계산 실패 시 0으로 설정하고 경고
                print(f"Warning: GT Physics loss calculation failed at epoch {epoch}: {str(e)}")
                gt_physics_loss = torch.tensor(0.0, device=cfg.device)

        # 총 손실 계산 - GT 기반 물리 손실 사용 (개선된 가중치)
        lambda_pg = min(1.0, epoch / cfg.pg_warmup) * cfg.pg_weight
        lambda_gt_physics = min(1.0, (epoch - cfg.physics_start_epoch) / cfg.physics_warmup) * cfg.physics_weight if epoch > cfg.physics_start_epoch else 0.0
        
        # 강도 복원을 위한 가중치 조정 (더 적극적으로)
        intensity_weight = 1.0 + 0.4 * min(1.0, epoch / 35.0)  # 0.3에서 0.4로, 40에서 35로 조정
        
        # 재균형된 손실 가중치 - 억제 완화
        loss = (intensity_weight * loss_unmeasured + 0.02 * loss_measured +  # measured 가중치 더 감소
                cfg.alpha_unmask * loss_all +
                lambda_pg * lap +
                cfg.center_lambda * loss_center +
                lambda_gt_physics * gt_physics_loss +  # GT 기반 물리 손실
                0.2 * suppression_loss +  # 원거리 억제 완화 (0.5→0.2)
                0.8 * measured_preservation_loss +  # 측정값 보존 완화 (1.0→0.8)
                0.1 * gradient_loss)  # 공간적 일관성 완화 (0.2→0.1)
        
        # 총 손실 안전장치 - 급격한 증가 방지
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at epoch {epoch}, skipping batch")
            continue
            
        if loss > 100.0:  # 비정상적으로 큰 손실
            print(f"Warning: Very large loss ({loss:.4f}) at epoch {epoch}")
            # GT 물리 손실 가중치를 일시적으로 줄임
            lambda_gt_physics_safe = lambda_gt_physics * 0.02  # 0.05에서 0.02로 더 감소
            loss = (intensity_weight * loss_unmeasured + 0.05 * loss_measured +
                    cfg.alpha_unmask * loss_all +
                    lambda_pg * lap +
                    cfg.center_lambda * loss_center +
                    lambda_gt_physics_safe * gt_physics_loss +
                    0.1 * suppression_loss +  # 억제 손실 더 완화
                    0.5 * measured_preservation_loss)  # 측정값 보존 완화
            print(f"Reduced loss: {loss:.4f}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()

        total += loss.item() * inp.size(0)
    return total / len(loader.dataset)



def eval_epoch(model, loader, cfg):
    model.eval()
    total_ssim, total_psnr = 0.0, 0.0
    with torch.no_grad():
        for inp, gt, _ in loader:
            inp, gt = inp.to(cfg.device), gt.to(cfg.device)
            pred = model(inp)  # sparse values already preserved in model
            total_ssim += ssim(pred, gt).item() * inp.size(0)
            mse = F.mse_loss(pred, gt, reduction="none").mean([1,2,3])
            psnr = (10 * torch.log10(1.0 / mse)).sum().item()
            total_psnr += psnr
    n = len(loader.dataset)
    return total_ssim / n, total_psnr / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--save_dir", type=Path, default=Path("./checkpoints/convnext_gt_physics_exp4"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    
    # 라플라시안 물리 손실 파라미터 (감소)
    p.add_argument("--pg_weight", type=float, default=0.1)   # 0.15에서 0.1로 감소
    p.add_argument("--pg_warmup", type=int, default=15)     # 10에서 15로 증가
    
    # GT 기반 역제곱 법칙 물리 손실 파라미터 (재조정)
    p.add_argument("--physics_weight", type=float, default=0.1)  # 0.1에서 0.05로 더 감소
    p.add_argument("--physics_warmup", type=int, default=25)     # 20에서 25로 증가
    p.add_argument("--physics_start_epoch", type=int, default=20) # 15에서 20으로 더 지연
    p.add_argument("--background_level", type=float, default=0.0)  # 0.0에서 0.001로 증가
    p.add_argument("--air_attenuation", type=float, default=0.005)  # 0.0에서 0.005로 증가
    
    # 기타 손실 파라미터
    p.add_argument("--alpha_unmask", type=float, default=0.1)
    p.add_argument("--center_lambda", type=float, default=0.2)
    
    cfg = p.parse_args()

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg),
          open(cfg.save_dir / "config.json", "w"),
          indent=2,
          default=str)    

    ds_train = RadFieldDataset(cfg.data_dir / "train.npz")
    ds_val   = RadFieldDataset(cfg.data_dir / "val.npz")

    dl_train = DataLoader(ds_train, batch_size=cfg.batch,
                          shuffle=True,  num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNeXtUNet(in_channels=6).to(cfg.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # ---------- ✦ 스케일·초기 출력 체크 ✦ ----------
    inp_dbg, gt_dbg, _ = next(iter(dl_train))        # 배치 하나
    with torch.no_grad():
        pred_dbg = model(inp_dbg.to(cfg.device)).cpu()[0, 0]
    gt_dbg = gt_dbg[0, 0]
    print("GT range :", gt_dbg.min().item(), gt_dbg.max().item())
    print("Pred range:", pred_dbg.min().item(), pred_dbg.max().item())
    print("Using GT-based physics loss for improved stability and accuracy")
    # ---------------------------------------------

    best_ssim = -1.0
    for ep in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, dl_train, opt, ep, cfg)
        ssim_val, psnr_val = eval_epoch(model, dl_val, cfg)
        scheduler.step()
        print(f"Ep{ep}: train={loss:.4f}  val_ssim={ssim_val:.4f}  psnr={psnr_val:.2f}")
        torch.save({"model": model.state_dict(), "epoch": ep}, cfg.save_dir / "ckpt_last.pth")
        if ssim_val > best_ssim:
            best_ssim = ssim_val
            torch.save({"model": model.state_dict(), "epoch": ep}, cfg.save_dir / "ckpt_best.pth")


if __name__ == "__main__":
    main()