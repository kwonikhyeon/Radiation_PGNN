"""Train script for PGNN‑UNet.
Run:
    $ python -m train.train \
        --data_dir dataset/npz \
        --save_dir runs/exp1
"""
from __future__ import annotations
import argparse, json, math, os, random, pathlib, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))            # …/src
from model.unet_deep_pgnn import PGNN_UNet, laplacian_loss

# -------------------------------------------------------------
# Dataset wrapper for .npz files
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Metric helpers (SSIM simplified)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------

def train_one_epoch(model, loader, opt, epoch, cfg):
    model.train()
    total = 0.0
    for inp, gt, _ in tqdm(loader, desc=f"Train {epoch}"):
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        mask = inp[:, 1:2]          # measured 위치
        dist = inp[:, 5:6]          # distance_map (정규화된 거리)

        # 모델 예측
        pred_raw = model(inp)

        # (1) 출력 마스킹: 측정된 위치는 gt 그대로 사용
        pred = pred_raw * (1 - mask) + gt * mask

        # (2) 거리 기반 soft attention weight (가까울수록 weight 높음)
        weight = torch.exp(-3.0 * dist)

        # (3) 손실 함수 (측정 위치는 약하게, 미측정은 정상 weight)
        loss_unmeasured = F.mse_loss(pred * (1 - mask) * weight, gt * (1 - mask) * weight)
        loss_measured   = F.mse_loss(pred_raw * mask, gt * mask)

        # (4) PGNN smooth loss
        lap = laplacian_loss(pred)
        λ_pg = min(1.0, epoch / cfg.pg_warmup) * cfg.pg_weight

        # (5) 총 손실
        loss = loss_unmeasured + 0.1 * loss_measured + cfg.alpha_unmask * F.mse_loss(pred, gt) + λ_pg * lap

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item() * inp.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, cfg):
    model.eval()
    total_ssim, total_psnr = 0.0, 0.0
    with torch.no_grad():
        for inp, gt, _ in loader:
            inp, gt = inp.to(cfg.device), gt.to(cfg.device)
            pred = model(inp)
            total_ssim += ssim(pred, gt).item() * inp.size(0)
            mse = F.mse_loss(pred, gt, reduction="none").mean([1,2,3])
            psnr = (10 * torch.log10(1.0 / mse)).sum().item()
            total_psnr += psnr
    n = len(loader.dataset)
    return total_ssim / n, total_psnr / n

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--save_dir", type=Path, default=Path("./checkpoints/unet_exp6"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pg_weight", type=float, default=0.1)
    p.add_argument("--pg_warmup", type=int, default=10)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--alpha_unmask", type=float, default=0.05,
                    help="weight for unmask MSE term")

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

    model = PGNN_UNet().to(cfg.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # ---------- ✦ 스케일·초기 출력 체크 ✦ ----------
    inp_dbg, gt_dbg, _ = next(iter(dl_train))        # 배치 하나
    with torch.no_grad():
        pred_dbg = model(inp_dbg.to(cfg.device)).cpu()[0, 0]
    gt_dbg = gt_dbg[0, 0]
    print("GT range :", gt_dbg.min().item(), gt_dbg.max().item())
    print("Pred range:", pred_dbg.min().item(), pred_dbg.max().item())
    # -----------------------------------------------

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