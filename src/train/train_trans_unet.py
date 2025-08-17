import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import argparse, json, math, os, random, pathlib, sys
from pathlib import Path
import json, argparse
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))            # …/src
from model.unet_deep_pgnn import PGNN_UNet, laplacian_loss
from model.trans_unet_pgnn import TransUNet
from dataset.dataset_generator import RadiationDataset

class RadFieldDataset(Dataset):
    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path)
        if "inp" in self.data:                       # (N,5,H,W)  형태
            self.inp = self.data["inp"].astype(np.float32)
        else:                                        # 개별 채널 → stack
            chans = [self.data[k].astype(np.float32)
                      for k in ["M", "mask", "logM", "X", "Y"]]
            self.inp = np.stack(chans, axis=1)       # (N,5,H,W)
        self.gt  = self.data["gt"].astype(np.float32)  # (N,1,H,W)
        self.names = [f"{npz_path.stem}_{i}" for i in range(len(self.inp))]

    def __len__(self):  return len(self.inp)

    def __getitem__(self, idx):
        return ( torch.from_numpy(self.inp[idx]),
                 torch.from_numpy(self.gt[idx]),
                 self.names[idx] )


def train_one_epoch(model, loader, opt, epoch, cfg):
    model.train()
    total = 0.0
    for inp, gt, _ in tqdm(loader, desc=f"Train {epoch}"):  # 세 번째 값(names)을 무시
        inp, gt = inp.to(cfg.device), gt.to(cfg.device)
        mask = inp[:, 1:2]          # measured 위치
        dist = inp[:, 5:6]          # distance_map (정규화된 거리)

        pred_raw = model(inp)

        # (1) 출력 마스킹: measured 위치는 gt로 유지
        pred = pred_raw * (1 - mask) + gt * mask

        # (2) 거리 기반 soft attention weight
        weight = torch.exp(-3.0 * dist)

        # (3) 손실 함수 구성
        loss_unmeasured = F.mse_loss(pred * (1 - mask) * weight, gt * (1 - mask) * weight)
        loss_measured   = F.mse_loss(pred_raw * mask, gt * mask)
        lap = laplacian_loss(pred)
        
        lambda_pg = min(1.0, epoch / cfg.pg_warmup) * cfg.pg_weight
        loss = loss_unmeasured + 0.1 * loss_measured + cfg.alpha_unmask * F.mse_loss(pred, gt) + lambda_pg * lap

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item() * inp.size(0)
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--save_dir", type=Path, default=Path("./checkpoints/transunet_exp1"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pg_weight", type=float, default=0.1)
    p.add_argument("--pg_warmup", type=int, default=10)
    p.add_argument("--alpha_unmask", type=float, default=0.05)
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

    model = TransUNet(in_channels=6).to(cfg.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_loss = float('inf')
    for ep in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, dl_train, opt, ep, cfg)
        scheduler.step()
        print(f"Ep{ep}: train_loss={loss:.6f}")
        torch.save({"model": model.state_dict(), "epoch": ep}, cfg.save_dir / "ckpt_last.pth")
        if loss < best_loss:
            best_loss = loss
            torch.save({"model": model.state_dict(), "epoch": ep}, cfg.save_dir / "ckpt_best.pth")


if __name__ == "__main__":
    main()