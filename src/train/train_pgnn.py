"""
훈련 스크립트 : UNeXt-PGNN  (mask-aware MSE + multi-scale PG loss + annealing + EMA)
"""

from __future__ import annotations
import math, time, argparse, pathlib, copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ← src 경로 추가
from dataset.dataset_generator import RadiationDataset
from model.unext_pgnn import UNEXT_PGNN

# ─────────────────────────────────────
# Laplacian & multi-scale PG-loss
# ─────────────────────────────────────
_PG_KERNEL = torch.tensor([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def laplacian(x):
    """Discrete Laplacian (same padding)"""
    k = _PG_KERNEL.to(x.device)
    return F.conv2d(x, k, padding=1)

def ms_pg_loss(pred):
    """multi-scale Laplacian² 평균"""
    loss, cur = 0.0, pred
    for lvl in range(3):               # H, H/2, H/4
        loss += (laplacian(cur) ** 2).mean()
        if lvl < 2:
            cur = F.avg_pool2d(cur, 2)
    return loss


# ─────────────────────────────────────
# Mask-aware MSE
# ─────────────────────────────────────
def masked_mse(pred, target, mask, alpha=0.85):
    """mask=1: 측정점(MSE 100%), mask=0: 미측정점(MSE α)"""
    w = alpha + (1 - alpha) * mask     # mask=1→1, mask=0→α
    return ((pred - target) ** 2 * w).mean()


# ─────────────────────────────────────
# Exponential Moving Average
# ─────────────────────────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}
        self.decay  = decay

    def update(self, model):
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {k: v.clone() for k, v in model.named_parameters()}
        for k in model.state_dict().keys():
            if k in self.shadow:
                model.state_dict()[k].copy_(self.shadow[k])

    def restore(self, model):
        for k in self.backup:
            model.state_dict()[k].copy_(self.backup[k])
        self.backup = {}


# ─────────────────────────────────────
# Train / Val step
# ─────────────────────────────────────
def run_epoch(model, loader, optim, device,
              λ_pg_max=1e-2, warm_steps=5_000,
              ema: EMA | None = None, train=True, global_step=0):
    model.train(train)
    tot_loss, tot_cnt = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = x[:, 1:2]                  # (N,1,H,W)

        pred, aux2, aux3 = model(x)
        # Deep supervision targets are same-scale resized
        y_ds2 = F.avg_pool2d(y, 2)
        y_ds3 = F.avg_pool2d(y, 4)
        mask2 = F.avg_pool2d(mask, 2)
        mask3 = F.avg_pool2d(mask, 4)

        mse_main = masked_mse(pred, y, mask)
        mse_aux2 = masked_mse(aux2, y_ds2, mask2)
        mse_aux3 = masked_mse(aux3, y_ds3, mask3)
        mse = mse_main + 0.4 * (mse_aux2 + mse_aux3)

        # λ annealing : 0 → λ_pg_max over warm_steps
        λ = λ_pg_max * min(1.0, global_step / warm_steps)
        pg = ms_pg_loss(pred)
        loss = mse + λ * pg

        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if ema:  # update shadow
                ema.update(model)

        tot_loss += loss.item() * x.size(0)
        tot_cnt  += x.size(0)
        global_step += 1

    return tot_loss / tot_cnt, global_step


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNEXT_PGNN().to(device)
    optim  = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema    = EMA(model, decay=0.999)

    train_loader = DataLoader(RadiationDataset("train"),
                              batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader   = DataLoader(RadiationDataset("val"),
                              batch_size=args.bs, shuffle=False, num_workers=2)

    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, global_step = run_epoch(model, train_loader, optim, device,
                                            ema=ema, train=True, global_step=global_step)
        # EMA eval
        ema.apply_shadow(model)
        val_loss, _ = run_epoch(model, val_loader, None, device,
                                train=False, global_step=global_step)
        ema.restore(model)
        dt = time.time() - t0
        print(f"[{epoch:02d}] train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")

        # checkpoint
        ckpt_dir = pathlib.Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"epoch{epoch:02d}.pth")

    print("done.")


# ─────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs",     type=int, default=8)
    main(p.parse_args())