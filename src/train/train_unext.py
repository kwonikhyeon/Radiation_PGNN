"""
훈련 스크립트 : UNeXt-PGNN
  - mask-aware MSE
  - multi-scale PG loss + annealing
  - EMA
  - AMP(O1) + Grad-Accumulation
  - base 채널/딥수퍼비전 토글
"""



from __future__ import annotations
import math, time, argparse, pathlib, os, sys
# env 옵션 (단편화 완화 권장)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True,max_split_size_mb:128")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# ───────── 경로 설정 ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))            # …/src
from dataset.dataset_generator import RadiationDataset
from model.unext_pgnn import UNEXT_PGNN

# ───────── Laplacian ─────────────────────────────────────────
_PG_KERNEL = torch.tensor([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]], dtype=torch.float32) \
            .unsqueeze(0).unsqueeze(0)

def laplacian(x):
    return F.conv2d(x, _PG_KERNEL.to(x.device), padding=1)

def ms_pg_loss(pred, clamp_val=1e3):
    """ PG-loss (Laplacian²)  −  항상 FP32 로 계산해 overflow 방지 """
    loss, cur = 0., pred.float()          # ← FP32
    for _ in range(3):
        lap = laplacian(cur)
        lap = torch.clamp(lap, -clamp_val, clamp_val)  # hard clip
        loss += (lap ** 2).mean()
        cur = F.avg_pool2d(cur, 2)
    return loss

# ───────── Masked MSE ───────────────────────────────────────
def masked_mse(pred, target, mask, alpha=0.1):
    w = alpha + (1 - alpha) * mask
    return ((pred - target) ** 2 * w).mean()

# ───────── EMA ──────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}
        self.decay  = decay
    def update(self, model):
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def apply_shadow(self, model):
        self.back  = {k: v.clone() for k, v in model.named_parameters()}
        for k in self.shadow:
            model.state_dict()[k].copy_(self.shadow[k])
    def restore(self, model):
        for k in self.back:
            model.state_dict()[k].copy_(self.back[k])

# ───────── Train/Val step ───────────────────────────────────
def run_epoch(model, loader, optim, scaler, device,
              accum=1, λ_pg_max=1e-2, warm_steps=5_000,
              ema: EMA | None = None, train=True, global_step=0,
              deep_supervision=True):
    torch.autograd.set_detect_anomaly(True)   # NaN/Inf 위치 트레이스
    model.train(train)
    tot_loss = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        mask = x[:, 1:2]

        with autocast():                       # AMP 영역
            pred, aux2, aux3 = model(x)
            y_ds2, y_ds3 = F.avg_pool2d(y, 2), F.avg_pool2d(y, 4)
            mask2, mask3 = F.avg_pool2d(mask, 2), F.avg_pool2d(mask, 4)

            mse_main = masked_mse(pred,  y,     mask)
            if deep_supervision:
                mse_aux2 = masked_mse(aux2, y_ds2, mask2)
                mse_aux3 = masked_mse(aux3, y_ds3, mask3)
                mse = mse_main + 0.4 * (mse_aux2 + mse_aux3)
            else:
                mse = mse_main

            λ = λ_pg_max * min(1.0, global_step / (warm_steps * 4))
            loss = mse + λ * ms_pg_loss(pred)
            loss = loss / accum                 # grad-accum 분할

        if train:
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                # grad-clip (FP16 안전)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

        tot_loss   += loss.item() * accum
        global_step += 1

    return tot_loss / len(loader), global_step


# ───────── Main ─────────────────────────────────────────────
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 학습 스크립트 예시
    model = UNEXT_PGNN(base=args.base, deep_supervision=args.deep_sup, use_coord=False)
    model = model.to(device)

    optim  = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=True)
    scaler = GradScaler()
    ema    = EMA(model, decay=0.999)

    train_loader = DataLoader(RadiationDataset("train"),
                              batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(RadiationDataset("val"),
                              batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    global_step = 0
    ckpt_dir = pathlib.Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    best_val = float("inf"); patience = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, global_step = run_epoch(model, train_loader, optim, scaler, device,
                                         accum=args.accum, ema=ema, train=True,
                                         global_step=global_step,
                                         deep_supervision=args.deep_sup)

        # EMA eval
        ema.apply_shadow(model)
        val_loss, _ = run_epoch(model, val_loader, None, scaler, device,
                                accum=1, train=False,
                                global_step=global_step,
                                deep_supervision=args.deep_sup)
        ema.restore(model)

        dt = time.time() - t0
        print(f"[{epoch:03d}] train {tr_loss:.4f}  |  val {val_loss:.4f}  |  {dt:.1f}s")

        # Early-Stopping
        if val_loss < best_val - 1e-4:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
        else:
            patience += 1
            if patience >= args.es_patience:
                print("Early stopping triggered."); break

        # epoch-별 체크포인트
        if epoch % args.ckpt_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch{epoch:03d}.pth")

        torch.cuda.empty_cache()   # 파편화 정리

    print("done.")


# ───────── CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--bs",          type=int, default=8,   help="physical batch size")
    p.add_argument("--accum",       type=int, default=2,    help="grad-accum steps")
    p.add_argument("--base",        type=int, default=32,   help="UNeXt base channels (32/24/16)")
    p.add_argument("--deep-sup",    dest="deep_sup", action="store_true")
    p.add_argument("--no-deep-sup", dest="deep_sup", action="store_false")
    p.set_defaults(deep_sup=True)
    p.add_argument("--ckpt-every",  type=int, default=10)
    p.add_argument("--es-patience", type=int, default=10,   help="early-stopping patience")
    main(p.parse_args())
