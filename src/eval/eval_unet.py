# ──────────────────────────────────────────────────────────────
# src/eval/eval.py
#   예)  $ python -m eval.eval \
#           --ckpt checkpoints/unet_exp1/ckpt_best.pth \
#           --data_file data/val.npz \
#           --out_dir  preds_vis
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, pathlib, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt            # ★ 추가

from kornia.metrics import ssim as kornia_ssim
from scipy.ndimage import binary_dilation  # 추가

# ------------------------- model ------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # …/src
sys.path.append(str(ROOT))
from model.unet_deep_pgnn import PGNN_UNet
# --------------------------------------------------------------

# --------------------- Dataset wrapper -----------------------
class RadNPZDataset(Dataset):
    """Load (inp, gt) pairs from a single .npz file."""
    def __init__(self, npz_path: Path):
        arr = np.load(npz_path)
        self.inp = arr["inp"].astype(np.float32)   # (N,5,H,W)
        self.gt  = arr["gt" ].astype(np.float32)   # (N,1,H,W)
        self.names = [f"{npz_path.stem}_{i}" for i in range(len(self.inp))]

    def __len__(self):  return len(self.inp)
    def __getitem__(self, idx):
        return ( torch.from_numpy(self.inp[idx]),
                 torch.from_numpy(self.gt[idx]),
                 self.names[idx] )
# --------------------------------------------------------------

def ssim(pred, gt):
    try:                           # kornia ≥0.6
        return kornia_ssim(pred, gt, window_size=11, reduction="none").mean()
    except TypeError:              # kornia <0.6
        return kornia_ssim(pred, gt, window_size=11).mean()

# --------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      type=Path, default="./checkpoints/unet_exp6/ckpt_best.pth", help=".pth checkpoint")
    ap.add_argument("--data_file", type=Path, default="./data/test.npz", help="dataset .npz")
    ap.add_argument("--out_dir",   type=Path, default="./eval_vis/exp6", help="output directory")
    ap.add_argument("--batch",     type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- model ----------
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    model = PGNN_UNet()
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    # ---------- data ----------
    ds = RadNPZDataset(args.data_file)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=2, pin_memory=True)

    # ---------- inference ----------
    tot_ssim, tot_psnr = 0.0, 0.0
    with torch.no_grad():
        for inp, gt, names in tqdm(dl, desc="Eval"):
            inp, gt = inp.to(device), gt.to(device)
            pred = model(inp)

            # ─── (A) 3-패널 시각화 저장 ──────────────────────
            for m, p, g, name in zip(inp.cpu(), pred.cpu(), gt.cpu(), names):
                meas  = m[0].numpy()         # M 채널
                mask  = m[1].numpy()
                sparse = meas * mask         # 실제 계측 위치만 값

                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
                for ax, img, ttl in zip(
                        axes,
                        [sparse, p[0].numpy(), g[0].numpy()],
                        ["Input: Sparse", "Prediction", "Ground-truth"]):
                    # vmin=0, vmax=1로 모든 데이터 동일한 기준으로 시각화
                    im = ax.imshow(img, cmap="hot", origin="lower", vmin=0, vmax=1)
                    ax.set_title(ttl, fontsize=10)
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # 계측 위치에 점 찍기 (모든 패널에 오버레이)
                    y_coords, x_coords = np.where(mask > 0)
                    ax.scatter(x_coords, y_coords, c="blue", s=10, label="Measured Points")
                    if ttl == "Input: Sparse":
                        ax.legend(loc="upper right", fontsize=8)

                plt.tight_layout()
                fig.savefig(args.out_dir / f"{name}_vis.png")
                plt.close(fig)
            # ------------------------------------------------

            # (선택) 원시 예측 npy 저장이 필요하면 아래 주석 해제
            # for p, name in zip(pred.cpu().numpy(), names):
            #     np.save(args.out_dir / f"{name}_pred.npy", p[0])

            # ---------- 메트릭 ----------
            tot_ssim += ssim(pred, gt).item() * inp.size(0)
            mse = F.mse_loss(pred, gt, reduction="none").mean([1,2,3])
            tot_psnr += (10 * torch.log10(1.0 / mse)).sum().item()

    n = len(ds)
    print(f"SSIM = {tot_ssim/n:.4f}   PSNR = {tot_psnr/n:.2f} dB")

# --------------------------- run ------------------------------
if __name__ == "__main__":
    main()
