# ───────── src/eval/eval_pgnn.py ─────────────────────────────
"""
U-NeXt PGNN 체크포인트 평가 스크립트
----------------------------------
예시:
    # 최신 체크포인트 자동 + PNG 저장
    python -m src.eval.eval_pgnn --save-img

    # 특정 ckpt + base 24 + deep_sup off
    python -m src.eval.eval_pgnn --ckpt checkpoints/best.pth --base 24 --no-deep-sup
"""
from __future__ import annotations
import argparse, sys, pathlib, glob, numpy as np, torch, torchvision.utils as vutils
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ── 경로 설정 ────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]  # ~/workspace
SRC  = ROOT / "src"
sys.path.append(str(SRC))

from model.unext_pgnn import UNEXT_PGNN
from dataset.dataset_generator import RadiationDataset

# ── 평가지표 ────────────────────────────────────────────────
def psnr(pred, target):
    return -10 * np.log10(((pred-target) ** 2).mean() + 1e-12)

def evaluate(model, loader, device):
    model.eval()
    ps, ss, rm = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            x, y = x.to(device), y.to(device)
            pred = model(x)[0].squeeze(1)
            for p, g in zip(pred.cpu().numpy(), y.cpu().numpy()):
                ps.append(psnr(p, g))
                ss.append(ssim(p, g, data_range=np.ptp(g)))   # ← 수정
                rm.append(np.sqrt(((p - g) ** 2).mean()))
    return dict(PSNR=np.mean(ps), SSIM=np.mean(ss), RMSE=np.mean(rm))

# ── 메인 ────────────────────────────────────────────────────
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) ckpt 자동 선택
    ckpt = args.ckpt
    if ckpt is None:
        lst = sorted(glob.glob(str(ROOT / "checkpoints" / "epoch*.pth")))
        if not lst:
            raise FileNotFoundError("no checkpoints found")
        ckpt = lst[-1]
        print(f"[INFO] ckpt not specified → using {ckpt}")

    # 2) 데이터
    ds = RadiationDataset(args.split)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)

    # 3) 모델 (훈련 시 설정과 동일하게!)
    net = UNEXT_PGNN(base=args.base, deep_supervision=args.deep_sup).to(device)

    # 4) state_dict 로드 (strict=False → 이름 달라도 무시)
    sd = torch.load(ckpt, map_location=device)
    miss, unexp = net.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[WARN] missing {len(miss)}  |  unexpected {len(unexp)} keys")

    # 5) 평가
    metrics = evaluate(net, dl, device)
    print(f"--- {args.split.upper()} ---")
    for k, v in metrics.items():
        print(f"{k:5s}: {v:.4f}")

    # 6) 예시 저장 / 시각화
    if args.save_img:
        out = ROOT / args.out_dir; out.mkdir(exist_ok=True)
        idx   = np.random.choice(len(ds), 8, replace=False)
        tiles = []                                      # 각 tile : (1,H,W)

        net.eval()
        with torch.no_grad():
            for i in idx:
                x, y = ds[i]                            # y: (H,W)
                y    = y.unsqueeze(0)                   # (1,H,W)
                meas = x[0].unsqueeze(0)                # (1,H,W)

                # ── 예측 (1,1,H,W) → [0] 로 batch 축 제거
                pred = net(x.unsqueeze(0).to(device))[0][0].cpu()  # (1,H,W)

                diff = torch.abs(y - pred)              # (1,H,W)
                tiles.extend([y, pred, diff, meas])     # GT,Pred,Err,Mask

        grid = torch.stack(tiles, 0)                    # (32,1,H,W)
        vutils.save_image(grid, out/"sample_grid.png",
                        nrow=4, normalize=True)

        if args.visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(4,4, figsize=(10,10))
            lbl = ["sparse","predict","ground-truth","|error|"]
            for r in range(4):
                gt, pred, diff, meas = tiles[r*4:r*4+4]
                for c,img in enumerate([meas,pred,gt,diff]):
                    ax[r,c].imshow(img.squeeze(), cmap="hot"); ax[r,c].axis("off")
                    if r==0: ax[r,c].set_title(lbl[c])
            plt.tight_layout(); plt.show()

# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint path (.pth)")
    ap.add_argument("--split", default="test", help="train/val/test")
    ap.add_argument("--base",  type=int, default=32, help="UNeXt base channels")
    ap.add_argument("--deep-sup",  dest="deep_sup", action="store_true")
    ap.add_argument("--no-deep-sup", dest="deep_sup", action="store_false")
    ap.set_defaults(deep_sup=True)
    ap.add_argument("--save-img",  default=True, help="PNG grid 저장")
    ap.add_argument("--visualize", default=True, help="matplotlib 시각화")
    ap.add_argument("--out-dir", default="eval")
    main(ap.parse_args())
