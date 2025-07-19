from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- local imports ----------------
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add src/ to PYTHONPATH
import dataset.generate_truth as gt
import dataset.trajectory_sampler as ts
# ------------------------------------------------

__all__ = ["generate_dataset", "RadiationDataset"]

# ---------------------- paths -------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------

# --------------- constants & helpers ------------
_rng_global = np.random.default_rng(None)


def _precompute_coord(grid: int):
    """Return (2,H,W) array with normalized X,Y in [-1,1]."""
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    ys = (ys / (grid - 1) * 2 - 1).astype(np.float32)
    xs = (xs / (grid - 1) * 2 - 1).astype(np.float32)
    return np.stack([ys, xs], axis=0)  # (2,H,W)

_COORD = _precompute_coord(gt.GRID)  # global cache


# ---------------- single‑sample synthesis ----------------

def _make_single_sample(rng: np.random.Generator):
    """Return (gt_field, meas, mask) all (H,W)."""
    n_src = int(rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1))
    c, a, s = gt.sample_sources(gt.GRID, n_src, rng=rng)
    field = gt.gaussian_field(gt.GRID, c, a, s)

    # ★ 음수 값 방지: field 값을 0으로 클리핑
    field = np.clip(field, 0, None)
    field /= 100.0  # ★ 전역 최대치(여유로 120)로 나눔

    waypoints = ts.generate_waypoints(rng=rng)
    meas, mask = ts.sparse_from_waypoints(field, waypoints, rng=rng)

    # ★ 음수 값 방지: meas 값을 0으로 클리핑
    meas = np.clip(meas, 0, None)

    # 로그 계산 시 음수 방지
    logM = np.log1p(meas)

    # 입력 데이터 생성
    inp = np.stack([meas, mask, logM, *_COORD], axis=0)  # (5,H,W)
    return field, inp, mask.astype(np.float32)


# ---------------- dataset generator API --------

def generate_dataset(num_samples: int, split: str):
    """Generate *num_samples* samples and save to data/{split}.npz."""
    inp_list, gt_list = [], []

    for _ in range(num_samples):
        gt_field, inp, mask = _make_single_sample(_rng_global)
        inp_list.append(inp)
        gt_list.append(gt_field[None])  # (1,H,W)

    out_path = DATA_DIR / f"{split}.npz"
    np.savez_compressed(out_path,
                        inp=np.stack(inp_list, axis=0),  # (N,5,H,W)
                        gt=np.stack(gt_list, axis=0))   # (N,1,H,W)
    print(f"[✓] Saved {split}: {num_samples} samples → {out_path.relative_to(ROOT_DIR)}")


# --------------------- torch Dataset ------------
class RadiationDataset(Dataset):
    """Loads pre‑generated {split}.npz and yields (inp, gt)."""

    def __init__(self, split: str):
        p = DATA_DIR / f"{split}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run generate_dataset() first.")
        cache = np.load(p)
        self.inp = cache["inp"].astype(np.float32)
        self.gt  = cache["gt" ].astype(np.float32)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return ( torch.from_numpy(self.inp[idx]),
                 torch.from_numpy(self.gt[idx]) )


# --------------------- CLI test ------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1800)
    ap.add_argument("--n_val",   type=int, default=200)
    ap.add_argument("--n_test",   type=int, default=200)
    args = ap.parse_args()

    generate_dataset(args.n_train, "train")
    generate_dataset(args.n_val,   "val")
    generate_dataset(args.n_test,   "test")

    # quick sanity check
    ds = RadiationDataset("train")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print("Batch shapes:", x.shape, y.shape)
