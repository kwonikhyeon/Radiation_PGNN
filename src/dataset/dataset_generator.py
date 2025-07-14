# src/dataset/dataset_generator.py
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ← src 경로 추가

import dataset.generate_truth as gt
import dataset.trajectory_sampler as ts


# ──────────────────────────────────────────────────────────
# 0. 출력 루트 디렉터리 정의  (…/data/synthetic)
# ──────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parents[2]     # 프로젝트 루트
DATA_DIR  = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)         # 없으면 생성


# ──────────────────────────────────────────────────────────
# 1. 단일 샘플 생성
# ──────────────────────────────────────────────────────────
# def make_sample(rng: np.random.Generator):
#     rng = rng or np.random.default_rng()

#     n_src = int(rng.integers(gt.N_SOURCES_RANGE[0],
#                              gt.N_SOURCES_RANGE[1] + 1))
#     coords, amps, sigmas = gt.sample_sources(gt.GRID, n_src)
#     field = gt.gaussian_field(gt.GRID, coords, amps, sigmas).astype(np.float32)

#     waypoints      = ts.generate_waypoints()
#     r_meas, mask   = ts.sparse_from_waypoints(field, waypoints)
#     return field, r_meas.astype(np.float32), mask.astype(np.uint8)

def make_sample(rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()

    # 1) Ground-truth 필드 생성 (256×256)
    n_src = int(rng.integers(gt.N_SOURCES_RANGE[0],
                             gt.N_SOURCES_RANGE[1] + 1))
    coords, amps, sigmas = gt.sample_sources(gt.GRID, n_src)
    field = gt.gaussian_field(gt.GRID, coords, amps, sigmas).astype(np.float32)

    # 2) 0-to-1 정규화 (★ 추가)
    f_max = field.max() + 1e-6          # 0 방지용 ε
    field /= f_max

    # 3) sparse 측정치 추출 → r_meas / mask
    waypoints = ts.generate_waypoints()
    r_meas, mask = ts.sparse_from_waypoints(field, waypoints)
    r_meas = r_meas.astype(np.float32)     # 같은 스케일
    mask   = mask.astype(np.uint8)

    return field, r_meas, mask                      # 모두 (H,W)


# ──────────────────────────────────────────────────────────
# 2. 데이터셋 생성 & 저장
# ──────────────────────────────────────────────────────────
def generate_dataset(num_samples: int,
                     split: str):
    """
    Parameters
    ----------
    num_samples : 저장할 샘플 수
    split       : 'train' | 'val' | 'test' …  (파일 이름에 사용)
    seed        : 난수 시드를 지정하면 재현 가능
    """
    rng = np.random.default_rng(None)
    truth_list, meas_list, mask_list = [], [], []

    for _ in range(num_samples):
        field, r_meas, mask = make_sample(rng)
        truth_list.append(field)
        meas_list.append(r_meas)
        mask_list.append(mask)

    out_path = DATA_DIR / f"{split}.npz"
    np.savez_compressed(
        out_path,
        truth=np.stack(truth_list),       # (N,H,W)
        r_meas=np.stack(meas_list),
        mask=np.stack(mask_list)
    )
    print(f"[✓] {split} ({num_samples}) → {out_path.relative_to(ROOT_DIR)}")


# ──────────────────────────────────────────────────────────
# 3. PyTorch Dataset (선택 사항)
# ──────────────────────────────────────────────────────────
class RadiationDataset(Dataset):
    def __init__(self, split: str):
        cache = np.load(DATA_DIR / f"{split}.npz")
        self.truth, self.r_meas, self.mask = \
            cache["truth"], cache["r_meas"], cache["mask"]

    def __len__(self): return len(self.truth)

    def __getitem__(self, idx):
        x = np.stack([self.r_meas[idx], self.mask[idx]], 0)   # (2,H,W)
        y = self.truth[idx]                                   # (H,W)
        return torch.from_numpy(x), torch.from_numpy(y)


# ──────────────────────────────────────────────────────────
# 4. 직접 실행 예시
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset(num_samples= 9000, split="train")
    generate_dataset(num_samples= 1000, split="val")
    generate_dataset(num_samples= 200, split="test")

    # DataLoader 사용법
    # train_loader = DataLoader(RadiationDataset("train"),
    #                           batch_size=8, shuffle=True)