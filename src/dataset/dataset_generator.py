from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

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
    """Return (gt_field, inp, mask) all (H,W).
    
    FIXED: Ensures data consistency between measured values and ground truth.
    """
    n_src = int(rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1))
    c, a, s = gt.sample_sources(gt.GRID, n_src, rng=rng)
    field = gt.gaussian_field(gt.GRID, c, a, s)

    # FIXED: Improved normalization strategy
    # Ensure non-negative values
    field = np.clip(field, 0, None)
    
    # FIXED: More conservative normalization to prevent extreme scaling
    # Use the 99th percentile instead of max to handle outliers
    field_99th = np.percentile(field[field > 0], 99) if np.any(field > 0) else 1.0
    normalization_factor = max(field_99th, 1.0)  # Ensure we don't divide by tiny numbers
    field = field / normalization_factor
    
    # Ensure the field stays in reasonable range [0, 1]
    field = np.clip(field, 0, 1.0)

    # Generate waypoints and measurements
    waypoints = ts.generate_waypoints(rng=rng)
    meas, mask = ts.sparse_from_waypoints(field, waypoints, rng=rng)

    # FIXED: No additional clipping needed since sparse_from_waypoints now handles this correctly
    # The measured values should now match the field values at measured positions
    
    # Verify data consistency (optional check)
    measured_positions = np.where(mask > 0)
    if len(measured_positions[0]) > 0:
        # Sample a few points to verify consistency
        for i in range(min(3, len(measured_positions[0]))):
            y, x = measured_positions[0][i], measured_positions[1][i]
            field_val = field[y, x]
            meas_val = meas[y, x]
            if abs(field_val - meas_val) > 1e-6:
                print(f"Warning: Data inconsistency at ({y},{x}): field={field_val:.6f}, meas={meas_val:.6f}")

    # Log calculation with proper handling
    logM = np.log1p(meas)

    # Distance map: measured positions = 0, others = normalized distance
    raw_dist = distance_transform_edt(1 - mask).astype(np.float32)
    raw_dist[mask == 1] = 0.0
    distance_map = raw_dist / (gt.GRID + 1e-6)  # avoid div by 0

    # Stack input channels: [meas, mask, logM, coordY, coordX, distance]
    inp = np.stack([meas, mask, logM, *_COORD, distance_map], axis=0)  # (6,H,W)
    
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
    ap.add_argument("--n_train", type=int, default=9000)
    ap.add_argument("--n_val",   type=int, default=1000)
    ap.add_argument("--n_test",   type=int, default=100)
    args = ap.parse_args()

    generate_dataset(args.n_train, "train")
    generate_dataset(args.n_val,   "val")
    generate_dataset(args.n_test,   "test")

    # quick sanity check
    ds = RadiationDataset("train")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print("Batch shapes:", x.shape, y.shape)
