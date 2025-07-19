# ──────────────────────────────────────────────────────────────
# dataset/trajectory_sampler.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# local import (assumes this file is under src/dataset)
sys.path.append(str(Path(__file__).resolve().parents[1]))
import dataset.generate_truth as gt

__all__ = [
    "generate_waypoints", "sparse_from_waypoints", "visualize_sparse"
]

# -------------------- sampling hyper‑parameters --------------------
GRID            = gt.GRID
WORLD_M         = 10.0
PX_PER_M        = GRID / WORLD_M
STEP_M_RANGE    = (0.4, 1.2)        # 0.4 m–1.2 m travel between samples
MIN_WP, MAX_WP  = 10, 100
TURN_LIMIT_DEG  = 120.0
GAUSSIAN_NOISE  = 0.05              # ±5 % σ noise on measurement values
POISSON_NOISE   = True
_rng_global     = np.random.default_rng(None)
# ------------------------------------------------------------------


def _random_heading(rng: np.random.Generator):
    return rng.uniform(0.0, 360.0)


def _step_in_heading(y: float, x: float, step_px: float, heading_deg: float):
    rad = np.deg2rad(heading_deg)
    return y + step_px * np.sin(rad), x + step_px * np.cos(rad)


def generate_waypoints(
    grid: int = GRID,
    *,
    min_pts: int = MIN_WP,
    max_pts: int = MAX_WP,
    rng: np.random.Generator | None = None,
):
    """Generate a *random walk* waypoint list that respects world bounds.

    The step length is sampled uniformly in *STEP_M_RANGE* (converted to pixels)
    at every iteration to mimic irregular sampling intervals.
    """
    rng = rng or _rng_global
    n_pts = int(rng.integers(min_pts, max_pts + 1))

    # initial pose
    y, x = rng.uniform(0.0, grid), rng.uniform(0.0, grid)
    heading_deg = _random_heading(rng)
    pts: list[tuple[int, int]] = [(round(y), round(x))]  # Ensure consistent tuple structure

    for _ in range(n_pts - 1):
        # random step length [m] → [px]
        step_px = rng.uniform(*STEP_M_RANGE) * PX_PER_M

        # sample a turning offset
        heading_new = (heading_deg + rng.uniform(-TURN_LIMIT_DEG, TURN_LIMIT_DEG)) % 360.0
        y_new, x_new = _step_in_heading(y, x, step_px, heading_new)

        # boundary check (with 50 retries)
        tries = 0
        while not (0.0 <= y_new < grid and 0.0 <= x_new < grid) and tries < 50:
            heading_new = (heading_deg + rng.uniform(-TURN_LIMIT_DEG, TURN_LIMIT_DEG)) % 360.0
            y_new, x_new = _step_in_heading(y, x, step_px, heading_new)
            tries += 1
        if tries == 50:
            break  # give up if no valid direction

        pts.append((round(y_new), round(x_new)))
        y, x = y_new, x_new
        heading_deg = heading_new

    return np.clip(np.asarray(pts, dtype=int), 0, grid - 1)


# ------------------------------------------------------------------
# sparse measurement generation
# ------------------------------------------------------------------

def sparse_from_waypoints(
    field: np.ndarray,
    waypoints: np.ndarray,
    *,
    gaussian_noise_sigma: float = GAUSSIAN_NOISE,
    poisson_noise: bool = POISSON_NOISE,
    rng: np.random.Generator | None = None,
):
    """Return (measurement, mask) arrays of *field* given *waypoints*.

    Adds sensor noise: multiplicative Gaussian ±σ and optional Poisson.
    """
    rng = rng or _rng_global
    r_meas = np.zeros_like(field, dtype=np.float32)
    mask   = np.zeros_like(field, dtype=np.uint8)

    for y, x in waypoints:
        val = field[y, x]
        # multiplicative Gaussian noise (5 % default)
        if gaussian_noise_sigma > 0:
            val *= rng.normal(loc=1.0, scale=gaussian_noise_sigma)
        # Poisson noise (~counting statistics)
        if poisson_noise and val > 0:
            val = rng.poisson(val)
        r_meas[y, x] = val
        mask[y, x] = 1
    return r_meas, mask


# ------------------------------------------------------------------
# visual helper (unchanged API)
# ------------------------------------------------------------------

def visualize_sparse(field: np.ndarray, waypoints: np.ndarray, r_meas: np.ndarray, mask: np.ndarray):
    """Plot ground‑truth + trajectory and corresponding sparse map."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=110)

    # (A) ground‑truth
    ax[0].imshow(field, cmap="hot", origin="lower")
    ax[0].plot(waypoints[:, 1], waypoints[:, 0], "-o", c="lime", lw=1.3, ms=3, label="Path")
    ax[0].set_title("Ground‑truth & Trajectory")
    ax[0].legend(frameon=False)
    ax[0].axis("off")

    # (B) sparse measurement
    im = ax[1].imshow(r_meas, cmap="hot", origin="lower")
    ax[1].plot(waypoints[:, 1], waypoints[:, 0], c="white", lw=0.6, alpha=0.35)
    ax[1].set_title(f"Sparse Measurements ({mask.sum()} pts)")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# sanity‑check when run directly
if __name__ == "__main__":
    n_src = _rng_global.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1)
    c, a, s = gt.sample_sources(GRID, n_src)
    field = gt.gaussian_field(GRID, c, a, s)
    wps = generate_waypoints()
    r_meas, mask = sparse_from_waypoints(field, wps)
    visualize_sparse(field, wps, r_meas, mask)
