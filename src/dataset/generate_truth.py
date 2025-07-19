# ──────────────────────────────────────────────────────────────
# dataset/generate_truth.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np

__all__ = [
    "GRID", "N_SOURCES_RANGE", "INTENSITY_RANGE", "SIGMA_RANGE", "SEED",
    "sample_sources", "gaussian_field", "get_param"
]

# ------------------- configurable hyper‑parameters -------------------
GRID             = 256                  # 256×256 map (≈10 m × 10 m @0.04 m/pixel)
N_SOURCES_RANGE  = (1, 4)               # 1–4 sources
INTENSITY_RANGE  = (30.0, 100.0)        # peak intensity (MeV scale)
SIGMA_RANGE      = (10.0, 20.0)          # σ in pixels (spatial spread)
SEED             = None                 # global seed (None ⇒ different every run)
# ---------------------------------------------------------------------

_rng_global = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# util functions
# ---------------------------------------------------------------------
def _min_dist_ok(new: tuple[int, int], coords: list[tuple[int, int]], min_px: float) -> bool:
    """Returns True if *new* is farther than *min_px* from every coord in *coords*."""
    for y, x in coords:
        if np.hypot(new[0] - y, new[1] - x) < min_px:
            return False
    return True


def sample_sources(grid: int, n: int, *, rng: np.random.Generator | None = None):
    """Sample **n** source parameters (position, amplitude, sigma).

    Ensures that sources are not too close to each other or to the image border.
    """
    rng = rng or _rng_global
    coords: list[tuple[int, int]] = []
    amps:   list[float] = []
    sigmas: list[float] = []

    border = 0.1 * grid          # keep sources away from border
    min_dist = 0.25 * grid       # min distance between two sources (in pixels)

    while len(coords) < n:
        y = rng.integers(border, grid - border)
        x = rng.integers(border, grid - border)
        if _min_dist_ok((y, x), coords, min_dist):
            coords.append((y, x))
            amps.append(rng.uniform(*INTENSITY_RANGE))
            sigmas.append(rng.uniform(*SIGMA_RANGE))
    return np.array(coords), np.array(amps), np.array(sigmas)


def gaussian_field(grid: int, coords: np.ndarray, amps: np.ndarray, sigmas: np.ndarray):
    """Return (H, W) field composed of isotropic 2‑D Gaussians."""
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    field = np.zeros((grid, grid), dtype=np.float32)
    for (y0, x0), A, s in zip(coords, amps, sigmas, strict=True):
        d2 = (yy - y0) ** 2 + (xx - x0) ** 2
        field += A * np.exp(-d2 / (2.0 * s ** 2))
    return field


def get_param():
    """Return current generator configuration as dict."""
    return {
        "GRID": GRID,
        "N_SOURCES_RANGE": N_SOURCES_RANGE,
        "INTENSITY_RANGE": INTENSITY_RANGE,
        "SIGMA_RANGE": SIGMA_RANGE,
        "SEED": SEED,
    }

# ──────────────────────────────────────────────────────────────
# Quick visual check when executed directly
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_src = _rng_global.integers(N_SOURCES_RANGE[0], N_SOURCES_RANGE[1] + 1)
    c, a, s = sample_sources(GRID, n_src)
    f = gaussian_field(GRID, c, a, s)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(f, cmap="hot", origin="lower")
    plt.scatter(c[:, 1], c[:, 0], c="lime", s=60, label="sources")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Ground‑truth radiation field")
    plt.show()