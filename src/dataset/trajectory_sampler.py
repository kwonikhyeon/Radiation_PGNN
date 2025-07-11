import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ← src 경로 추가
import dataset.generate_truth as gt

# generate_truth.get_param()을 통해 파라미터 가져오기
params = gt.get_param()
GRID = params['GRID']
N_SOURCES_RANGE = params['N_SOURCES_RANGE']
INTENSITY_RANGE = params['INTENSITY_RANGE']
SIGMA_RANGE = params['SIGMA_RANGE']
SEED = params['SEED']
N_SOURCES = params['N_SOURCES']

# ---------------- 하이퍼파라미터 ----------------
GRID      = 256
WORLD_M   = 10.0
PX_PER_M  = GRID / WORLD_M       # ≈ 25.6 px
STEP_M    = 1.0
STEP_PX   = STEP_M * PX_PER_M
MIN_WP, MAX_WP = 10, 100
TURN_LIMIT_DEG = 120             # ±120°
rng = np.random.default_rng(None)
# -----------------------------------------------

def generate_waypoints(grid=GRID, step_px=STEP_PX,
                       min_pts=MIN_WP, max_pts=MAX_WP,
                       turn_limit_deg=TURN_LIMIT_DEG,
                       rng=rng):
    n_pts = int(rng.integers(min_pts, max_pts + 1))

    # 시작 좌표
    y, x = rng.uniform(0, grid), rng.uniform(0, grid)
    pts  = [(y, x)]

    # 초기 방향 (0~360° 무작위)
    heading_deg = rng.uniform(0, 360)

    for _ in range(n_pts - 1):
        # 새 방향 = 현재 heading ± random_offset(−120°~+120°)
        offset = rng.uniform(-turn_limit_deg, turn_limit_deg)
        heading_deg_new = (heading_deg + offset) % 360
        heading_rad = np.deg2rad(heading_deg_new)

        y_new = y + step_px * np.sin(heading_rad)
        x_new = x + step_px * np.cos(heading_rad)

        # 경계 확인, 벗어나면 각도 다시 샘플
        tries = 0
        while not (0 <= y_new < grid and 0 <= x_new < grid) and tries < 50:
            offset = rng.uniform(-turn_limit_deg, turn_limit_deg)
            heading_deg_new = (heading_deg + offset) % 360
            heading_rad = np.deg2rad(heading_deg_new)
            y_new = y + step_px * np.sin(heading_rad)
            x_new = x + step_px * np.cos(heading_rad)
            tries += 1
        if tries == 50:   # 더 이상 유효한 방향 못 찾으면 종료
            break

        # 다음 스텝 확정
        pts.append((y_new, x_new))
        y, x = y_new, x_new
        heading_deg = heading_deg_new

    pts_int = np.round(pts).astype(int)
    pts_int = np.clip(pts_int, 0, grid-1)   # y,x 모두 클립
    return pts_int

# -------------------------------------------------
# 1) sparse_from_waypoints
# -------------------------------------------------
def sparse_from_waypoints(field: np.ndarray,
                          waypoints: np.ndarray):
    """
    field      : (H, W) 정답 방사선 필드
    waypoints  : (N, 2) int  배열  (row=y, col=x)
    
    반환
      r_meas : waypoint 위치에만 실제 값, 나머지 0
      mask   : waypoint 위치 1, 나머지 0
    """
    r_meas = np.zeros_like(field, dtype=np.float32)
    mask   = np.zeros_like(field, dtype=np.uint8)

    # waypoint 좌표마다 값 할당
    for y, x in waypoints:
        r_meas[y, x] = field[y, x]
        mask[y, x]   = 1
    return r_meas, mask


# -------------------------------------------------
# 2) visualize_sparse
# -------------------------------------------------
def visualize_sparse(field: np.ndarray,
                     waypoints: np.ndarray,
                     r_meas: np.ndarray,
                     mask: np.ndarray):
    """
    두 subplot:
      (A) 정답 필드 + 경로(선)
      (B) sparse 측정 시각화
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=110)

    # (A) Ground-truth + Trajectory
    ax[0].imshow(field, cmap='hot', origin='lower')
    ax[0].plot(waypoints[:, 1], waypoints[:, 0],
               '-o', c='lime', lw=1.3, ms=3, label='Path')
    ax[0].set_title("Ground-truth Field & Trajectory")
    ax[0].legend(frameon=False)
    ax[0].axis('off')

    # (B) Sparse 측정 표시
    background = np.ma.masked_where(mask, field)      # waypoint 제외 배경
    sparse_vis = np.ma.masked_where(mask == 0, r_meas)

    im = ax[1].imshow(r_meas, cmap='hot', origin='lower')    
    ax[1].plot(waypoints[:, 1], waypoints[:, 0],
               c='white', lw=0.6, alpha=0.35)          # 얇은 경로
    ax[1].set_title(f"Sparse Measurements ({mask.sum()} pts)")
    ax[1].axis('off')
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03,
                 label='Measured intensity')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- 단일 샘플 생성 ---
    coords, amps, sigmas = gt.sample_sources(GRID, N_SOURCES)
    field = gt.gaussian_field(GRID, coords, amps, sigmas)
    # ---------- 예시 실행 ----------
    # waypoint 생성
    wps = generate_waypoints()
    r_meas, mask = sparse_from_waypoints(field, wps)
    visualize_sparse(field, wps, r_meas, mask)