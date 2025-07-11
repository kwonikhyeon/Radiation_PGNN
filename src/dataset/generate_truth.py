import numpy as np
import matplotlib.pyplot as plt

# ------------------------- 파라미터 --------------------------
GRID = 256                   # 256×256 맵(10m x 10m 기준 0.04m/픽셀)
N_SOURCES_RANGE = (1, 4)     # 소스 개수 (1~4개 중 랜덤 선택)
INTENSITY_RANGE = (30, 120)  # 그림과 유사하게 강한 중앙(색바)
SIGMA_RANGE = (18.0, 30.0)    # σ: 넓게 퍼질수록 값 증가
SEED = None                  # None으로 설정하면 매번 다른 랜덤 시드 사용
# -------------------------------------------------------------

rng = np.random.default_rng(SEED)
N_SOURCES = rng.integers(N_SOURCES_RANGE[0], N_SOURCES_RANGE[1] + 1)  # 1~4개 중 랜덤 선택

def sample_sources(grid, n):
    """넓게 퍼지는 블롭용 파라미터 샘플링"""
    coords, amps, sigmas = [], [], []
    # 메인 rng와 독립적인 랜덤 생성기 사용 (매번 다른 결과)
    local_rng = np.random.default_rng()
    while len(coords) < n:
        y, x = local_rng.integers(0.15*grid, 0.85*grid, size=2)
        if all(np.hypot(y-yy, x-xx) > 0.25*grid for yy, xx in coords):
            coords.append((y, x))
            amps.append(local_rng.uniform(*INTENSITY_RANGE))
            sigmas.append(local_rng.uniform(*SIGMA_RANGE))   # ▲ 훨씬 큰 σ
    return np.array(coords), np.array(amps), np.array(sigmas)

def gaussian_field(grid, coords, amps, sigmas):
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    field = np.zeros((grid, grid), np.float32)
    for (y0, x0), A, s in zip(coords, amps, sigmas):
        d2 = (yy - y0) ** 2 + (xx - x0) ** 2
        field += A * np.exp(-d2 / (2 * s**2))
    return field

def get_param():
    """설정된 파라미터들을 딕셔너리로 반환"""
    return {
        'GRID': GRID,
        'N_SOURCES_RANGE': N_SOURCES_RANGE,
        'INTENSITY_RANGE': INTENSITY_RANGE,
        'SIGMA_RANGE': SIGMA_RANGE,
        'SEED': SEED,
        'N_SOURCES': N_SOURCES  # 현재 생성된 소스 개수
    }


if __name__ == "__main__":
    # --- 단일 샘플 생성 ---
    coords, amps, sigmas = sample_sources(GRID, N_SOURCES)
    field = gaussian_field(GRID, coords, amps, sigmas)
    
    # 생성된 필드와 소스 시각화
    plt.figure(figsize=(6, 6))
    im = plt.imshow(field, cmap='hot', origin='lower')
    plt.scatter(coords[:, 1], coords[:, 0], c='lime', marker='o', s=60, label="Sources")
    plt.colorbar(im, label="Radiation Intensity (MeV scale)")
    plt.title("Generated Radiation Field with Sources")
    plt.legend()
    plt.axis("off")
    plt.show()