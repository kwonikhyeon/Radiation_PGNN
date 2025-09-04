# 희소 측정 기반 **평가 전용(Eval-Only)** 불확실성 맵
**범위**: 10 m × 10 m 실내 정사각형 공간, 정적(고정) 벽, 희소한 방사선 측정.  
**목표**: 현재 네트워크의 예측 \(\hat{D}(\mathbf{x})\)와 희소 측정만을 이용해 **학습 수정 없이(eval 단계에서만)** 전역 **불확실성 맵** \(\sigma(\mathbf{x})\)을 구축한다.

---

## 표기
- \(\mathbf{x}=(x,y)\in[0,L]^2\): 공간 좌표 (그리드 해상도는 임의)
- \(S=\{s_k\}\): 측정 지점 집합
- \(M(\mathbf{x})\in\{0,1\}\): 측정 마스크(측정 지점=1, 그 외=0)
- \(\hat{D}(\mathbf{x})\): 모델이 예측한 선량률
- \(D_{\text{meas}}(\mathbf{x})\): 측정된 선량률(측정이 없으면 0)
- \(G_h(\mathbf{r})=\exp(-\|\mathbf{r}\|^2/(2h^2))\): 2D 가우시안 커널(필요 시 정규화)
- \(*\): 2D 컨볼루션, \(\varepsilon\): 아주 작은 수 \((10^{-6}\sim10^{-8})\)

---

## 방법 개요
두 개의 상보적인 신호를 결합한다:
1. **커버리지(Coverage)** — 주변에 측정이 많을수록 불확실성이 낮다.  
2. **잔차 전파(Residual Propagation)** — 측정과 예측의 불일치가 큰 지점 주변은 더 불확실하다.

최종 불확실성은 가중합으로 정의한다:
\[
\sigma(\mathbf{x}) \;=\; a\,U_{\text{cov}}(\mathbf{x}) \;+\; b\,U_{\text{res}}(\mathbf{x}).
\]

---

## 1) 커버리지 기반 불확실성 \(U_{\text{cov}}\)
가우시안 KDE로 국소 측정 밀도를 계산한다:
\[
\alpha(\mathbf{x}) \;=\; (M * G_{h})(\mathbf{x}), 
\qquad
U_{\text{cov}}(\mathbf{x}) \;=\; \frac{1}{\alpha(\mathbf{x})+\varepsilon}.
\]
- **직관**: \(\mathbf{x}\) 주변에 측정이 많으면 \(\alpha\)가 커져 \(U_{\text{cov}}\)가 작아진다(확실↑).  
- **대역폭 \(h\)**: 측정점들 간 **최근접 거리의 중앙값**으로 시작. 측정이 듬성듬성하면 \(h\)를 키우고, 조밀하면 줄인다.

> *대안/보조*: 최근접 측정점까지의 거리 \(d(\mathbf{x})\)로  
> \(U_{\text{dist}}(\mathbf{x})=1-\exp(-d(\mathbf{x})^2/(2h^2))\).  
> 단, 여러 점의 누적효과를 자연스럽게 반영하는 **KDE 기반 \(U_{\text{cov}}\)**가 일반적으로 더 적합하다.

---

## 2) 잔차 전파 \(U_{\text{res}}\)
측정 지점에서의 **예측–실측 절대오차**를 주변으로 퍼뜨린다:
\[
R(\mathbf{x}) \;=\; \big|\hat{D}(\mathbf{x})-D_{\text{meas}}(\mathbf{x})\big|\cdot M(\mathbf{x}),
\]
\[
U_{\text{res}}(\mathbf{x}) \;=\;
\frac{(R * G_{h_r})(\mathbf{x})}{(M * G_{h_r})(\mathbf{x})+\varepsilon}.
\]
- **직관**: 모델이 측정과 어긋난 지점 주변은 신뢰도가 낮다(불확실↑).  
- **커널 \(h_r\)**: 보통 \(h_r\approx h\)로 시작한다.

---

## 3) 최종 불확실성 \(\sigma(\mathbf{x})\) 및 정보이득(선택)
\[
\sigma(\mathbf{x}) \;=\; a\,U_{\text{cov}}(\mathbf{x}) \;+\; b\,U_{\text{res}}(\mathbf{x}).
\]
- 초기 가중치 권장: \(a=0.7,\; b=0.3\).  
- 수치적 안정성을 위해 상위 퍼센타일(예: 99%)로 **클리핑** 후 \([0,1]\) **정규화**를 권장.

센서 노이즈 분산 \(\sigma_n^2\) 추정치가 있다면, **정보이득(Information Gain)**으로 변환할 수 있다:
\[
I(\mathbf{x}) \;=\; \tfrac12 \log\!\Big(1+\tfrac{\sigma(\mathbf{x})^2}{\sigma_n^2}\Big).
\]

---

## 실무 파라미터 튜닝
- **\(h\)**: \(S\)의 최근접거리 **중앙값**으로 시작 → 점이 띄엄띄엄이면 ↑, 과도평활이면 ↓.  
- **\(h_r\)**: 일단 \(h\)와 동일하게 두고, 잔차 헤일로가 너무 넓으면 조금 줄인다.  
- **\(a:b\)**: 데이터가 균질하고 전반적으로 잘 맞으면 \(a↑\); 특정 구역 문제(불일치)가 잦으면 \(b↑\).  
- **\(\varepsilon\)**: \(10^{-6}\sim10^{-8}\).

---

## PyTorch 참조 구현(Eval 전용)
```python
import torch
import torch.nn.functional as F

def gaussian_kernel2d(kernel_size: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
    k = k / (k.sum() + 1e-8)
    return k[None, None]  # [1,1,K,K]

def uncertainty_eval_ko(
    pred: torch.Tensor,          # [1,1,H,W] 예측 선량
    meas: torch.Tensor,          # [1,1,H,W] 측정 선량 (없으면 0)
    meas_mask: torch.Tensor,     # [1,1,H,W] {0,1}
    h: float, h_r: float,        # 커널 표준편차 (픽셀/미터 단위 일관 필요)
    a: float = 0.7, b: float = 0.3,
    clip_p: float = 99.0
):
    B, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    # 커널 크기 ~ 6*sigma (홀수 보정)
    def ksize(sig):
        k = int(6*sig) | 1
        return max(k, 3)

    Kc = gaussian_kernel2d(ksize(h),    h,   device=device, dtype=dtype)
    Kr = gaussian_kernel2d(ksize(h_r),  h_r, device=device, dtype=dtype)

    # Coverage
    alpha = F.conv2d(meas_mask, Kc, padding=Kc.shape[-1]//2)
    U_cov = 1.0 / (alpha + 1e-6)

    # Residual propagation
    R = (pred - meas).abs() * meas_mask
    num = F.conv2d(R,         Kr, padding=Kr.shape[-1]//2)
    den = F.conv2d(meas_mask, Kr, padding=Kr.shape[-1]//2)
    U_res = num / (den + 1e-6)

    # Weighted sum
    sigma = a*U_cov + b*U_res

    # 퍼센타일 클리핑 & [0,1] 정규화
    q = torch.quantile(sigma.flatten(), clip_p/100.0)
    sigma = torch.clamp(sigma, max=q)
    sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)
    return sigma  # [1,1,H,W]
```

---

## 비고
- 본 절차는 **추론/평가 단계에서만** 수행되며, 모델 구조나 학습 파이프라인을 변경하지 않는다.  
- 시간 여유가 있다면, 소량의 **입력 노이즈 TTA 표준편차**를 \((+\,c\,U_{\text{tta}},\; c\approx0.1)\) 형태로 더해 **입력 견고성**을 보조적으로 반영할 수 있다.  
- \(\sigma\)가 높은 셀을 **탐사 후보 지점**으로 사용하거나, 필요 시 위의 \(I(\mathbf{x})\)로 변환하여 정보이론 기반 스코어링에 활용한다.
