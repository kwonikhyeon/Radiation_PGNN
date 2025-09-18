
# 위험도 / 정보 이득 / 주행 가능성 레이어 — 수식적 정의
*(예측된 방사선 필드 맵과 불확실성 맵을 보유한 상태에서 시작)*  
*버전: 2025-09-08 (KST)*

## 기호 및 입력값
- 격자 영역: \(\Omega \subset \mathbb{R}^2\), 픽셀 \(x \in \Omega\).
- 예측 방사선 필드(밀집): \(\hat{r}(x)\)  \([\text{counts}/\text{s}]\).
- 픽셀 단위 불확실성: \(\sigma_{\hat{r}}(x)\)  (표준편차, 또는 불확실성 추정치).
- 로봇 상태: 현재 위치와 방향 \((x_0, \theta_0)\).
- 선택적 보조 맵: 장애물 거리 변환 \(d_{\text{obs}}(x)\), 지역 속도 맵 \(v(x)\in(0, v_{\max}]\).

정규화를 위해 스칼라 필드 \(f(x)\)에 대해 다음과 같이 정의:
\[
\operatorname{norm}(f)(x) \;\triangleq\; \frac{f(x)-\min_{\Omega} f}{\max_{\Omega} f - \min_{\Omega} f + \varepsilon},
\qquad \varepsilon \approx 10^{-8}.
\]

---

## 1) 위험도 레이어 \(L_r(x)\)
**의도.** 방사선이 낮은(안전한) 지역을 더 선호하도록 가중치 부여.  

**정의 (슬라이드 수식과 일치):**
\[
L_r(x) \;=\; 1 - \operatorname{norm}\!\big(\hat{r}\big)(x).
\]
- \(\hat{r}(x)\) 값이 클수록 → \(L_r(x)\) 낮음 (위험, 회피 권장).
- \(\hat{r}(x)\) 값이 작을수록 → \(L_r(x)\) 높음 (안전, 선호).

**선택적 노출 제한 (소프트 컷):**
\[
L_r^{\text{clamp}}(x) \;=\; L_r(x)\cdot \sigma\!\big(\kappa(\tau_E-\hat{r}(x))\big),
\]
여기서 \(\sigma(\cdot)\)는 로지스틱 함수, \(\tau_E\)는 노출 한계치, \(\kappa>0\)는 급격도.

---

## 2) 정보 이득 레이어 \(L_i(x)\)
**의도.** 불확실성이 크거나 필드 변화가 큰 영역을 측정 우선순위로 배치.

### 2.1 예측 강도 기반 게이팅
임계값 \(\tau_r\)에 대해 부드러운 게이트 \(g(x)\):
\[
g(x) \;=\; \sigma\!\big(\gamma(\hat{r}(x)-\tau_r)\big), \qquad \gamma>0.
\]
그 후, 그래디언트 기반과 불확실성 기반을 결합:
\[
L_i(x) \;=\; \operatorname{norm}\!\big(\,\alpha_g\,\lVert\nabla \hat{r}(x)\rVert_2\big)\cdot g(x)
\;+\; \operatorname{norm}\!\big(\,\alpha_u\,\sigma_{\hat{r}}(x)\big)\cdot (1-g(x)),
\]
\(\alpha_g,\alpha_u>0\)는 가중치.

### 2.2 구현 팁
- \(\nabla \hat{r}\)는 Sobel 필터로 계산, 가우시안 블러로 잡음 완화.
- 불확실성 항은 Mutual Information 근사 등으로 대체 가능하지만, \(\sigma_{\hat{r}}\)가 단순·견고.

---

## 3) 주행 가능성 레이어 \(L_t(x)\)
**의도.** 현재 위치와 방향에서 효율적인 전진을 선호.

**슬라이드 기반 정의:**
\[
L_t(x) \;=\; \exp\!\Big(\alpha\,\lVert x-x_0\rVert_2 \;-\; \beta\,\big|\operatorname{wrap}_{\pi}(\angle(x-x_0)-\theta_0)\big| \Big),
\]
\(\alpha,\beta\ge 0\). 전진 거리가 길고 회전이 적을수록 값이 큼.

**경계화된 형태 (안정적 [0,1] 값):**
\[
L_t^{\star}(x) \;=\; 
\exp\!\Big(-\,\underbrace{\lambda_d\,\tfrac{\lVert x-x_0\rVert_2}{d_{\max}}}_{\text{이동 비용}} 
\;-\; \underbrace{\lambda_\theta\,\tfrac{|\operatorname{wrap}_{\pi}(\angle(x-x_0)-\theta_0)|}{\pi}}_{\text{회전 비용}}
\;-\; \underbrace{\lambda_o\,\operatorname{norm}\!\big(1/d_{\text{obs}}(x)\big)}_{\text{장애물 근접도}} 
\;-\; \underbrace{\lambda_v\,\operatorname{norm}\!\big(1/v(x)\big)}_{\text{속도 제한 지역}}
\Big).
\]

---

## 4) 가중 맵 및 경로 점수
세 레이어를 결합 (경로 선택 시 최대화):
\[
W(x) \;=\; \lambda_r\,L_r(x) \;+\; \lambda_i\,L_i(x) \;+\; \lambda_t\,L_t(x),
\qquad \lambda_r+\lambda_i+\lambda_t=1.
\]

**경로 점수 (n-step)**, \(P=\{x_1,\dots,x_n\}\):
\[
J(P) \;=\; \sum_{k=1}^{n} W(x_k),
\quad \text{s.t.} \quad 
\sum_{k=1}^{n} \hat{r}(x_k)\,\Delta t \le E_{\max},\;\; 충돌 없음.
\]

---

## 5) 파라미터 예시
- \(\gamma=5\sim20\), \(\tau_r\): \(\hat{r}\)의 60–80 분위값.
- \(\alpha_g=1,\alpha_u=1\).  
  \(\lambda_r=0.4,\;\lambda_i=0.4,\;\lambda_t=0.2\) (균등 혹은 목적에 따라 조정).
- \(L_t\) bounded form: \(\lambda_d=0.2\sim0.6,\;\lambda_\theta=0.4\sim0.8,\;\lambda_o=0.3\sim0.7\).

---

## 6) 간단한 의사코드
```python
# 입력: r_hat, sigma_r, x0, theta0, params
r_norm   = norm(r_hat)
L_r      = 1.0 - r_norm

g        = sigmoid(gamma * (r_hat - tau_r))
grad_mag = sobel_norm(r_hat)                # ||∇r_hat||_2
L_i      = norm(alpha_g*grad_mag) * g + norm(alpha_u*sigma_r) * (1 - g)

L_t      = exp(alpha*dist(x, x0) - beta*abs(wrap(angle(x-x0)-theta0)))
# 또는 L_t = bounded_traversability(...)

W        = lam_r*L_r + lam_i*L_i + lam_t*L_t
# 가중 맵을 이용해 RRT 후보 경로를 평가.
```

---

## 참고
- 위 수식은 PDF 제안서 슬라이드【19†Thesis Proposal.Print.pdf†L23-L27】의 **Risk, Information Gain, Traversability Layer** 정의를 기반으로 구현 관점에서 보완한 것임.
